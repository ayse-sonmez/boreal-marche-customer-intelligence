import json
import math
import platform
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .rag.agent import AgentRequest, run_agent

MODEL_PATH = Path("artifacts/model.joblib")
META_PATH = Path("artifacts/model_meta.json")
RULES_PATH = Path("artifacts/rules.json")

FEATURES_PATH = Path(
    "data/processed/customer_features_gold.parquet"
)

PREDICTION_LOG_DB = Path(
    "data/prediction_logs.db"
)

HTTP_METRICS_DB = Path("data/http_metrics.db")
TRAIN_FEATURES_PATH = Path("data/processed/customer_features_train.parquet")
TEST_FEATURES_PATH = Path("data/processed/customer_features_test.parquet")
MIN_LIVE_CUSTOMERS = 30
MIN_LABELED_CUSTOMERS = 30
_metrics_lock = Lock()


model = None
model_meta = None
rules = None
customer_features = None

request_count = 0
prediction_count = 0
recommendation_count = 0
chat_count = 0


# ---------------------------------------------------------
# Request / response models
# ---------------------------------------------------------


class CustomerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: int


class PredictionResponse(BaseModel):
    customer_id: int
    return_probability: float
    threshold: float
    at_risk: bool


class WhatIfRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: int
    recency_days: int | None = Field(default=None, ge=0)
    total_orders: int | None = Field(default=None, ge=1)
    total_spend: float | None = Field(default=None, ge=0)


class WhatIfResponse(BaseModel):
    customer_id: int
    original_probability: float
    scenario_probability: float
    threshold: float
    original_at_risk: bool
    scenario_at_risk: bool
    changed_features: dict[str, float | int]


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_ids: list[int]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    basket: list[str]

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class RecommendationItem(BaseModel):
    product: str
    because_product: str
    confidence: float
    lift: float


class RecommendationResponse(BaseModel):
    strategy: str
    recommendations: list[RecommendationItem]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    provider: str = "rule"


class ChatResponse(BaseModel):
    answer: str
    tool_used: str
    citations: list[str]
    refused: bool
    human_handoff: bool


# ---------------------------------------------------------
# Prediction logging
# ---------------------------------------------------------


def initialise_prediction_log_db():
    """
    Create the local prediction log database and table.
    """

    PREDICTION_LOG_DB.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(
        PREDICTION_LOG_DB
    ) as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                customer_id INTEGER NOT NULL,
                return_probability REAL NOT NULL,
                threshold REAL NOT NULL,
                at_risk INTEGER NOT NULL,
                model_version TEXT
            )
            """
        )


def log_prediction(
    prediction: PredictionResponse,
):
    """
    Store one structured prediction record.
    """

    with sqlite3.connect(
        PREDICTION_LOG_DB
    ) as connection:

        connection.execute(
            """
            INSERT INTO prediction_logs (
                timestamp_utc,
                customer_id,
                return_probability,
                threshold,
                at_risk,
                model_version
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(
                    UTC
                ).isoformat(),
                prediction.customer_id,
                prediction.return_probability,
                prediction.threshold,
                int(prediction.at_risk),
                model_meta.get(
                    "model_version"
                ),
            ),
        )


# ---------------------------------------------------------
# Persistent HTTP monitoring
# ---------------------------------------------------------


def initialise_http_metrics_db():
    """Keep HTTP observations across container restarts."""
    HTTP_METRICS_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(HTTP_METRICS_DB, timeout=5) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS http_request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                method TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                duration_ms REAL NOT NULL
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_http_request_time
            ON http_request_logs(timestamp_utc)
        """)


def record_http_request(method, endpoint, status_code, duration_ms):
    """Store metadata only: no request bodies, customer IDs or IPs."""
    try:
        with _metrics_lock, sqlite3.connect(HTTP_METRICS_DB, timeout=5) as connection:
                connection.execute(
                    """INSERT INTO http_request_logs
                       (timestamp_utc, method, endpoint, status_code, duration_ms)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        datetime.now(UTC).isoformat(),
                        method,
                        endpoint,
                        int(status_code),
                        round(float(duration_ms), 3),
                    ),
                )
    except sqlite3.Error as error:
        # Monitoring failures must not turn successful API calls into errors.
        print(f"HTTP monitoring write failed: {error}")


def http_metrics_summary():
    """Aggregate all recorded requests, including completed SSE streams."""
    with sqlite3.connect(HTTP_METRICS_DB, timeout=5) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("""
            SELECT method, endpoint, status_code, duration_ms
            FROM http_request_logs
        """).fetchall()

    count = len(rows)
    durations = sorted(float(row["duration_ms"]) for row in rows)
    errors = sum(int(row["status_code"]) >= 400 for row in rows)
    server_errors = sum(int(row["status_code"]) >= 500 for row in rows)
    endpoints = {}
    for row in rows:
        key = f'{row["method"]} {row["endpoint"]}'
        entry = endpoints.setdefault(key, {
            "requests": 0, "errors": 0, "total_latency_ms": 0.0
        })
        entry["requests"] += 1
        entry["errors"] += int(row["status_code"] >= 400)
        entry["total_latency_ms"] += float(row["duration_ms"])

    endpoint_stats = {}
    for key, entry in sorted(endpoints.items()):
        endpoint_stats[key] = {
            "requests": entry["requests"],
            "errors": entry["errors"],
            "avg_latency_ms": round(
                entry["total_latency_ms"] / entry["requests"], 2
            ),
        }

    return {
        "http_requests_total": count,
        "http_errors_total": errors,
        "http_server_errors_total": server_errors,
        "http_error_rate_pct": round(100 * errors / count, 2) if count else 0.0,
        "http_avg_latency_ms": round(sum(durations) / count, 2) if count else None,
        "http_p95_latency_ms": (
            round(durations[max(0, math.ceil(0.95 * count) - 1)], 2)
            if count else None
        ),
        "http_by_endpoint": endpoint_stats,
        "http_metrics_scope": "All recorded HTTP requests except GET /metrics; all-time since logging enabled",
        "http_latency_scope": "Full response duration, including completed SSE streaming",
    }


class HttpMetricsMiddleware:
    """ASGI middleware measures through the final response body event."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")
        # Exclude metrics reads so dashboards do not inflate their own totals.
        if path == "/metrics" and method == "GET":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_code = 500
        completed = False

        async def observed_send(message):
            nonlocal status_code, completed
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
            if (message["type"] == "http.response.body"
                    and not message.get("more_body", False)):
                completed = True

        try:
            await self.app(scope, receive, observed_send)
        except Exception:
            status_code = 500
            raise
        finally:
            # A broken/incomplete stream is an operational failure even if
            # HTTP 200 headers were already sent.
            if not completed:
                status_code = 500
            duration_ms = (time.perf_counter() - started) * 1000
            record_http_request(method, path, status_code, duration_ms)


# ---------------------------------------------------------
# Load artifacts once
# ---------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):

    global model
    global model_meta
    global rules
    global customer_features

    print("Loading API artifacts...")

    model = joblib.load(
        MODEL_PATH
    )

    model_meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    rules_data = json.loads(RULES_PATH.read_text(encoding="utf-8"))

    rules = rules_data[
        "rules"
    ]

    customer_features = pd.read_parquet(
        FEATURES_PATH
    )

    initialise_prediction_log_db()
    initialise_http_metrics_db()
    initialise_drift_tables()

    print("API artifacts loaded.")

    yield

    print("API shutting down.")


app = FastAPI(
    title="BorÃ©al MarchÃ© Customer Intelligence API",
    description=(
        "API for retention prediction, "
        "product recommendation and "
        "bilingual customer assistance."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(HttpMetricsMiddleware)


# ---------------------------------------------------------
# Helper: prediction
# ---------------------------------------------------------


def predict_customer(
    customer_id: int,
) -> PredictionResponse:

    customer = customer_features[
        customer_features["Customer ID"]
        == customer_id
    ].copy()

    if customer.empty:
        raise HTTPException(
            status_code=404,
            detail="Customer ID not found.",
        )

    X = customer.drop(
        columns=[
            "Customer ID",
            "label_90d",
            "first_purchase_date",
            "last_purchase_date",
        ],
        errors="ignore",
    )

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    threshold_value = model_meta.get(
        "threshold",
        0.73,
    )

    # Support either a direct numeric threshold
    # or a dictionary stored in metadata.
    if isinstance(
        threshold_value,
        dict,
    ):
        threshold = float(
            threshold_value.get(
                "value",
                0.73,
            )
        )
    else:
        threshold = float(
            threshold_value
        )

    prediction = PredictionResponse(
        customer_id=customer_id,
        return_probability=probability,
        threshold=threshold,
        at_risk=(
            probability < threshold
        ),
    )

    log_prediction(
        prediction
    )
    log_feature_snapshot(customer_id, X.iloc[0])

    return prediction


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ---------------------------------------------------------
# Readiness
# ---------------------------------------------------------


@app.get("/ready")
def ready():

    ready_state = all(
        item is not None
        for item in [
            model,
            model_meta,
            rules,
            customer_features,
        ]
    )

    if not ready_state:
        raise HTTPException(
            status_code=503,
            detail="API is not ready.",
        )

    return {
        "status": "ready"
    }


# ---------------------------------------------------------
# Version
# ---------------------------------------------------------


@app.get("/version")
def version():

    return {
        "api_version": "1.0.0",
        "model_version": (
            model_meta.get(
                "model_version"
            )
        ),
        "git_commit": (
            model_meta.get(
                "git_commit"
            )
        ),
        "python_version": (
            platform.python_version()
        ),
    }


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: CustomerRequest,
):

    global request_count
    global prediction_count

    request_count += 1
    prediction_count += 1

    return predict_customer(
        request.customer_id
    )


# ---------------------------------------------------------
# What-if prediction (scenario only; not logged as live traffic)
# ---------------------------------------------------------


@app.post(
    "/predict/what-if",
    response_model=WhatIfResponse,
)
def predict_what_if(request: WhatIfRequest):
    global request_count
    request_count += 1

    customer = customer_features.loc[
        customer_features["Customer ID"] == request.customer_id
    ]
    if customer.empty:
        raise HTTPException(status_code=404, detail="Customer ID not found.")

    X = customer.drop(
        columns=[
            "Customer ID",
            "label_90d",
            "first_purchase_date",
            "last_purchase_date",
        ],
        errors="ignore",
    ).copy()

    # Keep exactly the columns and order used during training.
    X = X.loc[:, list(model.feature_names_in_)]
    scenario = X.copy()
    changes = {}

    if request.recency_days is not None:
        scenario.loc[:, "recency_days"] = request.recency_days
        changes["recency_days"] = request.recency_days

    if request.total_orders is not None:
        scenario.loc[:, "total_orders"] = request.total_orders
        changes["total_orders"] = request.total_orders

    if request.total_spend is not None:
        scenario.loc[:, "total_spend"] = request.total_spend
        changes["total_spend"] = request.total_spend

    # Recompute direct derived features when their inputs change.
    if request.total_orders is not None or request.total_spend is not None:
        orders = float(scenario["total_orders"].iloc[0])
        spend = float(scenario["total_spend"].iloc[0])
        active_months = float(scenario["active_months"].iloc[0])
        products = float(scenario["unique_products"].iloc[0])

        scenario.loc[:, "average_order_value"] = spend / orders
        if active_months > 0:
            scenario.loc[:, "orders_per_active_month"] = orders / active_months
            scenario.loc[:, "spend_per_active_month"] = spend / active_months
        if products > 0:
            scenario.loc[:, "spend_per_product"] = spend / products

    original_probability = float(model.predict_proba(X)[0, 1])
    scenario_probability = float(model.predict_proba(scenario)[0, 1])

    threshold_value = model_meta.get("threshold", 0.73)
    if isinstance(threshold_value, dict):
        threshold = float(threshold_value.get("value", 0.73))
    else:
        threshold = float(threshold_value)

    return WhatIfResponse(
        customer_id=request.customer_id,
        original_probability=original_probability,
        scenario_probability=scenario_probability,
        threshold=threshold,
        original_at_risk=original_probability < threshold,
        scenario_at_risk=scenario_probability < threshold,
        changed_features=changes,
    )


# ---------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(
    request: BatchPredictionRequest,
):

    global request_count
    global prediction_count

    request_count += 1

    predictions = []

    for customer_id in request.customer_ids:

        prediction = predict_customer(
            customer_id
        )

        predictions.append(
            prediction
        )

        prediction_count += 1

    return BatchPredictionResponse(
        predictions=predictions
    )


# ---------------------------------------------------------
# Recommendation
# ---------------------------------------------------------


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(
    request: RecommendationRequest,
):

    global request_count
    global recommendation_count

    request_count += 1
    recommendation_count += 1

    basket = {
        str(product)
        for product
        in request.basket
    }

    candidates = {}

    for rule in rules:

        antecedent = str(
            rule["antecedent"]
        )

        consequent = str(
            rule["consequent"]
        )

        if antecedent not in basket:
            continue

        if consequent in basket:
            continue

        current = candidates.get(
            consequent
        )

        if (
            current is None
            or rule["lift"]
            > current["lift"]
        ):
            candidates[
                consequent
            ] = rule

    ranked = sorted(
        candidates.values(),
        key=lambda rule: (
            rule["lift"],
            rule["confidence"],
        ),
        reverse=True,
    )

    recommendation_items = []

    for rule in ranked[
        :request.top_k
    ]:

        recommendation_items.append(
            RecommendationItem(
                product=str(
                    rule[
                        "consequent"
                    ]
                ),
                because_product=str(
                    rule[
                        "antecedent"
                    ]
                ),
                confidence=float(
                    rule[
                        "confidence"
                    ]
                ),
                lift=float(
                    rule["lift"]
                ),
            )
        )

    strategy = (
        "association_rules"
        if recommendation_items
        else "no_matching_rule"
    )

    return RecommendationResponse(
        strategy=strategy,
        recommendations=(
            recommendation_items
        ),
    )


# ---------------------------------------------------------
# Chat
# ---------------------------------------------------------


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):

    global request_count
    global chat_count

    request_count += 1
    chat_count += 1

    result = run_agent(
        AgentRequest(
            question=(
                request.question
            ),
            provider=(
                request.provider
            ),
        )
    )

    return ChatResponse(
        answer=result.answer,
        tool_used=(
            result.tool_used
        ),
        citations=(
            result.citations
        ),
        refused=(
            result.refused
        ),
        human_handoff=(
            result.human_handoff
        ),
    )


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------




# ---------------------------------------------------------
# Streaming chat (SSE)
# ---------------------------------------------------------


def _sse_event(event_name: str, payload: dict) -> str:
    """Encode one UTF-8-safe Server-Sent Event."""
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Stream a completed agent response in chunks over SSE.

    This streams delivery, not token-by-token model generation.
    Existing /chat behavior remains unchanged.
    """
    global request_count
    global chat_count

    request_count += 1
    chat_count += 1

    # Execute before opening the stream so ordinary errors are
    # surfaced as HTTP errors rather than a truncated SSE response.
    result = run_agent(
        AgentRequest(question=request.question, provider=request.provider)
    )

    def events():
        yield _sse_event("start", {"provider": request.provider})

        # Preserve spaces and punctuation when joining streamed chunks.
        # Chunking is only for transport/presentation, not LLM generation.
        answer = result.answer
        chunk_size = 32
        for offset in range(0, len(answer), chunk_size):
            yield _sse_event(
                "delta", {"text": answer[offset:offset + chunk_size]}
            )

        yield _sse_event(
            "done",
            {
                "tool_used": result.tool_used,
                "citations": result.citations,
                "refused": result.refused,
                "human_handoff": result.human_handoff,
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/metrics")
def metrics():

    return {
        "requests_total": (
            request_count
        ),
        "predictions_total": (
            prediction_count
        ),
        "recommendations_total": (
            recommendation_count
        ),
        "chat_requests_total": (
            chat_count
        ),
        **http_metrics_summary(),
    }


# ---------------------------------------------------------
# Model drift and delayed-label monitoring
# ---------------------------------------------------------

def initialise_drift_tables():
    """Add monitoring tables without modifying historical prediction rows."""
    with sqlite3.connect(PREDICTION_LOG_DB, timeout=10) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS prediction_feature_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                customer_id INTEGER NOT NULL,
                features_json TEXT NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS delayed_labels (
                customer_id INTEGER PRIMARY KEY,
                label_90d INTEGER NOT NULL CHECK(label_90d IN (0, 1)),
                observed_at_utc TEXT NOT NULL,
                source TEXT NOT NULL
            )
        """)


def log_feature_snapshot(customer_id, row):
    """Store only the model inputs; no label or what-if scenario is included."""
    features = {}
    for name in model.feature_names_in_:
        value = row[name]
        if pd.isna(value):
            features[name] = None
        elif isinstance(value, (int, float, np.integer, np.floating)):
            features[name] = float(value)
        else:
            features[name] = str(value)
    with sqlite3.connect(PREDICTION_LOG_DB, timeout=10) as db:
        db.execute(
            """INSERT INTO prediction_feature_snapshots
               (timestamp_utc, customer_id, features_json) VALUES (?, ?, ?)""",
            (datetime.now(UTC).isoformat(), int(customer_id),
             json.dumps(features, allow_nan=False)),
        )


def population_stability_index(reference, observed, categorical=False):
    """PSI using reference quantile bins or categorical frequencies."""
    ref = pd.Series(reference).dropna()
    obs = pd.Series(observed).dropna()
    if ref.empty or obs.empty:
        return None
    if categorical:
        # Preserve top categories; group remaining values into Other.
        top = set(ref.astype(str).value_counts().head(10).index)
        a = ref.astype(str).map(lambda x: x if x in top else "Other")
        b = obs.astype(str).map(lambda x: x if x in top else "Other")
        categories = sorted(set(a.unique()) | set(b.unique()))
        expected = np.array([(a == c).mean() for c in categories])
        actual = np.array([(b == c).mean() for c in categories])
    else:
        a = pd.to_numeric(ref, errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(obs, errors="coerce").dropna().to_numpy(dtype=float)
        if not len(a) or not len(b):
            return None
        edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
        if len(edges) < 2:
            # Constant baseline: report a shift if observations differ.
            return 0.0 if np.all(b == a[0]) else None
        edges[0] = -np.inf
        edges[-1] = np.inf
        expected = np.histogram(a, bins=edges)[0] / len(a)
        actual = np.histogram(b, bins=edges)[0] / len(b)
    # Smoothing avoids log(0) while keeping scores finite.
    eps = 1e-4
    expected = np.maximum(expected, eps)
    actual = np.maximum(actual, eps)
    return round(float(np.sum((actual - expected) * np.log(actual / expected))), 4)


def ks_separation(y_true, probabilities):
    """Two-sample KS separation between positive and negative model scores."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    pos = np.sort(scores[y == 1])
    neg = np.sort(scores[y == 0])
    if not len(pos) or not len(neg):
        return None
    thresholds = np.sort(np.unique(scores))
    pos_cdf = np.searchsorted(pos, thresholds, side="right") / len(pos)
    neg_cdf = np.searchsorted(neg, thresholds, side="right") / len(neg)
    return round(float(np.max(np.abs(pos_cdf - neg_cdf))), 4)


class DelayedLabelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int
    label_90d: int = Field(ge=0, le=1)


def model_monitoring_summary():
    """Separate offline baseline checks from sparse real request observations."""
    features = list(model.feature_names_in_)
    train = pd.read_parquet(TRAIN_FEATURES_PATH)
    test = pd.read_parquet(TEST_FEATURES_PATH)
    offline_psi = {}
    for name in features:
        offline_psi[name] = population_stability_index(
            train[name], test[name], categorical=(name == "country")
        )
    offline_valid = [v for v in offline_psi.values() if v is not None]

    X_test = test.loc[:, features]
    y_test = pd.to_numeric(test["label_90d"], errors="coerce")
    valid = y_test.notna()
    offline_ks = ks_separation(
        y_test[valid].astype(int),
        model.predict_proba(X_test.loc[valid])[:, 1],
    ) if valid.any() else None

    with sqlite3.connect(PREDICTION_LOG_DB, timeout=10) as db:
        db.row_factory = sqlite3.Row
        snapshots = db.execute("""
            SELECT s.customer_id, s.features_json
            FROM prediction_feature_snapshots s
            JOIN (
                SELECT customer_id, MAX(id) AS latest_id
                FROM prediction_feature_snapshots GROUP BY customer_id
            ) latest ON s.id = latest.latest_id
        """).fetchall()
        labels = db.execute("""
            SELECT l.customer_id, l.label_90d, p.return_probability
            FROM delayed_labels l
            JOIN prediction_logs p ON p.id = (
                SELECT MAX(p2.id) FROM prediction_logs p2
                WHERE p2.customer_id = l.customer_id
                  AND p2.timestamp_utc <= l.observed_at_utc
            )
        """).fetchall()

    distinct = len(snapshots)
    live_psi = None
    if distinct >= MIN_LIVE_CUSTOMERS:
        frame = pd.DataFrame([json.loads(r["features_json"]) for r in snapshots])
        live_psi = {
            name: population_stability_index(
                train[name], frame[name], categorical=(name == "country")
            ) for name in features
        }

    live_ks = None
    if len(labels) >= MIN_LABELED_CUSTOMERS:
        live_ks = ks_separation(
            [r["label_90d"] for r in labels],
            [r["return_probability"] for r in labels],
        )

    http = http_metrics_summary()
    # Ignore Docker health probes when assessing application failures.
    operational = [
        v for k, v in http["http_by_endpoint"].items()
        if k not in ("GET /health", "GET /ready")
    ]
    operational_requests = sum(v["requests"] for v in operational)
    operational_errors = sum(v["errors"] for v in operational)
    operational_error_pct = (
        round(100 * operational_errors / operational_requests, 2)
        if operational_requests else None
    )
    alerts = []
    if operational_requests >= 20 and operational_error_pct >= 5:
        alerts.append({
            "severity": "warning", "type": "http_error_rate",
            "message": "Application HTTP error rate is at least 5%."
        })
    if live_psi is not None:
        elevated = [name for name, score in live_psi.items()
                    if score is not None and score >= 0.25]
        if elevated:
            alerts.append({
                "severity": "warning", "type": "feature_drift",
                "message": "PSI >= 0.25 for: " + ", ".join(elevated)
            })

    return {
        "offline_reference": {
            "type": "training_vs_held_out_test_not_live_drift",
            "train_customers": len(train),
            "test_customers": len(test),
            "psi_by_feature": offline_psi,
            "max_psi": max(offline_valid) if offline_valid else None,
            "ks_model_separation": offline_ks,
        },
        "live": {
            "distinct_customers_with_feature_snapshots": distinct,
            "minimum_distinct_customers_for_psi": MIN_LIVE_CUSTOMERS,
            "psi_status": "available" if live_psi is not None else "insufficient_data",
            "psi_by_feature": live_psi,
            "matched_delayed_labels": len(labels),
            "minimum_labels_for_ks": MIN_LABELED_CUSTOMERS,
            "ks_status": "available" if live_ks is not None else "insufficient_data",
            "ks_model_separation": live_ks,
            "note": "Only real prediction feature snapshots are used. Repeated customer queries count once for PSI. Labels must be supplied after outcomes are observed.",
        },
        "operational": {
            "application_requests": operational_requests,
            "application_errors": operational_errors,
            "application_error_rate_pct": operational_error_pct,
        },
        "alerts": alerts,
        "alert_policy": "HTTP >=5% errors with >=20 non-health requests; feature PSI >=0.25 with >=30 distinct customers.",
    }


@app.get("/monitoring/drift")
def monitoring_drift():
    return model_monitoring_summary()


@app.post("/monitoring/labels")
def submit_delayed_label(request: DelayedLabelRequest):
    """Accept an externally observed 90-day outcome; never infer it from test data."""
    with sqlite3.connect(PREDICTION_LOG_DB, timeout=10) as db:
        exists = db.execute(
            "SELECT 1 FROM prediction_logs WHERE customer_id = ? LIMIT 1",
            (request.customer_id,)
        ).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="No prediction for this customer.")
        db.execute("""
            INSERT INTO delayed_labels (customer_id, label_90d, observed_at_utc, source)
            VALUES (?, ?, ?, 'externally_submitted')
            ON CONFLICT(customer_id) DO UPDATE SET
                label_90d=excluded.label_90d,
                observed_at_utc=excluded.observed_at_utc,
                source=excluded.source
        """, (request.customer_id, request.label_90d,
              datetime.now(UTC).isoformat()))
    return {"status": "recorded", "customer_id": request.customer_id,
            "label_90d": request.label_90d,
            "note": "Only submit verified outcomes after the 90-day observation window."}

