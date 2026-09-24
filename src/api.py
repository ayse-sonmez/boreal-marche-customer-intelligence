from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import json
import platform
import sqlite3

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
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
                    timezone.utc
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

    with open(
        META_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        model_meta = json.load(
            file
        )

    with open(
        RULES_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        rules_data = json.load(
            file
        )

    rules = rules_data[
        "rules"
    ]

    customer_features = pd.read_parquet(
        FEATURES_PATH
    )

    initialise_prediction_log_db()

    print("API artifacts loaded.")

    yield

    print("API shutting down.")


app = FastAPI(
    title="Boréal Marché Customer Intelligence API",
    description=(
        "API for retention prediction, "
        "product recommendation and "
        "bilingual customer assistance."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


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
    }