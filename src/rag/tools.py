from pathlib import Path
import json

import faiss
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path("artifacts/index/rag.index")
METADATA_PATH = Path("artifacts/index/metadata.json")
RULES_PATH = Path("artifacts/rules.json")
MODEL_PATH = Path("artifacts/model.joblib")
FEATURES_PATH = Path(
    "data/processed/customer_features_gold.parquet"
)
CATALOGUE_DATA_PATH = Path(
    "data/processed/online_retail_ii_silver.parquet"
)


# ---------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------

class PolicySearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = Field(default=3, ge=1, le=5)


class PolicyResult(BaseModel):
    source: str
    chunk_id: int
    similarity: float
    text: str


class PolicySearchResponse(BaseModel):
    results: list[PolicyResult]


class RetentionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: int


class RetentionResponse(BaseModel):
    customer_id: int
    return_probability: float
    threshold: float
    at_risk: bool


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    basket: list[str]
    top_k: int = Field(default=5, ge=1, le=10)


class RecommendationItem(BaseModel):
    product: str
    because_product: str
    confidence: float
    lift: float


class RecommendationResponse(BaseModel):
    strategy: str
    recommendations: list[RecommendationItem]


class ProductLookupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str


class ProductLookupResponse(BaseModel):
    found: bool
    product_id: str
    description: str | None = None


# ---------------------------------------------------------
# Tool 1: Policy search
# ---------------------------------------------------------

def policy_search(
    request: PolicySearchRequest,
) -> PolicySearchResponse:

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    model = SentenceTransformer(
        metadata["embedding_model"]
    )

    query_embedding = model.encode(
        [request.query],
        convert_to_numpy=True,
    ).astype(np.float32)

    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = index.search(
        query_embedding,
        request.top_k,
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0],
    ):

        chunk = metadata["chunks"][
            int(index_id)
        ]

        results.append(
            PolicyResult(
                source=chunk["source"],
                chunk_id=chunk["chunk_id"],
                similarity=float(score),
                text=chunk["text"],
            )
        )

    return PolicySearchResponse(
        results=results
    )


# ---------------------------------------------------------
# Tool 2: Retention score
# ---------------------------------------------------------

def retention_score(
    request: RetentionRequest,
) -> RetentionResponse:

    model = joblib.load(
        MODEL_PATH
    )

    df = pd.read_parquet(
        FEATURES_PATH
    )

    customer = df[
        df["Customer ID"]
        == request.customer_id
    ].copy()

    if customer.empty:
        raise ValueError(
            "Customer ID not found."
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

    threshold = 0.73

    # Low return probability means
    # the customer is considered at risk.
    at_risk = probability < threshold

    return RetentionResponse(
        customer_id=request.customer_id,
        return_probability=probability,
        threshold=threshold,
        at_risk=at_risk,
    )


# ---------------------------------------------------------
# Tool 3: Recommendations
# ---------------------------------------------------------

def recommendations(
    request: RecommendationRequest,
) -> RecommendationResponse:

    with open(
        RULES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    rules = data["rules"]

    basket = {
        str(product)
        for product in request.basket
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

    output = []

    for rule in ranked[
        : request.top_k
    ]:

        output.append(
            RecommendationItem(
                product=str(
                    rule["consequent"]
                ),
                because_product=str(
                    rule["antecedent"]
                ),
                confidence=float(
                    rule["confidence"]
                ),
                lift=float(
                    rule["lift"]
                ),
            )
        )

    strategy = (
        "association_rules"
        if output
        else "no_matching_rule"
    )

    return RecommendationResponse(
        strategy=strategy,
        recommendations=output,
    )


# ---------------------------------------------------------
# Tool 4: Product lookup
# ---------------------------------------------------------

def product_lookup(
    request: ProductLookupRequest,
) -> ProductLookupResponse:

    df = pd.read_parquet(
        CATALOGUE_DATA_PATH,
        columns=[
            "StockCode",
            "Description",
        ],
    )

    df["StockCode"] = (
        df["StockCode"]
        .astype(str)
    )

    product = df[
        df["StockCode"]
        == str(request.product_id)
    ]

    if product.empty:

        return ProductLookupResponse(
            found=False,
            product_id=request.product_id,
            description=None,
        )

    description = str(
        product.iloc[0][
            "Description"
        ]
    )

    return ProductLookupResponse(
        found=True,
        product_id=request.product_id,
        description=description,
    )


# ---------------------------------------------------------
# Simple test
# ---------------------------------------------------------

def main():

    print("Testing typed tools...\n")

    policy_result = policy_search(
        PolicySearchRequest(
            query=(
                "How many days do I have "
                "to return a product?"
            ),
            top_k=1,
        )
    )

    print("1. Policy search:")
    print(
        policy_result.model_dump_json(
            indent=2
        )
    )

    recommendation_result = recommendations(
        RecommendationRequest(
            basket=["22748"],
            top_k=5,
        )
    )

    print(
        "\n2. Recommendations:"
    )

    print(
        recommendation_result
        .model_dump_json(
            indent=2
        )
    )

    product_result = product_lookup(
        ProductLookupRequest(
            product_id="22745"
        )
    )

    print(
        "\n3. Product lookup:"
    )

    print(
        product_result.model_dump_json(
            indent=2
        )
    )

    # Use one real customer from Gold
    customer_df = pd.read_parquet(
        FEATURES_PATH,
        columns=["Customer ID"],
    )

    example_customer = int(
        customer_df.iloc[0][
            "Customer ID"
        ]
    )

    retention_result = retention_score(
        RetentionRequest(
            customer_id=example_customer
        )
    )

    print(
        "\n4. Retention score:"
    )

    print(
        retention_result.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()