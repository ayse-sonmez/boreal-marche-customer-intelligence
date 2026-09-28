import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Paths and API
# ---------------------------------------------------------

INDEX_PATH = Path("artifacts/index/rag.index")
METADATA_PATH = Path("artifacts/index/metadata.json")

CATALOGUE_DATA_PATH = Path(
    "data/processed/online_retail_ii_silver.parquet"
)

API_BASE_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------
# Load RAG resources once
# ---------------------------------------------------------

print("Loading RAG resources...")

RAG_INDEX = faiss.read_index(
    str(INDEX_PATH)
)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:
    RAG_METADATA = json.load(file)

EMBEDDING_MODEL = SentenceTransformer(
    RAG_METADATA["embedding_model"]
)

print("RAG resources loaded.")


# ---------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------

class PolicySearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str

    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
    )


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

    query_embedding = EMBEDDING_MODEL.encode(
        [request.query],
        convert_to_numpy=True,
    ).astype(np.float32)

    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = RAG_INDEX.search(
        query_embedding,
        request.top_k,
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0],
    ):

        if index_id < 0:
            continue

        chunk = RAG_METADATA[
            "chunks"
        ][int(index_id)]

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
# Calls our FastAPI /predict endpoint
# ---------------------------------------------------------

def retention_score(
    request: RetentionRequest,
) -> RetentionResponse:

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "customer_id":
                    request.customer_id
            },
            timeout=10,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        raise RuntimeError(
            "Retention API request failed."
        ) from exc

    return RetentionResponse(
        **response.json()
    )


# ---------------------------------------------------------
# Tool 3: Recommendations
# Calls our FastAPI /recommend endpoint
# ---------------------------------------------------------

def recommendations(
    request: RecommendationRequest,
) -> RecommendationResponse:

    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json={
                "basket":
                    request.basket,
                "top_k":
                    request.top_k,
            },
            timeout=10,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        raise RuntimeError(
            "Recommendation API request failed."
        ) from exc

    return RecommendationResponse(
        **response.json()
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
            product_id=(
                request.product_id
            ),
            description=None,
        )

    description = str(
        product.iloc[0][
            "Description"
        ]
    )

    return ProductLookupResponse(
        found=True,
        product_id=(
            request.product_id
        ),
        description=description,
    )


# ---------------------------------------------------------
# Simple tool test
# ---------------------------------------------------------

def main():

    print(
        "\nTesting typed tools...\n"
    )

    # Policy search
    policy_result = policy_search(
        PolicySearchRequest(
            query=(
                "How many days do "
                "I have to return "
                "a product?"
            ),
            top_k=1,
        )
    )

    print(
        "1. Policy search:"
    )

    print(
        policy_result.model_dump_json(
            indent=2
        )
    )

    # Recommendation through API
    recommendation_result = (
        recommendations(
            RecommendationRequest(
                basket=["22748"],
                top_k=5,
            )
        )
    )

    print(
        "\n2. Recommendations "
        "(through API):"
    )

    print(
        recommendation_result
        .model_dump_json(
            indent=2
        )
    )

    # Product lookup
    product_result = product_lookup(
        ProductLookupRequest(
            product_id="22745"
        )
    )

    print(
        "\n3. Product lookup:"
    )

    print(
        product_result
        .model_dump_json(
            indent=2
        )
    )

    # Retention through API
    retention_result = retention_score(
        RetentionRequest(
            customer_id=12346
        )
    )

    print(
        "\n4. Retention score "
        "(through API):"
    )

    print(
        retention_result
        .model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()