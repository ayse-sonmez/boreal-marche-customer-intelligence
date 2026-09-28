import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path(
    "artifacts/index/rag.index"
)

METADATA_PATH = Path(
    "artifacts/index/metadata.json"
)

TOP_K = 3


def main():

    print("Loading RAG index...")

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    model_name = metadata[
        "embedding_model"
    ]

    chunks = metadata[
        "chunks"
    ]

    print(
        f"Vectors loaded: {index.ntotal}"
    )

    print(
        f"Embedding model: {model_name}"
    )

    model = SentenceTransformer(
        model_name
    )

    # English and French retrieval tests
    questions = [
        "How many days do I have to return a product?",
        "Combien de jours ai-je pour retourner un produit?",
    ]

    for question in questions:

        print(
            f"\nQuestion: {question}"
        )

        query_embedding = model.encode(
            [question],
            convert_to_numpy=True,
        ).astype(
            np.float32
        )

        faiss.normalize_L2(
            query_embedding
        )

        scores, indices = index.search(
            query_embedding,
            TOP_K,
        )

        for rank, (
            score,
            index_id,
        ) in enumerate(
            zip(
                scores[0],
                indices[0],
            ),
            start=1,
        ):

            result = chunks[
                int(index_id)
            ]

            print(
                f"\nResult {rank}"
            )

            print(
                f"Source: "
                f"{result['source']}"
            )

            print(
                f"Similarity: "
                f"{score:.4f}"
            )

            print(
                "Text:"
            )

            print(
                result["text"][:400]
            )


if __name__ == "__main__":
    main()