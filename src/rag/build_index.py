from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


CORPUS_DIR = Path("docs/corpus")
INDEX_DIR = Path("artifacts/index")

INDEX_PATH = INDEX_DIR / "rag.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_text(text):
    """Split document text into overlapping chunks."""

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def main():

    print("Loading corpus...")

    documents = sorted(
        CORPUS_DIR.glob("*.md")
    )

    print(
        f"Documents found: {len(documents)}"
    )

    all_chunks = []
    metadata = []

    for document in documents:

        text = document.read_text(
            encoding="utf-8"
        )

        chunks = chunk_text(text)

        print(
            f"{document.name}: "
            f"{len(chunks)} chunks"
        )

        for chunk_number, chunk in enumerate(
            chunks
        ):

            all_chunks.append(chunk)

            metadata.append(
                {
                    "source": document.name,
                    "chunk_id": chunk_number,
                    "text": chunk,
                }
            )

    print(
        f"\nTotal chunks: {len(all_chunks)}"
    )

    print(
        f"Loading multilingual model: "
        f"{MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Creating embeddings...")

    embeddings = model.encode(
        all_chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.astype(
        np.float32
    )

    # Normalize so inner product works
    # as cosine similarity.
    faiss.normalize_L2(
        embeddings
    )

    dimension = embeddings.shape[1]

    print(
        f"Embedding dimension: {dimension}"
    )

    # Simple FAISS cosine similarity index
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "embedding_model": MODEL_NAME,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "document_count": len(
                    documents
                ),
                "chunk_count": len(
                    all_chunks
                ),
                "chunks": metadata,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nVector index saved to: "
        f"{INDEX_PATH}"
    )

    print(
        f"Metadata saved to: "
        f"{METADATA_PATH}"
    )

    print(
        f"Vectors indexed: "
        f"{index.ntotal}"
    )


if __name__ == "__main__":
    main()