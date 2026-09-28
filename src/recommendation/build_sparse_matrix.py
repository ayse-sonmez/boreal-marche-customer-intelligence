import json
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, save_npz

INPUT_PATH = Path(
    "data/processed/recommendation_transactions.parquet"
)

MATRIX_PATH = Path(
    "artifacts/recommendation_basket_matrix.npz"
)

PRODUCTS_PATH = Path(
    "artifacts/recommendation_products.json"
)

TRANSACTIONS_PATH = Path(
    "artifacts/recommendation_transactions.json"
)

REPORT_PATH = Path(
    "reports/recommendation_matrix_report.json"
)


def bytes_to_mb(number_bytes):
    return number_bytes / (1024 ** 2)


def main():

    print("Loading recommendation transactions...")

    df = pd.read_parquet(INPUT_PATH)

    print(
        f"Rows: {len(df):,}"
    )

    # ---------------------------------------------------------
    # Create integer IDs for invoices and products
    # ---------------------------------------------------------

    invoice_codes, invoice_values = pd.factorize(
        df["Invoice"],
        sort=True,
    )

    product_codes, product_values = pd.factorize(
        df["StockCode"],
        sort=True,
    )

    number_transactions = len(invoice_values)
    number_products = len(product_values)

    print(
        f"Transactions: {number_transactions:,}"
    )

    print(
        f"Products: {number_products:,}"
    )

    # ---------------------------------------------------------
    # Build sparse binary basket matrix
    #
    # rows    = transactions
    # columns = products
    # 1       = product appeared in transaction
    # ---------------------------------------------------------

    values = [1] * len(df)

    sparse_matrix = csr_matrix(
        (
            values,
            (
                invoice_codes,
                product_codes,
            ),
        ),
        shape=(
            number_transactions,
            number_products,
        ),
        dtype="uint8",
    )

    # Because invoice/product duplicates were already removed,
    # each non-zero value should be 1.

    non_zero_values = sparse_matrix.nnz

    total_cells = (
        number_transactions
        * number_products
    )

    density = (
        non_zero_values
        / total_cells
    )

    # ---------------------------------------------------------
    # Memory comparison
    # ---------------------------------------------------------

    # A dense uint8 matrix would require one byte per cell.
    dense_memory_bytes = total_cells

    sparse_memory_bytes = (
        sparse_matrix.data.nbytes
        + sparse_matrix.indices.nbytes
        + sparse_matrix.indptr.nbytes
    )

    dense_memory_mb = bytes_to_mb(
        dense_memory_bytes
    )

    sparse_memory_mb = bytes_to_mb(
        sparse_memory_bytes
    )

    memory_reduction_percent = (
        1
        - (
            sparse_memory_bytes
            / dense_memory_bytes
        )
    ) * 100

    print(
        "\nBasket matrix summary"
    )

    print(
        "----------------------------------"
    )

    print(
        f"Shape: "
        f"{number_transactions:,} x "
        f"{number_products:,}"
    )

    print(
        f"Total cells: {total_cells:,}"
    )

    print(
        f"Non-zero cells: {non_zero_values:,}"
    )

    print(
        f"Density: {density:.6f}"
    )

    print(
        f"Dense memory estimate: "
        f"{dense_memory_mb:.2f} MB"
    )

    print(
        f"Sparse memory usage: "
        f"{sparse_memory_mb:.2f} MB"
    )

    print(
        f"Memory reduction: "
        f"{memory_reduction_percent:.2f}%"
    )

    # ---------------------------------------------------------
    # Save artifacts
    # ---------------------------------------------------------

    MATRIX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_npz(
        MATRIX_PATH,
        sparse_matrix,
    )

    with open(
        PRODUCTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [
                str(value)
                for value in product_values
            ],
            file,
            indent=2,
        )

    with open(
        TRANSACTIONS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [
                str(value)
                for value in invoice_values
            ],
            file,
            indent=2,
        )

    report = {
        "transactions": int(
            number_transactions
        ),
        "products": int(
            number_products
        ),
        "matrix_shape": [
            int(number_transactions),
            int(number_products),
        ],
        "total_cells": int(
            total_cells
        ),
        "non_zero_cells": int(
            non_zero_values
        ),
        "density": float(
            density
        ),
        "dense_memory_mb_uint8": float(
            dense_memory_mb
        ),
        "sparse_memory_mb": float(
            sparse_memory_mb
        ),
        "memory_reduction_percent": float(
            memory_reduction_percent
        ),
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        f"\nSparse matrix saved to: "
        f"{MATRIX_PATH}"
    )

    print(
        f"Memory report saved to: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()