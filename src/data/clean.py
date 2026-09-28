from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/online_retail_ii.parquet")
OUTPUT_PATH = Path("data/processed/online_retail_ii_silver.parquet")
REPORT_PATH = Path("reports/cleaning_report.json")


# Stock codes confirmed from the dataset as non-product/service records.
NON_PRODUCT_STOCK_CODES = {
    "ADJUST",
    "ADJUST2",
    "BANK CHARGES",
    "C2",
    "D",
    "DOT",
    "M",
    "POST",
    "TEST001",
    "TEST002",
}


def record_step(
    report: list[dict],
    step: str,
    before_rows: int,
    after_rows: int,
    rationale: str,
) -> None:
    report.append(
        {
            "step": step,
            "before_rows": before_rows,
            "after_rows": after_rows,
            "rows_removed": before_rows - after_rows,
            "rationale": rationale,
        }
    )


def clean_data() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    report: list[dict] = []

    print(f"Initial rows: {len(df):,}")

    # -----------------------------------------------------
    # 1. Identify cancelled transactions
    # -----------------------------------------------------
    df["IsCancelled"] = (
        df["Invoice"]
        .fillna("")
        .str.upper()
        .str.startswith("C")
    )

    cancelled_count = int(df["IsCancelled"].sum())

    report.append(
        {
            "step": "identify_cancelled_transactions",
            "before_rows": len(df),
            "after_rows": len(df),
            "rows_removed": 0,
            "cancelled_rows_identified": cancelled_count,
            "rationale": (
                "Cancelled transactions are retained in the Silver layer "
                "because basket information may still be useful for "
                "recommendation. They are flagged so prediction pipelines "
                "can exclude them when necessary."
            ),
        }
    )

    # -----------------------------------------------------
    # 2. Remove invalid quantities
    # Keep negative quantities only when they represent
    # explicitly cancelled transactions.
    # -----------------------------------------------------
    before = len(df)

    valid_quantity = (
        (df["Quantity"] > 0)
        | (
            df["IsCancelled"]
            & (df["Quantity"] < 0)
        )
    )

    df = df[valid_quantity].copy()

    record_step(
        report=report,
        step="remove_invalid_quantities",
        before_rows=before,
        after_rows=len(df),
        rationale=(
            "Zero quantities and negative quantities that are not marked "
            "as cancellations are removed. Negative quantities belonging "
            "to cancelled invoices are retained for later model-specific handling."
        ),
    )

    # -----------------------------------------------------
    # 3. Remove non-positive prices
    # -----------------------------------------------------
    before = len(df)

    df = df[df["Price"] > 0].copy()

    record_step(
        report=report,
        step="remove_non_positive_price",
        before_rows=before,
        after_rows=len(df),
        rationale=(
            "Rows with zero or negative prices are not considered "
            "valid commercial product transactions."
        ),
    )

    # -----------------------------------------------------
    # 4. Remove missing customer IDs
    # -----------------------------------------------------
    before = len(df)

    df = df[df["Customer ID"].notna()].copy()

    record_step(
        report=report,
        step="remove_missing_customer_id",
        before_rows=before,
        after_rows=len(df),
        rationale=(
            "Customer-level prediction requires a valid customer identifier."
        ),
    )

    # -----------------------------------------------------
    # 5. Canonicalise descriptions
    # -----------------------------------------------------
    before = len(df)

    df["Description"] = (
        df["Description"]
        .fillna("")
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )

    record_step(
        report=report,
        step="canonicalise_description",
        before_rows=before,
        after_rows=len(df),
        rationale=(
            "Descriptions are normalised for consistent product-level analysis."
        ),
    )

    # -----------------------------------------------------
    # 6. Remove non-product and service stock codes
    # -----------------------------------------------------
    before = len(df)

    df = df[
        ~df["StockCode"].isin(NON_PRODUCT_STOCK_CODES)
    ].copy()

    record_step(
        report=report,
        step="remove_non_product_service_skus",
        before_rows=before,
        after_rows=len(df),
        rationale=(
            "Confirmed non-product records such as postage, carriage, "
            "bank charges, discounts, manual entries, adjustments and "
            "test transactions are excluded from product purchase behaviour."
        ),
    )

    # -----------------------------------------------------
    # 7. Outlier review
    # Do not automatically remove extreme values.
    # They are documented and handled later according
    # to the model use case.
    # -----------------------------------------------------
    quantity_999 = float(df["Quantity"].abs().quantile(0.999))
    price_999 = float(df["Price"].quantile(0.999))

    quantity_outliers = int(
        (df["Quantity"].abs() > quantity_999).sum()
    )

    price_outliers = int(
        (df["Price"] > price_999).sum()
    )

    report.append(
        {
            "step": "review_outliers",
            "before_rows": len(df),
            "after_rows": len(df),
            "rows_removed": 0,
            "quantity_abs_99_9_percentile": quantity_999,
            "price_99_9_percentile": price_999,
            "quantity_rows_above_threshold": quantity_outliers,
            "price_rows_above_threshold": price_outliers,
            "rationale": (
                "Extreme quantity and price values were reviewed rather than "
                "automatically removed. Some high values are legitimate products, "
                "while some extreme quantities are related to cancelled transactions. "
                "The final treatment will depend on the prediction or recommendation use case."
            ),
        }
    )

    # -----------------------------------------------------
    # Save cleaned Silver dataset
    # -----------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
    )

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("Silver cleaning completed.")
    print(f"Final rows: {len(df):,}")
    print(f"Cancelled rows retained: {int(df['IsCancelled'].sum()):,}")
    print(f"Saved dataset: {OUTPUT_PATH}")
    print(f"Cleaning report: {REPORT_PATH}")


if __name__ == "__main__":
    clean_data()