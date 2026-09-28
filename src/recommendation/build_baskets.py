from pathlib import Path

import pandas as pd

INPUT_PATH = Path(
    "data/processed/online_retail_ii_silver.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/recommendation_transactions.parquet"
)


def main():
    print("Loading Silver transaction data...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"Silver rows: {len(df):,}")

    # ---------------------------------------------------------
    # Recommendation data
    # ---------------------------------------------------------
    # For recommendation, we keep the basket information.
    # Cancellation rows can still contain useful information
    # about which products originally appeared together.
    #
    # We only need valid invoice/product pairs here.
    # ---------------------------------------------------------

    recommendation_df = df[
        [
            "Invoice",
            "StockCode",
            "Description",
            "InvoiceDate",
            "Customer ID",
            "Country",
            "IsCancelled",
        ]
    ].copy()

    recommendation_df = recommendation_df.dropna(
        subset=[
            "Invoice",
            "StockCode",
        ]
    )

    # Remove exact duplicate product entries
    # inside the same transaction.
    recommendation_df = (
        recommendation_df
        .drop_duplicates(
            subset=[
                "Invoice",
                "StockCode",
            ]
        )
        .reset_index(drop=True)
    )

    number_transactions = (
        recommendation_df["Invoice"]
        .nunique()
    )

    number_products = (
        recommendation_df["StockCode"]
        .nunique()
    )

    average_basket_size = (
        recommendation_df
        .groupby("Invoice")["StockCode"]
        .nunique()
        .mean()
    )

    single_item_baskets = (
        recommendation_df
        .groupby("Invoice")["StockCode"]
        .nunique()
        .eq(1)
        .sum()
    )

    print("\nRecommendation transaction summary")
    print("----------------------------------")
    print(
        f"Rows: {len(recommendation_df):,}"
    )
    print(
        f"Transactions: {number_transactions:,}"
    )
    print(
        f"Unique products: {number_products:,}"
    )
    print(
        "Average unique products per basket: "
        f"{average_basket_size:.2f}"
    )
    print(
        "Single-product baskets: "
        f"{single_item_baskets:,}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendation_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved recommendation transactions to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()