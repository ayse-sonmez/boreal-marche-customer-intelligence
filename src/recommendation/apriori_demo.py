import json
import time
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori

INPUT_PATH = Path(
    "data/processed/recommendation_transactions.parquet"
)

REPORT_PATH = Path(
    "reports/apriori_demo_report.json"
)

# Apriori is used only as a demonstration.
# FP-Growth will be used later for production.
MIN_SUPPORT = 0.02

# To keep the Apriori demonstration manageable,
# use the most frequent products.
TOP_PRODUCTS = 500


def main():

    print("Loading recommendation transactions...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"Rows: {len(df):,}")

    # ---------------------------------------------------------
    # Select most frequent products for Apriori demonstration
    # ---------------------------------------------------------

    product_frequency = (
        df.groupby("StockCode")["Invoice"]
        .nunique()
        .sort_values(ascending=False)
    )

    top_products = (
        product_frequency
        .head(TOP_PRODUCTS)
        .index
    )

    demo_df = df[
        df["StockCode"].isin(top_products)
    ].copy()

    print(
        f"Products used for Apriori demo: "
        f"{len(top_products):,}"
    )

    # ---------------------------------------------------------
    # Build transaction-product basket
    # ---------------------------------------------------------

    basket = pd.crosstab(
        demo_df["Invoice"],
        demo_df["StockCode"],
    )

    basket = basket.gt(0)

    print(
        f"Apriori basket shape: "
        f"{basket.shape[0]:,} x "
        f"{basket.shape[1]:,}"
    )

    # ---------------------------------------------------------
    # Run and time Apriori
    # ---------------------------------------------------------

    print(
        f"\nRunning Apriori with "
        f"min_support={MIN_SUPPORT}..."
    )

    start_time = time.perf_counter()

    frequent_itemsets = apriori(
        basket,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=3,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    frequent_itemsets = (
        frequent_itemsets
        .sort_values(
            "support",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\nApriori completed.")
    print(
        f"Runtime: {elapsed_seconds:.2f} seconds"
    )
    print(
        f"Frequent itemsets found: "
        f"{len(frequent_itemsets):,}"
    )

    if not frequent_itemsets.empty:

        print(
            "\nTop 10 frequent itemsets:"
        )

        print(
            frequent_itemsets.head(10).to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "purpose": (
            "Apriori demonstration and timing. "
            "FP-Growth will be used for production mining."
        ),
        "input_rows": len(df),
        "products_used": len(top_products),
        "basket_transactions": int(
            basket.shape[0]
        ),
        "basket_products": int(
            basket.shape[1]
        ),
        "min_support": MIN_SUPPORT,
        "max_itemset_length": 3,
        "runtime_seconds": float(
            elapsed_seconds
        ),
        "frequent_itemsets_found": len(frequent_itemsets),
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
        f"\nReport saved to: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()