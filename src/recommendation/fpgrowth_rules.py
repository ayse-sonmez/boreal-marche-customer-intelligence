from pathlib import Path
import hashlib
import json
import time
from datetime import datetime, timezone

import pandas as pd
from mlxtend.frequent_patterns import (
    fpgrowth,
    association_rules,
)


INPUT_PATH = Path(
    "data/processed/recommendation_transactions.parquet"
)

RULES_PATH = Path(
    "artifacts/rules.json"
)

REPORT_PATH = Path(
    "reports/fpgrowth_report.json"
)

# Production mining parameters
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.20
MIN_LIFT = 1.10


def calculate_sha256(file_path):
    """
    Calculate SHA-256 hash of the source data file.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(8192),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():

    print(
        "Loading recommendation transactions..."
    )

    df = pd.read_parquet(
        INPUT_PATH
    )

    print(
        f"Rows: {len(df):,}"
    )

    # ---------------------------------------------------------
    # Calculate source data hash
    # ---------------------------------------------------------

    print(
        "Calculating source data SHA-256..."
    )

    source_hash = calculate_sha256(
        INPUT_PATH
    )

    # ---------------------------------------------------------
    # Build sparse basket matrix
    # ---------------------------------------------------------

    print(
        "Building basket matrix..."
    )

    basket = (
        df.assign(value=True)
        .pivot_table(
            index="Invoice",
            columns="StockCode",
            values="value",
            aggfunc="max",
            fill_value=False,
        )
        .astype(bool)
    )

    basket = basket.astype(
        pd.SparseDtype(
            bool,
            False,
        )
    )

    print(
        f"Basket shape: "
        f"{basket.shape[0]:,} x "
        f"{basket.shape[1]:,}"
    )

    # ---------------------------------------------------------
    # FP-Growth
    # ---------------------------------------------------------

    print(
        f"\nRunning FP-Growth "
        f"(min_support={MIN_SUPPORT})..."
    )

    start_time = time.perf_counter()

    frequent_itemsets = fpgrowth(
        basket,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=3,
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    print(
        f"FP-Growth runtime: "
        f"{runtime:.2f} seconds"
    )

    print(
        f"Frequent itemsets: "
        f"{len(frequent_itemsets):,}"
    )

    # ---------------------------------------------------------
    # Generate association rules
    # ---------------------------------------------------------

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE,
    )

    rules_before_filter = len(
        rules
    )

    print(
        f"Rules before lift filter: "
        f"{rules_before_filter:,}"
    )

    # ---------------------------------------------------------
    # Filter rules
    # ---------------------------------------------------------

    rules = rules[
        rules["lift"] >= MIN_LIFT
    ].copy()

    # Keep simple product -> product rules.
    rules = rules[
        (
            rules["antecedents"]
            .apply(len)
            == 1
        )
        &
        (
            rules["consequents"]
            .apply(len)
            == 1
        )
    ].copy()

    rules = rules.sort_values(
        [
            "lift",
            "confidence",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    print(
        f"Final rules after filtering: "
        f"{len(rules):,}"
    )

    # ---------------------------------------------------------
    # Convert rules to JSON format
    # ---------------------------------------------------------

    exported_rules = []

    for _, row in rules.iterrows():

        antecedent = str(
            next(
                iter(
                    row["antecedents"]
                )
            )
        )

        consequent = str(
            next(
                iter(
                    row["consequents"]
                )
            )
        )

        exported_rules.append(
            {
                "antecedent": antecedent,
                "consequent": consequent,
                "support": float(
                    row["support"]
                ),
                "confidence": float(
                    row["confidence"]
                ),
                "lift": float(
                    row["lift"]
                ),
            }
        )

    # ---------------------------------------------------------
    # Save rules.json
    # ---------------------------------------------------------

    output = {
        "header": {
            "mining_date_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "method": "FP-Growth",
            "parameters": {
                "min_support": (
                    MIN_SUPPORT
                ),
                "min_confidence": (
                    MIN_CONFIDENCE
                ),
                "min_lift": (
                    MIN_LIFT
                ),
                "max_itemset_length": 3,
            },
            "source": str(
                INPUT_PATH
            ),
            "source_data_sha256": (
                source_hash
            ),
            "transaction_count": int(
                basket.shape[0]
            ),
            "product_count": int(
                basket.shape[1]
            ),
            "rule_count": int(
                len(exported_rules)
            ),
        },
        "rules": exported_rules,
    }

    RULES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RULES_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Save simple mining report
    # ---------------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "method": "FP-Growth",
        "input_rows": int(
            len(df)
        ),
        "transactions": int(
            basket.shape[0]
        ),
        "products": int(
            basket.shape[1]
        ),
        "min_support": (
            MIN_SUPPORT
        ),
        "min_confidence": (
            MIN_CONFIDENCE
        ),
        "min_lift": (
            MIN_LIFT
        ),
        "runtime_seconds": float(
            runtime
        ),
        "frequent_itemsets": int(
            len(frequent_itemsets)
        ),
        "rules_before_filter": int(
            rules_before_filter
        ),
        "final_rules": int(
            len(rules)
        ),
        "source_data_sha256": (
            source_hash
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

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print(
        f"\nRules saved to: "
        f"{RULES_PATH}"
    )

    print(
        f"Report saved to: "
        f"{REPORT_PATH}"
    )

    print(
        f"Source SHA-256: "
        f"{source_hash}"
    )

    if exported_rules:

        print(
            "\nExample rule:"
        )

        print(
            json.dumps(
                exported_rules[0],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()