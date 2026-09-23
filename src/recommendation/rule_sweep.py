from pathlib import Path
import json

import pandas as pd
from mlxtend.frequent_patterns import (
    fpgrowth,
    association_rules,
)


INPUT_PATH = Path(
    "data/processed/recommendation_transactions.parquet"
)

REPORT_PATH = Path(
    "reports/rule_sweep_report.json"
)

SUPPORT_VALUES = [
    0.005,
    0.01,
    0.02,
]

CONFIDENCE_VALUES = [
    0.20,
    0.40,
    0.60,
]

MIN_LIFT = 1.10


def main():

    print("Loading recommendation transactions...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"Rows: {len(df):,}")

    print("Building basket matrix...")

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
        pd.SparseDtype(bool, False)
    )

    print(
        f"Basket shape: "
        f"{basket.shape[0]:,} x "
        f"{basket.shape[1]:,}"
    )

    results = []

    print("\nRunning support/confidence sweep...")

    for min_support in SUPPORT_VALUES:

        frequent_itemsets = fpgrowth(
            basket,
            min_support=min_support,
            use_colnames=True,
            max_len=3,
        )

        for min_confidence in CONFIDENCE_VALUES:

            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence,
            )

            simple_rules = rules[
                (rules["antecedents"].apply(len) == 1)
                & (rules["consequents"].apply(len) == 1)
            ].copy()

            useful_rules = simple_rules[
                simple_rules["lift"] >= MIN_LIFT
            ]

            result = {
                "min_support": min_support,
                "min_confidence": min_confidence,
                "frequent_itemsets": int(
                    len(frequent_itemsets)
                ),
                "simple_rules_before_lift": int(
                    len(simple_rules)
                ),
                "rules_after_lift_filter": int(
                    len(useful_rules)
                ),
            }

            results.append(result)

            print(
                f"support={min_support:.3f}, "
                f"confidence={min_confidence:.2f} "
                f"-> {len(useful_rules):,} useful rules"
            )

    # ---------------------------------------------------------
    # Find a real high-confidence but low-lift rule
    # ---------------------------------------------------------

    frequent_itemsets = fpgrowth(
        basket,
        min_support=0.005,
        use_colnames=True,
        max_len=3,
    )

    all_rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.40,
    )

    simple_rules = all_rules[
        (all_rules["antecedents"].apply(len) == 1)
        & (all_rules["consequents"].apply(len) == 1)
    ].copy()

    discarded = simple_rules[
        simple_rules["lift"] < MIN_LIFT
    ].sort_values(
        "confidence",
        ascending=False,
    )

    discarded_example = None

    if not discarded.empty:

        row = discarded.iloc[0]

        discarded_example = {
            "antecedent": str(
                next(iter(row["antecedents"]))
            ),
            "consequent": str(
                next(iter(row["consequents"]))
            ),
            "support": float(
                row["support"]
            ),
            "confidence": float(
                row["confidence"]
            ),
            "lift": float(
                row["lift"]
            ),
            "reason_discarded": (
                "The rule has relatively high confidence, "
                "but lift is below 1.10, so the association "
                "does not add enough value over the "
                "consequent's normal popularity."
            ),
        }

        print(
            "\nHigh-confidence / low-lift "
            "discarded example:"
        )

        print(
            json.dumps(
                discarded_example,
                indent=2,
            )
        )

    else:
        print(
            "\nNo high-confidence low-lift rule "
            "was found with these settings."
        )

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------

    report = {
        "support_values": SUPPORT_VALUES,
        "confidence_values": CONFIDENCE_VALUES,
        "minimum_lift": MIN_LIFT,
        "sweep_results": results,
        "selected_production_parameters": {
            "min_support": 0.01,
            "min_confidence": 0.20,
            "min_lift": 1.10,
            "reason": (
                "Selected as a simple balance between "
                "rule coverage and rule quality."
            ),
        },
        "discarded_low_lift_example": (
            discarded_example
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        f"\nReport saved to: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()