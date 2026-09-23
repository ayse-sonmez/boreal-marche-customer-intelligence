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
    "reports/recommendation_evaluation.json"
)

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.20
MIN_LIFT = 1.10
TOP_K = 5


def build_basket_matrix(df):
    """Create a sparse transaction-product matrix."""

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

    return basket.astype(
        pd.SparseDtype(bool, False)
    )


def build_rule_index(rules):
    """Index rules by antecedent product."""

    rule_index = {}

    for _, rule in rules.iterrows():

        antecedent = str(
            next(iter(rule["antecedents"]))
        )

        consequent = str(
            next(iter(rule["consequents"]))
        )

        rule_data = {
            "consequent": consequent,
            "confidence": float(
                rule["confidence"]
            ),
            "lift": float(
                rule["lift"]
            ),
        }

        rule_index.setdefault(
            antecedent,
            []
        ).append(rule_data)

    return rule_index


def recommend_from_rules(
    basket,
    rule_index,
    top_k=5,
):
    """Return top-k recommendations."""

    basket = {
        str(product)
        for product in basket
    }

    candidates = {}

    for product in basket:

        for rule in rule_index.get(
            product,
            [],
        ):

            consequent = rule[
                "consequent"
            ]

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
        candidates.items(),
        key=lambda item: (
            item[1]["lift"],
            item[1]["confidence"],
        ),
        reverse=True,
    )

    return [
        product
        for product, _ in ranked[:top_k]
    ]


def main():

    print(
        "Loading recommendation transactions..."
    )

    df = pd.read_parquet(
        INPUT_PATH
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"]
    )

    # ---------------------------------------------------------
    # Temporal holdout
    # First 80% of time = training
    # Last 20% of time = testing
    # ---------------------------------------------------------

    cutoff = df[
        "InvoiceDate"
    ].quantile(0.80)

    train = df[
        df["InvoiceDate"] <= cutoff
    ].copy()

    test = df[
        df["InvoiceDate"] > cutoff
    ].copy()

    print(
        f"Temporal cutoff: {cutoff}"
    )

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    # ---------------------------------------------------------
    # Mine rules ONLY from training period
    # ---------------------------------------------------------

    print(
        "\nMining training-period rules..."
    )

    train_basket = build_basket_matrix(
        train
    )

    frequent_itemsets = fpgrowth(
        train_basket,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=3,
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE,
    )

    rules = rules[
        (rules["lift"] >= MIN_LIFT)
        & (
            rules["antecedents"]
            .apply(len)
            == 1
        )
        & (
            rules["consequents"]
            .apply(len)
            == 1
        )
    ].copy()

    print(
        f"Training rules: {len(rules):,}"
    )

    rule_index = build_rule_index(
        rules
    )

    # ---------------------------------------------------------
    # Global popularity baseline
    # Training data only
    # ---------------------------------------------------------

    popular_products = (
        train.groupby(
            "StockCode"
        )["Invoice"]
        .nunique()
        .sort_values(
            ascending=False
        )
        .head(TOP_K)
        .index
        .astype(str)
        .tolist()
    )

    print(
        f"Popularity baseline: "
        f"{popular_products}"
    )

    # ---------------------------------------------------------
    # Build test baskets
    # ---------------------------------------------------------

    test_baskets = (
        test.groupby("Invoice")[
            "StockCode"
        ]
        .apply(
            lambda x: list(
                dict.fromkeys(
                    x.astype(str)
                )
            )
        )
    )

    rule_hits = 0
    popularity_hits = 0

    evaluated = 0

    recommended_products = set()

    all_test_products = set(
        test["StockCode"]
        .astype(str)
        .unique()
    )

    # ---------------------------------------------------------
    # Offline evaluation
    #
    # Hide one product from each test basket.
    # Use remaining products as input.
    # Check whether hidden product appears in top 5.
    # ---------------------------------------------------------

    for products in test_baskets:

        if len(products) < 2:
            continue

        actual_product = products[-1]

        input_basket = products[:-1]

        rule_recommendations = (
            recommend_from_rules(
                input_basket,
                rule_index,
                TOP_K,
            )
        )

        popularity_recommendations = [
            product
            for product
            in popular_products
            if product
            not in input_basket
        ][:TOP_K]

        evaluated += 1

        recommended_products.update(
            rule_recommendations
        )

        if (
            actual_product
            in rule_recommendations
        ):
            rule_hits += 1

        if (
            actual_product
            in popularity_recommendations
        ):
            popularity_hits += 1

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    rule_hit_rate = (
        rule_hits / evaluated
        if evaluated
        else 0
    )

    popularity_hit_rate = (
        popularity_hits / evaluated
        if evaluated
        else 0
    )

    coverage = (
        len(recommended_products)
        / len(all_test_products)
        if all_test_products
        else 0
    )

    print(
        f"\nEvaluated baskets: "
        f"{evaluated:,}"
    )

    print(
        f"Rule Hit-rate@5: "
        f"{rule_hit_rate:.4f}"
    )

    print(
        f"Popularity Hit-rate@5: "
        f"{popularity_hit_rate:.4f}"
    )

    print(
        f"Recommendation coverage: "
        f"{coverage:.4f}"
    )

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------

    report = {
        "evaluation": (
            "temporal_holdout"
        ),
        "cutoff_date": str(
            cutoff
        ),
        "training_rows": int(
            len(train)
        ),
        "test_rows": int(
            len(test)
        ),
        "training_rules": int(
            len(rules)
        ),
        "top_k": TOP_K,
        "evaluated_baskets": int(
            evaluated
        ),
        "rule_hit_rate_at_5": float(
            rule_hit_rate
        ),
        "popularity_hit_rate_at_5": float(
            popularity_hit_rate
        ),
        "coverage": float(
            coverage
        ),
        "parameters": {
            "min_support": MIN_SUPPORT,
            "min_confidence": (
                MIN_CONFIDENCE
            ),
            "min_lift": MIN_LIFT,
        },
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
        f"\nReport saved to: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()