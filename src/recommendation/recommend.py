from pathlib import Path
import json


RULES_PATH = Path(
    "artifacts/rules.json"
)


def load_rules():
    """
    Load previously mined recommendation rules.
    """

    with open(
        RULES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data["rules"]


def build_rule_index(rules):
    """
    Index rules by antecedent product.

    This makes online recommendation lookup fast.
    """

    rule_index = {}

    for rule in rules:

        antecedent = rule["antecedent"]

        if antecedent not in rule_index:
            rule_index[antecedent] = []

        rule_index[antecedent].append(rule)

    return rule_index


def recommend(
    basket,
    rule_index,
    top_k=5,
):
    """
    Recommend products for a basket.

    - Uses matching association rules.
    - Excludes products already in the basket.
    - Deduplicates recommendations.
    - Keeps the rule with the highest lift.
    - Returns the rule that caused each recommendation.
    """

    basket = {
        str(product)
        for product in basket
    }

    candidates = {}

    for product in basket:

        matching_rules = rule_index.get(
            product,
            [],
        )

        for rule in matching_rules:

            consequent = rule["consequent"]

            # Do not recommend something
            # already in the customer's basket.
            if consequent in basket:
                continue

            existing = candidates.get(
                consequent
            )

            # If several rules recommend the same
            # product, keep the highest-lift rule.
            if (
                existing is None
                or rule["lift"] > existing["lift"]
            ):
                candidates[consequent] = rule

    ranked_rules = sorted(
        candidates.values(),
        key=lambda rule: (
            rule["lift"],
            rule["confidence"],
        ),
        reverse=True,
    )

    recommendations = []

    for rule in ranked_rules[:top_k]:

        recommendations.append(
            {
                "product": rule["consequent"],
                "because_product": (
                    rule["antecedent"]
                ),
                "support": rule["support"],
                "confidence": rule["confidence"],
                "lift": rule["lift"],
            }
        )

    if recommendations:
        strategy = "association_rules"
    else:
        strategy = "no_matching_rule"

    return {
        "strategy": strategy,
        "basket": sorted(basket),
        "recommendations": recommendations,
    }


def main():

    print("Loading recommendation rules...")

    rules = load_rules()

    print(
        f"Rules loaded: {len(rules):,}"
    )

    rule_index = build_rule_index(
        rules
    )

    print(
        f"Indexed antecedent products: "
        f"{len(rule_index):,}"
    )

    # Use a real antecedent from rules.json
    # so we can demonstrate the lookup.
    if rules:

        example_product = (
            rules[0]["antecedent"]
        )

        example_basket = [
            example_product
        ]

        print(
            f"\nExample basket: "
            f"{example_basket}"
        )

        result = recommend(
            basket=example_basket,
            rule_index=rule_index,
            top_k=5,
        )

        print(
            "\nRecommendation result:"
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    # Test empty basket
    empty_result = recommend(
        basket=[],
        rule_index=rule_index,
        top_k=5,
    )

    print(
        "\nEmpty basket strategy: "
        f"{empty_result['strategy']}"
    )

    # Test product with no matching rule
    no_match_result = recommend(
        basket=["UNKNOWN_PRODUCT"],
        rule_index=rule_index,
        top_k=5,
    )

    print(
        "No-match strategy: "
        f"{no_match_result['strategy']}"
    )


if __name__ == "__main__":
    main()