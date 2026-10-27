import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

MODEL_PATH = Path("artifacts/model.joblib")
META_PATH = Path("artifacts/model_meta.json")
TEST_PATH = Path("data/processed/customer_features_test.parquet")
OUTPUT_PATH = Path("reports/fairness_audit.json")

MIN_GROUP_SIZE = 30
GAP_THRESHOLD = 0.05


def calculate_metrics(y_true, probabilities, threshold):
    predicted_return = probabilities >= threshold
    selected_for_offer = probabilities < threshold

    return {
        "n": int(len(y_true)),
        "actual_return_rate": float(np.mean(y_true)),
        "return_recall": float(
            recall_score(y_true, predicted_return, zero_division=0)
        ),
        "return_precision": float(
            precision_score(y_true, predicted_return, zero_division=0)
        ),
        "offer_selection_rate": float(np.mean(selected_for_offer)),
        "actual_positive_count": int(np.sum(y_true)),
        "predicted_positive_count": int(np.sum(predicted_return)),
    }


def evaluate_groups(frame, group_column, threshold):
    results = []

    for group_name, group in frame.groupby(group_column, dropna=False):
        metrics = calculate_metrics(
            group["label_90d"].to_numpy(),
            group["return_probability"].to_numpy(),
            threshold,
        )
        metrics["group"] = str(group_name)
        metrics["small_sample"] = metrics["n"] < MIN_GROUP_SIZE
        results.append(metrics)

    eligible = [r for r in results if not r["small_sample"]]

    gaps = {}
    for metric in [
        "return_recall",
        "return_precision",
        "offer_selection_rate",
    ]:
        if len(eligible) >= 2:
            values = [r[metric] for r in eligible]
            gap = float(max(values) - min(values))
            gaps[metric] = {
                "gap": round(gap, 4),
                "exceeds_5_percentage_points": gap > GAP_THRESHOLD,
            }
        else:
            gaps[metric] = {
                "gap": None,
                "exceeds_5_percentage_points": None,
                "note": "Insufficient groups with at least 30 customers",
            }

    return {
        "groups": results,
        "eligible_group_count": len(eligible),
        "gaps": gaps,
    }


def main():
    with META_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)

    threshold = float(metadata["threshold"])
    features = metadata["features_in_order"]

    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(TEST_PATH).copy()

    missing = [
        column for column in features + ["label_90d"]
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    probabilities = model.predict_proba(df[features])[:, 1]
    df["return_probability"] = probabilities

    # Use actual source country. Small groups are reported,
    # but excluded from the cross-group gap calculation.
    df["audit_country"] = df["country"].fillna("Unknown").astype(str)

    # Customer lifetime is measured in days.
    # Rank-based grouping handles repeated values.
    df["audit_tenure"] = pd.qcut(
        df["customer_lifetime_days"].rank(method="first"),
        q=4,
        labels=["Q1_shortest", "Q2", "Q3", "Q4_longest"],
    )

    df["audit_spend"] = pd.qcut(
        df["total_spend"].rank(method="first"),
        q=4,
        labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"],
    )

    report = {
        "project": "Boreal Marche",
        "model_version": metadata["model_version"],
        "evaluation_dataset": str(TEST_PATH),
        "test_customers": int(len(df)),
        "return_probability_threshold": threshold,
        "offer_selection_rule": "return_probability < threshold",
        "gap_threshold_percentage_points": 5,
        "minimum_group_size_for_gap_comparison": MIN_GROUP_SIZE,
        "metric_definitions": {
            "return_recall": (
                "Recall for actual 90-day returners; positive label is 1."
            ),
            "return_precision": (
                "Precision for predicted 90-day returners."
            ),
            "offer_selection_rate": (
                "Share of customers below the return threshold "
                "and therefore selected for a hypothetical offer."
            ),
        },
        "audits": {
            "country": evaluate_groups(df, "audit_country", threshold),
            "customer_tenure_quartiles": evaluate_groups(
                df, "audit_tenure", threshold
            ),
            "customer_spending_quartiles": evaluate_groups(
                df, "audit_spend", threshold
            ),
        },
        "limitations": [
            "Historical UCI data is not representative of Quebec customers.",
            "Country is a model input and may act as a geographic proxy.",
            "Small groups have unstable metrics and are excluded from gap comparisons.",
            "Quartiles are constructed from ranked values to handle ties.",
            "Observed disparities do not by themselves prove discrimination.",
            "No real retention campaign or causal impact was evaluated.",
        ],
        "proposed_mitigations": [
            "Investigate flagged gaps with larger representative samples.",
            "Compare performance with and without the country feature.",
            "Evaluate group-specific calibration and error rates.",
            "Require human review before using scores for real interventions.",
        ],
        "mitigation_tradeoffs": [
            "Removing country may reduce predictive performance.",
            "Group-specific adjustments may improve one metric while worsening another.",
            "Human review increases operating cost but improves oversight.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("FAIRNESS AUDIT COMPLETED")
    print(f"Test customers: {len(df)}")
    print(f"Threshold: {threshold}")
    print()

    for name, audit in report["audits"].items():
        print(name)
        print(f"  Groups: {len(audit['groups'])}")
        print(f"  Eligible groups: {audit['eligible_group_count']}")
        for metric, gap in audit["gaps"].items():
            if gap["gap"] is None:
                print(f"  {metric}: insufficient data")
            else:
                print(
                    f"  {metric}: {gap['gap'] * 100:.2f} pp; "
                    f"flagged={gap['exceeds_5_percentage_points']}"
                )
        print()

    print(f"Report saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
