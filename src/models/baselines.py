from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


DATA_PATH = Path("data/processed/customer_features_gold.parquet")


def print_metrics(
    name: str,
    y_true: pd.Series,
    probabilities,
) -> None:
    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    brier = brier_score_loss(
        y_true,
        probabilities,
    )

    print()
    print(name)
    print("-" * len(name))
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Brier:   {brier:.4f}")


def run_baselines() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Gold dataset not found: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    y = df["label_90d"]

    print(f"Customers: {len(df):,}")
    print(f"Positive rate: {y.mean() * 100:.2f}%")

    # -----------------------------------------------------
    # Baseline 1: DummyClassifier
    #
    # This baseline predicts using the overall class
    # distribution without learning customer behaviour.
    # -----------------------------------------------------
    dummy_x = df[["recency_days"]]

    dummy = DummyClassifier(
        strategy="prior",
        random_state=42,
    )

    dummy.fit(dummy_x, y)

    dummy_probabilities = dummy.predict_proba(
        dummy_x
    )[:, 1]

    print_metrics(
        "DummyClassifier Baseline",
        y,
        dummy_probabilities,
    )

    # -----------------------------------------------------
    # Baseline 2: Simple business rule
    #
    # Rule:
    # If the customer purchased within the last 90 days,
    # predict that the customer will return.
    # -----------------------------------------------------
    business_probabilities = (
        df["recency_days"] <= 90
    ).astype(float)

    print_metrics(
        "Business Rule Baseline",
        y,
        business_probabilities,
    )

    # Simple classification accuracy for the business rule
    business_predictions = (
        df["recency_days"] <= 90
    ).astype(int)

    business_accuracy = (
        business_predictions == y
    ).mean()

    print(
        f"Business Rule Accuracy: "
        f"{business_accuracy:.4f}"
    )


if __name__ == "__main__":
    run_baselines()