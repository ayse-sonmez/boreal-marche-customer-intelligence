import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

TRAIN_PATH = Path(
    "data/processed/customer_features_train.parquet"
)

TEST_PATH = Path(
    "data/processed/customer_features_test.parquet"
)

RESULTS_PATH = Path(
    "reports/baseline_test_results.csv"
)


def recall_top_decile(
    y_true,
    scores,
):
    """
    Recall captured in the 10% of customers with
    the highest score.
    """

    number_top = max(
        1,
        int(np.ceil(len(y_true) * 0.10)),
    )

    ranking = pd.DataFrame(
        {
            "actual": np.asarray(y_true),
            "score": np.asarray(scores),
        }
    )

    ranking = ranking.sort_values(
        "score",
        ascending=False,
    )

    top_decile = ranking.head(number_top)

    total_positives = ranking[
        "actual"
    ].sum()

    if total_positives == 0:
        return 0.0

    positives_in_top_decile = (
        top_decile["actual"].sum()
    )

    return float(
        positives_in_top_decile
        / total_positives
    )


def evaluate(
    name,
    y_true,
    scores,
    fit_time,
):
    """
    Calculate the same evaluation metrics used
    for the retention models.
    """

    return {
        "model": name,
        "pr_auc": average_precision_score(
            y_true,
            scores,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            scores,
        ),
        "brier": brier_score_loss(
            y_true,
            scores,
        ),
        "recall_top_decile": recall_top_decile(
            y_true,
            scores,
        ),
        "fit_time_seconds": fit_time,
    }


def main():

    print(
        "Loading training and final test data..."
    )

    train_df = pd.read_parquet(
        TRAIN_PATH
    )

    test_df = pd.read_parquet(
        TEST_PATH
    )

    print(
        f"Training customers: "
        f"{len(train_df):,}"
    )

    print(
        f"Final test customers: "
        f"{len(test_df):,}"
    )

    y_train = train_df[
        "label_90d"
    ].astype(int)

    y_test = test_df[
        "label_90d"
    ].astype(int)

    # We only need a simple input column for
    # DummyClassifier because it ignores features.
    X_train_dummy = train_df[
        ["recency_days"]
    ]

    X_test_dummy = test_df[
        ["recency_days"]
    ]

    results = []

    # -----------------------------------------------------
    # 1. DUMMY BASELINE
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("DUMMY CLASSIFIER")
    print("=" * 60)

    dummy_model = DummyClassifier(
        strategy="prior",
        random_state=42,
    )

    start_time = time.time()

    dummy_model.fit(
        X_train_dummy,
        y_train,
    )

    dummy_fit_time = (
        time.time() - start_time
    )

    dummy_probabilities = (
        dummy_model.predict_proba(
            X_test_dummy
        )[:, 1]
    )

    dummy_result = evaluate(
        "Dummy Classifier",
        y_test,
        dummy_probabilities,
        dummy_fit_time,
    )

    results.append(
        dummy_result
    )

    print(
        f"PR-AUC: "
        f"{dummy_result['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{dummy_result['roc_auc']:.4f}"
    )

    print(
        f"Brier: "
        f"{dummy_result['brier']:.4f}"
    )

    print(
        f"Recall@Top10%: "
        f"{dummy_result['recall_top_decile']:.4f}"
    )

    print(
        f"Fit time: "
        f"{dummy_result['fit_time_seconds']:.4f}s"
    )

    # -----------------------------------------------------
    # 2. BUSINESS RULE BASELINE
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("BUSINESS RULE")
    print("=" * 60)

    # Simple business heuristic:
    # customers who purchased within the last
    # 90 days are predicted as likely to return.
    business_scores = (
        test_df["recency_days"] <= 90
    ).astype(float)

    # No training is required for this rule.
    business_fit_time = 0.0

    business_result = evaluate(
        "Business Rule: Recency <= 90 days",
        y_test,
        business_scores,
        business_fit_time,
    )

    results.append(
        business_result
    )

    print(
        "Rule: recency_days <= 90"
    )

    print(
        f"PR-AUC: "
        f"{business_result['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{business_result['roc_auc']:.4f}"
    )

    print(
        f"Brier: "
        f"{business_result['brier']:.4f}"
    )

    print(
        f"Recall@Top10%: "
        f"{business_result['recall_top_decile']:.4f}"
    )

    print(
        f"Fit time: "
        f"{business_result['fit_time_seconds']:.4f}s"
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL BASELINE RESULTS")
    print("=" * 70)

    print(
        results_df.round(4).to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to: "
        f"{RESULTS_PATH}"
    )

    print()
    print(
        "Note: The business rule was defined "
        "before final model evaluation. "
        "The held-out test set is used here only "
        "for final baseline reporting."
    )


if __name__ == "__main__":
    main()