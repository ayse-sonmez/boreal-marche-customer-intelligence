from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRAIN_PATH = Path(
    "data/processed/customer_features_train.parquet"
)

TEST_PATH = Path(
    "data/processed/customer_features_test.parquet"
)

MODEL_PATH = Path(
    "artifacts/model.joblib"
)

META_PATH = Path(
    "artifacts/model_meta.json"
)

RANDOM_STATE = 42

BEST_PARAMS = {
    "n_estimators": 157,
    "learning_rate": 0.01436244069943827,
    "max_depth": 3,
}

# Selected using validation data only
PROFIT_THRESHOLD = 0.73

OFFER_COST = 5.0
VALUE_OF_SUCCESSFUL_TARGET = 20.0


def calculate_profit(
    y_true,
    return_probabilities,
    threshold,
):
    """
    Customers below the threshold are treated as at risk
    and receive a retention offer.

    The profit values are project assumptions,
    not observed causal campaign profit.
    """

    targeted = return_probabilities < threshold

    profit = np.where(
        targeted & (y_true == 0),
        VALUE_OF_SUCCESSFUL_TARGET - OFFER_COST,
        np.where(
            targeted & (y_true == 1),
            -OFFER_COST,
            0.0,
        ),
    )

    return float(profit.sum())


def recall_top_decile(
    y_true,
    probabilities,
):
    """
    Recall among the 10% of customers with the highest
    predicted probability of returning.
    """

    number_top = max(
        1,
        int(np.ceil(len(y_true) * 0.10)),
    )

    ranking = pd.DataFrame(
        {
            "actual": np.asarray(y_true),
            "probability": probabilities,
        }
    )

    ranking = ranking.sort_values(
        "probability",
        ascending=False,
    )

    top_decile = ranking.head(number_top)

    total_positives = ranking["actual"].sum()

    positives_in_top_decile = (
        top_decile["actual"].sum()
    )

    if total_positives == 0:
        return 0.0

    return float(
        positives_in_top_decile
        / total_positives
    )


def main():

    print("Loading training and final test data...")

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    print(
        f"Training customers: {len(train_df):,}"
    )

    print(
        f"Final test customers: {len(test_df):,}"
    )

    excluded_columns = [
        "Customer ID",
        "first_purchase_date",
        "last_purchase_date",
        "label_90d",
    ]

    X_train = train_df.drop(
        columns=excluded_columns
    )

    y_train = train_df[
        "label_90d"
    ].astype(int)

    X_test = test_df.drop(
        columns=excluded_columns
    )

    y_test = test_df[
        "label_90d"
    ].astype(int)

    categorical_features = [
        "country"
    ]

    numeric_features = [
        column
        for column in X_train.columns
        if column not in categorical_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    base_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                    **BEST_PARAMS,
                ),
            ),
        ]
    )

    # Calibrate using only training data.
    final_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
    )

    print(
        "\nTraining final calibrated model..."
    )

    final_model.fit(
        X_train,
        y_train,
    )

    print(
        "Evaluating final held-out test set..."
    )

    test_probabilities = (
        final_model.predict_proba(
            X_test
        )[:, 1]
    )

    pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    brier = brier_score_loss(
        y_test,
        test_probabilities,
    )

    top_decile_recall = recall_top_decile(
        y_test,
        test_probabilities,
    )

    estimated_profit = calculate_profit(
        y_test.to_numpy(),
        test_probabilities,
        PROFIT_THRESHOLD,
    )

    estimated_profit_05 = calculate_profit(
        y_test.to_numpy(),
        test_probabilities,
        0.50,
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "FINAL HELD-OUT TEST RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"PR-AUC: {pr_auc:.4f}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"Brier: {brier:.4f}"
    )

    print(
        f"Recall @ top decile: "
        f"{top_decile_recall:.4f}"
    )

    print(
        f"Profit threshold: "
        f"{PROFIT_THRESHOLD:.2f}"
    )

    print(
        f"Scenario profit at "
        f"{PROFIT_THRESHOLD:.2f}: "
        f"${estimated_profit:,.2f}"
    )

    print(
        "Scenario profit at 0.50: "
        f"${estimated_profit_05:,.2f}"
    )

    # Save complete inference model
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    # Save simple model metadata
    metadata = {
        "model_name": (
            "GradientBoostingClassifier"
        ),
        "calibration": "sigmoid",
        "training_rows": int(
            len(train_df)
        ),
        "test_rows": int(
            len(test_df)
        ),
        "feature_count": int(
            X_train.shape[1]
        ),
        "threshold": (
            PROFIT_THRESHOLD
        ),
        "test_pr_auc": float(
            pr_auc
        ),
        "test_roc_auc": float(
            roc_auc
        ),
        "test_brier": float(
            brier
        ),
        "test_recall_top_decile": float(
            top_decile_recall
        ),
        "best_parameters": (
            BEST_PARAMS
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "profit_assumptions": {
            "offer_cost": (
                OFFER_COST
            ),
            "successful_target_value": (
                VALUE_OF_SUCCESSFUL_TARGET
            ),
            "note": (
                "Scenario-based assumptions, "
                "not observed causal profit."
            ),
        },
    }

    with open(
        META_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"\nModel saved to: "
        f"{MODEL_PATH}"
    )

    print(
        f"Metadata saved to: "
        f"{META_PATH}"
    )

    print(
        "\nIMPORTANT: "
        "The held-out test set has now been "
        "used for final evaluation. "
        "Do not tune the model based on "
        "these test results."
    )


if __name__ == "__main__":
    main()
    