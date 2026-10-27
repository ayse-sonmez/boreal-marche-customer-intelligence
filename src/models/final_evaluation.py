import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
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

# Selected previously using validation data only.
# Do not change this based on final test results.
PROFIT_THRESHOLD = 0.73

OFFER_COST = 5.0
VALUE_OF_SUCCESSFUL_TARGET = 20.0


def calculate_profit(
    y_true,
    return_probabilities,
    threshold,
):
    """
    Customers below the return-probability threshold
    are treated as at risk and receive an offer.

    Profit values are scenario assumptions,
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
    Fraction of all positive customers captured
    in the top 10% ranked by predicted return probability.
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


def get_git_commit():
    """
    Return the current Git commit SHA.
    """

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except (OSError, __import__('subprocess').CalledProcessError):
        return "unavailable"


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

    feature_names = list(
        X_train.columns
    )

    categorical_features = [
        "country"
    ]

    numeric_features = [
        column
        for column in feature_names
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

    # Calibration is performed only with training data.
    final_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
    )

    print(
        "\nTraining final calibrated model..."
    )

    fit_start = time.perf_counter()

    final_model.fit(
        X_train,
        y_train,
    )

    fit_time_seconds = (
        time.perf_counter() - fit_start
    )

    print(
        f"Fit time: "
        f"{fit_time_seconds:.2f} seconds"
    )

    print(
        "\nEvaluating final held-out test set..."
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

    profit_improvement = (
        estimated_profit
        - estimated_profit_05
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
        "Recall @ top decile: "
        f"{top_decile_recall:.4f}"
    )

    print(
        f"Fit time: "
        f"{fit_time_seconds:.2f} seconds"
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

    print(
        "Scenario profit improvement: "
        f"${profit_improvement:,.2f}"
    )

    # ---------------------------------------------------------
    # Save complete inference pipeline
    # ---------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    # ---------------------------------------------------------
    # Model sidecar metadata
    # ---------------------------------------------------------

    metadata = {
        "model_version": "1.0",
        "model_name": (
            "GradientBoostingClassifier"
        ),
        "calibration": "sigmoid",
        "training_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_count": int(
            X_train.shape[1]
        ),
        "features_in_order": (
            feature_names
        ),
        "categorical_features": (
            categorical_features
        ),
        "numeric_features": (
            numeric_features
        ),
        "threshold": (
            PROFIT_THRESHOLD
        ),
        "threshold_selected_on": (
            "validation data only"
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
        "fit_time_seconds": float(
            fit_time_seconds
        ),
        "best_parameters": (
            BEST_PARAMS
        ),
        "git_commit": (
            get_git_commit()
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "libraries": {
            "python": (
                platform.python_version()
            ),
            "pandas": (
                pd.__version__
            ),
            "numpy": (
                np.__version__
            ),
            "scikit_learn": (
                sklearn.__version__
            ),
            "joblib": (
                joblib.__version__
            ),
        },
        "profit_results": {
            "threshold_0_73": float(
                estimated_profit
            ),
            "threshold_0_50": float(
                estimated_profit_05
            ),
            "improvement": float(
                profit_improvement
            ),
        },
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
        f"\nFinal calibrated model saved to: "
        f"{MODEL_PATH}"
    )

    print(
        f"Model metadata saved to: "
        f"{META_PATH}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The held-out test set has now been used "
        "for final evaluation."
    )

    print(
        "Do not change the model, hyperparameters "
        "or threshold based on these test results."
    )


if __name__ == "__main__":
    main()