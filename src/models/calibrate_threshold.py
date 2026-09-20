from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/processed/customer_features_train.parquet")

RANDOM_STATE = 42

# Best parameters found with Optuna
BEST_PARAMS = {
    "n_estimators": 157,
    "learning_rate": 0.01436244069943827,
    "max_depth": 3,
}

# Simple business assumptions for the retention campaign
OFFER_COST = 5.0
VALUE_OF_SUCCESSFUL_TARGET = 20.0


def calculate_profit(y_true, return_probabilities, threshold):
    """
    Customers with a return probability below the threshold
    are considered at risk and receive an offer.

    This is a simple business scenario, not measured causal uplift.
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

    return profit.sum()


def main():
    # Load training data only.
    # The final test set is still untouched.
    df = pd.read_parquet(DATA_PATH)

    excluded_columns = [
        "Customer ID",
        "first_purchase_date",
        "last_purchase_date",
        "label_90d",
    ]

    X = df.drop(columns=excluded_columns)
    y = df["label_90d"].astype(int)

    categorical_features = ["country"]

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    # Numeric preprocessing
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Country preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
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

    # Split the training data again into model training
    # and validation data.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Best Gradient Boosting model from Optuna
    base_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                    **BEST_PARAMS,
                ),
            ),
        ]
    )

    # ---------------------------------------------------------
    # 1. UNCALIBRATED MODEL
    # ---------------------------------------------------------

    base_model.fit(X_train, y_train)

    uncalibrated_prob = base_model.predict_proba(
        X_validation
    )[:, 1]

    print("UNCALIBRATED MODEL")

    print(
        f"PR-AUC: "
        f"{average_precision_score(y_validation, uncalibrated_prob):.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc_score(y_validation, uncalibrated_prob):.4f}"
    )

    print(
        f"Brier: "
        f"{brier_score_loss(y_validation, uncalibrated_prob):.4f}"
    )

    # ---------------------------------------------------------
    # 2. CALIBRATED MODEL
    # ---------------------------------------------------------

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
    )

    calibrated_model.fit(X_train, y_train)

    calibrated_prob = calibrated_model.predict_proba(
        X_validation
    )[:, 1]

    print("\nCALIBRATED MODEL")

    print(
        f"PR-AUC: "
        f"{average_precision_score(y_validation, calibrated_prob):.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc_score(y_validation, calibrated_prob):.4f}"
    )

    print(
        f"Brier: "
        f"{brier_score_loss(y_validation, calibrated_prob):.4f}"
    )

    # ---------------------------------------------------------
    # 3. RELIABILITY CURVE
    # ---------------------------------------------------------

    prob_true_before, prob_pred_before = calibration_curve(
        y_validation,
        uncalibrated_prob,
        n_bins=10,
        strategy="uniform",
    )

    prob_true_after, prob_pred_after = calibration_curve(
        y_validation,
        calibrated_prob,
        n_bins=10,
        strategy="uniform",
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        prob_pred_before,
        prob_true_before,
        marker="o",
        label="Before calibration",
    )

    plt.plot(
        prob_pred_after,
        prob_true_after,
        marker="o",
        label="After calibration",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Gradient Boosting Reliability Curve")
    plt.legend()
    plt.tight_layout()

    Path("reports").mkdir(exist_ok=True)

    plt.savefig(
        "reports/calibration_reliability_curve.png",
        dpi=150,
    )

    plt.close()

    print(
        "\nReliability curve saved to: "
        "reports/calibration_reliability_curve.png"
    )

    # ---------------------------------------------------------
    # 4. PROFIT-BASED THRESHOLD
    # ---------------------------------------------------------

    thresholds = np.arange(0.05, 0.96, 0.01)

    profits = [
        calculate_profit(
            y_validation.to_numpy(),
            calibrated_prob,
            threshold,
        )
        for threshold in thresholds
    ]

    best_index = int(np.argmax(profits))

    best_threshold = thresholds[best_index]
    best_profit = profits[best_index]

    profit_at_05 = calculate_profit(
        y_validation.to_numpy(),
        calibrated_prob,
        0.50,
    )

    print("\nPROFIT-BASED THRESHOLD")

    print(
        f"Offer cost assumption: "
        f"${OFFER_COST:.2f}"
    )

    print(
        "Value of successfully targeting an at-risk customer: "
        f"${VALUE_OF_SUCCESSFUL_TARGET:.2f}"
    )

    print(
        f"Best threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Estimated profit at best threshold: "
        f"${best_profit:,.2f}"
    )

    print(
        f"Estimated profit at threshold 0.50: "
        f"${profit_at_05:,.2f}"
    )

    print(
        "\nNote: Profit values are scenario-based assumptions "
        "for the project, not observed causal campaign profit."
    )


if __name__ == "__main__":
    main()