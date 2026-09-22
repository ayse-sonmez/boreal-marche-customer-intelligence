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


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = Path("data/processed/customer_features_train.parquet")
REPORTS_DIR = Path("reports")

RANDOM_STATE = 42

# Best parameters found with Optuna on the training data
N_ESTIMATORS = 157
LEARNING_RATE = 0.01436244069943827
MAX_DEPTH = 3

# Simple business assumptions for threshold selection
OFFER_COST = 5.0
VALUE_OF_SUCCESSFUL_TARGET = 20.0


# ---------------------------------------------------------
# Build model pipeline
# ---------------------------------------------------------

def build_model():
    model = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )

    return model


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")

    df = pd.read_parquet(DATA_PATH)

    y = df["label_90d"]

    columns_to_drop = [
        "Customer ID",
        "first_purchase_date",
        "last_purchase_date",
        "label_90d",
    ]

    X = df.drop(columns=columns_to_drop)

    categorical_features = ["country"]

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # ---------------------------------------------------------
    # Internal train / validation split
    # ---------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Sub-training customers: {len(X_train):,}")
    print(f"Validation customers: {len(X_val):,}")

    # ---------------------------------------------------------
    # Uncalibrated model
    # ---------------------------------------------------------

    base_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_model()),
        ]
    )

    print("\nTraining uncalibrated model...")

    base_model.fit(X_train, y_train)

    uncalibrated_prob = base_model.predict_proba(X_val)[:, 1]

    uncalibrated_pr_auc = average_precision_score(
        y_val,
        uncalibrated_prob,
    )

    uncalibrated_roc_auc = roc_auc_score(
        y_val,
        uncalibrated_prob,
    )

    uncalibrated_brier = brier_score_loss(
        y_val,
        uncalibrated_prob,
    )

    # ---------------------------------------------------------
    # Calibrated model
    # ---------------------------------------------------------

    calibrated_base_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_model()),
        ]
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=calibrated_base_model,
        method="sigmoid",
        cv=5,
    )

    print("Training calibrated model...")

    calibrated_model.fit(X_train, y_train)

    calibrated_prob = calibrated_model.predict_proba(X_val)[:, 1]

    calibrated_pr_auc = average_precision_score(
        y_val,
        calibrated_prob,
    )

    calibrated_roc_auc = roc_auc_score(
        y_val,
        calibrated_prob,
    )

    calibrated_brier = brier_score_loss(
        y_val,
        calibrated_prob,
    )

    # ---------------------------------------------------------
    # Reliability curve
    # ---------------------------------------------------------

    uncalibrated_true, uncalibrated_pred = calibration_curve(
        y_val,
        uncalibrated_prob,
        n_bins=10,
        strategy="uniform",
    )

    calibrated_true, calibrated_pred = calibration_curve(
        y_val,
        calibrated_prob,
        n_bins=10,
        strategy="uniform",
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        uncalibrated_pred,
        uncalibrated_true,
        marker="o",
        label="Before calibration",
    )

    plt.plot(
        calibrated_pred,
        calibrated_true,
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
    plt.title("Retention Model Reliability Curve")
    plt.legend()
    plt.tight_layout()

    reliability_path = (
        REPORTS_DIR / "calibration_reliability_curve.png"
    )

    plt.savefig(
        reliability_path,
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------
    # Profit-based threshold
    # ---------------------------------------------------------

    thresholds = np.arange(
        0.05,
        0.96,
        0.01,
    )

    profit_results = []

    for threshold in thresholds:

        # Low return probability means customer is considered at risk.
        targeted = calibrated_prob < threshold

        profit = np.where(
            targeted & (y_val.to_numpy() == 0),
            VALUE_OF_SUCCESSFUL_TARGET - OFFER_COST,
            np.where(
                targeted & (y_val.to_numpy() == 1),
                -OFFER_COST,
                0.0,
            ),
        ).sum()

        profit_results.append(
            {
                "threshold": threshold,
                "profit": profit,
                "customers_targeted": int(targeted.sum()),
            }
        )

    profit_df = pd.DataFrame(profit_results)

    best_row = profit_df.loc[
        profit_df["profit"].idxmax()
    ]

    best_threshold = float(best_row["threshold"])
    best_profit = float(best_row["profit"])

    # Profit using the common 0.50 threshold
    threshold_050_row = profit_df.iloc[
        (profit_df["threshold"] - 0.50).abs().argsort()[:1]
    ]

    profit_at_050 = float(
        threshold_050_row["profit"].iloc[0]
    )

    profit_improvement = best_profit - profit_at_050

    # Save threshold results
    threshold_report_path = (
        REPORTS_DIR / "profit_threshold_results.csv"
    )

    profit_df.to_csv(
        threshold_report_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\nUNCALIBRATED MODEL")
    print("------------------")
    print(f"PR-AUC:  {uncalibrated_pr_auc:.4f}")
    print(f"ROC-AUC: {uncalibrated_roc_auc:.4f}")
    print(f"Brier:   {uncalibrated_brier:.4f}")

    print("\nCALIBRATED MODEL")
    print("----------------")
    print(f"PR-AUC:  {calibrated_pr_auc:.4f}")
    print(f"ROC-AUC: {calibrated_roc_auc:.4f}")
    print(f"Brier:   {calibrated_brier:.4f}")

    print("\nPROFIT-BASED THRESHOLD")
    print("----------------------")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Estimated scenario profit: ${best_profit:,.0f}")
    print(f"Profit at threshold 0.50: ${profit_at_050:,.0f}")
    print(
        f"Improvement over 0.50: ${profit_improvement:,.0f}"
    )

    print(
        "\nNote: Profit values are scenario-based estimates, "
        "not observed causal uplift."
    )

    print(
        f"\nReliability curve saved to: "
        f"{reliability_path}"
    )

    print(
        f"Threshold results saved to: "
        f"{threshold_report_path}"
    )


if __name__ == "__main__":
    main()