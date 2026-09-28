from __future__ import annotations

from pathlib import Path

import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(
    "data/processed/customer_features_gold.parquet"
)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Gold dataset not found: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    y = df["label_90d"]

    X = df.drop(
        columns=[
            "Customer ID",
            "first_purchase_date",
            "last_purchase_date",
            "label_90d",
        ]
    )

    categorical_features = ["country"]

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
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

    return X, y, preprocessor


def tune_logistic_regression(
    X,
    y,
    preprocessor,
):
    def objective(trial):

        c_value = trial.suggest_float(
            "C",
            0.01,
            10.0,
            log=True,
        )

        model = LogisticRegression(
            C=c_value,
            max_iter=1000,
            random_state=42,
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="average_precision",
            n_jobs=-1,
        )

        return scores.mean()

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=20,
    )

    return study


def tune_gradient_boosting(
    X,
    y,
    preprocessor,
):
    def objective(trial):

        n_estimators = trial.suggest_int(
            "n_estimators",
            50,
            200,
        )

        learning_rate = trial.suggest_float(
            "learning_rate",
            0.01,
            0.2,
            log=True,
        )

        max_depth = trial.suggest_int(
            "max_depth",
            1,
            4,
        )

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="average_precision",
            n_jobs=-1,
        )

        return scores.mean()

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=20,
    )

    return study


def main():
    X, y, preprocessor = load_data()

    print(f"Customers: {len(X):,}")
    print(f"Positive rate: {y.mean() * 100:.2f}%")

    print()
    print("=" * 60)
    print("TUNING LOGISTIC REGRESSION")
    print("=" * 60)

    logistic_study = tune_logistic_regression(
        X,
        y,
        preprocessor,
    )

    print()
    print("Best Logistic Regression PR-AUC:")
    print(f"{logistic_study.best_value:.4f}")

    print("Best parameters:")
    print(logistic_study.best_params)

    print()
    print("=" * 60)
    print("TUNING GRADIENT BOOSTING")
    print("=" * 60)

    gradient_study = tune_gradient_boosting(
        X,
        y,
        preprocessor,
    )

    print()
    print("Best Gradient Boosting PR-AUC:")
    print(f"{gradient_study.best_value:.4f}")

    print("Best parameters:")
    print(gradient_study.best_params)


if __name__ == "__main__":
    main()