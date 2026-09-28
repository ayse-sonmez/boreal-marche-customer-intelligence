from __future__ import annotations

from pathlib import Path

import mlflow
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(
   "data/processed/customer_features_train.parquet"
) 


MLFLOW_EXPERIMENT = "boreal-marche-retention-tuning"


def load_data():
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


def evaluate_with_cv(
    trial,
    model,
    model_name,
    X,
    y,
    preprocessor,
):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_scores = []

    with mlflow.start_run(
        run_name=f"{model_name}_trial_{trial.number}"
    ):
        mlflow.set_tag(
            "model_name",
            model_name,
        )

        mlflow.set_tag(
            "optuna_trial",
            trial.number,
        )

        mlflow.log_params(
            trial.params
        )

        for fold_number, (
            train_index,
            validation_index,
        ) in enumerate(
            cv.split(X, y),
            start=1,
        ):
            X_train = X.iloc[train_index]
            X_validation = X.iloc[
                validation_index
            ]

            y_train = y.iloc[train_index]
            y_validation = y.iloc[
                validation_index
            ]

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

            pipeline.fit(
                X_train,
                y_train,
            )

            probabilities = (
                pipeline.predict_proba(
                    X_validation
                )[:, 1]
            )

            pr_auc = average_precision_score(
                y_validation,
                probabilities,
            )

            fold_scores.append(pr_auc)

            current_mean = (
                sum(fold_scores)
                / len(fold_scores)
            )

            mlflow.log_metric(
                f"fold_{fold_number}_pr_auc",
                pr_auc,
            )

            trial.report(
                current_mean,
                step=fold_number,
            )

            if trial.should_prune():

                mlflow.set_tag(
                    "trial_status",
                    "pruned",
                )

                mlflow.log_metric(
                    "pr_auc_before_pruning",
                    current_mean,
                )

                raise optuna.TrialPruned()

        mean_pr_auc = (
            sum(fold_scores)
            / len(fold_scores)
        )

        mlflow.log_metric(
            "mean_pr_auc",
            mean_pr_auc,
        )

        mlflow.set_tag(
            "trial_status",
            "completed",
        )

        return mean_pr_auc


def logistic_objective(
    trial,
    X,
    y,
    preprocessor,
):
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

    return evaluate_with_cv(
        trial=trial,
        model=model,
        model_name="logistic_regression",
        X=X,
        y=y,
        preprocessor=preprocessor,
    )


def gradient_objective(
    trial,
    X,
    y,
    preprocessor,
):
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

    return evaluate_with_cv(
        trial=trial,
        model=model,
        model_name="gradient_boosting",
        X=X,
        y=y,
        preprocessor=preprocessor,
    )


def create_study():
    sampler = optuna.samplers.TPESampler(
        seed=42
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2,
    )

    return optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    X, y, preprocessor = load_data()

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    print(
        f"Customers: {len(X):,}"
    )

    print(
        f"Positive rate: "
        f"{y.mean() * 100:.2f}%"
    )

    print()
    print("=" * 60)
    print("LOGISTIC REGRESSION - OPTUNA + MLFLOW")
    print("=" * 60)

    logistic_study = create_study()

    logistic_study.optimize(
        lambda trial: logistic_objective(
            trial,
            X,
            y,
            preprocessor,
        ),
        n_trials=20,
    )

    print()
    print("Best Logistic Regression PR-AUC:")
    print(
        f"{logistic_study.best_value:.4f}"
    )

    print("Best parameters:")
    print(
        logistic_study.best_params
    )

    print()
    print("=" * 60)
    print("GRADIENT BOOSTING - OPTUNA + MLFLOW")
    print("=" * 60)

    gradient_study = create_study()

    gradient_study.optimize(
        lambda trial: gradient_objective(
            trial,
            X,
            y,
            preprocessor,
        ),
        n_trials=20,
    )

    print()
    print("Best Gradient Boosting PR-AUC:")
    print(
        f"{gradient_study.best_value:.4f}"
    )

    print("Best parameters:")
    print(
        gradient_study.best_params
    )

    print()
    print(
        "MLflow experiment:",
        MLFLOW_EXPERIMENT,
    )


if __name__ == "__main__":
    main()