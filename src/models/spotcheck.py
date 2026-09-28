from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(
    "data/processed/customer_features_train.parquet"
)

RESULTS_PATH = Path(
    "reports/spotcheck_results.csv"
)

BOXPLOT_PATH = Path(
    "reports/spotcheck_pr_auc_boxplot.png"
)


def recall_top_decile(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """
    Recall captured in the 10% of customers with the
    highest predicted probability of returning.
    """

    number_top = max(
        1,
        int(np.ceil(len(y_true) * 0.10)),
    )

    ranking = pd.DataFrame(
        {
            "actual": y_true.to_numpy(),
            "probability": probabilities,
        }
    )

    ranking = ranking.sort_values(
        "probability",
        ascending=False,
    )

    top_decile = ranking.head(number_top)

    total_positives = ranking["actual"].sum()

    if total_positives == 0:
        return 0.0

    positives_in_top_decile = (
        top_decile["actual"].sum()
    )

    return float(
        positives_in_top_decile
        / total_positives
    )


def main() -> None:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------
    y = df["label_90d"].astype(int)

    # -----------------------------------------------------
    # Remove columns that should not be model inputs
    # -----------------------------------------------------
    X = df.drop(
        columns=[
            "Customer ID",
            "first_purchase_date",
            "last_purchase_date",
            "label_90d",
        ]
    )

    # -----------------------------------------------------
    # Column groups
    # -----------------------------------------------------
    categorical_features = ["country"]

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    print(f"Customers: {len(df):,}")
    print(f"Numeric features: {len(numeric_features)}")
    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )
    print(f"Positive rate: {y.mean() * 100:.2f}%")

    # -----------------------------------------------------
    # Same preprocessing for every model
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # 8 model families
    # -----------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
        ),

        "Gaussian Naive Bayes": GaussianNB(),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),

        "AdaBoost": AdaBoostClassifier(
            random_state=42,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # -----------------------------------------------------
    # Same 5 folds for every model
    # -----------------------------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    # Keep individual fold PR-AUC values for boxplot
    boxplot_results = {}

    # -----------------------------------------------------
    # Spot-check
    # -----------------------------------------------------
    for model_name, model in models.items():

        print()
        print("=" * 60)
        print(f"Testing: {model_name}")
        print("=" * 60)

        fold_pr_auc = []
        fold_roc_auc = []
        fold_brier = []
        fold_recall_top_decile = []
        fold_fit_times = []

        for fold_number, (
            train_index,
            validation_index,
        ) in enumerate(
            cv.split(X, y),
            start=1,
        ):

            X_train = X.iloc[train_index]
            X_validation = X.iloc[validation_index]

            y_train = y.iloc[train_index]
            y_validation = y.iloc[validation_index]

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

            start_time = time.time()

            pipeline.fit(
                X_train,
                y_train,
            )

            fit_time = (
                time.time()
                - start_time
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

            roc_auc = roc_auc_score(
                y_validation,
                probabilities,
            )

            brier = brier_score_loss(
                y_validation,
                probabilities,
            )

            top_decile_recall = recall_top_decile(
                y_validation,
                probabilities,
            )

            fold_pr_auc.append(pr_auc)
            fold_roc_auc.append(roc_auc)
            fold_brier.append(brier)
            fold_recall_top_decile.append(
                top_decile_recall
            )
            fold_fit_times.append(fit_time)

            print(
                f"Fold {fold_number}: "
                f"PR-AUC={pr_auc:.4f} | "
                f"ROC-AUC={roc_auc:.4f} | "
                f"Brier={brier:.4f} | "
                f"Recall@10%={top_decile_recall:.4f} | "
                f"Fit={fit_time:.2f}s"
            )

        boxplot_results[model_name] = fold_pr_auc

        result = {
            "model": model_name,

            "mean_pr_auc": (
                sum(fold_pr_auc)
                / len(fold_pr_auc)
            ),

            "mean_roc_auc": (
                sum(fold_roc_auc)
                / len(fold_roc_auc)
            ),

            "mean_brier": (
                sum(fold_brier)
                / len(fold_brier)
            ),

            "mean_recall_top_decile": (
                sum(fold_recall_top_decile)
                / len(fold_recall_top_decile)
            ),

            "mean_fit_time_seconds": (
                sum(fold_fit_times)
                / len(fold_fit_times)
            ),
        }

        results.append(result)

    # -----------------------------------------------------
    # Final comparison
    # -----------------------------------------------------
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="mean_pr_auc",
        ascending=False,
    ).reset_index(drop=True)

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # PR-AUC boxplot
    # -----------------------------------------------------
    ordered_models = results_df["model"].tolist()

    plot_data = [
        boxplot_results[model_name]
        for model_name in ordered_models
    ]

    plt.figure(figsize=(12, 7))

    plt.boxplot(
        plot_data,
        tick_labels=ordered_models,
    )

    plt.ylabel("PR-AUC")
    plt.xlabel("Model")
    plt.title(
        "Spot-Check PR-AUC Across 5 Cross-Validation Folds"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        BOXPLOT_PATH,
        dpi=150,
    )

    plt.close()

    # -----------------------------------------------------
    # Print final results
    # -----------------------------------------------------
    print()
    print("=" * 70)
    print("FINAL SPOT-CHECK RESULTS")
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

    print(
        f"PR-AUC boxplot saved to: "
        f"{BOXPLOT_PATH}"
    )


if __name__ == "__main__":
    main()