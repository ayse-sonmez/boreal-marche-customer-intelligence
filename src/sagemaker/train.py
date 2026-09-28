import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-estimators", type=int, default=157)
    parser.add_argument("--learning-rate", type=float, default=0.01436244069943827)
    parser.add_argument("--max-depth", type=int, default=3)

    return parser.parse_args()


def main():
    args = parse_args()

    # SageMaker provides these paths automatically during a training job.
    # The default values let us run the same script locally.
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "data/processed")
    model_dir = os.environ.get("SM_MODEL_DIR", "artifacts")

    train_path = Path(train_dir) / "customer_features_train.parquet"

    print(f"Reading training data from: {train_path}")

    df = pd.read_parquet(train_path)

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
        column for column in X.columns
        if column not in categorical_features
    ]

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

    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("Training Gradient Boosting model...")

    pipeline.fit(X, y)

    os.makedirs(model_dir, exist_ok=True)

    model_path = Path(model_dir) / "model.joblib"
    joblib.dump(pipeline, model_path)

    print("Training completed.")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()