from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/customer_features_gold.parquet")
TRAIN_PATH = Path("data/processed/customer_features_train.parquet")
TEST_PATH = Path("data/processed/customer_features_test.parquet")

TEST_SIZE = 0.20
RANDOM_STATE = 42


def main():
    df = pd.read_parquet(DATA_PATH)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label_90d"],
    )

    train_df.to_parquet(TRAIN_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)

    print(f"Total customers: {len(df):,}")
    print(f"Training customers: {len(train_df):,}")
    print(f"Final test customers: {len(test_df):,}")

    print(
        f"Training positive rate: "
        f"{train_df['label_90d'].mean():.2%}"
    )
    print(
        f"Test positive rate: "
        f"{test_df['label_90d'].mean():.2%}"
    )

    print("\nFinal test set is reserved and should not be used for tuning.")


if __name__ == "__main__":
    main()