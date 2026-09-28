from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/online_retail_ii_silver.parquet")
OUTPUT_PATH = Path("data/processed/customer_features_gold.parquet")

CUTOFF_DATE = pd.Timestamp("2011-09-10 23:59:59")
LABEL_END_DATE = CUTOFF_DATE + pd.Timedelta(days=90)


def build_customer_features() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    print(f"Silver rows: {len(df):,}")
    print(f"Cutoff date: {CUTOFF_DATE}")
    print(f"Label end date: {LABEL_END_DATE}")

    # -----------------------------------------------------
    # Completed purchases only for Head A
    # -----------------------------------------------------
    df = df[~df["IsCancelled"]].copy()

    # Observation period
    observation_df = df[
        df["InvoiceDate"] <= CUTOFF_DATE
    ].copy()

    # Future 90-day label period
    label_df = df[
        (df["InvoiceDate"] > CUTOFF_DATE)
        & (df["InvoiceDate"] <= LABEL_END_DATE)
    ].copy()

    print(f"Observation rows: {len(observation_df):,}")
    print(f"Label-window rows: {len(label_df):,}")

    # -----------------------------------------------------
    # Basic transaction columns
    # -----------------------------------------------------
    observation_df["LineTotal"] = (
        observation_df["Quantity"] * observation_df["Price"]
    )

    observation_df["PurchaseDate"] = (
        observation_df["InvoiceDate"].dt.date
    )

    observation_df["PurchaseMonth"] = (
        observation_df["InvoiceDate"].dt.to_period("M")
    )

    observation_df["PurchaseWeekday"] = (
        observation_df["InvoiceDate"].dt.dayofweek
    )

    observation_df["PurchaseHour"] = (
        observation_df["InvoiceDate"].dt.hour
    )

    # -----------------------------------------------------
    # Customer-level basic features
    # -----------------------------------------------------
    customer_features = (
        observation_df
        .groupby("Customer ID")
        .agg(
            # RFM
            recency_days=(
                "InvoiceDate",
                lambda x: (CUTOFF_DATE - x.max()).days,
            ),
            total_orders=("Invoice", "nunique"),
            total_spend=("LineTotal", "sum"),

            # Basket / product behaviour
            total_quantity=("Quantity", "sum"),
            unique_products=("StockCode", "nunique"),
            average_price=("Price", "mean"),
            max_price=("Price", "max"),
            min_price=("Price", "min"),
            average_line_value=("LineTotal", "mean"),
            max_line_value=("LineTotal", "max"),

            # Activity
            active_days=("PurchaseDate", "nunique"),
            active_months=("PurchaseMonth", "nunique"),

            # Time
            average_purchase_hour=("PurchaseHour", "mean"),

            # Customer history
            first_purchase_date=("InvoiceDate", "min"),
            last_purchase_date=("InvoiceDate", "max"),
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # Simple calculated features
    # -----------------------------------------------------

    # 16
    customer_features["average_order_value"] = (
        customer_features["total_spend"]
        / customer_features["total_orders"]
    )

    # 17
    customer_features["average_quantity_per_order"] = (
        customer_features["total_quantity"]
        / customer_features["total_orders"]
    )

    # 18
    customer_features["average_products_per_order"] = (
        customer_features["unique_products"]
        / customer_features["total_orders"]
    )

    # 19
    customer_features["customer_lifetime_days"] = (
        customer_features["last_purchase_date"]
        - customer_features["first_purchase_date"]
    ).dt.days

    # 20
    customer_features["orders_per_active_month"] = (
        customer_features["total_orders"]
        / customer_features["active_months"]
    )

    # 21
    customer_features["spend_per_active_month"] = (
        customer_features["total_spend"]
        / customer_features["active_months"]
    )

    # 22
    customer_features["quantity_per_product"] = (
        customer_features["total_quantity"]
        / customer_features["unique_products"]
    )

    # 23
    customer_features["spend_per_product"] = (
        customer_features["total_spend"]
        / customer_features["unique_products"]
    )

    # -----------------------------------------------------
    # Weekend behaviour
    # -----------------------------------------------------
    weekend = (
        observation_df["PurchaseWeekday"] >= 5
    )

    weekend_orders = (
        observation_df[weekend]
        .groupby("Customer ID")["Invoice"]
        .nunique()
        .rename("weekend_orders")
        .reset_index()
    )

    customer_features = customer_features.merge(
        weekend_orders,
        on="Customer ID",
        how="left",
    )

    customer_features["weekend_orders"] = (
        customer_features["weekend_orders"].fillna(0)
    )

    # 24
    customer_features["weekend_order_ratio"] = (
        customer_features["weekend_orders"]
        / customer_features["total_orders"]
    )

    # -----------------------------------------------------
    # Repeat purchase behaviour
    # -----------------------------------------------------
    product_counts = (
        observation_df
        .groupby(["Customer ID", "StockCode"])
        .size()
        .reset_index(name="purchase_count")
    )

    repeat_products = (
        product_counts[
            product_counts["purchase_count"] > 1
        ]
        .groupby("Customer ID")["StockCode"]
        .nunique()
        .rename("repeat_products")
        .reset_index()
    )

    customer_features = customer_features.merge(
        repeat_products,
        on="Customer ID",
        how="left",
    )

    customer_features["repeat_products"] = (
        customer_features["repeat_products"].fillna(0)
    )

    # 25
    customer_features["repeat_product_ratio"] = (
        customer_features["repeat_products"]
        / customer_features["unique_products"]
    )

    # -----------------------------------------------------
    # Purchase cadence
    # -----------------------------------------------------
    order_dates = (
        observation_df[
            ["Customer ID", "Invoice", "InvoiceDate"]
        ]
        .drop_duplicates(
            subset=["Customer ID", "Invoice"]
        )
        .sort_values(
            ["Customer ID", "InvoiceDate"]
        )
    )

    order_dates["days_between_orders"] = (
        order_dates
        .groupby("Customer ID")["InvoiceDate"]
        .diff()
        .dt.total_seconds()
        / 86400
    )

    cadence = (
        order_dates
        .groupby("Customer ID")["days_between_orders"]
        .agg(
            average_days_between_orders="mean",
            max_days_between_orders="max",
        )
        .reset_index()
    )

    customer_features = customer_features.merge(
        cadence,
        on="Customer ID",
        how="left",
    )

    # Customers with one order have no gap.
    customer_features[
        [
            "average_days_between_orders",
            "max_days_between_orders",
        ]
    ] = customer_features[
        [
            "average_days_between_orders",
            "max_days_between_orders",
        ]
    ].fillna(0)

    # -----------------------------------------------------
    # Geography
    # Keep the most common country for each customer.
    # -----------------------------------------------------
    customer_country = (
        observation_df
        .groupby("Customer ID")["Country"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("country")
        .reset_index()
    )

    customer_features = customer_features.merge(
        customer_country,
        on="Customer ID",
        how="left",
    )

    # -----------------------------------------------------
    # Create 90-day target
    # -----------------------------------------------------
    returning_customers = set(
        label_df["Customer ID"].dropna().unique()
    )

    customer_features["label_90d"] = (
        customer_features["Customer ID"]
        .isin(returning_customers)
        .astype(int)
    )

    # -----------------------------------------------------
    # Save Gold table
    # -----------------------------------------------------
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    customer_features.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    positive_labels = int(
        customer_features["label_90d"].sum()
    )

    negative_labels = int(
        (customer_features["label_90d"] == 0).sum()
    )

    positive_rate = (
        customer_features["label_90d"].mean() * 100
    )

    # Exclude ID, dates and target from the simple feature count.
    excluded_columns = {
        "Customer ID",
        "first_purchase_date",
        "last_purchase_date",
        "label_90d",
    }

    feature_columns = [
        column
        for column in customer_features.columns
        if column not in excluded_columns
    ]

    print()
    print("Gold customer feature table created.")
    print(f"Customers: {len(customer_features):,}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Returned within 90 days: {positive_labels:,}")
    print(f"Did not return: {negative_labels:,}")
    print(f"Positive rate: {positive_rate:.2f}%")
    print(f"Saved file: {OUTPUT_PATH}")

    print()
    print("Features:")
    for number, feature in enumerate(
        feature_columns,
        start=1,
    ):
        print(f"{number}. {feature}")

    print()
    print("Missing values:")
    print(
        customer_features[
            feature_columns
        ].isna().sum()
    )


if __name__ == "__main__":
    build_customer_features()