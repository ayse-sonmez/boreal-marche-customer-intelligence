from pathlib import Path

import pandas as pd

INPUT_PATH = Path(
    "data/processed/online_retail_ii_silver.parquet"
)

OUTPUT_PATH = Path(
    "docs/corpus/product_catalogue.md"
)

MAX_PRODUCTS = 200


def main():

    print("Loading Silver dataset...")

    df = pd.read_parquet(
        INPUT_PATH,
        columns=[
            "StockCode",
            "Description",
            "Invoice",
        ],
    )

    # Remove empty descriptions
    df = df[
        df["Description"].notna()
        & (df["Description"].str.strip() != "")
    ].copy()

    # Count how often each product appears
    products = (
        df.groupby(
            ["StockCode", "Description"]
        )["Invoice"]
        .nunique()
        .reset_index(
            name="transaction_count"
        )
        .sort_values(
            "transaction_count",
            ascending=False,
        )
    )

    # Keep the most common products
    products = products.head(
        MAX_PRODUCTS
    )

    print(
        f"Products selected: {len(products):,}"
    )

    # Build Markdown catalogue
    lines = [
        "# Boréal Marché — Product Catalogue",
        "# Catalogue de produits",
        "",
        "Document type: Product Catalogue",
        "Version: 1.0",
        "Source: UCI Online Retail II dataset, "
        "reskinned for the Boréal Marché academic project.",
        "",
        "---",
        "",
        "This catalogue contains product identifiers and "
        "descriptions from the project dataset.",
        "",
    ]

    for _, row in products.iterrows():

        stock_code = str(
            row["StockCode"]
        )

        description = str(
            row["Description"]
        )

        lines.extend(
            [
                f"## Product {stock_code}",
                "",
                f"Product ID: {stock_code}",
                "",
                f"Description: {description}",
                "",
            ]
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"Catalogue saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()