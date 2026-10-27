from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MANIFEST_PATH = Path("data/manifest.json")


# ---------------------------------------------------------
# SHA-256 checksum
# ---------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Calculate SHA-256 checksum for a file."""

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ---------------------------------------------------------
# Manifest
# ---------------------------------------------------------

def save_manifest(
    source_file: Path,
    parquet_file: Path,
    row_count: int,
) -> None:
    """Save dataset provenance information."""

    manifest = {
        "dataset_name": "UCI Online Retail II",
        "source": "UCI Machine Learning Repository",
        "source_file": str(source_file),
        "processed_file": str(parquet_file),
        "retrieval_date_utc": datetime.now(UTC).isoformat(),
        "sha256": sha256_file(source_file),
        "row_count": row_count,
        "licence": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "project_dataset_name": "Boreal Marche",
        "provenance_note": (
            "Public UCI Online Retail II dataset re-skinned as "
            "Boreal Marche for a pedagogical project."
        ),
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ---------------------------------------------------------
# Excel -> Parquet
# ---------------------------------------------------------

def convert_excel_to_parquet(source_file: Path) -> None:
    """Read all Excel sheets, apply explicit schema and save as Parquet."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading UCI Online Retail II dataset...")

    sheets = pd.read_excel(
        source_file,
        sheet_name=None,
    )

    frames = []

    for sheet_name, dataframe in sheets.items():
        dataframe["source_sheet"] = sheet_name
        frames.append(dataframe)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    print(f"Raw rows loaded: {len(combined):,}")

    # -----------------------------------------------------
    # Explicit schema
    # -----------------------------------------------------
    # Invoice must be string because cancelled invoices
    # can contain values such as C489449.

    combined["Invoice"] = combined["Invoice"].astype("string")

    combined["StockCode"] = combined["StockCode"].astype("string")

    combined["Description"] = combined["Description"].astype("string")

    combined["Quantity"] = pd.to_numeric(
        combined["Quantity"],
        errors="coerce",
    ).astype("Int64")

    combined["InvoiceDate"] = pd.to_datetime(
        combined["InvoiceDate"],
        errors="coerce",
    )

    combined["Price"] = pd.to_numeric(
        combined["Price"],
        errors="coerce",
    ).astype("float64")

    combined["Customer ID"] = pd.to_numeric(
        combined["Customer ID"],
        errors="coerce",
    ).astype("Int64")

    combined["Country"] = combined["Country"].astype("string")

    combined["source_sheet"] = combined["source_sheet"].astype("string")

    # -----------------------------------------------------
    # Save Parquet
    # -----------------------------------------------------

    parquet_file = (
        PROCESSED_DIR
        / "online_retail_ii.parquet"
    )

    combined.to_parquet(
        parquet_file,
        index=False,
        engine="pyarrow",
    )

    # -----------------------------------------------------
    # Save manifest
    # -----------------------------------------------------

    save_manifest(
        source_file=source_file,
        parquet_file=parquet_file,
        row_count=len(combined),
    )

    # -----------------------------------------------------
    # Console summary
    # -----------------------------------------------------

    print()
    print("Ingestion completed successfully.")
    print(f"Rows: {len(combined):,}")
    print(f"Columns: {len(combined.columns)}")
    print(f"Parquet: {parquet_file}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"SHA-256: {sha256_file(source_file)}")

    print()
    print("Schema:")
    print(combined.dtypes)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    source = RAW_DIR / "online_retail_II.xlsx"

    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found: {source}\n"
            "Place the UCI Online Retail II Excel file "
            "inside data/raw/"
        )

    convert_excel_to_parquet(source)