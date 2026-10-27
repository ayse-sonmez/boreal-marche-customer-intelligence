# DATA CARD — Boréal Marché Customer Intelligence Platform

## 1. Dataset Overview

**Project:** Boréal Marché — Customer Intelligence Platform  
**Dataset:** UCI Online Retail II  
**Source:** UCI Machine Learning Repository  
**Purpose:** Educational AI capstone project  
**Data retrieval date (UTC):** 2026-09-09  
**License recorded in the project manifest:** Creative Commons Attribution 4.0 International (CC BY 4.0)

Boréal Marché is a fictional retail business used to demonstrate customer retention prediction, product recommendations, and a bilingual AI customer assistant.

The underlying transactional data comes from UCI Online Retail II. It is not transaction data collected from an actual Québec retailer.

## 2. Data Provenance and Integrity

The original dataset is stored locally at:

`data/raw/online_retail_II.xlsx`

The project records dataset provenance in:

`data/manifest.json`

**Original transaction rows:** 1,067,371

**SHA-256:**

`bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`

The SHA-256 checksum was independently recalculated using PowerShell and matched the value recorded in the manifest.

The original dataset is tracked through DVC metadata rather than committed directly to Git.

## 3. Data Processing Architecture

The project follows a Bronze–Silver–Gold data-processing approach.

**Bronze:** Original UCI Online Retail II source data and its provenance metadata.

**Silver:** Cleaned transaction records with normalized descriptions, transaction validation, and cancellation indicators.

**Gold:** Customer-level features and the 90-day return-purchase target used for retention modelling.

Relevant project files include:

- `src/data/ingest.py`
- `src/data/clean.py`
- `src/features/build_customer_features.py`
- `reports/cleaning_report.json`
- `data/processed/customer_features_gold.parquet`

## 4. Data Cleaning

The cleaning pipeline begins with 1,067,371 transaction rows.

| Cleaning operation | Rows removed |
|---|---:|
| Invalid quantities | 3,457 |
| Non-positive prices | 2,750 |
| Missing customer identifiers | 236,871 |
| Non-product or service records | 3,692 |
| Description normalization | 0 |
| Outlier review | 0 |
| **Total removed** | **246,770** |

**Rows remaining after cleaning: 820,601.**

The pipeline identifies 19,494 cancelled transaction rows. These records are flagged and retained in the Silver layer rather than automatically removed from every downstream dataset.

For customer retention feature generation, cancelled transactions are excluded before computing customer purchase behaviour.

Product descriptions are normalized to support consistent product analysis.

Non-product entries, including certain service charges, adjustments, and test transactions, are excluded from the cleaned product-transaction dataset.

### Outlier treatment

Extreme quantities and prices are reviewed rather than automatically discarded.

The cleaning report records:

- Absolute quantity 99.9th percentile: 600
- Price 99.9th percentile: 35.95
- Quantity rows above the review threshold: 686
- Price rows above the review threshold: 800

No additional rows were removed during the outlier-review step. This preserves potentially legitimate high-value transactions while allowing downstream tasks to apply use-case-specific treatment.

## 5. Prediction Objective and Time Windows

The retention model predicts whether an observed customer will make another completed purchase within the following 90 days.

**Observation cutoff:** 2011-09-10 23:59:59

**Prediction horizon:** 90 days after the observation cutoff

The feature-generation pipeline uses transactions up to the observation cutoff to calculate customer behaviour.

The binary target `label_90d` is created using purchases after the cutoff and within the subsequent 90-day window.

- `1`: The customer returned within the prediction window.
- `0`: No return purchase was observed within the prediction window.

This temporal separation is intended to prevent future purchases from being used directly as prediction features.

## 6. Customer Features

The final retention model uses **28 input features**: 27 numerical features and one categorical feature.

### Recency and purchase activity

1. `recency_days`
2. `total_orders`
3. `total_spend`
4. `total_quantity`
5. `unique_products`
6. `average_price`
7. `max_price`
8. `min_price`
9. `average_line_value`
10. `max_line_value`
11. `active_days`
12. `active_months`
13. `average_purchase_hour`

### Order value, frequency, and product behaviour

14. `average_order_value`
15. `average_quantity_per_order`
16. `average_products_per_order`
17. `customer_lifetime_days`
18. `orders_per_active_month`
19. `spend_per_active_month`
20. `quantity_per_product`
21. `spend_per_product`

### Weekend and repeat-purchase behaviour

22. `weekend_orders`
23. `weekend_order_ratio`
24. `repeat_products`
25. `repeat_product_ratio`

### Purchase cadence

26. `average_days_between_orders`
27. `max_days_between_orders`

### Geography

28. `country`

The final feature list and ordering are recorded in `artifacts/model_meta.json`.

Customer identifiers, raw first and last purchase dates, and the prediction target are excluded from the model's input feature matrix.

## 7. Missing Values and Encoding

The retention model includes preprocessing inside its saved scikit-learn pipeline.

Numerical features are processed using:

- Median imputation
- StandardScaler

The categorical feature `country` is processed using:

- Most-frequent-value imputation
- One-hot encoding with unknown categories ignored

These preprocessing operations are fitted within the training pipeline rather than independently fitted to the final test dataset.

## 8. Dataset Splits

The final model metadata records:

- Training customers: 4,204
- Held-out test customers: 1,052
- Total modelled customers: 5,256

The final test set is used for evaluation and must not be reused to select model hyperparameters or decision thresholds.

The saved model metadata states that the operating threshold was selected using validation data only.

## 9. Data Quality and Limitations

The following limitations must be considered when interpreting project results:

1. **Historical data:** The source transactions are historical and may not represent current retail behaviour.
2. **Geographical mismatch:** The dataset does not represent actual Québec retail customers.
3. **Selection effects:** Removing rows without customer identifiers limits customer-level analysis to identifiable transactions in the source dataset.
4. **Observation limits:** A customer labelled as not returning may have purchased outside the observed 90-day window or through another channel.
5. **Cancellation handling:** Cancelled transactions require different treatment depending on the prediction or recommendation task.
6. **Outliers:** Extreme values were reviewed but not automatically eliminated.
7. **Feature sensitivity:** Country and spending-related features may reflect differences between customer groups and require fairness assessment.
8. **Educational context:** Results should not be treated as evidence of performance for a real Québec retail deployment.

## 10. Privacy and Responsible Use

The project uses a publicly available historical dataset rebranded for an educational demonstration.

The fictional Boréal Marché branding must not be interpreted as evidence that the source transactions originated from a real Boréal Marché business.

Customer identifiers in the source dataset are used for aggregation and target construction. They are excluded from the retention model's input features.

However, the existence of customer identifiers means the dataset should not automatically be described as anonymous or free of privacy risk.

A real deployment would require a documented privacy assessment, appropriate access controls, retention and deletion rules, customer notices, and human oversight of decisions affecting individuals.

Québec Law 25 considerations are documented separately in `GOVERNANCE_LOI25.md`.

This data card does not certify legal compliance.

## 11. Reproducibility and Supporting Evidence

Key supporting files:

- `data/manifest.json`
- `data/raw/online_retail_II.xlsx.dvc`
- `reports/cleaning_report.json`
- `src/data/ingest.py`
- `src/data/clean.py`
- `src/features/build_customer_features.py`
- `src/models/final_evaluation.py`
- `artifacts/model_meta.json`

**Dataset documentation status:** Based on the existing project code, manifest, cleaning report, and saved model metadata.

**Intended use:** Academic demonstration and reproducible analysis.

**Not intended for:** Unreviewed automated decisions about real customers.
