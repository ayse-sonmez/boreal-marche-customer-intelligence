# MODEL CARD — Boréal Marché Customer Retention Prediction

## 1. Model Overview

**Project:** Boréal Marché — Customer Intelligence Platform  
**Model version:** 1.0  
**Model type:** GradientBoostingClassifier  
**Framework:** scikit-learn  
**Calibration:** Sigmoid (5-fold cross-validation)  
**Prediction horizon:** 90 days  
**Purpose:** Educational customer retention prediction

This model estimates the probability that a customer will make another completed purchase within the next 90 days, based on historical transaction behaviour.

Boréal Marché is a fictional retailer. The model was developed using the historical UCI Online Retail II dataset.

The model is intended for academic demonstration, not autonomous decision-making about real customers.

## 2. Intended Use

The model supports:

- Customer retention analysis
- Identification of customers with lower predicted return probability
- Educational demonstration of machine learning classification
- What-if analysis of customer behaviour
- Evaluation of hypothetical customer retention strategies

The model must not be used as the sole basis for decisions affecting real individuals.

## 3. Prediction Target

**Target variable:** `label_90d`

- `1`: The customer makes another completed purchase within 90 days.
- `0`: No completed return purchase is observed during the 90-day window.

**Observation cutoff:** 2011-09-10 23:59:59

Only transactions occurring on or before the observation cutoff are used to generate predictive features.

Purchases occurring after the cutoff are used to construct the future outcome label.

Cancelled transactions are excluded from the retention feature-generation process.

## 4. Input Features

The final model uses **28 features**:

**27 numerical features:**

1. recency_days
2. total_orders
3. total_spend
4. total_quantity
5. unique_products
6. average_price
7. max_price
8. min_price
9. average_line_value
10. max_line_value
11. active_days
12. active_months
13. average_purchase_hour
14. average_order_value
15. average_quantity_per_order
16. average_products_per_order
17. customer_lifetime_days
18. orders_per_active_month
19. spend_per_active_month
20. quantity_per_product
21. spend_per_product
22. weekend_orders
23. weekend_order_ratio
24. repeat_products
25. repeat_product_ratio
26. average_days_between_orders
27. max_days_between_orders

**1 categorical feature:**

28. country

Customer identifiers, raw first and last purchase dates, and the target label are excluded from the final feature matrix.

The authoritative feature ordering is stored in `artifacts/model_meta.json`.

## 5. Data Preparation

The model uses a scikit-learn preprocessing pipeline.

**Numerical features:**

- Median imputation for missing values
- StandardScaler for feature scaling

**Categorical feature:**

- Most-frequent-value imputation
- OneHotEncoder with unknown categories ignored

The preprocessing pipeline is included in the saved model artifact.

This helps ensure that training and inference use consistent transformations.

## 6. Training and Evaluation

**Training observations:** 4,204 customers  
**Held-out test observations:** 1,052 customers  
**Total observations:** 5,256 customers

The final estimator is a GradientBoostingClassifier wrapped in CalibratedClassifierCV.

**Calibration method:** Sigmoid  
**Calibration cross-validation:** 5 folds  
**Random state:** 42

### Selected hyperparameters

- `n_estimators`: 157
- `learning_rate`: 0.01436244069943827
- `max_depth`: 3

The saved metadata states that the operating threshold was selected using validation data only.

The held-out test set was used for final evaluation. It must not be reused to choose new hyperparameters or modify the operating threshold.

## 7. Final Held-Out Test Results

| Metric | Result |
|---|---:|
| PR-AUC / Average Precision | 0.7951 |
| ROC-AUC | 0.8192 |
| Brier Score | 0.1699 |
| Recall at Top 10% | 0.2096 |
| Model fitting time | 11.97 seconds |

These results describe performance on the project's held-out historical test data.

They do not establish equivalent performance for current retail customers or a real Québec business.

## 8. Decision Threshold

**Selected threshold:** 0.73

The application uses the predicted probability of returning within 90 days.

Customers with a predicted return probability below the threshold are classified as potentially at risk for the educational retention scenario.

The threshold was selected before the final held-out test evaluation, according to the saved model metadata.

The threshold should not be interpreted as a universal business rule.

## 9. Scenario-Based Business Value

The project evaluates a hypothetical retention-offer scenario.

**Assumptions:**

- Cost per targeted offer: 5 monetary units
- Value assigned to a successfully targeted non-returning customer: 20 monetary units

**Recorded results:**

| Threshold | Scenario value |
|---|---:|
| 0.50 | 6,665 |
| 0.73 | 7,190 |
| Difference | +525 |

These values are calculated under simplified assumptions.

They are **not observed campaign revenue, measured causal uplift, or verified real-world profit**.

A production system would require actual campaign outcomes and appropriate experimental validation.

## 10. Model Strengths

The model demonstrates:

- Customer-level feature engineering
- Temporal separation of observations and labels
- Reusable preprocessing and inference
- Probability calibration
- Evaluation on held-out data
- Validation-selected decision threshold
- Business-oriented scenario analysis
- Integration with a FastAPI service and Streamlit interface

## 11. Limitations and Risks

### Historical dataset

The model uses historical retail transactions and may not generalize to modern purchasing patterns.

### Domain mismatch

The underlying dataset does not represent actual customers of a Québec retailer.

### Incomplete customer observation

A customer may purchase through another channel or outside the observation window.

Therefore, the absence of an observed return purchase is not necessarily evidence of permanent customer churn.

### Prediction uncertainty

Predicted probabilities are estimates, not guarantees.

A model score must not be presented as a certain statement about an individual's future behaviour.

### Dataset selection

Transactions without customer identifiers are excluded from customer-level modelling.

This may introduce selection effects.

### Geographic feature

The `country` feature may capture geographical differences that require additional fairness assessment.

### Spending-related features

Features related to customer spending may create unequal treatment across economic groups if used in real business decisions.

### Model drift

Changes in purchasing patterns, product availability, or customer composition may reduce predictive performance.

## 12. Fairness and Responsible AI

The model is intended to support analysis rather than automatically approve or deny customer benefits.

Before use with real customers, the following would be necessary:

- Subgroup performance analysis
- Assessment of selection rates and error-rate differences
- Review of geographical and spending-related disparities
- Documented human oversight
- Customer-facing explanations where applicable
- Privacy impact assessment
- Monitoring for performance degradation and drift

The existence of a fairness requirement does not mean that a complete fairness audit has already been performed.

Any completed fairness results must be documented separately with supporting evidence.

## 13. Explainability and Human Oversight

The application provides model probabilities and a What-if interface for exploring how selected input changes affect predictions.

What-if results are hypothetical model outputs, not causal estimates.

A real retention campaign should include human review of proposed interventions, especially when outcomes could materially affect customers.

The model must not be used as an unreviewed automated decision system.

## 14. Deployment Context

The Boréal Marché educational application is deployed on **Google Cloud Run** in the **Montréal region (`northamerica-northeast1`)**.

The deployed architecture includes:

- Streamlit frontend
- FastAPI backend
- Saved scikit-learn model pipeline
- FP-Growth recommendation artifacts
- RAG assistant components

The public frontend communicates with the backend using authenticated service-to-service requests.

Cloud deployment demonstrates application accessibility. It does not independently establish legal compliance, production readiness, or model fairness.

## 15. Monitoring and Maintenance

The project includes application monitoring and drift-related functionality.

Relevant monitoring concepts include:

- Request counts
- Error rates
- Response latency
- Population Stability Index (PSI)
- Kolmogorov–Smirnov (KS) statistics
- Prediction distribution changes
- Delayed outcome monitoring when sufficient observations become available

Monitoring results must distinguish between metrics that have been calculated and metrics that cannot yet be estimated reliably because of insufficient live data.

## 16. Model Artifacts and Reproducibility

**Saved model:** `artifacts/model.joblib`  
**Metadata:** `artifacts/model_meta.json`  
**Final evaluation code:** `src/models/final_evaluation.py`  
**Feature generation:** `src/features/build_customer_features.py`  
**Data documentation:** `DATA_CARD.md`  
**Governance documentation:** `GOVERNANCE_LOI25.md`

**Model metadata creation date:** 2026-09-22 (UTC)

**Recorded training environment:**

- Python 3.11.0
- pandas 3.0.5
- NumPy 2.4.6
- scikit-learn 1.9.0
- joblib 1.6.0

## 17. Responsible Use Statement

This model is a research and educational demonstration.

Its outputs must not be used alone to make decisions about real individuals.

Deployment in a real business would require additional evaluation, monitoring, privacy controls, fairness review, and human oversight.

This model card does not certify that the system is production-ready or compliant with Québec Law 25.
