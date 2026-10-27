# Boréal Marché — AI Governance and Québec Law 25

## 1. Project Scope

Boréal Marché is an academic AI demonstration using the historical UCI Online Retail II dataset. It is not a production system processing real Québec retail customers.

The platform includes:
- A 90-day customer return prediction model.
- Product recommendations based on purchasing patterns.
- A bilingual English/French AI customer assistant.

The application is deployed on Google Cloud Run in Montréal. This document describes governance practices, observed evaluation results, and safeguards required before any real-world deployment. It is not a certification of legal compliance.

## 2. Automated Decision Register

| System | Output | Potential impact | Human oversight |
|---|---|---|---|
| Retention model | Probability of purchasing within 90 days | Selection for hypothetical retention offers | Human review required before real campaigns |
| Recommendation engine | Suggested products | Influence on product visibility | Business review of recommendation rules |
| AI assistant | Answers and tool-generated information | Incorrect or incomplete customer guidance | Escalation to customer support for consequential requests |

The academic prototype does not establish that a production human-review workflow has been implemented.

## 3. Transparency and Customer Rights

The existing bilingual customer policy is maintained in `docs/corpus/privacy_policy.md`.

For real-world use, customers must receive clear information about relevant personal-information processing and applicable automated decisions. Processes must support requests for access, correction, explanations, and human review where legally required.

Publication of the policy inside the customer interface and operational handling of these requests must be verified before production use.

## 4. Data Minimization and Retention

The academic dataset contains transaction history and pseudonymous customer identifiers. Such identifiers remain privacy-relevant.

The retention model uses 28 features, including transaction-based behavioural information and country. Each feature should be reviewed for necessity, predictive value, and potential proxy effects before real-world use.

Production safeguards must include:
- Purpose-based access restrictions.
- Defined retention and deletion schedules.
- Secure handling of customer identifiers.
- A documented process for access and correction requests.
- Review of any transfer of personal information to external AI providers.

A production retention/deletion workflow has not been verified. Cloud Run local storage and temporary SQLite data must not be represented as durable audit storage.

## 5. Fairness Evaluation

An offline fairness audit was conducted on 1,052 held-out test customers using the existing model and a return-probability threshold of 0.73.

The audit examined:
- Country.
- Customer-tenure quartiles.
- Customer-spending quartiles.

Observed maximum differences:

| Metric | Tenure groups | Spending groups |
|---|---:|---:|
| Return recall | 63.78 percentage points | 75.00 percentage points |
| Offer selection rate | 51.33 percentage points | 61.22 percentage points |

These differences exceed the project's five-percentage-point review threshold.

Precision requires caution: groups with no predicted positive cases have undefined precision, although the initial audit software displayed zero. Its reported 100-percentage-point precision gaps should therefore not be treated as reliable findings without recalculation.

Country comparison was inconclusive: 26 countries were represented, but only one country had at least 30 test customers.

These results indicate performance disparities requiring investigation; they do not independently establish unlawful discrimination. The historical dataset is not representative of Québec customers.

Evidence: `reports/fairness_audit.json`.

## 6. Proposed Fairness Mitigations

1. Evaluate model performance and calibration using larger, representative customer samples.
2. Compare models with and without the country feature to assess geographic proxy effects.
3. Require human review of proposed targeting decisions and monitor outcomes across customer groups.

Trade-offs include possible predictive-performance reductions, additional evaluation costs, and increased operational workload. No mitigation is claimed to have been deployed or proven effective.

## 7. AI Assistant Governance

The assistant uses a policy knowledge base and tools for customer-facing answers.

The bilingual privacy policy states that real customer personal information must not be transmitted to externally hosted language models without appropriate authorization and protections.

Before production use:
- Verify retrieval accuracy and response quality.
- Prevent disclosure of personal information through prompts and tool outputs.
- Provide a route to human customer support.
- Assess privacy and cross-border transfer implications of any external AI provider.

## 8. Security, Monitoring, and Incident Response

The prototype uses a public frontend and a private backend on Google Cloud Run. Access controls must be maintained and periodically reviewed.

Monitoring currently includes application metrics and offline model checks. A durable production audit trail, verified automated drift alert, and complete incident-response process have not been established.

Before production deployment, assign an accountable privacy owner, document incident escalation, define monitoring thresholds, test alerts, and establish procedures for investigating and reporting confidentiality incidents when required.

## 9. Québec Law 25 Readiness

Québec Law 25 readiness requires more than publishing a privacy policy. A real deployment would need an assessment of applicable obligations, including transparency, automated-decision rights, privacy impact assessment where required, data retention, access controls, and incident management.

This academic prototype documents those considerations but does not claim full legal or operational compliance.

## 10. Governance Conclusion

Boréal Marché demonstrates customer intelligence, recommendation, and bilingual AI assistance capabilities.

The fairness audit identified substantial differences across tenure and spending groups, while country-level comparisons lacked sufficient data. The appropriate next step for any real deployment is representative validation, privacy assessment, human oversight, and operational governance before customer-impacting decisions are made.
