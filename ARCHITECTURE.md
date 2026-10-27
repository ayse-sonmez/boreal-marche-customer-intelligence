@'
# Boréal Marché — System Architecture

## 1. Overview

Boréal Marché is an academic AI Customer Intelligence Platform built using the UCI Online Retail II dataset.

The platform combines three capabilities:

1. Customer retention prediction.
2. Product recommendations.
3. Bilingual English/French AI customer assistance.

The application is deployed on Google Cloud Run in the Montréal region. Google Cloud was approved by the instructor as the deployment platform.

## 2. High-Level Architecture

```text
                     CUSTOMER / USER
                            |
                            v
                 Streamlit Web Frontend
                   Google Cloud Run
                            |
                    Authenticated API
                            |
                            v
                     FastAPI Backend
                   Google Cloud Run
                            |
           +----------------+----------------+
           |                |                |
           v                v                v
     Retention Model   Recommendation    RAG Assistant
     Scikit-learn      FP-Growth Rules   FAISS + Tools
           |                |                |
           v                v                v
     Model Artifacts    Rules Artifact   Policy Documents