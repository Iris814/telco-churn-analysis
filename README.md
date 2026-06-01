# Telco Customer Churn — Analysis & Prediction

> 🚧 **Work in progress** — private until complete.

Predicting telecom customer churn and turning the drivers into a retention-targeting
view. A product-analytics project: business question first, model as a tool.

## Dataset

IBM Telco Customer Churn — 7,043 customers × 21 fields (contract, tenure, charges,
services, churn flag). Public sample dataset, included in `data/`.

## Status

| Stage | State |
|---|---|
| Data cleaning (`TotalCharges` → numeric, NaNs) | ✅ |
| EDA — churn rate by contract / tenure / charges | ✅ |
| Encoding + stratified train/test split | ✅ |
| Logistic-regression baseline | ✅ |
| Tree / XGBoost model + comparison | ⏳ in progress |
| Evaluation beyond accuracy (precision / recall / ROC-AUC) | ⏳ |
| Churn drivers (feature importance) → retention targeting + ROI | ⏳ |

## Run

```bash
pip install -r requirements.txt
jupyter lab notebooks/03_churn_classification.ipynb
```
