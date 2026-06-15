# Telco Customer Churn — Analysis & Prediction

Predicting telecom customer churn and turning the drivers into a retention-targeting
view. A product-analytics project: **business question first, model as a tool.**

## Headline results

- **Who churns:** month-to-month contracts (43%), electronic-check payers (45%), and
  fiber-optic internet (42%) are the highest-risk segments; two-year contracts barely
  churn (3%).
- **Catching churners:** a class-balanced model flags ~80% of churners vs. 56% for a
  default logistic model — at the cost of more false alarms.
- **The real lever is the decision threshold, not the algorithm.** Every model *ranks*
  churners similarly well (AUC ≈ 0.82–0.84); where you draw the "flag as churn" line is
  a business call — the cost of a wasted retention offer vs. a lost customer.

## Dataset

IBM Telco Customer Churn — 7,043 customers × 21 fields (contract, tenure, charges,
services, churn flag). Public sample dataset, included in `data/`.

## Approach

| Stage | State |
|---|---|
| Data cleaning (`TotalCharges` → numeric, NaNs) | ✅ |
| EDA — churn rate by contract / tenure / charges / payment | ✅ |
| Encoding + stratified train/test split | ✅ |
| Logistic-regression baseline | ✅ |
| Decision Tree / Random Forest / XGBoost + comparison | ✅ |
| Evaluation beyond accuracy (precision / recall / ROC-AUC) | ✅ |
| Churn drivers + retention-ROI decision framing | ✅ |

## Model comparison (test set = 2,113 customers)

| Model | Churn recall | Churn precision | F1 | Accuracy |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.56 | 0.67 | 0.61 | 0.81 |
| Logistic Regression (balanced) | **0.80** | 0.51 | 0.62 | 0.74 |
| Random Forest | 0.48 | 0.61 | 0.54 | 0.78 |
| XGBoost (`scale_pos_weight=2.8`) | 0.66 | 0.53 | 0.59 | 0.76 |

## Next steps

- Behavioral features (recent usage drop, support tickets, tenure milestones)
- Validate the chosen decision threshold against real offer economics

## Run

```bash
pip install -r requirements.txt
jupyter lab notebooks/03_churn_classification.ipynb
```
