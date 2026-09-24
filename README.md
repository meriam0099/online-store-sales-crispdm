# Online Store Sales Analysis — CRISP-DM

A full CRISP-DM analysis of a European online store's sales data
(~1,000 orders, 2019–2020, 15 countries).

**Two goals:**
1. Segment customers for retention strategy.
2. Evaluate whether order value can be predicted from available features.

---

## TL;DR — Three Findings

### 1. 🎯 RFM segmentation found a 5-customer VIP cohort worth 26% of revenue

Using Gaussian Mixture Models (chosen over K-Means because a single whale customer
distorted K-Means into a degenerate solution), three clean segments emerged:

| Segment | Customers | Avg Recency | Avg Frequency | Avg Spend |
|---|---|---|---|---|
| 👑 VIP / High-Value | 5 (6.7%) | 17 days | 53 orders | €5.86M |
| 🟢 Core Loyal | 62 (82.7%) | 77 days | 12 orders | €1.34M |
| 🔴 Dormant | 8 (10.7%) | 288 days | 1.6 orders | €0.14M |

**The VIP cohort alone accounts for ~26% of total revenue.**

### 2. 🚨 The regression model was an illusion

Three models were trained (Random Forest, Gradient Boosting, Ridge).
The best achieved **Test R² = 0.31** — but a feature importance check
revealed that **84% of the model's signal came from `month` + `day`**.

An ablation study confirmed the diagnosis:
- Full feature set → Test R² = **0.31**
- Remove `month` + `day` → Test R² = **−0.45** (worse than predicting the mean)

**Conclusion:** the model was memorizing monthly averages from a 24-month
window, not learning generalizable revenue drivers. It is not deployable.

### 3. ✅ The methodology is the deliverable

- Correct exclusion of `cost` (which would leak the target).
- Fair model comparison (Ridge properly scaled in a Pipeline).
- An ablation study that falsified the regression before deployment.
- A reference inference pipeline with honest metadata warnings.

---

## Visuals

### RFM Segments (GMM, k=3)
![RFM Segments](figures/rfm_segments.png)

### Ablation Study — Why the Regression Doesn't Work
![Ablation](figures/ablation_study.png)

---

## Methodology

Followed **CRISP-DM** across all six phases:

| Phase | Deliverable |
|---|---|
| 1. Business Understanding | Objectives + success criteria |
| 2. Data Understanding | Profiling, distributions, missing values |
| 3. Data Preparation | Feature engineering, encoding, temporal features |
| 4. Modeling | Random Forest baseline + hyperparameter tuning |
| 5. Evaluation | Multi-model comparison + ablation study |
| 6. Deployment | Reference pipeline + metadata + inference demo |

### Key techniques

- **Unsupervised:** Gaussian Mixture Models for customer segmentation
- **Supervised regression:** Random Forest, Gradient Boosting, Ridge
- **Supervised classification:** Random Forest with class weighting
- **Model interrogation:** Feature importance analysis + ablation study
- **Fair comparison:** Scaled Pipelines, stratified splits, cross-validation

---

## ⚠️ Limitations & Honest Assessment

- The dataset contains only ~30 weakly informative features. Order value
  is driven by **basket composition, product mix, and pricing** — none
  of which are present.
- The regression model is a **reference baseline**, not a deployable tool.
- The VIP segment has only 5 customers — statistics are directional, not precise.

---

## 🚀 How to Run

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn
jupyter notebook online_store_sales_crispdm.ipynb
