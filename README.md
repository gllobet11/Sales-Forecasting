# Time Series Sales Forecasting — Retail Monthly Sales Prediction

## Overview

This project tackles the challenge of **forecasting monthly sales** for thousands of store–product combinations in a retail chain. Accurate forecasting enables better inventory management, pricing strategy, and financial planning.

**Goal:** build a machine learning model that minimizes the **Root Mean Squared Error (RMSE)** in next-month sales prediction.

---

## Methodology

The pipeline integrates advanced feature engineering with state-of-the-art gradient boosting models to capture temporal dynamics and nonlinear interactions.

1. **Data Cleaning and Preprocessing**  
   - Outlier capping by product category.  
   - Hierarchical imputation of missing prices (product → category → global).  

2. **Feature Engineering**  
   - Lag features, rolling windows, trend and seasonality signals.  
   - Price ratios and historical consistency indicators.  

3. **Categorical Encoding**  
   - *Time-aware target encoding* ensuring complete causality (no data leakage).  

4. **Missing Value Imputation**  
   - Variable-type specific imputation for lags, ratios, and volatility features.  

5. **Modeling and Temporal Validation**  
   - Models trained: **LightGBM**, **XGBoost**, **CatBoost**, **RandomForest**.  
   - Validation scheme: temporal hold-out (month N−1 as validation, N as test).  

6. **Model Ensembling**  
   - A Ridge regression meta-model blends individual predictions, reducing RMSE.  

---

## Models and Results

| Model | Objective / Loss Function | RMSE (Validation) | Target Scale |
|--------|----------------------------|--------------------|---------------|
| CatBoost | RMSE | **2.7067** | log |
| LightGBM Tweedie | Tweedie (variance_power=1.56) | 2.7445 | original |
| XGBoost | reg:squarederror | 2.7537 | log |
| LightGBM Poisson | Poisson | 2.7604 | original |
| LightGBM L1 | regression_l1 | 2.7645 | log |
| RandomForest | MSE | 2.7474 | log |

**Final ensemble RMSE:** **2.6311**, outperforming all individual models.

---

## Validation Strategy

A **time-based hold-out** validation approach was implemented:

- **Train:** all months except the last two  
- **Validation:** second-to-last month  
- **Test:** last month  

**Rationale:**  
This design mirrors real-world forecasting conditions, preserves temporal causality, and prevents data leakage. Traditional K-Fold CV was avoided due to heterogeneous time histories across series.

---

## Results Analysis

### Key Predictive Features
- **monthly_sales_rolling_mean_3 / 6** — short- and mid-term sales trends  
- **monthly_sales_lag_1** — immediate autocorrelation  
- **item_id / item_category_id** — product and category identifiers  
- **price_vs_historical_avg_ratio** — price sensitivity indicator  

### Low-Impact Features
- **shop_id / year** — their influence is captured by derived trend features  

---

## Error Analysis

The model tends to **underestimate volatile products** and items with occasional promotions.  
Largest residuals occur in the medium sales range (13–21 units), where historical data is sparse.

Examples:
- **ID 6_5822:** erratic behavior; model regresses toward the mean.  
- **ID 42_6497:** extreme peaks; Tweedie objective penalizes them less strongly.  

---

## Model Interpretability

Two complementary techniques were used:

1. **Native Feature Importance**  
   Based on gain/split metrics (LightGBM, CatBoost).

2. **Permutation Importance**  
   Measures RMSE degradation when perturbing individual features.

Both consistently highlight **lag** and **rolling mean** features as primary drivers.

---

## Tech Stack

- **Language:** Python 3.10  
- **Core libraries:**  
  `pandas`, `numpy`, `lightgbm`, `xgboost`, `catboost`, `scikit-learn`, `plotly`, `matplotlib`, `seaborn`  
- **Environment:** Jupyter Notebook  
- **Ensembling:** Ridge regression meta-model  

---

## Project Structure

```
Time-Series-Sales-Forecasting/
│
├── data/
│   └── ts_kaggle_train.csv
├── notebooks/
│   └── Time_Series_Sales_Forecast_Gerard.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── utils.py
├── results/
│   ├── feature_importance/
│   ├── error_analysis/
│   └── predictions.csv
└── README.md
```

---

## Reproducibility

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run main notebook**
   ```bash
   jupyter notebook notebooks/Time_Series_Sales_Forecast_Gerard.ipynb
   ```

3. **Set data paths**
   ```python
   DATA_PATH = "data/"
   TRAIN_FILE = "ts_kaggle_train.csv"
   ```

---

## Next Steps

- Deploy a **rolling forecast pipeline** (monthly retraining).  
- Evaluate **deep learning** architectures (N-BEATS, Temporal Fusion Transformer).  
- Incorporate **exogenous features** such as promotions, weather, and calendar effects.  
