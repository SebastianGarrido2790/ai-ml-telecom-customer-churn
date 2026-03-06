# Exploratory Data Analysis (EDA): Telco Customer Churn

## Overview
This document summarizes the Exploratory Data Analysis performed in `notebooks/01_eda_telco_churn.ipynb` on the raw `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset. The goal is to establish the baseline "Hard Signals" distribution before moving to Phase 2: AI-driven Data Enrichment for "Soft Signals" (Synthetic Ticket Notes).

## Findings

### 1. Data Cleaning
- `TotalCharges` was initially parsed as an `object` (string) due to empty space characters. It was coercively mapped to a numeric format. 
- There were 11 missing values (representing brand-new customers with `tenure = 0` and no charges yet), which were subsequently dropped for a clean baseline.

### 2. Target Variable (Churn) Distribution
- The dataset exhibits a significant class imbalance (roughly 27% Churn to 73% No Churn). 
- **Actionable Insight**: The training pipeline (to be configured in MLflow/Optuna) will likely need to employ `scale_pos_weight` in XGBoost/LightGBM or oversampling techniques (e.g., SMOTE) to maximize the F1-Score recall.

### 3. Quantitative Variables (Hard Signals)
- **Tenure**: Extremely strong negative correlation with Churn. New customers (low tenure) have the highest churn propensity. The risk begins to stabilize significantly after ~24 months.
- **Monthly Charges**: Higher monthly charges generally correlate with higher churn rates, pointing to potential pricing sensitivity.
- **Total Charges**: Lower total charges correlate with churn, primarily driven by the confounding effect of lower tenure.

### 4. Categorical Variables (Context for AI Syndication)
- **Internet Service**: Fiber Optic customers churn significantly higher than DSL customers. 
- **Support Services**: Lack of "Tech Support", "Device Protection", or "Online Security" dramatically increases the churn rate.
- **Contract Type**: Month-to-month contracts represent the overwhelming majority of churned users. Extended contracts (1 or 2 years) show very high retention.
- **Payment Method**: "Electronic check" users feature the highest churn rate among all payment types.

### 5. Statistical & Multicollinearity Analysis (Correlation & VIF)
Before feeding data into the models, we analyzed feature correlations and calculated the Variance Inflation Factor (VIF).
- **Multicollinearity Discovery**: `MonthlyCharges` (VIF: 210.7), `PhoneService` (VIF: 47.0), and `TotalCharges` (VIF: 21.3) demonstrate extreme levels of multicollinearity over the standard threshold of 5-10. This is expected since `TotalCharges` is a direct product of `MonthlyCharges` and `tenure`.
- **Actionable Insight**: Tree-based models (XGBoost/LightGBM) to be used in the 'Brawn' pipeline handle multicollinearity well natively, but linear models (like Logistic Regression) would fall apart here without heavy regularization. We will retain these variables but be mindful of feature importance degradation during model evaluation.

## Strategic Link to Phase 2 (AI Data Enrichment)
The findings from this baseline justify the Agentic approach. While hard signals predict *who* might churn, they are deterministic and lack the nuance of a real operations center. 

To bridge the gap outlined in the `user_stories.md`, the AI Agent in **Phase 2** will use these hard signals as conditioning inputs to generate realistic "Ticket Notes" and extract Sentiment:
- *Example Condition 1*: If `InternetService == 'Fiber optic'` and `TechSupport == 'No'` and `Churn == 'Yes'`, the Agent might synthesize a note illustrating a frustrated customer experiencing frequent outages with no technical help.
- *Example Condition 2*: If `MonthlyCharges` is high and `Contract == 'Month-to-month'`, the Agent might synthesize a note regarding competitive dissatisfaction over unexpected billing spikes.

## Visual Artifacts
Generated EDA figures for stakeholder review:
- `reports/figures/eda/churn_distribution.png`
- `reports/figures/eda/quantitative_vs_churn.png`
- `reports/figures/eda/qualitative_vs_churn.png`
- `reports/figures/eda/correlation_map.png`
- `reports/figures/eda/vif_results.csv`
