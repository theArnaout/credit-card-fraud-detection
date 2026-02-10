# Credit Card Fraud Detection Pipeline – Project Plan

## Goal
Build an end-to-end ML pipeline to detect credit card fraud, showing both data engineering and advanced analytics/AI skills. Using a realistic dataset with merchant categories, locations, and personal details to enable richer analysis, feature engineering, and ethical AI discussions.

## Dataset
- Credit Card Transactions Fraud Detection (from Kaggle: https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- ~1M+ transactions
- Key features: trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender, street, city, state, zip, lat, long, city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud
- Highly imbalanced (fraud ~0.57%, based on samples where is_fraud=1)
- Notes: Includes sensitive PII (names, gender, dob) – to be avoided in models for ethical reasons

## Major Milestones
1. Planning
2. Data acquisition & basic exploration
3. ETL pipeline (cleaning, feature engineering e.g., transaction hour, merchant-user distance; handle categoricals and PII)
4. EDA – understand fraud patterns (e.g., by category, time, location)
5. 
    - Modeling – baseline + XGBoost, handle imbalance 
    - Evaluation – ROC-AUC, PR curve, SHAP explainability
6. Ethical AI section – bias/fairness checks (e.g., on gender/dob), privacy discussion
7. Streamlit demo app
8. Documentation & README

## Success Criteria
- Recall > 90% on fraud class (critical for banking to catch fraud)
- ROC-AUC > 0.95
- Clean, reproducible code on GitHub
- Simple web demo (e.g., input transaction details, get fraud score with explanation)
- Clear explanation of business value for a bank (e.g., reduced losses, fewer false alerts)
- Ethical focus: Demonstrate responsible handling of data (e.g., no gender in final model)

## Tools I Plan to Use
- Python, Pandas, PySpark
- scikit-learn, XGBoost, imbalanced-learn
- Haversine formula (for lat/long distance calcs)
- Streamlit for deployment
- Git/GitHub for version control
