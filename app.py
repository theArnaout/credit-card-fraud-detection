import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_model():
    model_path = "models/xgb_fraud_model.pkl"
    try:
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

st.title("Credit Card Fraud Detection Demo")
st.markdown("Enter transaction details to get a fraud probability. Provide some context about your normal behavior for better accuracy.")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=75.0, step=1.0, format="%.2f")
        trans_hour = st.slider("Transaction Hour (0-23)", 0, 23, 14)
        category = st.selectbox("Merchant Category", [
            'grocery_pos', 'gas_transport', 'shopping_net', 'misc_net', 'food_dining',
            'home', 'travel', 'grocery_net', 'shopping_pos', 'other'
        ])

    with col2:
        typical_amt = st.number_input("Your typical transaction amount ($)", min_value=0.0, value=50.0, step=1.0)
        typical_hour = st.slider("Your typical transaction hour", 0, 23, 18)
        distance_from_usual = st.number_input("How far from your usual locations? (km)", 0.0, 500.0, 10.0)
        time_since_last = st.number_input("Hours since your last transaction", 0.0, 168.0, 12.0)  # up to 1 week

    submitted = st.form_submit_button("Check for Fraud")

if submitted:
    # Calculate derived features
    log_amt = np.log1p(amt)
    is_late_night = 1 if trans_hour >= 22 or trans_hour <= 5 else 0
    is_night = 1 if trans_hour >= 21 or trans_hour <= 5 else 0
    amt_deviation = amt / typical_amt if typical_amt > 0 else 1.0  # relative to usual

    # Build input DataFrame (must match ALL training columns)
    input_data = pd.DataFrame({
        'category': [category],
        'amt': [amt],
        'log_amt': [log_amt],
        'trans_hour': [trans_hour],
        'is_late_night': [is_late_night],
        'trans_month': [6],                    # default
        'is_holiday_season': [0],
        'distance_km': [distance_from_usual],  # approximate
        'time_since_last_trans': [time_since_last],
        'count_30_days': [2.0],                # default
        'count_7_days': [1.0],
        'count_1_day': [0.0],
        'gender': ['M'],                       # default
        'lat': [43.65], 'long': [-79.38],
        'merch_lat': [43.70], 'merch_long': [-79.40],
        'trans_dayofweek': [3],
        'is_night': [is_night],
        'usual_lat': [43.65], 'usual_long': [-79.38],
        'distance_from_usual_km': [distance_from_usual],
        'city_pop': [1000000],
    })

    if model is None:
        # Placeholder (improve later)
        probability = 0.05 + (amt_deviation - 1) * 0.3 + (is_late_night * 0.2)
        probability = min(max(probability, 0.0), 1.0)
    else:
        probability = model.predict_proba(input_data)[0][1]

    # Display
    st.subheader("Fraud Probability")
    st.metric("Probability of Fraud", f"{probability:.2%}")

    if probability >= 0.30:
        st.error("⚠️ HIGH RISK — Potential Fraud Detected")
    elif probability >= 0.10:
        st.warning("⚠️ MODERATE RISK — Review Recommended")
    else:
        st.success("✅ LOW RISK — Likely Legitimate")

st.markdown("---")
st.caption("Model: XGBoost | Tuned for high recall | Results are illustrative and depend on input accuracy")