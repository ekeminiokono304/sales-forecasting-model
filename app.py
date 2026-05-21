import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

st.set_page_config(page_title="Retail Sales Forecast Engine", page_icon="🏬", layout="wide")

st.title("🏬 Enterprise Retail Operations Sales Prediction Engine")
st.markdown("""
This production dashboard serves real-time predictive analysis out of our optimized LightGBM regression pipeline. 
Adjust the temporal parameters, promotions, and external indices in the left sidebar to generate point-in-time sales forecasts.
""")
st.write("---")

# --- SAFE BUNDLE LOADING ---
@st.cache_resource
def load_production_pipeline_bundle():
    try:
        return joblib.load("sales_forecast_production_bundle.pkl"), False
    except Exception:
        # Fallback dummy model if the user didn't run the training cells yet
        class DummyModel:
            def predict(self, df): return np.array([250.0 + df['onpromotion'].values[0] * 2.5 - df['dcoilwtico'].values[0] * 0.1])
        
        fallback_bundle = {
            "model": DummyModel(),
            "encoders": {"family": type('Classes', (object,), {'classes_': np.array(['AUTOMOTIVE', 'BEVERAGES', 'BREAD/BAKERY', 'DAIRY', 'GROCERY I', 'POULTRY'])})},
            "features": ["store_nbr", "family_encoded", "onpromotion", "year", "month", "day", "day_of_week", "dcoilwtico"]
        }
        return fallback_bundle, True

bundle, is_fallback = load_production_pipeline_bundle()
model = bundle["model"]
encoders = bundle["encoders"]
features = bundle["features"]

if is_fallback:
    st.warning("⚠️ **Running in Demo Mode:** 'sales_forecast_production_bundle.pkl' was not found. Please run the training cells above to activate your fully optimized LightGBM model.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Forecast Feature Inputs Panel")
store_nbr = st.sidebar.number_input("Store Location Number (ID)", min_value=1, max_value=54, value=1)
family_selection = st.sidebar.selectbox("Product Line Category Family", options=list(encoders["family"].classes_))
onpromotion = st.sidebar.number_input("Active Promotional Catalog Count", min_value=0, value=15)
oil_price = st.sidebar.slider("WTI Crude Oil Price Index ($ per barrel)", min_value=10.0, max_value=150.0, value=65.40)
target_date = st.sidebar.date_input("Target Forecast Evaluation Horizon Date", datetime.date(2017, 8, 16))

# Map inputs to match features format
family_encoded = np.where(encoders["family"].classes_ == family_selection)[0][0]
input_data = {
    "store_nbr": store_nbr, "family_encoded": family_encoded, "onpromotion": onpromotion,
    "year": target_date.year, "month": target_date.month, "day": target_date.day,
    "day_of_week": target_date.weekday(), "dcoilwtico": oil_price
}

# Match structural columns of features exactly
input_record = pd.DataFrame([input_data])
for col in features:
    if col not in input_record.columns:
        input_record[col] = 0
input_record = input_record[features]

# --- MAIN PAGE INFERENCE ---
main_col, meta_col = st.columns([2, 1])
with main_col:
    st.subheader("🎯 Real-Time Execution")
    if st.button("🔮 Compute Optimization Forecast Metrics", type="primary"):
        prediction = max(0.0, float(model.predict(input_record)[0]))
        st.success("Inference processing cycle ran completely out of error boundaries!")
        st.metric(label=f"Predicted Unit Sales Estimate for {family_selection}", value=f"{prediction:,.2f} Units")

with meta_col:
    st.subheader("📊 Pipeline Metadata")
    st.markdown(f"* **Model Engine:** {'Fallback Engine' if is_fallback else 'LightGBM Regressor'}\n* **Features Loaded:** {len(features)} variables")
