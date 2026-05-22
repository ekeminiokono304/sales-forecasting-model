import streamlit as st
import joblib
import os
import gdown

# 🔴 PASTE YOUR GOOGLE DRIVE ID HERE 🔴
GOOGLE_DRIVE_FILE_ID = "1Zlj2HSfXf3qzbFC4xBEUFSwVyOIfJEfR"

model_filename = 'sales_forecast_production_bundle.pkl'

@st.cache_resource
def load_production_model():
    if not os.path.exists(model_filename):
        with st.spinner("Downloading full LightGBM model from cloud storage..."):
            url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
            # Downloads directly into the Streamlit container instance memory
            gdown.download(url, model_filename, quiet=False)
    return joblib.load(model_filename)

try:
    production_bundle = load_production_model()
    st.sidebar.success("🚀 Real LightGBM Production Pipeline Active!")
except Exception as e:
    st.sidebar.error(f"Fallback to Demo mode. Error details: {str(e)}")
    production_bundle = None
