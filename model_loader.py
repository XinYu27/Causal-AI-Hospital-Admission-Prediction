import streamlit as st
import joblib
import zipfile
import os
@st.cache_resource
def load_model():
    try:
        # Unzip if compressed file exists
        if os.path.exists("rf_best_model.zip"):
            with zipfile.ZipFile("rf_best_model.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
        
        # Load unzipped model
        return joblib.load("rf_best_model.joblib")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
# Load the scaler
@st.cache_resource
def load_scaler():
    try:
        return joblib.load('scaler.joblib')
    except FileNotFoundError:
        st.error("Scaler file not found. Please ensure the model files are properly installed.")
        return None


