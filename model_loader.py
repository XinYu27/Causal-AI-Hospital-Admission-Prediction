import streamlit as st
import joblib
import zipfile
import os

@st.cache_resource
def load_model():
    try:
        model_path = "rf_best_model.joblib"
        
        # Try to load zipped model first if it exists
        if os.path.exists("rf_best_model.zip"):
            with zipfile.ZipFile("rf_best_model.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found")
            return None
            
        # Load model
        model = joblib.load(model_path)
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
# Load the scaler
@st.cache_resource
def load_scaler():
    try:
        scaler_path = 'scaler.joblib'
        if not os.path.exists(scaler_path):
            st.error(f"Scaler file '{scaler_path}' not found")
            return None
            
        scaler = joblib.load(scaler_path)
        return scaler
        
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None