import streamlit as st
import joblib

# Load the scaler
@st.cache_resource
def load_scaler():
    try:
        return joblib.load('scaler.joblib')
    except FileNotFoundError:
        st.error("Scaler file not found. Please ensure the model files are properly installed.")
        return None
    
# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load("rf_best_model.joblib")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model files are properly installed.")
        return None

