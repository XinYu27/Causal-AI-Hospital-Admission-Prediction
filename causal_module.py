import streamlit as st
import pandas as pd
import numpy as np
from model_loader import load_scaler
from feature_processing import arrival_mode_mapping

@st.cache_data
def load_causal_effects():
    data = {
        'Feature': [
            'triage_vital_o2_device', 'cc_pregnancy_diseases',
            'cc_health_services', 'esi', 'cc_skin_diseases',
            'cc_ear_diseases', 'cc_nervous_system_diseases',
            'cc_respiratory_diseases', 'cc_digestive_diseases',
            'digestive_diseases', 'immune_diseases',
            'cc_mental_disorders', 'cc_special_purposes',
            'cc_musculoskeletal_diseases', 'ear_diseases',
            'arrivalhour_bin_sin', 'respiratory_diseases',
            'arrivalmode', 'triage_vital_hr', 'triage_vital_dbp',
            'triage_vital_sbp', 'cc_endocrine_diseases'
        ],
        'Causal Effect': [
            0.6548768278194967, 0.640449438202247, -0.366666666666666,
            -0.36146452447260646, -0.35999999999999954, -0.29123089300080407,
            -0.25387105826880174, 0.211312518786675, 0.1751599905190806,
            0.12379149226411262, 0.10526800423337235, 0.07346919371322846,
            -0.06912442396313373, 0.06601731601731614, -0.034722414115235745,
            -0.017967278863523783, -0.013121513529256945, 0.012849167852241106,
            0.006745588158942217, -0.005887899165885946, 0.005560075619394922,
            1.6653345369377348e-16
        ],
    }

    df = pd.DataFrame(data)

    standard_error = 0.05
    z_value = 1.96

    df['CI Lower'] = df['Causal Effect'] - z_value * standard_error
    df['CI Upper'] = df['Causal Effect'] + z_value * standard_error

    return df

def calculate_individual_treatment_effects(model, input_df, features_to_analyze, continuous_features, arrival_time_bin=None):
    """
    Calculate individual treatment effects by comparing feature states directly
    rather than relative to base prediction.
    
    Parameters:
    - model: trained model
    - input_df: DataFrame with input features
    - features_to_analyze: list of features to analyze
    - continuous_features: list of continuous features
    - arrival_time_bin: string, the arrival time bin from the UI (optional)
    """
    individual_effects = []
    scaler = load_scaler()

    reverse_arrival_mode = {v: k for k, v in arrival_mode_mapping.items()}

    if 'esi' in input_df.columns:
        input_df['esi'] = pd.to_numeric(input_df['esi'], errors='coerce')
    
    for feature in features_to_analyze:
        if feature not in input_df.columns:
            continue
            
        original_value = input_df[feature].iloc[0]
        if pd.isna(original_value):
            continue
            
        try:
            modified_df = input_df.copy()
            
            if feature == 'esi':
                # ESI calculation remains the same
                modified_df[feature] = 1
                high_acuity_pred = model.predict_proba(modified_df)[0][1]
                modified_df[feature] = 5
                low_acuity_pred = model.predict_proba(modified_df)[0][1]
                effect_size = (high_acuity_pred - low_acuity_pred) / 4
                confidence_interval = 0.05

            elif feature == 'arrivalmode':
                # Arrival mode calculation remains the same
                original_text = reverse_arrival_mode.get(original_value, "Unknown")
                baseline_pred = model.predict_proba(modified_df)[0][1]
                
                effects = []
                for mode_value in arrival_mode_mapping.values():
                    if mode_value != original_value:
                        modified_df[feature] = mode_value
                        new_pred = model.predict_proba(modified_df)[0][1]
                        effects.append(new_pred - baseline_pred)
                
                effect_size = np.mean(effects)
                confidence_interval = 0.05
                original_value = original_text

            elif feature == 'arrivalhour_bin_sin':
                # Now using passed arrival_time_bin parameter
                if arrival_time_bin is None or arrival_time_bin == "Unknown":
                    continue
                    
                time_bins = ['23-02', '03-06', '07-10', '11-14', '15-18', '19-22']
                
                # Baseline prediction with original time bin
                baseline_df = modified_df.copy()
                baseline_df['arrivalhour_bin_sin'] = np.sin(time_bins.index(arrival_time_bin) / 6 * 2 * np.pi)
                baseline_pred = model.predict_proba(baseline_df)[0][1]
                
                effect_sizes = []
                for new_time_bin in time_bins:
                    if new_time_bin == arrival_time_bin:
                        continue
                    
                    temp_df = modified_df.copy()
                    temp_df['arrivalhour_bin_sin'] = np.sin(time_bins.index(new_time_bin) / 6 * 2 * np.pi)
                    new_pred = model.predict_proba(temp_df)[0][1]
                    effect_sizes.append(new_pred - baseline_pred)
                
                effect_size = np.mean(effect_sizes)
                confidence_interval = 0.05
                original_value = arrival_time_bin

            elif feature in continuous_features:
                # Continuous features calculation remains the same
                std_dev = scaler.scale_[list(scaler.feature_names_in_).index(feature)]
                
                modified_df[feature] = original_value + std_dev
                increased_pred = model.predict_proba(modified_df)[0][1]
                
                modified_df[feature] = original_value - std_dev
                decreased_pred = model.predict_proba(modified_df)[0][1]
                
                effect_size = (increased_pred - decreased_pred) / 2
                confidence_interval = abs(increased_pred - decreased_pred) * 0.1
                
            else:  # Binary features calculation remains the same
                original_value = input_df[feature].iloc[0]
                if original_value == 1:
                    modified_df[feature] = 0
                    absent_pred = model.predict_proba(modified_df)[0][1]

                    modified_df[feature] = 1
                    present_pred = model.predict_proba(modified_df)[0][1]
                    
                    effect_size = absent_pred - present_pred
                else: 
                    modified_df[feature] = 1
                    present_pred = model.predict_proba(modified_df)[0][1]

                    modified_df[feature] = 0
                    absent_pred = model.predict_proba(modified_df)[0][1]

                    effect_size = present_pred - absent_pred
                
                confidence_interval = 0.05
    
            individual_effects.append({
                'Feature': feature,
                'Original_Value': original_value,
                'Individual_Effect': effect_size,
                'CI_Lower': effect_size - confidence_interval,
                'CI_Upper': effect_size + confidence_interval,
                'Direction': 'Positive' if effect_size > 0 else 'Negative'
            })
            
        except Exception as e:
            st.warning(f"Could not calculate effect for {feature}: {str(e)}")
            continue

    return pd.DataFrame(individual_effects) if individual_effects else pd.DataFrame()

@st.cache_data
def prepare_causal_effects_data(causal_effects_df, feature_description_mapping):
    # Add human-readable names
    df = causal_effects_df.copy()
    df["Human-Readable Feature"] = df["Feature"].map(feature_description_mapping).fillna(df["Feature"])
    return df