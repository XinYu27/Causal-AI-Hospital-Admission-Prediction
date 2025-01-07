from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import streamlit as st

# Mappings and Encoders
arrival_mode_mapping = {
    "Ambulance": 6, "Car": 0, "Walk-in": 4, "Police": 2,
    "Public Transportation": 3, "Wheel Chair": 5, "Other": 1
}

# Initialize and fit the encoders with predefined categories
arrival_month_encoder = OrdinalEncoder(categories=[[  
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]])


arrival_day_encoder = OrdinalEncoder(categories=[[  
    'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'
]])


arrival_hour_encoder = OrdinalEncoder(categories=[['23-02', '03-06', '07-10', '11-14', '15-18', '19-22']])


cc_category = {
    "Infectious or Parasitic Diseases" : "cc_infectious_parasitic_diseases",
    "Neoplasms" : "cc_neoplasms",
    "Blood or Blood-Forming Organs Diseases" : "cc_blood_diseases",
    "Immune System Diseases" : "cc_immune_diseases",
    "Endocrine, Nutritional or Metabolic Diseases" : "cc_endocrine_diseases",
    "Mental, Behavioural or Neurodevelopmental Disorders" : "cc_mental_disorders",
    "Nervous System Diseases" : "cc_nervous_system_diseases",
    "Visual System Diseases" : "cc_visual_system_diseases",
    "Ear or Mastoid Process Diseases" : "cc_ear_diseases",
    "Circulatory System Diseases" : "cc_circulatory_diseases",
    "Respiratory System Diseases" : "cc_respiratory_diseases",
    "Digestive System Diseases" : "cc_digestive_diseases",
    "Skin and Subcutaneous Tissue Diseases" : "cc_skin_diseases",
    "Musculoskeletal System or Connective Tissue Diseases" : "cc_musculoskeletal_diseases",
    "Genitourinary System Diseases" : "cc_genitourinary_diseases",
    "Pregnancy, Childbirth or the Puerperium" : "cc_pregnancy_diseases",
    "Symptoms, Signs or Clinical Findings, Not Elsewhere Classified" : "cc_clinical_findings",
    "Injury, Poisoning or Certain Other Consequences of External Causes and External Causes of Morbidity or Mortality" : "cc_injuries_and_external_causes",
    "Factors Influencing Health Status or Contact with Health Services" : "cc_health_services",
    "Other" : "cc_special_purposes"
}

pmh_category = {
    "Infectious or Parasitic Diseases": "infectious_parasitic_diseases",
    "Neoplasms" : "neoplasms",
    "Blood or Blood-Forming Organs Diseases" : "blood_diseases",
    "Immune System Diseases" : "immune_diseases",
    "Endocrine, Nutritional or Metabolic Diseases" : "endocrine_diseases",
    "Mental, Behavioural or Neurodevelopmental Disorders" : "mental_disorders",
    "Nervous System Diseases" : "nervous_system_diseases",
    "Visual System Diseases" : "visual_system_diseases",
    "Ear or Mastoid Process Diseases" : "ear_diseases",
    "Circulatory System Diseases" : "circulatory_diseases",
    "Respiratory System Diseases" : "respiratory_diseases",
    "Digestive System Diseases" : "digestive_diseases",
    "Skin and Subcutaneous Tissue Diseases" : "skin_diseases",
    "Musculoskeletal System or Connective Tissue Diseases" : "musculoskeletal_diseases",
    "Genitourinary System Diseases" : "genitourinary_diseases",
    "Pregnancy, Childbirth or the Puerperium" : "pregnancy_diseases",
    "Certain Conditions Originating in the Perinatal Period" : "perinatal_diseases",
    "Developmental Anomalies" : "developmental_anomalies",
    "Symptoms, Signs or Clinical Findings, Not Elsewhere Classified" : "clinical_findings",
    "Injury, Poisoning or Certain Other Consequences of External Causes and External Causes of Morbidity or Mortality" : "injuries_and_external_causes",
    "Factors Influencing Health Status or Contact with Health Services" : "health_services",
    "Other" : "unclassified"
}

# Feature transformation
def preprocess_inputs(data):
    data = data.copy() 
    data = data.replace({None: np.nan, "": np.nan})

    # Encode arrivalmonth, arrivalday, and arrivalhour_bin
    if 'arrival_month' in data.columns and data['arrival_month'].notna().all():
        data['arrival_month'] = data['arrival_month'].astype(str)
        data[['arrival_month']] = arrival_month_encoder.transform(data[['arrival_month']])
        data['arrivalmonth_sin'] = np.sin(data['arrival_month'] / 12 * 2 * np.pi)
        data['arrivalmonth_cos'] = np.cos(data['arrival_month'] / 12 * 2 * np.pi)
        
    if 'arrival_day' in data.columns and data['arrival_day'].notna().all():
        data['arrival_day'] = data['arrival_day'].astype(str)
        data[['arrival_day']] = arrival_day_encoder.transform(data[['arrival_day']])
        data['arrivalday_sin'] = np.sin(data['arrival_day'] / 7 * 2 * np.pi)
        data['arrivalday_cos'] = np.cos(data['arrival_day'] / 7 * 2 * np.pi)
        
    if 'arrival_hour_bin' in data.columns and data['arrival_hour_bin'].notna().all():
        data['arrival_hour_bin'] = data['arrival_hour_bin'].astype(str)
        data[['arrival_hour_bin']] = arrival_hour_encoder.transform(data[['arrival_hour_bin']])
        data['arrivalhour_bin_sin'] = np.sin(data['arrival_hour_bin'] / 6 * 2 * np.pi)
        data['arrivalhour_bin_cos'] = np.cos(data['arrival_hour_bin'] / 6 * 2 * np.pi)

    # Map arrival mode
    if 'arrival_mode' in data.columns:
        data['arrivalmode'] = data['arrival_mode'].map(arrival_mode_mapping)

    # Ensure all required features are present
    required_features = [
        'triage_vital_o2_device', 'cc_pregnancy_diseases', 'cc_health_services',
        'esi', 'cc_skin_diseases', 'cc_ear_diseases', 'cc_nervous_system_diseases',
        'cc_respiratory_diseases', 'cc_digestive_diseases', 'digestive_diseases',
        'immune_diseases', 'cc_mental_disorders', 'cc_special_purposes',
        'cc_musculoskeletal_diseases', 'ear_diseases', 'arrivalhour_bin_sin',
        'respiratory_diseases', 'arrivalmode', 'triage_vital_hr', 'triage_vital_dbp',
        'triage_vital_sbp', 'cc_endocrine_diseases'
    ]
    
    for feature in required_features:
        if feature not in data.columns:
            data[feature] = 0

    # Drop intermediate columns
    columns_to_drop = ['arrival_month', 'arrival_day', 'arrival_hour_bin', 'arrival_mode']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    return data[required_features]

def display_input_info(display_inputs):
    # Convert dictionary to DataFrame and handle types properly
    display_df = pd.DataFrame.from_dict(display_inputs, orient='index', columns=['Values'])
    
    # Convert all values to strings to avoid Arrow serialization issues
    display_df['Values'] = display_df['Values'].astype(str)
    
    # Display the DataFrame
    st.dataframe(display_df, use_container_width=True)

@st.cache_data
def get_feature_mappings():
    feature_description_mapping = {
        'triage_vital_o2_device': 'Use of Supplementary Oxygen Device',
        'cc_pregnancy_diseases': 'Pregnancy-Related Conditions (Chief Complaint)',
        'cc_health_services': 'Health Service Factors (Chief Complaint)',
        'esi': 'Emergency Severity Index',
        'cc_skin_diseases': 'Skin Diseases (Chief Complaint)',
        'cc_ear_diseases': 'Ear Diseases (Chief Complaint)',
        'cc_nervous_system_diseases': 'Nervous System Diseases (Chief Complaint)',
        'cc_respiratory_diseases': 'Respiratory Diseases (Chief Complaint)',
        'cc_digestive_diseases': 'Digestive Diseases (Chief Complaint)',
        'digestive_diseases': 'Digestive Diseases (Medical History)',
        'immune_diseases': 'Immune Diseases (Medical History)',
        'cc_mental_disorders': 'Mental Disorders (Chief Complaint)',
        'cc_special_purposes': 'Special Purpose Factors (Chief Complaint)',
        'cc_musculoskeletal_diseases': 'Musculoskeletal Diseases (Chief Complaint)',
        'ear_diseases': 'Ear Diseases (Past History)',
        'arrivalhour_bin_sin': 'Time of Arrival (Sinusoidal)',
        'respiratory_diseases': 'Respiratory Diseases (Medical History)',
        'arrivalmode': 'Arrival Mode',
        'triage_vital_hr': 'Heart Rate',
        'triage_vital_dbp': 'Diastolic Blood Pressure',
        'triage_vital_sbp': 'Systolic Blood Pressure',
        'cc_endocrine_diseases': 'Endocrine Diseases (Chief Complaint)'
    }
    return feature_description_mapping
