import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import plotly.express as px

# Custom CSS for styling
st.markdown("""
<style>
    /* Custom styles for input fields */
    .stTextInput, .stNumberInput, .stSelectbox, .stMultiselect {
        background-color: #F8F9FA;
        border-radius: 8px;
        border: 1px solid #E9ECEF;
        padding: 8px;
        box-shadow: none;
    }

    /* Style for buttons */
    .stButton > button {
        background-color: #5AA1E3;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 8px 24px;
        font-weight: 500;
    }

    /* Hover effect for buttons */
    .stButton > button:hover {
        background-color: #4A90D9;
        border: none;
    }

    /* Custom header styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Larger header for the main title */
    h1 {
        font-size: 3rem; /* Larger than section headers */
        font-weight: 700;
        color: #262730;
    }

    /* Smaller headers for page sections */
    h2 {
        font-size: 1.8rem;
        font-weight: 600;
        color: #4A4A4A;
    }

    /* Multi-select dropdown customization */
    .stMultiSelect div[role="listbox"] {
        background-color: #D6EAF8 !important; /* Background color of multiselect dropdown */
    }

    /* Style for selected items in multiselect */
    .stMultiSelect [data-baseweb="tag"] {
        color: #1A1A1A !important;
        background-color: #D6EAF8 !important;
    }

    /* Checkbox customization */
    div[data-baseweb="checkbox"] > div[data-checked="true"] {
        background-color: #4A90D9 !important; /* Darker blue when checked */
        color: white !important;
    }

    /* Hover effect for unchecked checkboxes */
    div[data-baseweb="checkbox"] > div:hover {
        background-color: #E5F1FB !important;
    }

    /* Custom styles for radio buttons */
    div[role="radiogroup"] input:checked+label {
        background-color: #4A90D9; /* Darker blue background for selected radio */
        border-color: #4A90D9;
        color: white; /* Optional: Change text color for better contrast */
    }

    /* Default appearance for radio buttons */
    div[role="radiogroup"] label {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 8px;
        padding: 8px 16px;
        margin: 4px;
        cursor: pointer;
        transition: background-color 0.3s, color 0.3s;
    }

    /* Hover effect for radio buttons */
    div[role="radiogroup"] label:hover {
        background-color: #E5F1FB; /* Lighter hover color */
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_scaler():
    try:
        return joblib.load('scaler.joblib')
    except FileNotFoundError:
        st.error("Scaler file not found. Please ensure the model files are properly installed.")
        return None

@st.cache_data
def prepare_error_bars(df):
    df = df.copy()
    df['Error_Positive'] = df['CI Upper'] - df['Causal Effect']
    df['Error_Negative'] = df['Causal Effect'] - df['CI Lower']
    return df

def calculate_individual_treatment_effects(model, input_df, features_to_analyze, continuous_features):
    """
    Calculate individual treatment effects by comparing feature states directly
    rather than relative to base prediction.
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
                # For ESI (1-5 scale)
                if pd.isna(original_value):
                    continue
                    
                # Calculate effect between adjacent ESI levels
                modified_df[feature] = 1
                high_acuity_pred = model.predict_proba(modified_df)[0][1]
                modified_df[feature] = 5
                low_acuity_pred = model.predict_proba(modified_df)[0][1]
                effect_size = (high_acuity_pred - low_acuity_pred) / 4  # Average effect per level
                confidence_interval = 0.05

            elif feature == 'arrivalmode':
                # For arrival mode, use the actual text value
                original_text = reverse_arrival_mode.get(original_value, "Unknown")
                
                # Calculate effect between different modes
                baseline_pred = model.predict_proba(modified_df)[0][1]
                
                # Try other modes and get average effect
                effects = []
                for mode_value in arrival_mode_mapping.values():
                    if mode_value != original_value:
                        modified_df[feature] = mode_value
                        new_pred = model.predict_proba(modified_df)[0][1]
                        effects.append(new_pred - baseline_pred)
                
                effect_size = np.mean(effects)
                confidence_interval = 0.05
                original_value = original_text  # Use text value for display
                
            elif feature == 'arrivalhour_bin_sin':
                # For arrival time, use the original time bin
                original_time_bin = inputs.get('arrival_hour_bin', "Unknown")
                if original_time_bin == "Unknown":
                    continue
                # Define a range of time bins to compare
                time_bins = ['23-02', '03-06', '07-10', '11-14', '15-18', '19-22']
                
                # Baseline prediction with original time bin
                baseline_df = modified_df.copy()
                baseline_df['arrivalhour_bin_sin'] = np.sin(time_bins.index(original_time_bin) / 6 * 2 * np.pi)
                baseline_pred = model.predict_proba(baseline_df)[0][1]
                
                # Calculate predictions for other time bins
                effect_sizes = []
                for new_time_bin in time_bins:
                    if new_time_bin == original_time_bin:
                        continue  # Skip the original time bin
                    
                    temp_df = modified_df.copy()
                    temp_df['arrivalhour_bin_sin'] = np.sin(time_bins.index(new_time_bin) / 6 * 2 * np.pi)
                    new_pred = model.predict_proba(temp_df)[0][1]
                    effect_sizes.append(new_pred - baseline_pred)
                
                # Effect size as the average change across different time bins
                effect_size = np.mean(effect_sizes)
                confidence_interval = 0.05

            elif feature in continuous_features:
                # For continuous features, use standardized unit changes
                std_dev = scaler.scale_[list(scaler.feature_names_in_).index(feature)]
                
                # Calculate predictions for increased and decreased values
                modified_df[feature] = original_value + std_dev
                increased_pred = model.predict_proba(modified_df)[0][1]
                
                modified_df[feature] = original_value - std_dev
                decreased_pred = model.predict_proba(modified_df)[0][1]
                
                # Effect is the slope between these points
                effect_size = (increased_pred - decreased_pred) / 2
                confidence_interval = abs(increased_pred - decreased_pred) * 0.1
                
            else:  # Binary features or categorical
                # Calculate effect between presence and absence
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

@st.cache_data
def prepare_causal_effects_data(causal_effects_df, feature_description_mapping):
    # Add human-readable names
    df = causal_effects_df.copy()
    df["Human-Readable Feature"] = df["Feature"].map(feature_description_mapping).fillna(df["Feature"])
    return df

@st.cache_data
def create_plotly_figure(filtered_df):
    # Prepare error bars
    filtered_df = prepare_error_bars(filtered_df)
    
    fig = px.bar(
        filtered_df,
        x="Human-Readable Feature",
        y="Causal Effect",  # Use signed values
        error_y="Error_Positive",      # Correct error bar settings
        error_y_minus="Error_Negative",
        title="Causal Effect Visualization",
        labels={
            "Causal Effect": "Causal Effect (Probability Change)",
            "Human-Readable Feature": "Feature"
        },
        color="Causal Effect",
        color_continuous_scale="RdBu",  # Diverging color scale
    )
    
    fig.update_layout(
        yaxis=dict(range=[filtered_df["Causal Effect"].min() * 1.2, filtered_df["Causal Effect"].max() * 1.2]),
        xaxis_tickangle=-45
    )
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0,
        y1=0,
        xref='paper',  # Span the entire x-axis
        yref='y',
        line=dict(color="black", width=2)
    )

    return fig

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("rf_best_model.joblib")

model = load_model()

# Initialize session state
if "current_tab" not in st.session_state:
    st.session_state["current_tab"] = 0
if "inputs" not in st.session_state:
    st.session_state["inputs"] = {}

# Navigation Functions
def next_tab():
    st.session_state["current_tab"] += 1
    st.rerun()

def previous_tab():
    st.session_state["current_tab"] -= 1
    st.rerun()

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
    data = data.copy()  # Create a copy to avoid modifying the original
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

causal_effects_df = load_causal_effects()

# Tabs
st.title("Hospital Admission Prediction")
tab_titles = ["Sociodemographic", "Triage Assessment", "Chief Complaint", "Past Medical History", "Result"]
current_tab = st.session_state["current_tab"]

st.sidebar.title("Sections")
for i, title in enumerate(tab_titles):
    if st.sidebar.button(title, key=f"tab_{i}"):
        st.session_state["current_tab"] = i
        st.rerun()

inputs = st.session_state["inputs"]

# Tab 0: Sociodemographic Information
if current_tab == 0:
    st.header("Sociodemographic Information")
    inputs['age'] = st.number_input("Age", 0, 120, value=None, key="age")  # Set a default value to avoid None
    gender = st.selectbox("Gender", ["Not specified", "Male", "Female"], key="gender")
    if gender == "Male":
        inputs['gender_binary'] = 1
    elif gender == "Female":
        inputs['gender_binary'] = 0
    else:
        inputs['gender_binary'] = None
    if st.button("Next", key="next_0"):
        next_tab()

# Tab 1: Triage Assessment
elif current_tab == 1:
    st.header("Triage Assessment")
    inputs['hr'] = st.number_input("Heart Rate", 0, 240, value=None, key="hr")  # Set default
    inputs['sbp'] = st.number_input("Systolic BP", 0, 300, value=None, key="sbp")
    inputs['dbp'] = st.number_input("Diastolic BP", 0, 200, value=None, key="dbp")
    inputs['rr'] = st.number_input("Respiratory Rate", 0, 80, value=None, key="rr")
    inputs['o2'] = st.number_input("Oxygen Saturation", 0, 100, value=None, key="o2")
    inputs['o2_device'] = 1 if st.checkbox("Supplementary O2 Device", key="o2_device") else 0
    temp_input = st.number_input("Temperature (°C)", 25.0, 42.0, step=0.1, format="%.1f", value=None, key="temp")
    inputs['temp_c'] = round(temp_input, 1) if temp_input is not None else None
    if inputs['temp_c'] is not None:
        inputs['temp'] = (inputs['temp_c'] * 9/5) + 32  # Convert C to F
    else:
        inputs['temp'] = None
    
    arrival_mode_options = ["Not specified"] + list(arrival_mode_mapping.keys())
    inputs['arrival_mode'] = st.selectbox("Arrival Method", arrival_mode_options, key="arrival_mode")
    if inputs['arrival_mode'] == "Not specified":
        inputs['arrival_mode'] = None
    
    month_options = ['Not specified', 'January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']
    inputs['arrival_month'] = st.selectbox("Arrival Month", month_options, key="arrival_month")
    if inputs['arrival_month'] == "Not specified":
        inputs['arrival_month'] = None
    
    day_options = ['Not specified', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    inputs['arrival_day'] = st.selectbox("Arrival Day", day_options, key="arrival_day")
    if inputs['arrival_day'] == "Not specified":
        inputs['arrival_day'] = None

    hour_options = ['Not specified', '23-02', '03-06', '07-10', '11-14', '15-18', '19-22']
    inputs['arrival_hour_bin'] = st.selectbox("Arrival Hour", hour_options, key="arrival_hour_bin")
    if inputs['arrival_hour_bin'] == "Not specified":
        inputs['arrival_hour_bin'] = None
    
    esi_options = ["Not specified", "1", "2", "3", "4", "5"]
    inputs['esi'] = st.selectbox("Emergency Severity Index (ESI)", esi_options)
    if inputs['esi'] == "Not specified":
        inputs['esi'] = None

    if st.button("Previous", key="prev_1"): 
        previous_tab()
    if st.button("Next", key="next_1"): 
        next_tab()

# Tab 2: Chief Complaints
elif current_tab == 2:
    st.header("Chief Complaints")
    
    inputs['selected_cc'] = st.multiselect("Chief Complaints", cc_category.keys(), key="selected_cc")
    if st.button("Previous", key="prev_2"): 
        previous_tab()
    if st.button("Next", key="next_2"): 
        next_tab()

# Tab 3: Past Medical History
elif current_tab == 3:
    st.header("Past Medical History")
    
    inputs['selected_pmh'] = st.multiselect("Past Medical History", pmh_category.keys(), key="selected_pmh")
    if st.button("Previous", key="prev_3"): 
        previous_tab()
    # Add validation before allowing submission
    has_meaningful_input = any([
        inputs.get('age') is not None,
        inputs.get('gender_binary') is not None,
        inputs.get('hr') is not None,
        inputs.get('sbp') is not None,
        inputs.get('dbp') is not None,
        inputs.get('rr') is not None,
        inputs.get('o2') is not None,
        inputs.get('temp_c') is not None,
        inputs.get('esi') not in [None, "Not specified"],
        inputs.get('arrival_mode') not in [None, "Not specified"],
        inputs.get('arrival_month') not in [None, "Not specified"],
        inputs.get('arrival_day') not in [None, "Not specified"],
        inputs.get('arrival_hour_bin') not in [None, "Not specified"],
        inputs.get('selected_cc') and len(inputs['selected_cc']) > 0,
        inputs.get('selected_pmh') and len(inputs['selected_pmh']) > 0
    ])

    if st.button("Submit", key="submit"):
        if has_meaningful_input:
            st.session_state["form_submitted"] = True
            next_tab()
        else:
            st.warning("Please provide at least one input before submitting.")

# Initialize session state flag for form submission
if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False

# Tab 4: Result
elif current_tab == 4:
    if not st.session_state["form_submitted"]:
        st.warning("No input has been submitted. Please complete the form and submit to view the result.")
        if st.button("Go Back to Start"):
            st.session_state["current_tab"] = 0
            st.session_state["inputs"] = {}
            st.session_state["form_submitted"] = False
            st.rerun()
    else:
        st.header("Prediction Result")
    
        # Check if ANY meaningful data was entered
        has_data = False
        for key, value in inputs.items():
            # Skip o2_device in this check
            if key == 'o2_device':
                continue
            # Check if the value is meaningful
            if value is not None and value != "Not specified":
                if isinstance(value, list):  # For selected_cc and selected_pmh
                    if len(value) > 0:
                        has_data = True
                        break
                else:
                    has_data = True
                    break

        if not has_data:
            st.error("No input values were provided. Please go back and fill in some information.")
            if st.button("Go Back"):
                st.session_state["current_tab"] = 0
                st.session_state["inputs"] = {}
                st.session_state["form_submitted"] = False
                st.rerun()
        else:
            # Create display_inputs only if we have actual data
            display_inputs = {}

            if inputs.get('age') is not None:
                display_inputs['Age'] = inputs['age']
                
            if inputs.get('gender_binary') is not None:
                display_inputs['Gender'] = 'Male' if inputs['gender_binary'] == 1 else 'Female'
                
            if inputs.get('hr') is not None:
                display_inputs['Heart Rate'] = inputs['hr']
                
            if inputs.get('o2_device') is not None:
                display_inputs['Oxygen Device'] = 'Yes' if inputs['o2_device'] == 1 else 'No'
                
            if inputs.get('sbp') is not None:
                display_inputs['Systolic BP'] = inputs['sbp']
                
            if inputs.get('dbp') is not None:
                display_inputs['Diastolic BP'] = inputs['dbp']
                
            if inputs.get('temp_c') is not None:
                display_inputs['Temperature (°C)'] = inputs['temp_c']
                
            if inputs.get('esi') is not None:
                display_inputs['ESI Level'] = inputs['esi']
                
            if inputs.get('arrival_mode') is not None and inputs['arrival_mode'] != "Not specified":
                display_inputs['Arrival Mode'] = inputs['arrival_mode']
                
            if inputs.get('arrival_month') is not None and inputs['arrival_month'] != "Not specified":
                display_inputs['Arrival Month'] = inputs['arrival_month']
                
            if inputs.get('arrival_day') is not None and inputs['arrival_day'] != "Not specified":
                display_inputs['Arrival Day'] = inputs['arrival_day']
                
            if inputs.get('arrival_hour_bin') is not None and inputs['arrival_hour_bin'] != "Not specified":
                display_inputs['Arrival Time'] = inputs['arrival_hour_bin']

            # Add Chief Complaints
            if inputs.get('selected_cc') and len(inputs['selected_cc']) > 0:
                display_inputs['Chief Complaints'] = ', '.join(inputs['selected_cc'])
        
            # Add Past Medical History
            if inputs.get('selected_pmh') and len(inputs['selected_pmh']) > 0:
                display_inputs['Past Medical History'] = ', '.join(inputs['selected_pmh'])

            # Remove None values for cleaner display
            display_inputs = {k: v for k, v in display_inputs.items() if v is not None}
        
            if display_inputs:
                try:
                    # Create processing DataFrame for prediction
                    user_inputs = {
                        'age': inputs.get('age'),
                        'gender': inputs.get('gender_binary'),
                        'triage_vital_hr': inputs.get('hr'),
                        'triage_vital_o2_device': inputs.get('o2_device'),
                        'triage_vital_sbp': inputs.get('sbp'),
                        'triage_vital_dbp': inputs.get('dbp'),
                        'triage_vital_temp': inputs.get('temp'),
                        'esi': inputs.get('esi'),
                        'arrivalmode': arrival_mode_mapping.get(inputs.get('arrival_mode')),
                        'arrivalmonth': inputs.get('arrival_month'),
                        'arrivalday': inputs.get('arrival_day'),
                        'arrivalhour_bin': inputs.get('arrival_hour_bin')
                    }

                    # Initialize all 'cc_*' and 'pmh_*' features to 0
                    for cc_feature in cc_category.values():
                        user_inputs[cc_feature] = 0
                    for pmh_feature in pmh_category.values():
                        user_inputs[pmh_feature] = 0

                    # Set selected chief complaints to 1
                    for selected in inputs.get('selected_cc', []):
                        feature = cc_category.get(selected)
                        if feature:
                            user_inputs[feature] = 1

                    # Set selected past medical history to 1
                    for selected in inputs.get('selected_pmh', []):
                        feature = pmh_category.get(selected)
                        if feature:
                            user_inputs[feature] = 1

                    # Create DataFrame
                    input_df = pd.DataFrame([user_inputs])

                    # Preprocess Inputs
                    input_df = preprocess_inputs(input_df)

                    # Load the features used during training
                    trained_features = [
                        'triage_vital_o2_device', 'cc_pregnancy_diseases', 'cc_health_services',
                        'esi', 'cc_skin_diseases', 'cc_ear_diseases',
                        'cc_nervous_system_diseases', 'cc_respiratory_diseases',
                        'cc_digestive_diseases', 'digestive_diseases', 'immune_diseases',
                        'cc_mental_disorders', 'cc_special_purposes', 'cc_musculoskeletal_diseases',
                        'ear_diseases', 'arrivalhour_bin_sin', 'respiratory_diseases',
                        'arrivalmode', 'triage_vital_hr', 'triage_vital_dbp',
                        'triage_vital_sbp', 'cc_endocrine_diseases'
                    ]

                    # Align DataFrame
                    input_df = input_df.reindex(columns=trained_features, fill_value=0)
                    # Ensure no NaN overwrites valid inputs
                    input_df.update(input_df.fillna(np.nan))

                    # Make prediction
                    prediction = model.predict(input_df)
                    prediction_proba = model.predict_proba(input_df)[0][1]

                    if prediction[0] == 1:
                        st.markdown(f' <h2 style="color:red;">Admission</h2>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<h2 style="color:green;">Discharge</h2>', unsafe_allow_html=True)

                    # Display the input information
                    st.subheader("Patient Information")
                    display_input_info(display_inputs)

                    # Back to first page
                    if st.button("Start New Prediction"):
                        st.session_state["current_tab"] = 0
                        st.session_state["inputs"] = {}
                        st.session_state["form_submitted"] = False
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
            else:
                st.error("No input values were provided. Please go back and fill in some information.")
                if st.button("Go Back"):
                    st.session_state["current_tab"] = 0
                    st.session_state["inputs"] = {}
                    st.session_state["form_submitted"] = False
                    st.rerun()

        # Get cached mappings
        feature_description_mapping = get_feature_mappings()
        reverse_feature_mapping = {v: k for k, v in feature_description_mapping.items()}
        
        # Prepare data once
        prepared_causal_effects = prepare_causal_effects_data(causal_effects_df, feature_description_mapping)
        
        # Get human-readable options
        human_readable_options = prepared_causal_effects["Human-Readable Feature"].tolist()
        
        if current_tab == 4 and st.session_state["form_submitted"]:
            st.subheader("Causal Effect Analysis")
            
            # Define continuous features
            continuous_features = [
                'triage_vital_sbp', 'triage_vital_hr', 'triage_vital_dbp',
                'arrivalhour_bin_sin', 'arrivalhour_bin_cos'
            ]
            
            try:
                # Get all possible features from the causal effects data
                all_features = [f for f in causal_effects_df['Feature'].tolist() 
                                    if f in input_df.columns and not pd.isna(input_df[f].iloc[0])]

                # Calculate individual treatment effects for all features
                ite_results = calculate_individual_treatment_effects(
                    model, 
                    input_df,
                    all_features,
                    continuous_features
                )
                
                # Merge with feature descriptions for display
                ite_results['Human-Readable Feature'] = ite_results['Feature'].map(
                    feature_description_mapping
                ).fillna(ite_results['Feature'])
                
                # Get human-readable options only for features with actual input values
                available_human_readable = ite_results['Human-Readable Feature'].tolist()
                
                if available_human_readable:
                    # Filter the causal effects dataframe to only show effects for features that have input values
                    filtered_causal_effects = causal_effects_df[
                        causal_effects_df['Feature'].isin(all_features)
                    ].copy()
                    
                    # Add human-readable feature names
                    filtered_causal_effects['Human-Readable Feature'] = filtered_causal_effects['Feature'].map(
                        feature_description_mapping
                    ).fillna(filtered_causal_effects['Feature'])
                    
                    selected_features = st.multiselect(
                        "Select Features to View Effects:",
                        available_human_readable,
                        default=[]
                    )
                    
                    if selected_features:
                        # Filter results based on selection for the plot
                        plot_data = filtered_causal_effects[
                            filtered_causal_effects['Human-Readable Feature'].isin(selected_features)
                        ].sort_values(by='Causal Effect', ascending=False)
                        
                        # Create visualization using average causal effects
                        fig_causal = px.bar(
                            plot_data,
                            x='Human-Readable Feature',
                            y='Causal Effect',
                            error_y='CI Upper',
                            error_y_minus='CI Lower',
                            title='Global Causal Effects',
                            labels={
                                'Causal Effect': 'Effect on Admission Probability',
                                'Human-Readable Feature': 'Feature'
                            },
                            color='Causal Effect',
                            color_continuous_scale='RdBu'  # Diverging color scale
                        )

                        fig_causal.update_layout(
                            xaxis_tickangle=-45,
                            showlegend=False,
                            height=500
                        )

                        # Add a horizontal line at y=0
                        fig_causal.add_shape(
                            type='line',
                            x0=0,
                            x1=1,
                            y0=0,
                            y1=0,
                            xref='paper',
                            yref='y',
                            line=dict(color='Black', width=2)
                        )

                        # Update y-axis range
                        fig_causal.update_yaxes(
                            range=[plot_data['Causal Effect'].min() * 1.2, 
                                plot_data['Causal Effect'].max() * 1.2]
                        )

                        # Display the plot
                        st.plotly_chart(fig_causal, use_container_width=True)
                        
                        # Add detailed analysis for selected features
                        st.write("### Individual Counterfactual Hypothesis")
                        for _, row in ite_results.iterrows():
                            if row['Human-Readable Feature'] in selected_features:
                                feature = row['Human-Readable Feature']
                                effect = row['Individual_Effect'] * 100
                                value = row['Original_Value']
                                ci_lower = row['CI_Lower'] * 100
                                ci_upper = row['CI_Upper'] * 100
                                
                                # Create expandable sections for each feature
                                with st.expander(f"🔍 {feature}"):
                                    # For ESI
                                    if row['Feature'] == 'esi':
                                        is_high_acuity = value <= 2
                                        next_level = min(5, value + 1) if is_high_acuity else max(1, value - 1)
                                        
                                        st.markdown(f"""
                                        **Current Value:** {value}
                                        
                                        **Effect on Admission Probability:** {effect:.2f}% (95% CI: {ci_lower:.2f}% to {ci_upper:.2f}%)
                                        
                                        **Counterfactual Analysis:** If the ESI level were {next_level} ({"lower" if is_high_acuity else "higher"} acuity), the admission probability would {"decrease" if is_high_acuity else "increase"} by approximately **{abs(effect):.2f}%**.
                                        """)

                                    # For arrival mode
                                    elif row['Feature'] == 'arrivalmode':
                                        current_mode = row['Original_Value']
                                        st.markdown(f"""
                                        **Current Mode:** {current_mode}
                                        
                                        **Effect on Admission Probability:** {effect:.2f}% (95% CI: {ci_lower:.2f}% to {ci_upper:.2f}%)
                                        
                                        **Counterfactual Analysis:** Changing from {current_mode} to other arrival modes would affect the admission probability by approximately **{abs(effect):.2f}%**.
                                        """)

                                    # For arrival time
                                    elif row['Feature'] == 'arrivalhour_bin_sin':
                                        current_time = row['Original_Value']
                                        st.markdown(f"""
                                        **Current Time:** {current_time}
                                        
                                        **Effect on Admission Probability:** {effect:.2f}% (95% CI: {ci_lower:.2f}% to {ci_upper:.2f}%)
                                        
                                        **Counterfactual Analysis:** Arriving at a different time would affect the admission probability by approximately **{abs(effect):.2f}%**.
                                        """)

                                    # For continuous vital signs
                                    elif row['Feature'] in ['triage_vital_sbp', 'triage_vital_hr', 'triage_vital_dbp']:
                                        direction = "increase" if effect > 0 else "decrease"
                                        st.markdown(f"""
                                        **Current Value:** {value}
                                        
                                        **Effect on Admission Probability:** {effect:.2f}% (95% CI: {ci_lower:.2f}% to {ci_upper:.2f}%)
                                        
                                        **Counterfactual Analysis:** A meaningful change in this vital sign would affect the admission probability by approximately **{abs(effect):.2f}%**.
                                        """)

                                    else:
                                        status = "Present" if value == 1 else "Absent"
                                        if value == 1:
                                            counterfactual_direction = "decrease" if effect > 0 else "increase"
                                        else:
                                            counterfactual_direction = "increase" if effect > 0 else "decrease"
                                        st.markdown(f"""
                                        **Current Status:** {status}
                                        
                                        **Effect on Admission Probability:** {effect:.2f}% (95% CI: {ci_lower:.2f}% to {ci_upper:.2f}%)
                                        
                                        **Counterfactual Analysis:** If this condition were {'absent' if value == 1 else 'present'}, the admission probability would **{counterfactual_direction}** by approximately **{abs(effect):.2f}%**.
                                        """)
            except Exception as e:
                st.error(f"Error calculating effects: {str(e)}")
