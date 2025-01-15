import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from causal_module import load_causal_effects, calculate_individual_treatment_effects
from feature_processing import preprocess_inputs, cc_category, pmh_category, get_feature_mappings, arrival_mode_mapping, display_input_info
from model_loader import load_model, load_scaler

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

# Initialize session state
if "current_tab" not in st.session_state:
    st.session_state["current_tab"] = -1 #Landing  Page
if "inputs" not in st.session_state:
    st.session_state["inputs"] = {}
if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False 

# Navigation Functions
def next_tab():
    st.session_state["current_tab"] += 1
    st.rerun()

def previous_tab():
    st.session_state["current_tab"] -= 1
    st.rerun()

def start_prediction():
    st.session_state['current_tab'] = 0
    st.rerun()

@st.cache_data
def get_model_prediction(_model, df):
    return _model.predict_proba(df)[0][1]

# Load data and models
if "model" not in st.session_state:
    st.session_state["model"] = load_model() 
if "scaler" not in st.session_state:
    st.session_state["scaler"] = load_scaler()  
if "causal_effects" not in st.session_state:
    st.session_state["causal_effects"] = load_causal_effects()
st.session_state["model_loaded"] = True

feature_description_mapping = get_feature_mappings()

if "page_transition" not in st.session_state:
    st.session_state["page_transition"] = False

def validate_sociodemographic(inputs):
    """Allow empty values but validate ranges if values are provided"""
    age = inputs.get('age')
    return age is None or (0 <= age <= 120)

def validate_triage(inputs):
    """Validate triage vital signs are within acceptable ranges if provided"""
    validations = [
        inputs.get('hr') is None or (35.0 <= inputs['hr'] <= 200.0),
        inputs.get('sbp') is None or (50.0 <= inputs['sbp'] <= 260.0),
        inputs.get('dbp') is None or (20.0 <= inputs['dbp'] <= 160.0),
        inputs.get('rr') is None or (10.0 <= inputs['rr'] <= 70.0),
        inputs.get('o2') is None or (90.0 <= inputs['o2'] <= 100.0),
        inputs.get('temp_c') is None or (34.0 <= inputs['temp_c'] <= 42.0)
    ]
    return all(validations)

def show_triage_validation_warnings(inputs):
    """Show specific warnings for out-of-range values"""
    if inputs.get('hr') is not None and not (35.0 <= inputs['hr'] <= 200.0):
        st.warning("Heart Rate must be between 35 and 200")
    if inputs.get('sbp') is not None and not (50.0 <= inputs['sbp'] <= 260.0):
        st.warning("Systolic BP must be between 50 and 260")
    if inputs.get('dbp') is not None and not (20.0 <= inputs['dbp'] <= 160.0):
        st.warning("Diastolic BP must be between 20 and 160")
    if inputs.get('rr') is not None and not (10.0 <= inputs['rr'] <= 70.0):
        st.warning("Respiratory Rate must be between 10 and 70")
    if inputs.get('o2') is not None and not (90.0 <= inputs['o2'] <= 100.0):
        st.warning("Oxygen Saturation must be between 90 and 100")
    if inputs.get('temp_c') is not None and not (34.0 <= inputs['temp_c'] <= 42.0):
        st.warning("Temperature must be between 34.0°C and 42.0°C")

# Initialize session state
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False

# Landing Page Content
if "landing_page_shown" not in st.session_state:
    st.session_state["landing_page_shown"] = False

# Landing Page
if "landing_page_shown" not in st.session_state:
    st.session_state["landing_page_shown"] = False

def show_landing_page():
    st.markdown("""
        <style>
        /* Hero section styling */
        .hero-section {
            padding: 2rem 0;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        
        
        /* Feature card styling */
        .feature-card {
            background: #5AA1E3;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #4A90D9;
            margin-bottom: 1rem;
            transition: transform 0.2s;
            color: white;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            background-color: #4A90D9;
        }
        
        /* Feature card text styling */
        .feature-card h3 {
            color: white !important;
            margin-bottom: 1rem;
            font-weight: 500;
        }
        
        .feature-card p {
            color: rgba(255, 255, 255, 0.9) !important;
            line-height: 1.5;
        }
        
        /* Icon styling */
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 1rem;
            color: #4A90D9;
        }
                
        /* Button Container Styling */
        .button-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 2rem;
        }
                
        /* Button styling */
        .stButton > button {
            background-color: #5AA1E3 !important;
            color: white !important;
            border-radius: 20px !important;
            border: none !important;
            padding: 10px 30px !important;
            font-weight: 500 !important;
            font-size: 1.2rem !important;
            min-width: 200px;
        }

        .stButton > button:hover {
            background-color: #4A90D9 !important;
            border: none !important;
            transition: all 0.3s ease;
        }
        
        /* Override spinner color to match theme */
        .stSpinner > div {
            border-color: #5AA1E3 !important;
        }
                
        /* Add specific centering for the button container */
        .button-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 2rem auto;
            width: 100%;
        }
        
        /* Ensure button takes appropriate width */
        .stButton > button {
            display: inline-block;
            background-color: #5AA1E3 !important;
            color: white !important;
            border-radius: 20px !important;
            border: none !important;
            padding: 8px 24px !important;
            font-weight: 500 !important;
            min-width: 200px;  /* Set minimum width for the button */
        }

        .stButton > button:hover {
            background-color: #4A90D9 !important;
            border: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <h1 style="color: #262730; font-size: 3rem; margin-bottom: 1rem;">
                Hospital Admission Prediction
            </h1>
            <p style="color: #4A4A4A; font-size: 1.2rem; max-width: 800px; margin: 0 auto;">
                Make data-driven decisions using machine learning model and causal analysis system
            </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Features Section using columns
    st.markdown("<h2 style='text-align: center; color: #4A4A4A; margin-bottom: 2rem;'>Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h3 style="color: #262730;">Disposition Predictions</h3>
                <p style="color: #4A4A4A;">
                    Machine Learning model to predict disposition based on patient data
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <h3 style="color: #262730;">Causal Analysis</h3>
                <p style="color: #4A4A4A;">
                    View the key features influencing admission decisions with detailed analysis
                </p>
            </div>
        """, unsafe_allow_html=True)
        

    # How to Use Section
    st.markdown("<h2 style='text-align: center; color: #4A4A4A; margin: 3rem 0 2rem;'>How to Use</h2>", unsafe_allow_html=True)
    
    steps_col1, steps_col2, steps_col3 = st.columns(3)
    
    with steps_col1:
        st.markdown("""
            <div class="feature-card">
                <h3 style="color: #262730;">1. Input Data 📝</h3>
                <p style="color: #4A4A4A;">
                    Fill in patient details across different sections including demographics, vital signs, chief complaints and medical history
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with steps_col2:
        st.markdown("""
            <div class="feature-card">
                <h3 style="color: #262730;">2. Get Prediction 🔮</h3>
                <p style="color: #4A4A4A;">
                    Submit the form to receive a real-time prediction (Complete input ensures better prediction!)
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with steps_col3:
        st.markdown("""
            <div class="feature-card">
                <h3 style="color: #262730;">3. Analyze Results 📊</h3>
                <p style="color: #4A4A4A;">
                    Explore detailed causal analysis and hypotheses to understand the factors behind the prediction
                </p>
            </div>
        """, unsafe_allow_html=True)

    
    # Create a container div for the button
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    
    # Use a single column with a specific width for the button
    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        with st.spinner("Preparing the application..."):
            if st.button("Start Prediction", key="start_button", type="primary", use_container_width=True):
                st.session_state["landing_page_shown"] = True
                st.session_state["current_tab"] = 0
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


# Show landing page if not shown yet
if not st.session_state["landing_page_shown"]:
    show_landing_page()

else:
    model = st.session_state.get("model")
    scaler = st.session_state.get("scaler")
    causal_effects_df = st.session_state.get("causal_effects")

    st.title("Hospital Admission Prediction")
    tab_titles = ["Sociodemographic", "Triage Assessment", "Chief Complaint", "Past Medical History", "Result", "Causal Analysis"]
    current_tab = st.session_state["current_tab"]

    st.sidebar.title("Sections")
    for i, title in enumerate(tab_titles):
        if st.sidebar.button(title, key=f"tab_{i}"):
            st.session_state["current_tab"] = i
            st.rerun()

    inputs = st.session_state["inputs"]

    # Helper Function for Input Validation
    def has_meaningful_input(inputs):
        required_fields = [
            'age', 'gender_binary', 'hr', 'sbp', 'dbp', 'rr', 'o2',
            'temp_c', 'esi', 'arrival_mode', 'arrival_month',
            'arrival_day', 'arrival_hour_bin', 'selected_cc', 'selected_pmh'
        ]
        for field in required_fields:
            value = inputs.get(field)
            if isinstance(value, list):
                if value:
                    return True
            else:
                if value is not None and value != "Not specified":
                    return True
        return False

    # Tab 0: Sociodemographic Information
    if current_tab == 0:
        st.header("Sociodemographic Information")
        inputs['age'] = st.number_input("Age", 0, 120, value=None, key="age")
        gender = st.selectbox("Gender", ["Not specified", "Male", "Female"], key="gender")
        if gender == "Male":
            inputs['gender_binary'] = 1
        elif gender == "Female":
            inputs['gender_binary'] = 0
        else:
            inputs['gender_binary'] = None
            
        is_valid = validate_sociodemographic(inputs)
        if not is_valid:
            st.warning("Age must be between 0 and 120")
        if st.button("Next", key="next_0", disabled=not is_valid):
            next_tab()

    # Tab 1: Triage Assessment
    elif current_tab == 1:
        st.header("Triage Assessment")
        inputs['hr'] = st.number_input("Heart Rate", 35.0, 200.0, step=0.1, format="%.1f", value=None, key="hr")  # Set default
        inputs['sbp'] = st.number_input("Systolic BP", 50.0, 260.0, step=0.1, format="%.1f", value=None, key="sbp")
        inputs['dbp'] = st.number_input("Diastolic BP", 20.0, 160.0, step=0.1, format="%.1f", value=None, key="dbp")
        inputs['rr'] = st.number_input("Respiratory Rate", 10.0, 70.0, step=0.1, format="%.1f", value=None, key="rr")
        inputs['o2'] = st.number_input("Oxygen Saturation", 90.0, 100.0, step=0.1, format="%.1f", value=None, key="o2")
        inputs['o2_device'] = 1 if st.checkbox("Supplementary O2 Device", key="o2_device") else 0
        temp_input = st.number_input("Temperature (°C)", 34.0, 42.0, step=0.1, format="%.1f", value=None, key="temp")
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
        
        is_valid = validate_triage(inputs)
        if not is_valid:
            show_triage_validation_warnings(inputs)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Previous", key="prev_1"): 
                previous_tab()
        with col2:
            if st.button("Next", key="next_1", disabled=not is_valid): 
                next_tab()

    # Tab 2: Chief Complaints
    elif current_tab == 2:
        st.header("Chief Complaints")
        
        inputs['selected_cc'] = st.multiselect("Chief Complaints", cc_category.keys(), key="selected_cc")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Previous", key="prev_2"): 
                previous_tab()
        with col2:
            if st.button("Next", key="next_2"): 
                next_tab()

    # Tab 3: Past Medical History
    elif current_tab == 3:
        st.header("Past Medical History")
        
        inputs['selected_pmh'] = st.multiselect("Past Medical History", pmh_category.keys(), key="selected_pmh")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Previous", key="prev_3"): 
                previous_tab()
        
        with col2:
        # Validation before allowing submission
            if st.button("Submit", key="submit"):
                if has_meaningful_input(inputs):
                    st.session_state["form_submitted"] = True
                    next_tab()
                else:
                    st.warning("Please provide at least one input before submitting.")

    # Tab 4: Result
    elif current_tab == 4:
        if not st.session_state["form_submitted"]:
            st.warning("No input has been submitted. Please complete the form and submit to view the result.")
            if st.button("Go Back to Start", key="go_back_to_start_tab4"):
                st.session_state["current_tab"] = 0
                st.session_state["inputs"] = {}
                st.session_state["form_submitted"] = False
                st.session_state.pop('input_df', None)  # Clear input_df
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
                if st.button("Go Back", key="go_back_start_tab4"):
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
                        # Check if model is loaded
                        if not st.session_state.get("model_loaded", False):
                            st.warning("Please wait while the model is loading...")
                            st.stop()
                            
                        model = st.session_state.get("model")
                        if model is None:
                            st.error("Failed to load the model. Please try refreshing the page.")
                            st.stop()
                        
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

                        # Store input_df in session state for access in Tab 5
                        st.session_state['input_df'] = input_df

                        # Make prediction
                        prediction = model.predict(input_df)
                        prediction_proba = get_model_prediction(model, input_df)



                        if prediction_proba > 0.5:
                            st.markdown(f' <h2 style="color:red;">Admission</h2>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<h2 style="color:green;">Discharge</h2>', unsafe_allow_html=True)

                        
                        # Display the input information
                        st.subheader("Patient Information")
                        display_input_info(display_inputs)

                        def handle_causal_analysis():
                            st.session_state["page_transition"] = True
                            st.session_state["current_tab"] = 5

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Start New Prediction", key="start_new_pred_result_tab4"):
                                st.session_state["current_tab"] = 0
                                st.session_state["inputs"] = {}
                                st.session_state["form_submitted"] = False
                                st.session_state.pop('input_df', None)
                                st.rerun()
                        
                        with col2:
                            st.button("View Causal Analysis", key="view_causal_tab4", on_click=handle_causal_analysis)


                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")

    # Tab 5: Causal Analysis
    elif current_tab == 5:
        if st.session_state.get("page_transition"):
            st.session_state["page_transition"] = False

        if not st.session_state["form_submitted"]:
            st.warning("No input has been submitted. Please complete the form first to view causal analysis.")
            if st.button("Go Back to Start", key="go_back_start_tab5"):
                st.session_state["current_tab"] = 0
                st.session_state["inputs"] = {}
                st.session_state["form_submitted"] = False
                st.session_state.pop('input_df', None) 
                st.rerun()
        else:
            input_df = st.session_state.get('input_df')
            
            if input_df is None or input_df.isnull().all().all():
                st.error("No valid input data found. Please complete the prediction form first.")
                if st.button("Go Back to Start", key="go_back_to_start_tab5"):
                    st.session_state["current_tab"] = 0
                    st.session_state["inputs"] = {}
                    st.session_state["form_submitted"] = False
                    st.rerun()
            else:
                st.header("Causal Effect Analysis")
                
                # Define continuous features
                continuous_features = [
                    'triage_vital_sbp', 'triage_vital_hr', 'triage_vital_dbp',
                    'arrivalhour_bin_sin', 'arrivalhour_bin_cos'
                ]

                try:
                    # Get all possible features from the causal effects data
                    all_features = [f for f in causal_effects_df['Feature'].tolist() 
                                        if f in input_df.columns and not pd.isna(input_df[f].iloc[0]) and input_df[f].iloc[0] != 0]

                    if not all_features:
                        st.warning("No applicable features found for causal analysis based on the provided inputs.")
                        if st.button("Go Back to Start", key="causal_go_back_to_start_tab5"):
                            st.session_state["current_tab"] = 0
                    else:
                        # Calculate individual treatment effects for all features
                        ite_results = calculate_individual_treatment_effects(
                            model, 
                            input_df,
                            all_features,
                            continuous_features,
                            arrival_time_bin=inputs.get('arrival_hour_bin')
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
                                default=available_human_readable[:3]
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
                                    title='Global Causal Effects',
                                    labels={
                                        'Causal Effect': 'Effect on Admission Probability',
                                        'Human-Readable Feature': 'Feature'
                                    },
                                    color='Causal Effect',
                                    color_continuous_scale='Turbo'  # Diverging color scale
                                )

                                fig_causal.update_layout(
                                    autosize=True,
                                    margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins as needed
                                    xaxis=dict(
                                        tickangle=-45,
                                        autorange=True
                                    ),
                                    yaxis=dict(
                                        autorange=True 
                                    ),
                                    showlegend=False
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

                # Buttons to navigate after causal analysis
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Back to Result", key="back_to_result"):
                        st.session_state["current_tab"] = 4
                        st.rerun()

                with col2:
                    if st.button("Start New Prediction", key="start_new_pred_causal"):
                        st.session_state["current_tab"] = 0
                        st.session_state["inputs"] = {}
                        st.session_state["form_submitted"] = False
                        st.session_state.pop('input_df', None)  
                        st.rerun()