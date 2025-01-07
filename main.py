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
    st.session_state["current_tab"] = 0
if "inputs" not in st.session_state:
    st.session_state["inputs"] = {}
if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False  # Initialize early for consistency

# Navigation Functions
def next_tab():
    st.session_state["current_tab"] += 1
    st.rerun()

def previous_tab():
    st.session_state["current_tab"] -= 1
    st.rerun()

causal_effects_df = load_causal_effects()
model = load_model()
scaler = load_scaler()
feature_description_mapping = get_feature_mappings()

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
        if st.button("Go Back to Start"):
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

                    # Store input_df in session state for access in Tab 5
                    st.session_state['input_df'] = input_df

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

                    # Layout for buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Start New Prediction"):
                            st.session_state["current_tab"] = 0
                            st.session_state["inputs"] = {}
                            st.session_state["form_submitted"] = False
                            st.session_state.pop('input_df', None)  
                            st.rerun()

                    with col2:
                        if st.button("View Causal Analysis", key="view_causal"):
                            st.session_state["current_tab"] = 5
                            st.rerun()

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

# Tab 5: Causal Analysis
elif current_tab == 5:
    if not st.session_state["form_submitted"]:
        st.warning("No input has been submitted. Please complete the form first to view causal analysis.")
        if st.button("Go Back to Start"):
            st.session_state["current_tab"] = 0
            st.session_state["inputs"] = {}
            st.session_state["form_submitted"] = False
            st.session_state.pop('input_df', None)  # Clear input_df
            st.rerun()
    else:
        input_df = st.session_state.get('input_df')
        
        if input_df is None:
            st.error("No input data found. Please complete the prediction form first.")
            if st.button("Go Back to Start"):
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
                                    if f in input_df.columns and not pd.isna(input_df[f].iloc[0])]

                if not all_features:
                    st.warning("No applicable features found for causal analysis based on the provided inputs.")
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
                                title='Global Causal Effects',
                                labels={
                                    'Causal Effect': 'Effect on Admission Probability',
                                    'Human-Readable Feature': 'Feature'
                                },
                                color='Causal Effect',
                                color_continuous_scale='Turbo'  # Diverging color scale
                            )

                            fig_causal.update_layout(
                                xaxis_tickangle=-45,
                                showlegend=False,
                                height=800
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
                if st.button("Go back to Prediction Result"):
                    previous_tab()

            with col2:
                if st.button("Start New Prediction"):
                    st.session_state["current_tab"] = 0
                    st.session_state["inputs"] = {}
                    st.session_state["form_submitted"] = False
                    st.session_state.pop('input_df', None)  
                    st.rerun()
