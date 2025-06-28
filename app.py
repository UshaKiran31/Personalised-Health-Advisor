import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from urllib.parse import urlparse, parse_qs

# Try to import Ollama, but handle the case where it's not available
try:
    from ollama import Client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Personalised Health Advisor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Handle navigation via query params for info-box links
query_params = st.query_params
if 'page' in query_params:
    page_param = query_params['page'][0]
    if page_param == 'HealthAnalytics':
        st.session_state.current_page = "📊 Health Analytics"
    elif page_param == 'DataInsights':
        st.session_state.current_page = "📈 Data Insights"
    elif page_param == 'Dashboard':
        st.session_state.current_page = "🏠 Dashboard"

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #1B2631;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        color: #EAECEE;
    }
    .warning-box {
        background-color: #2E2913;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #EAECEE;
    }
    .success-box {
        background-color: #142F23;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #EAECEE;
    }
    .stSelectbox > div > div {
        background-color: #2C3E50;
        color: white;
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: none !important;
        transform: none !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        color: white !important;
    }
    .stButton > button:active, .stButton > button:focus {
        color: white !important;
    }
    [data-testid="stSidebar"] .stButton button {
        text-align: left;
        justify-content: flex-start;
    }
    .floating-chat-icon {
        position: relative;
        display: flex;
        justify-content: center;
        margin-bottom: 1.5rem;
    }
    .chat-circle {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        cursor: pointer;
        border: 3px solid #fff;
        transition: box-shadow 0.2s;
    }
    .chat-circle:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .chat-modal {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: #232946;
        color: #fff;
        border-radius: 18px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        z-index: 9999;
        width: 350px;
        max-width: 95vw;
        max-height: 80vh;
        padding: 0;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        border: 2px solid #764ba2;
    }
    .chat-modal-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        font-size: 1.2rem;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .chat-modal-close {
        cursor: pointer;
        font-size: 1.3rem;
        color: #fff;
        margin-left: 1rem;
    }
    .chat-modal-body {
        flex: 1;
        padding: 1rem 1.5rem 0.5rem 1.5rem;
        overflow-y: auto;
        background: #232946;
    }
    .chat-modal-footer {
        padding: 0.75rem 1.5rem 1rem 1.5rem;
        background: #232946;
        border-top: 1px solid #444;
    }
    .chat-message-user {
        background: #764ba2;
        color: #fff;
        border-radius: 12px 12px 4px 12px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        align-self: flex-end;
        max-width: 80%;
        word-break: break-word;
    }
    .chat-message-bot {
        background: #2c3e50;
        color: #fff;
        border-radius: 12px 12px 12px 4px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        align-self: flex-start;
        max-width: 80%;
        word-break: break-word;
    }
    .chat-modal::-webkit-scrollbar {
        width: 8px;
        background: #232946;
    }
    .chat-modal::-webkit-scrollbar-thumb {
        background: #444;
        border-radius: 4px;
    }
    /* Tooltip styles */
    .symptom-tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .symptom-tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #2c3e50;
        color: #fff;
        text-align: left;
        border-radius: 8px;
        padding: 12px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9rem;
        line-height: 1.4;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 2px solid #3498db;
    }
    .symptom-tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #2c3e50 transparent transparent transparent;
    }
    .symptom-tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    /* Make the entire checkbox area trigger tooltip */
    .stCheckbox > div {
        position: relative;
    }
    .stCheckbox > div:hover .symptom-tooltip .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .symptom-checkbox {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 6px;
        transition: background-color 0.2s;
        border: 1px solid transparent;
    }
    .symptom-checkbox:hover {
        background-color: rgba(52, 152, 219, 0.1);
        border-color: #3498db;
    }
    .symptom-checkbox input[type="checkbox"] {
        margin-right: 8px;
    }
    .symptom-label {
        font-weight: 500;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all datasets and models"""
    try:
        # Define base path to ensure files are found regardless of execution context
        base_path = os.path.dirname(__file__)

        # Load datasets
        diseases_df = pd.read_csv(os.path.join(base_path, 'datasets/symptoms/unique_diseases.csv'), encoding='latin1')
        symptoms_df = pd.read_csv(os.path.join(base_path, 'datasets/symptoms/unique_symptoms.csv'), encoding='latin1')
        severity_df = pd.read_csv(os.path.join(base_path, 'datasets/symptoms/refined data/Symptom-severity.csv'), encoding='latin1')
        description_df = pd.read_csv(os.path.join(base_path, 'datasets/symptoms/description.csv'), encoding='latin1')
        medications_df = pd.read_csv(os.path.join(base_path, 'datasets/symptoms/medications.csv'), encoding='latin1')
        diets_df = pd.read_csv(os.path.join(base_path, 'datasets/symptoms/diets.csv'), encoding='latin1')
        precautions_df = pd.read_csv(os.path.join(base_path, 'datasets/symptoms/precautions_df.csv'), encoding='latin1')
        train_data = pd.read_csv(os.path.join(base_path, 'datasets/symptoms/refined data/Train Data.csv'), encoding='latin1')
        
        # Load symptom descriptions
        symptom_descriptions_df = pd.read_csv(os.path.join(base_path, 'datasets/symptoms/symptom_descriptions.csv'), encoding='utf-8')
        
        # Load doctor consultation dataset
        doctor_consult_df = pd.read_csv(os.path.join(base_path, 'datasets/symptoms/disease_doctor_to_consult.csv'), encoding='utf-8')
        
        # Load additional datasets
        heart_df = pd.read_csv(os.path.join(base_path, 'datasets/heart/heart.csv'), encoding='utf-8-sig')
        diabetes_df = pd.read_csv(os.path.join(base_path, 'datasets/diabetes/diabetes.csv'), encoding='utf-8-sig')
        
        if 'ï»¿age' in heart_df.columns:
            heart_df.rename(columns={'ï»¿age': 'age'}, inplace=True)
        
        # Load trained models
        with open(os.path.join(base_path, 'models/NaiveBayes.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        with open(os.path.join(base_path, 'models/label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        
        return {
            'diseases': diseases_df,
            'symptoms': symptoms_df,
            'severity': severity_df,
            'description': description_df,
            'medications': medications_df,
            'diets': diets_df,
            'precautions': precautions_df,
            'train_data': train_data,
            'heart': heart_df,
            'diabetes': diabetes_df,
            'model': model,
            'label_encoder': label_encoder,
            'symptom_descriptions': symptom_descriptions_df,
            'doctor_consult': doctor_consult_df
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def predict_disease(selected_symptoms, data_dict):
    """Predict disease based on selected symptoms"""
    try:
        # Create feature vector
        all_symptoms = data_dict['symptoms']['symptom'].tolist()
        feature_vector = np.zeros(len(all_symptoms))
        
        for symptom in selected_symptoms:
            if symptom in all_symptoms:
                idx = all_symptoms.index(symptom)
                feature_vector[idx] = 1
        
        # Make prediction
        prediction = data_dict['model'].predict([feature_vector])
        predicted_disease = data_dict['label_encoder'].inverse_transform(prediction)[0]
        
        # Get prediction probability
        probabilities = data_dict['model'].predict_proba([feature_vector])[0]
        max_prob = max(probabilities)
        
        return predicted_disease, max_prob
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, 0

def get_disease_info(disease_name, data_dict):
    """Get comprehensive disease information"""
    info = {}
    
    # Get description
    desc_row = data_dict['description'][data_dict['description']['Disease'] == disease_name]
    if not desc_row.empty:
        info['description'] = desc_row.iloc[0]['Description']
    
    # Get medications
    med_row = data_dict['medications'][data_dict['medications']['Disease'] == disease_name]
    if not med_row.empty:
        meds = med_row.iloc[0].filter(like='Medication').dropna().tolist()
        info['medications'] = ', '.join(meds)
    
    # Get diet
    diet_row = data_dict['diets'][data_dict['diets']['Disease'] == disease_name]
    if not diet_row.empty:
        diets = diet_row.iloc[0].filter(like='Diet').dropna().tolist()
        info['diet'] = ', '.join(diets)
    
    # Get precautions
    prec_row = data_dict['precautions'][data_dict['precautions']['Disease'] == disease_name]
    if not prec_row.empty:
        precautions = prec_row.iloc[0].filter(like='Precaution').dropna().tolist()
        info['precautions'] = ', '.join(precautions)
    
    # Get doctor to consult
    doctor_row = data_dict['doctor_consult'][data_dict['doctor_consult']['Disease'] == disease_name]
    if not doctor_row.empty:
        info['doctor'] = doctor_row.iloc[0]['Doctor to Consult']
    
    return info

def format_symptom_key(symptom_key):
    """Convert snake_case symptom key to human-readable format."""
    return symptom_key.replace('_', ' ').title()

def get_symptom_description(symptom_key, data_dict):
    """Get description for a symptom from the symptom descriptions dataset."""
    try:
        descriptions_df = data_dict['symptom_descriptions']
        description_row = descriptions_df[descriptions_df['Symptom'] == symptom_key]
        if not description_row.empty:
            return description_row.iloc[0]['Description']
        else:
            return "Description not available for this symptom."
    except Exception as e:
        return "Description not available for this symptom."

def main():
    # Move the main header to the sidebar
    st.sidebar.markdown('<h1 class="main-header" style="text-align:left; font-size:2.2rem; margin-bottom:1rem;">🏥 Personalised Health Advisor</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation with buttons instead of dropdown
    # Use session state to track current page
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard"
    
    # Navigation buttons
    if st.sidebar.button("🏠 Dashboard", use_container_width=True):
        st.session_state.current_page = "🏠 Dashboard"
    if st.sidebar.button("🔍 Symptom Checker", use_container_width=True):
        st.session_state.current_page = "🔍 Symptom Checker"
    if st.sidebar.button("💊 Disease Information", use_container_width=True):
        st.session_state.current_page = "💊 Disease Information"
    if st.sidebar.button("❤️ Heart Health", use_container_width=True):
        st.session_state.current_page = "❤️ Heart Health"
    if st.sidebar.button("🩸 Diabetes Risk", use_container_width=True):
        st.session_state.current_page = "🩸 Diabetes Risk"
    # Add AI Chatbot button
    if st.sidebar.button("🤖 AI Chatbot", use_container_width=True):
        st.session_state.current_page = "🤖 AI Chatbot"
    # Add About Us button
    if st.sidebar.button("ℹ️ About Us", use_container_width=True):
        st.session_state.current_page = "ℹ️ About Us"
    
    # Add disclaimer below navigation
    st.sidebar.markdown("""
    <div style='background-color:#2E2913; border-left:4px solid #ffc107; padding:1rem; margin:1.5rem 0 0 0; border-radius:5px; color:#EAECEE; font-size:0.95rem;'>
        <strong>Disclaimer:</strong> This tool is for educational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

    # Load data
    data_dict = load_data()
    if data_dict is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Page routing based on session state
    page = st.session_state.current_page
    
    if page == "🏠 Dashboard":
        show_dashboard(data_dict)
    elif page == "🔍 Symptom Checker":
        show_symptom_checker(data_dict)
    elif page == "💊 Disease Information":
        show_disease_info_page(data_dict)
    elif page == "❤️ Heart Health":
        show_heart_health(data_dict)
    elif page == "🩸 Diabetes Risk":
        show_diabetes_risk(data_dict)
    elif page == "ℹ️ About Us":
        show_about_us(data_dict)
    elif page == "📊 Health Analytics":
        show_health_analytics(data_dict)
    elif page == "📈 Data Insights":
        show_data_insights(data_dict)
    elif page == "🤖 AI Chatbot":
        show_ai_chatbot()

def show_dashboard(data_dict):
    """Show main dashboard"""
    st.markdown('<h2 class="sub-header">Welcome to Your Personal Health Assistant</h2>', unsafe_allow_html=True)
    
    st.markdown('<p> The Personalized Health Advice App is a  virtual assistant that provides tailored health recommendations using user inputs and a trained machine learning model. It helps users assess health risks and receive lifestyle and medical suggestions based on their profile.</p>', unsafe_allow_html=True)
    
    # Quick access features
    st.markdown('<h3 class="sub-header">Quick Access Features</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🔍 Symptom Checker</h4>
            <p>Select your symptoms and get instant disease predictions with detailed information about treatments, medications, and precautions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>❤️ Heart Health Assessment</h4>
            <p>Analyze your heart health risk factors using our comprehensive heart disease dataset with 14 medical parameters.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🩸 Diabetes Risk Calculator</h4>
            <p>Assess your diabetes risk based on various health indicators and get personalized recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>💊 Disease Information</h4>
            <p>Browse comprehensive information about diseases, including descriptions, medications, dietary recommendations, and precautions.</p>
        </div>
        """, unsafe_allow_html=True)

def show_symptom_checker(data_dict):
    """Show enhanced symptom checker interface with modern UI"""
    st.markdown('<h2 class="sub-header">🔍 Symptom Checker</h2>', unsafe_allow_html=True)
    
    # Introduction section
    st.markdown("""
    <div class="info-box">
        <h4>🤖 Smart Health Analysis</h4>
        <p>Our AI-powered diagnostic tool analyzes your symptoms along with personal health factors to provide accurate health insights. 
        Please provide your information below for a comprehensive assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Personal Information Section
    st.markdown("###  Personal Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input(
            "Age", 
            min_value=1, 
            max_value=120, 
            value=30,
            help="Your current age in years"
        )
    
    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"],
            help="Your biological gender"
        )
    
    with col3:
        urgency = st.selectbox(
            "Symptom Urgency",
            ["Mild", "Moderate", "Severe", "Emergency"],
            help="How severe are your symptoms?"
        )
    
    # Additional Health Context
    st.markdown("#### Health Context")
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.selectbox(
            "How long have you had these symptoms?",
            ["Less than 1 day", "1-3 days", "4-7 days", "1-2 weeks", "More than 2 weeks"],
            help="Duration of symptoms"
        )
    
    with col2:
        medical_history = st.multiselect(
            "Do you have any of these conditions?",
            ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "Allergies"],
            help="Select any existing medical conditions"
        )
    
    # Symptom Selection Section
    st.markdown("### Symptom Selection")
    
    # Get all symptoms and organize them
    all_symptoms = data_dict['symptoms']['symptom'].tolist()
    # Build a mapping from key to display value
    symptom_display_dict = {key: format_symptom_key(key) for key in all_symptoms}
    
    # Create symptom categories for better organization
    symptom_categories = {
        "General Symptoms": [s for s in all_symptoms if any(word in s.lower() for word in ['fever', 'fatigue', 'weakness', 'weight', 'appetite'])],
        "Pain & Discomfort": [s for s in all_symptoms if any(word in s.lower() for word in ['pain', 'ache', 'cramp', 'burning', 'stiff'])],
        "Digestive": [s for s in all_symptoms if any(word in s.lower() for word in ['nausea', 'vomit', 'diarrhea', 'constipation', 'stomach', 'abdominal'])],
        "Respiratory": [s for s in all_symptoms if any(word in s.lower() for word in ['cough', 'breathe', 'chest', 'throat', 'sinus'])],
        "Neurological": [s for s in all_symptoms if any(word in s.lower() for word in ['headache', 'dizziness', 'confusion', 'memory', 'seizure'])],
        "Skin & External": [s for s in all_symptoms if any(word in s.lower() for word in ['rash', 'skin', 'itch', 'swelling', 'bruising'])],
        "Other": []
    }
    # Add uncategorized symptoms to "Other"
    categorized_symptoms = set()
    for symptoms in symptom_categories.values():
        categorized_symptoms.update(symptoms)
    symptom_categories["Other"] = [s for s in all_symptoms if s not in categorized_symptoms]
    # Remove empty categories
    symptom_categories = {k: v for k, v in symptom_categories.items() if v}
    # Symptom selection with categories
    selected_symptoms = []
    # Option 1: Quick symptom search
    search_term = st.text_input("Search for symptoms...", placeholder="Type to search symptoms")
    if search_term:
        filtered_symptoms = [s for s in all_symptoms if search_term.lower() in s.lower()]
        if filtered_symptoms:
            # Create search results with tooltips
            for symptom in filtered_symptoms:
                description = get_symptom_description(symptom, data_dict)
                display_name = symptom_display_dict[symptom]
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    is_checked = st.checkbox("", key=f"search_{symptom}")
                    if is_checked and symptom not in selected_symptoms:
                        selected_symptoms.append(symptom)
                
                with col2:
                    st.markdown(f"""
                    <div style="margin-top: 8px;">
                        <span class="symptom-tooltip">
                            <span style="font-weight: 500; color: #2c3e50; cursor: pointer;">
                                {display_name}
                            </span>
                            <span class="tooltiptext">{description}</span>
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
    st.markdown("#### 📋 Browse by Category")
    # Tabs for symptom categories
    tabs = st.tabs(list(symptom_categories.keys()))
    for i, (category, symptoms) in enumerate(symptom_categories.items()):
        with tabs[i]:
            if symptoms:
                cols = st.columns(2)
                for j, symptom in enumerate(symptoms):
                    with cols[j % 2]:
                        description = get_symptom_description(symptom, data_dict)
                        display_name = symptom_display_dict[symptom]
                        checkbox_key = f"{category}_{symptom}"
                        row = st.columns([1, 8])
                        with row[0]:
                            is_checked = st.checkbox("", key=checkbox_key, label_visibility="collapsed")
                        with row[1]:
                            st.markdown(f'<span class="symptom-tooltip" style="color:#fff; font-weight:600; font-size:1rem; cursor:pointer;">{display_name}<span class="tooltiptext">{description}</span></span>', unsafe_allow_html=True)
                        if is_checked and symptom not in selected_symptoms:
                            selected_symptoms.append(symptom)
    # Remove duplicates
    selected_symptoms = list(set(selected_symptoms))
    # Display selected symptoms
    if selected_symptoms:
        st.markdown("### ✅ Selected Symptoms")
        # Create symptom chips with tooltips
        cols = st.columns(min(len(selected_symptoms), 4))
        for i, symptom in enumerate(selected_symptoms):
            with cols[i % 4]:
                description = get_symptom_description(symptom, data_dict)
                st.markdown(f"""
                <div class="symptom-tooltip" style="margin: 0.2rem;">
                    <div style="background: #1B2631; 
                               color: white; padding: 0.5rem; border-radius: 20px; 
                               text-align: center; font-size: 0.9rem; cursor: help;">
                        {symptom_display_dict[symptom]}
                    </div>
                    <span class="tooltiptext">{description}</span>
                </div>
                """, unsafe_allow_html=True)
        st.write(f"**Total symptoms selected:** {len(selected_symptoms)}")
    
    # Analysis button
    if st.button("🔍 Analyze My Symptoms", type="primary", use_container_width=True):
        if len(selected_symptoms) == 0:
            st.error("❌ Please select at least one symptom to proceed with the analysis.")
        else:
            # Create loading animation
            with st.spinner("🤖 AI is analyzing your symptoms..."):
                import time
                time.sleep(2)  # Simulate processing time
                
                predicted_disease, confidence = predict_disease(selected_symptoms, data_dict)
                
                if predicted_disease:
                    # Success message
                    st.success("✅ Analysis Complete! Here are your personalized health insights:")
                    
                    # Create results layout
                    result_col1, result_col2 = st.columns([2, 1])
                    
                    with result_col1:
                        # Main result card
                        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                        st.markdown(f"""
                        <div style="background: #1B2631; 
                                   color: white; padding: 2rem; border-radius: 15px; margin: 1rem 0;">
                            <h2 style="margin: 0; color: white;"> Predicted Condition</h2>
                            <h1 style="margin: 0.5rem 0; color: white;">{predicted_disease}</h1>
                            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; 
                                       border-radius: 10px; display: inline-block;">
                                <strong>Confidence Score: {confidence:.1%}</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk assessment based on age, gender, and urgency
                        risk_factors = []
                        if age > 65:
                            risk_factors.append("Advanced age increases risk")
                        if urgency in ["Severe", "Emergency"]:
                            risk_factors.append("High symptom severity")
                        if "More than 2 weeks" in duration:
                            risk_factors.append("Chronic symptoms")
                        
                        if risk_factors:
                            st.markdown("# Additional Risk Factors")
                            for factor in risk_factors:
                                st.markdown(f"• {factor}")
                        
                        # Personalized recommendations based on demographics
                        st.markdown("### Personalized Recommendations")
                        
                        recommendations = []
                        
                        if urgency == "Emergency":
                            recommendations.append("🚨 **URGENT**: Seek immediate medical attention")
                        elif urgency == "Severe":
                            recommendations.append("⚡ Consider urgent care or ER visit")
                        elif age > 65:
                            recommendations.append("👴 Given your age, consider consulting a physician soon")
                        elif duration == "More than 2 weeks":
                            recommendations.append("📅 Chronic symptoms warrant medical evaluation")
                        
                        if gender == "Female" and age >= 18:
                            recommendations.append("👩 Consider gynecological factors if relevant")
                        
                        for rec in recommendations[:3]:  # Show top 3 recommendations
                            st.markdown(f"• {rec}")
                    
                    with result_col2:

                        # Quick stats
                        st.markdown("### 📈 Quick Stats")
                        st.metric("Symptoms Analyzed", len(selected_symptoms))
                        st.metric("Age Factor", f"{age} years")
                        st.metric("Symptom Duration", duration)
                    
                    
                    disease_info = get_disease_info(predicted_disease, data_dict)
                    
                    if disease_info:
                        info_tabs = st.tabs(["Description", "Treatment", "Diet", "Precautions", "Doctor to Consult"])
                        
                        with info_tabs[0]:
                            if 'description' in disease_info:
                                st.markdown("### About the Condition")
                                st.markdown(f"""
                                <div>{disease_info['description']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info("Detailed description not available for this condition.")
                        
                        with info_tabs[1]:
                            if 'medications' in disease_info:
                                st.markdown("### Recommended Medications")
                                medications = disease_info['medications'].split(', ')
                                for med in medications:
                                    st.markdown(f"• **{med.strip()}**")
                            else:
                                st.info("Medication information not available.")
                        
                        with info_tabs[2]:
                            if 'diet' in disease_info:
                                st.markdown("### Dietary Guidelines")
                                diet_items = disease_info['diet'].split(', ')
                                for item in diet_items:
                                    st.markdown(f"• {item.strip()}")
                            else:
                                st.info("Dietary information not available.")
                        
                        with info_tabs[3]:
                            if 'precautions' in disease_info:
                                st.markdown("### Important Precautions")
                                precautions = disease_info['precautions'].split(', ')
                                for precaution in precautions:
                                    st.markdown(f"• {precaution.strip()}")
                            else:
                                st.info("Precautionary information not available.")
                        
                        with info_tabs[4]:
                            if 'doctor' in disease_info:
                                st.markdown("### Recommended Specialist")
                                st.markdown(f"""
                                    <h4 style="margin: 0; color: #3498db;">{disease_info['doctor']}</h4>
                                    <p style="margin: 0.5rem 0 0 0; color: #EAECEE;">
                                        This specialist is recommended for the diagnosis and treatment of {predicted_disease}.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Additional guidance
                                st.markdown("#### 📋 Consultation Tips")
                                st.markdown("""
                                • **Prepare your symptoms**: Write down all symptoms and their duration<br>
                                • **Medical history**: Bring any relevant medical records<br>
                                • **Questions**: Prepare a list of questions for your doctor<br>
                                • **Follow-up**: Schedule follow-up appointments as recommended<br>
                                • **Emergency**: If symptoms worsen, seek immediate medical attention
                                """, unsafe_allow_html=True)
                            else:
                                st.info("Specialist information not available for this condition.")
                    
                    
                    # Enhanced disclaimer
                    st.markdown("""
                    <div style="background: #2E2913; 
                               color: white; padding: 1.5rem; border-radius: 15px; margin: 2rem 0;">
                        <h4 style="margin: 0; color: white;">⚠️ Important Medical Disclaimer</h4>
                        <p style="margin: 0.5rem 0 0 0; color: white;">
                            This AI-powered analysis is for educational and informational purposes only. 
                            It should NOT replace professional medical diagnosis or treatment. 
                            Always consult with qualified healthcare professionals for proper medical care, 
                            especially if you have severe symptoms or emergency conditions.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                 
                else:
                    st.error("❌ Unable to analyze symptoms. Please try again or consult a healthcare professional.")
    
    # Help section
    with st.expander("❓ How to use this tool effectively"):
        st.markdown("""
        **Tips for better results:**
        
        1. **Be specific**: Select symptoms that closely match what you're experiencing
        2. **Provide accurate information**: Enter correct age, gender, and medical history
        3. **Consider timing**: Note how long you've had symptoms
        4. **Don't ignore severity**: Mark urgent symptoms appropriately
        5. **Seek professional help**: Use this as a starting point, not a final diagnosis
        
        **When to seek immediate medical attention:**
        - Chest pain or difficulty breathing
        - Severe headache or confusion
        - High fever (>103°F)
        - Severe allergic reactions
        - Loss of consciousness
        - Severe bleeding or trauma
        """)

        
def show_health_analytics(data_dict):
    """Show health analytics dashboard"""
    st.markdown('<h2 class="sub-header">📊 Health Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Disease distribution
    st.markdown("### 🏥 Disease Categories Distribution")
    
    # Analyze disease categories
    diseases = data_dict['diseases']['Disease'].tolist()
    categories = {}
    
    for disease in diseases:
        if 'cancer' in disease.lower() or 'carcinoma' in disease.lower():
            categories['Cancer'] = categories.get('Cancer', 0) + 1
        elif 'heart' in disease.lower() or 'cardiac' in disease.lower():
            categories['Heart Disease'] = categories.get('Heart Disease', 0) + 1
        elif 'diabetes' in disease.lower():
            categories['Diabetes'] = categories.get('Diabetes', 0) + 1
        elif 'infection' in disease.lower() or 'viral' in disease.lower():
            categories['Infections'] = categories.get('Infections', 0) + 1
        elif 'mental' in disease.lower() or 'depression' in disease.lower() or 'anxiety' in disease.lower():
            categories['Mental Health'] = categories.get('Mental Health', 0) + 1
        else:
            categories['Other'] = categories.get('Other', 0) + 1
    
    fig = px.pie(
        values=list(categories.values()),
        names=list(categories.keys()),
        title="Disease Categories Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Symptom severity analysis
    st.markdown("### ⚠️ Symptom Severity Analysis")
    
    severity_data = data_dict['severity']
    fig = px.bar(
        severity_data.head(20),
        x='Symptom',
        y='weight',
        title="Top 20 Most Severe Symptoms",
        labels={'weight': 'Severity Weight', 'Symptom': 'Symptom Name'}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Training data statistics
    st.markdown("### 📈 Training Data Statistics")
    
    train_data = data_dict['train_data']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(train_data))
    
    with col2:
        st.metric("Unique Diseases", train_data['Disease'].nunique())
    
    with col3:
        st.metric("Symptom Features", len(train_data.columns) - 1)  # Exclude Disease column

def show_disease_info_page(data_dict):
    """Show disease information page"""
    st.markdown('<h2 class="sub-header">💊 Disease Information Center</h2>', unsafe_allow_html=True)
    
    # Disease selector
    diseases = data_dict['diseases']['Disease'].tolist()
    selected_disease = st.selectbox("Select a disease to learn more:", diseases)
    
    if selected_disease:
        disease_info = get_disease_info(selected_disease, data_dict)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if disease_info:
                if 'description' in disease_info:
                    st.markdown("#### 📖 Description")
                    st.write(disease_info['description'])
                
                if 'medications' in disease_info:
                    st.markdown("#### 💊 Medications")
                    st.write(disease_info['medications'])
                
                if 'diet' in disease_info:
                    st.markdown("#### 🍎 Dietary Recommendations")
                    st.write(disease_info['diet'])
                
                if 'precautions' in disease_info:
                    st.markdown("#### ⚠️ Precautions")
                    st.write(disease_info['precautions'])
            else:
                st.info("Detailed information not available for this disease.")
        
        with col2:
            st.markdown("### 🔍 Quick Facts")
            st.write(f"**Disease Name:** {selected_disease}")
            st.write(f"**Category:** General Health")
            
            # Check if disease exists in training data
            train_data = data_dict['train_data']
            if selected_disease in train_data['Disease'].values:
                disease_data = train_data[train_data['Disease'] == selected_disease]
                symptom_count = disease_data.iloc[0, 1:].sum()
                st.write(f"**Common Symptoms:** {int(symptom_count)}")
            
            # Add doctor consultation information
            if disease_info and 'doctor' in disease_info:
                st.markdown("### 👨‍⚕️ Recommended Specialist")
                st.markdown(f"""
                <div style="background: #1B2631; 
                           color: white; padding: 1rem; border-radius: 8px; 
                           border-left: 4px solid #3498db; margin: 1rem 0;">
                    <h4 style="margin: 0; color: #3498db; font-size: 1rem;">{disease_info['doctor']}</h4>
                </div>
                """, unsafe_allow_html=True)

def show_heart_health(data_dict):
    """Show heart health assessment"""
    st.markdown('<h2 class="sub-header">❤️ Heart Health Assessment</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>Heart Disease Risk Assessment</h4>
        <p>Enter your health parameters below to assess your heart disease risk. This analysis is based on our comprehensive heart disease dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        thalach = st.slider("Maximum Heart Rate", 70, 202, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
    
    if st.button("❤️ Assess Heart Health", type="primary"):
        # Simple risk assessment based on heart dataset patterns
        risk_score = 0
        
        # Age factor
        if age > 65:
            risk_score += 2
        elif age > 45:
            risk_score += 1
        
        # Sex factor (males have higher risk)
        if sex == "Male":
            risk_score += 1
        
        # Blood pressure
        if trestbps > 140:
            risk_score += 2
        elif trestbps > 120:
            risk_score += 1
        
        # Cholesterol
        if chol > 300:
            risk_score += 2
        elif chol > 200:
            risk_score += 1
        
        # Blood sugar
        if fbs == "Yes":
            risk_score += 1
        
        # Exercise angina
        if exang == "Yes":
            risk_score += 2
        
        # Calculate risk percentage
        max_risk = 10
        risk_percentage = (risk_score / max_risk) * 100
        
        # Display results
        st.markdown("### 📊 Heart Health Assessment Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if risk_percentage < 30:
                st.success(f"🟢 Low Risk: {risk_percentage:.1f}%")
                st.write("Your heart health risk is low. Continue maintaining a healthy lifestyle!")
            elif risk_percentage < 60:
                st.warning(f"🟡 Moderate Risk: {risk_percentage:.1f}%")
                st.write("You have moderate risk factors. Consider lifestyle changes and regular check-ups.")
            else:
                st.error(f"🔴 High Risk: {risk_percentage:.1f}%")
                st.write("You have high risk factors. Please consult a healthcare professional.")
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percentage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Heart Disease Risk"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if risk_percentage < 30:
            st.markdown("""
            - Maintain regular exercise routine
            - Eat a balanced diet
            - Get regular check-ups
            - Avoid smoking and excessive alcohol
            """)
        elif risk_percentage < 60:
            st.markdown("""
            - Increase physical activity
            - Monitor blood pressure regularly
            - Reduce salt and saturated fat intake
            - Consider stress management techniques
            - Schedule regular medical check-ups
            """)
        else:
            st.markdown("""
            - **Immediate medical consultation recommended**
            - Strict dietary modifications
            - Regular monitoring of vital signs
            - Medication compliance if prescribed
            - Lifestyle changes under medical supervision
            """)

def show_diabetes_risk(data_dict):
    """Show diabetes risk assessment"""
    st.markdown('<h2 class="sub-header">🩸 Diabetes Risk Assessment</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>Diabetes Risk Calculator</h4>
        <p>Enter your health parameters to assess your diabetes risk. This analysis is based on our diabetes dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.slider("Number of Pregnancies", 0, 17, 0)
        glucose = st.slider("Glucose Level (mg/dl)", 44, 199, 120)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 24, 122, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 7, 99, 20)
    
    with col2:
        insulin = st.slider("Insulin Level (mu U/ml)", 14, 846, 80)
        bmi = st.slider("BMI", 18.0, 67.1, 25.0, 0.1)
        diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5, 0.01)
        age = st.slider("Age", 21, 81, 35)
    
    if st.button("🩸 Assess Diabetes Risk", type="primary"):
        # Simple risk assessment based on diabetes dataset patterns
        risk_score = 0
        
        # Glucose level (most important factor)
        if glucose > 140:
            risk_score += 4
        elif glucose > 120:
            risk_score += 2
        elif glucose > 100:
            risk_score += 1
        
        # BMI
        if bmi > 30:
            risk_score += 2
        elif bmi > 25:
            risk_score += 1
        
        # Age
        if age > 45:
            risk_score += 1
        
        # Blood pressure
        if blood_pressure > 90:
            risk_score += 1
        
        # Insulin
        if insulin > 140:
            risk_score += 1
        
        # Calculate risk percentage
        max_risk = 10
        risk_percentage = (risk_score / max_risk) * 100
        
        # Display results
        st.markdown("### 📊 Diabetes Risk Assessment Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if risk_percentage < 30:
                st.success(f"🟢 Low Risk: {risk_percentage:.1f}%")
                st.write("Your diabetes risk is low. Continue maintaining a healthy lifestyle!")
            elif risk_percentage < 60:
                st.warning(f"🟡 Moderate Risk: {risk_percentage:.1f}%")
                st.write("You have moderate risk factors. Consider lifestyle changes and regular monitoring.")
            else:
                st.error(f"🔴 High Risk: {risk_percentage:.1f}%")
                st.write("You have high risk factors. Please consult a healthcare professional.")
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percentage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Diabetes Risk"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if risk_percentage < 30:
            st.markdown("""
            - Maintain healthy diet
            - Regular exercise
            - Monitor blood sugar occasionally
            - Maintain healthy weight
            """)
        elif risk_percentage < 60:
            st.markdown("""
            - Reduce sugar and refined carbs
            - Increase physical activity
            - Monitor blood sugar regularly
            - Consider weight management
            - Regular medical check-ups
            """)
        else:
            st.markdown("""
            - **Immediate medical consultation recommended**
            - Strict dietary control
            - Regular blood sugar monitoring
            - Weight management program
            - Medication if prescribed
            """)

def show_about_us(data_dict):
    """Show About Us page with key metrics and info-boxes for Health Analytics and Data Insights"""
    st.markdown('<h2 class="sub-header">About Us</h2>', unsafe_allow_html=True)
    # Key metrics at the top (reuse dashboard metrics)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(data_dict['diseases'])}</h3>
            <p>Diseases Covered</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(data_dict['symptoms'])}</h3>
            <p>Symptoms Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(data_dict['train_data'])}</h3>
            <p>Training Records</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>AI-Powered</h3>
            <p>Naive Bayes Model</p>
        </div>
        """, unsafe_allow_html=True)
    # Info-boxes as clickable buttons for navigation
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''
        <div class="info-box" style="cursor:pointer;">
            <h4>📊 Health Analytics</h4>
            <p>Explore comprehensive health statistics, disease distributions, and symptom severity analysis.</p>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("Go to Health Analytics", key="about_health_analytics", use_container_width=True):
            st.session_state.current_page = "📊 Health Analytics"
    with col2:
        st.markdown('''
        <div class="info-box" style="cursor:pointer;">
            <h4>📈 Data Insights</h4>
            <p>Gain insights from our datasets, including disease, symptom, heart, and diabetes data, to better understand health trends.</p>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("Go to Data Insights", key="about_data_insights", use_container_width=True):
            st.session_state.current_page = "📈 Data Insights"

def show_data_insights(data_dict):
    """Show data insights and statistics"""
    st.markdown('<h2 class="sub-header">📈 Data Insights & Statistics</h2>', unsafe_allow_html=True)
    # Dataset overview
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Diseases", len(data_dict['diseases']))
        st.metric("Total Symptoms", len(data_dict['symptoms']))
    with col2:
        st.metric("Training Records", len(data_dict['train_data']))
        st.metric("Heart Disease Records", len(data_dict['heart']))
    with col3:
        st.metric("Diabetes Records", len(data_dict['diabetes']))
        st.metric("Model Accuracy", "~85% (Estimated)")
    # Top diseases by symptom count
    st.markdown("### 🏥 Top Diseases by Symptom Count")
    train_data = data_dict['train_data']
    disease_symptom_counts = []
    for _, row in train_data.iterrows():
        disease = row['Disease']
        symptom_count = row.iloc[1:].sum()
        disease_symptom_counts.append({'Disease': disease, 'Symptom_Count': symptom_count})
    disease_counts_df = pd.DataFrame(disease_symptom_counts)
    disease_counts_df = disease_counts_df.sort_values('Symptom_Count', ascending=False).head(15)
    fig = px.bar(
        disease_counts_df,
        x='Disease',
        y='Symptom_Count',
        title="Top 15 Diseases by Number of Associated Symptoms",
        labels={'Symptom_Count': 'Number of Symptoms', 'Disease': 'Disease Name'}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    # Symptom frequency analysis
    st.markdown("### 🔍 Most Common Symptoms")
    symptom_frequencies = train_data.iloc[:, 1:].sum().sort_values(ascending=False).head(20)
    fig = px.bar(
        x=symptom_frequencies.index,
        y=symptom_frequencies.values,
        title="Top 20 Most Common Symptoms Across All Diseases",
        labels={'x': 'Symptom', 'y': 'Frequency'}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    # Heart disease dataset insights
    st.markdown("### ❤️ Heart Disease Dataset Insights")
    heart_df = data_dict['heart']
    col1, col2 = st.columns(2)
    with col1:
        # Age distribution
        fig = px.histogram(
            heart_df,
            x='age',
            title="Age Distribution in Heart Disease Dataset",
            labels={'age': 'Age', 'count': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Gender distribution
        gender_counts = heart_df['sex'].value_counts()
        fig = px.pie(
            values=gender_counts.values,
            names=['Female' if x == 0 else 'Male' for x in gender_counts.index],
            title="Gender Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    # Diabetes dataset insights
    st.markdown("### 🩸 Diabetes Dataset Insights")
    diabetes_df = data_dict['diabetes']
    col1, col2 = st.columns(2)
    with col1:
        # Glucose distribution
        fig = px.histogram(
            diabetes_df,
            x='Glucose',
            title="Glucose Level Distribution",
            labels={'Glucose': 'Glucose Level (mg/dl)', 'count': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # BMI distribution
        fig = px.histogram(
            diabetes_df,
            x='BMI',
            title="BMI Distribution",
            labels={'BMI': 'Body Mass Index', 'count': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
def show_ai_chatbot():

    st.markdown('<h2 style="font-size: 2.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem;">🤖 AI Chatbot</h2>', unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1rem; border-bottom: 2px solid #3498db;'> I'm not a licensed medical professional, but I can provide general information and answer questions about various health topics. Keep in mind that I'm not capable of diagnosing medical conditions or providing personalized advice.<br>If you have a specific concern or question, I'll do my best to:<br>1. Provide general information on the topic<br>2. Offer suggestions for further research or consultation with a healthcare professional</p>", unsafe_allow_html=True)

    # Check if Ollama is available
    if not OLLAMA_AVAILABLE:
        st.warning("⚠️ AI Chatbot is not available in this environment. This feature requires Ollama to be installed and running locally.")
        st.info("To use this feature, please run the application locally with Ollama installed.")
        return

    try:
        # Initialize Ollama client (default host: http://localhost:11434)
        ollama_client = Client()
        
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Input box and send button
        prompt = st.chat_input("Ask your query here...")
        if prompt:
            # Add user message immediately
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.rerun()

        # Chat window (show all messages)
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

        # If the last message is from the user, get bot response
        if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
            with st.spinner("Thinking..."):
                try:
                    history = []
                    for m in st.session_state["messages"]:
                        if m["role"] == "user":
                            history.append({"role": "user", "content": m["content"]})
                        else:
                            history.append({"role": "assistant", "content": m["content"]})
                    response = ollama_client.chat(
                        model="llama3",
                        messages=history
                    )
                    answer = response["message"]["content"]
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as e:
                    error_msg = f"Error communicating with Ollama: {str(e)}"
                    st.error(error_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": error_msg})
                    st.rerun()
    except Exception as e:
        st.error(f"Error initializing AI Chatbot: {str(e)}")
        st.info("This feature requires Ollama to be installed and running locally.") 

if __name__ == "__main__":
    main() 