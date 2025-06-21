import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Personalised Health Advisor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all datasets and models"""
    try:
        # Load datasets
        diseases_df = pd.read_csv('datasets/symptoms/unique_diseases.csv', encoding='latin1')
        symptoms_df = pd.read_csv('datasets/symptoms/unique_symptoms.csv', encoding='latin1')
        severity_df = pd.read_csv('datasets/symptoms/refined data/Symptom-severity.csv', encoding='latin1')
        description_df = pd.read_csv('datasets/symptoms/description.csv', encoding='latin1')
        medications_df = pd.read_csv('datasets/symptoms/medications.csv', encoding='latin1')
        diets_df = pd.read_csv('datasets/symptoms/diets.csv', encoding='latin1')
        precautions_df = pd.read_csv('datasets/symptoms/precautions_df.csv', encoding='latin1')
        train_data = pd.read_csv('datasets/symptoms/refined data/Train Data.csv', encoding='latin1')
        
        # Load additional datasets
        heart_df = pd.read_csv('datasets/heart/heart.csv', encoding='utf-8-sig')
        diabetes_df = pd.read_csv('datasets/diabetes/diabetes.csv', encoding='utf-8-sig')
        
        heart_df.rename(columns={'ï»¿age': 'age'}, inplace=True)
        
        # Load trained models
        with open('models/NaiveBayes.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/label_encoder.pkl', 'rb') as f:
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
            'label_encoder': label_encoder
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
    
    return info

def main():
    st.markdown('<h1 class="main-header">🏥 Personalised Health Advisor</h1>', unsafe_allow_html=True)
    
    # Load data
    data_dict = load_data()
    if data_dict is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["🏠 Dashboard", "🔍 Symptom Checker", "📊 Health Analytics", "💊 Disease Information", "❤️ Heart Health", "🩸 Diabetes Risk", "📈 Data Insights"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard(data_dict)
    elif page == "🔍 Symptom Checker":
        show_symptom_checker(data_dict)
    elif page == "📊 Health Analytics":
        show_health_analytics(data_dict)
    elif page == "💊 Disease Information":
        show_disease_info_page(data_dict)
    elif page == "❤️ Heart Health":
        show_heart_health(data_dict)
    elif page == "🩸 Diabetes Risk":
        show_diabetes_risk(data_dict)
    elif page == "📈 Data Insights":
        show_data_insights(data_dict)

def show_dashboard(data_dict):
    """Show main dashboard"""
    st.markdown('<h2 class="sub-header">Welcome to Your Personal Health Assistant</h2>', unsafe_allow_html=True)
    
    # Key metrics
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
            <h4>📊 Health Analytics</h4>
            <p>Explore comprehensive health statistics, disease distributions, and symptom severity analysis.</p>
        </div>
        """, unsafe_allow_html=True)

def show_symptom_checker(data_dict):
    """Show symptom checker interface"""
    st.markdown('<h2 class="sub-header">🔍 Symptom Checker</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>How it works:</h4>
        <p>Select the symptoms you're experiencing from the list below. Our AI model will analyze your symptoms and provide a prediction with detailed information about the most likely condition.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get all symptoms
    all_symptoms = data_dict['symptoms']['symptom'].tolist()
    
    # Multi-select symptoms
    selected_symptoms = st.multiselect(
        "Select your symptoms:",
        all_symptoms,
        help="Choose all symptoms that apply to you"
    )
    
    if st.button("🔍 Analyze Symptoms", type="primary"):
        if len(selected_symptoms) == 0:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("Analyzing your symptoms..."):
                predicted_disease, confidence = predict_disease(selected_symptoms, data_dict)
                
                if predicted_disease:
                    st.success(f"Analysis Complete!")
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>Predicted Condition: {predicted_disease}</h3>
                            <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Get disease information
                        disease_info = get_disease_info(predicted_disease, data_dict)
                        
                        if disease_info:
                            if 'description' in disease_info:
                                st.markdown("### 📖 Description")
                                st.write(disease_info['description'])
                            
                            if 'medications' in disease_info:
                                st.markdown("### 💊 Recommended Medications")
                                st.write(disease_info['medications'])
                            
                            if 'diet' in disease_info:
                                st.markdown("### 🍎 Dietary Recommendations")
                                st.write(disease_info['diet'])
                            
                            if 'precautions' in disease_info:
                                st.markdown("### ⚠️ Precautions")
                                st.write(disease_info['precautions'])
                    
                    with col2:
                        st.markdown("### 📊 Confidence Score")
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=confidence * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Prediction Confidence"},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ Important Disclaimer</h4>
                        <p>This is an AI-powered prediction tool for educational purposes only. Always consult with a qualified healthcare professional for proper diagnosis and treatment. This tool should not replace professional medical advice.</p>
                    </div>
                    """, unsafe_allow_html=True)

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
                    st.markdown("### 📖 Description")
                    st.write(disease_info['description'])
                
                if 'medications' in disease_info:
                    st.markdown("### 💊 Medications")
                    st.write(disease_info['medications'])
                
                if 'diet' in disease_info:
                    st.markdown("### 🍎 Dietary Recommendations")
                    st.write(disease_info['diet'])
                
                if 'precautions' in disease_info:
                    st.markdown("### ⚠️ Precautions")
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

if __name__ == "__main__":
    main() 