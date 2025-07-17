import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

# Custom CSS for dark theme
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-bg: #0f1419;
        --secondary-bg: #1a1f2e;
        --card-bg: #23242b;
        --accent-color: #647dee;
        --accent-gradient: linear-gradient(135deg, #7f53ac 0%, #647dee 100%);
        --text-primary: #ffffff;
        --text-secondary: #b3b8c5;
        --text-muted: #8892b0;
        --border-color: #2d3142;
        --success-color: #00d2d3;
        --warning-color: #ffab00;
        --error-color: #ff6b6b;
        --shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        --shadow-light: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Main app styling */
    .stApp {
        background: var(--primary-bg);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main header styling */
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
    
    /* Form styling */
    .stForm {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    
    /* Input field styling */
    .stSelectbox > div > div {
        background: var(--secondary-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 2px rgba(100, 125, 222, 0.2);
    }
    
    .stNumberInput > div > div > input {
        background: var(--secondary-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
        padding: 0.75rem;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 2px rgba(100, 125, 222, 0.2);
        outline: none;
    }
    
    .stSlider > div > div > div {
        background: var(--secondary-bg);
        border-radius: 8px;
    }
    
    .stMultiSelect > div > div {
        background: var(--secondary-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--accent-gradient);
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        color: white;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-light);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(100, 125, 222, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Form submit button */
    .stFormSubmitButton > button {
        background: var(--accent-gradient);
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        color: white;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-light);
    }
    
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(100, 125, 222, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;         /* was 12px */
        padding: 0.75rem;           /* was 1.5rem */
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: var(--shadow-light);
        min-width: 120px;           /* optional: ensures cards don't get too small */
        max-width: 180px;           /* optional: limits max width */
        margin: 0 auto;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow);
    }
    
    .metric-value {
        font-size: 1.7rem;          /* was 2.5rem */
        font-weight: 700;
        color: var(--accent-color);
        margin: 0.25rem 0;
    }
    
    .metric-label {
        font-size: 0.8rem;          /* was 0.9rem */
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
    }
    
    /* Status cards */
    .status-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
        box-shadow: var(--shadow-light);
        transition: transform 0.3s ease;
    }
    
    .status-card:hover {
        transform: translateX(4px);
    }
    
    .status-underweight {
        background: rgba(241, 196, 15, 0.1);
        border-left-color: #f1c40f;
        color: #f1c40f;
    }
    
    .status-normal {
        background: rgba(0, 210, 211, 0.1);
        border-left-color: var(--success-color);
        color: var(--success-color);
    }
    
    .status-overweight {
        background: rgba(255, 171, 0, 0.1);
        border-left-color: var(--warning-color);
        color: var(--warning-color);
    }
    
    .status-obese {
        background: rgba(255, 107, 107, 0.1);
        border-left-color: var(--error-color);
        color: var(--error-color);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--secondary-bg);
        border-radius: 12px;
        padding: 0.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary);
        background: transparent;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.95rem;
        border: none;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--card-bg);
        color: var(--text-primary);
        transform: translateY(-1px);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-gradient) !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(100, 125, 222, 0.3);
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background: none !important;
    }
    
    /* Tab content styling */
    .tab-content-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: var(--text-primary);
        box-shadow: var(--shadow-light);
        line-height: 1.6;
    }
    
    .tab-content-card ul {
        margin: 0.5rem 0;
        padding-left: 1.2rem;
    }
    
    .tab-content-card li {
        margin-bottom: 0.5rem;
        color: var(--text-secondary);
    }
    
    .tab-content-card b {
        color: var(--accent-color);
    }
    
    /* Meal recommendation cards */
    .meal-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-light);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .meal-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow);
        border-color: var(--accent-color);
    }
    
    .meal-item {
        margin-bottom: 0.75rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border-color);
    }
    
    .meal-item:last-child {
        border-bottom: none;
    }
    
    .meal-label {
        color: var(--accent-color);
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .meal-description {
        color: var(--text-primary);
        font-size: 1rem;
        margin-top: 0.25rem;
    }
    
    .nutrition-info {
        background: var(--secondary-bg);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }
    
    .nutrition-info b {
        color: var(--accent-color);
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 2rem 0;
    }
    
    /* Columns styling */
    .stColumn {
        padding: 0 0.5rem;
    }
    
    /* Subheader styling */
    .stSubheader {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Labels styling */
    .stSelectbox label,
    .stNumberInput label,
    .stSlider label,
    .stMultiSelect label {
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    /* Plotly chart styling */
    .js-plotly-plot {
        background: var(--card-bg);
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-light);
    }
    
    /* Risk level indicators */
    .risk-low {
        background: rgba(0, 210, 211, 0.1);
        border-left: 4px solid var(--success-color);
        color: var(--success-color);
    }
    
    .risk-medium {
        background: rgba(255, 171, 0, 0.1);
        border-left: 4px solid var(--warning-color);
        color: var(--warning-color);
    }
    
    .risk-high {
        background: rgba(255, 107, 107, 0.1);
        border-left: 4px solid var(--error-color);
        color: var(--error-color);
    }
    
    /* Exercise recommendation cards */
    .exercise-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        box-shadow: var(--shadow-light);
        transition: all 0.3s ease;
    }
    
    .exercise-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow);
        border-color: var(--accent-color);
    }
    
    .exercise-name {
        color: var(--accent-color);
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .exercise-stats {
        color: var(--text-secondary);
        font-size: 0.9rem;
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .exercise-stat {
        background: var(--secondary-bg);
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        border: 1px solid var(--border-color);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1.5rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
        
        .stForm {
            padding: 1rem;
        }
        
        .info-box {
            padding: 1rem;
        }
        
        .meal-card,
        .exercise-card,
        .tab-content-card {
            padding: 1rem;
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--secondary-bg);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #5a6fd8;
    }
    
    /* Animation for loading states */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError, .stInfo, .stWarning {
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-light);
    }
    
    .stSuccess {
        background: rgba(0, 210, 211, 0.1);
        border-left: 4px solid var(--success-color);
    }
    
    .stError {
        background: rgba(255, 107, 107, 0.1);
        border-left: 4px solid var(--error-color);
    }
    
    .stInfo {
        background: rgba(100, 125, 222, 0.1);
        border-left: 4px solid var(--accent-color);
    }
    
    .stWarning {
        background: rgba(255, 171, 0, 0.1);
        border-left: 4px solid var(--warning-color);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_csv('datasets/Food_and_Nutrition__.csv')
    df['Disease'] = df['Disease'].str.split(', ')
    data_exploded = df.explode('Disease')
    return df, data_exploded

df, data_exploded = load_data()

def load_health_data():
    if os.path.exists('health_data.csv'):
        return pd.read_csv('datasets/health_data.csv')
    return pd.DataFrame({
        'activity_type': ['Walking', 'Swimming', 'Cycling', 'Weight Training', 'Strength Training', 'Yoga', 'Running'],
        'calories_burned': [200, 350, 300, 250, 270, 150, 400],
        'avg_heart_rate': [100, 120, 110, 105, 108, 90, 130],
        'duration_minutes': [30, 45, 40, 35, 40, 60, 30],
    })

def load_food_data():
    if os.path.exists('food_data.csv'):
        return pd.read_csv('food_data.csv')
    return df

def load_sleep_data():
    if os.path.exists('sleep_data.csv'):
        return pd.read_csv('datasets/sleep_data.csv')
    return pd.DataFrame({'user_id': [1,2,3], 'date': ['2024-06-01']*3, 'sleep_hours': [7, 6.5, 8]})

health_data = load_health_data()
food_data = load_food_data()
sleep_data = load_sleep_data()

activity_map = {
    'Sedentary': 'Sedentary',
    'Lightly Active': 'Lightly Active',
    'Moderately Active': 'Moderately Active',
    'Very Active': 'Very Active',
    'Extremely Active': 'Extremely Active',
}
dietary_map = {
    'Pescatarian': 'Pescatarian',
    'Vegetarian': 'Vegetarian',
    'Vegan': 'Vegan',
    'Omnivore': 'Omnivore',
}

def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

def infer_health_goals(height, weight):
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        return ['Weight Gain']
    elif bmi >= 25:
        return ['Weight Loss']
    else:
        return []

def recommend_meals(user_input):
    filtered = data_exploded.copy()
    if user_input['Activity Level']:
        filtered = filtered[filtered['Activity Level'].str.lower().str.contains(user_input['Activity Level'].lower(), na=False)]
    if user_input['Dietary Preference']:
        filtered = filtered[filtered['Dietary Preference'].str.lower().str.contains(user_input['Dietary Preference'].lower(), na=False)]
    if user_input['Disease']:
        if isinstance(filtered, pd.DataFrame):
            filtered = filtered[filtered['Disease'].isin(user_input['Disease']) | filtered['Disease'].isin([d.lower() for d in user_input['Disease']])]
    if not isinstance(filtered, pd.DataFrame):
        return pd.DataFrame()
    filtered = filtered.drop_duplicates(subset=['Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion', 'Snack Suggestion'])
    return filtered[['Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion', 'Snack Suggestion',
                     'Calories', 'Protein', 'Carbohydrates', 'Fat', 'Fiber', 'Sugar', 'Sodium']].head(3)

def get_exercise_recommendations(user_data):
    bmi_category = user_data.get('bmi_category')
    available_activities = health_data['activity_type'].unique()
    if bmi_category == 'Underweight':
        preferred_activities = ['Weight Training', 'Strength Training', 'Yoga']
    elif bmi_category == 'Obese':
        preferred_activities = ['Walking', 'Swimming', 'Cycling']
    else:
        preferred_activities = available_activities
    filtered_activities = [act for act in preferred_activities if act in available_activities]
    if not filtered_activities:
        filtered_activities = available_activities[:5]
    activity_stats = health_data[health_data['activity_type'].isin(filtered_activities)].groupby('activity_type').agg({
        'calories_burned': 'mean',
        'avg_heart_rate': 'mean',
        'duration_minutes': 'mean'
    }).reset_index()
    recommendations = []
    for _, activity in activity_stats.head(5).iterrows():
        recommendations.append({
            'activity': activity['activity_type'],
            'calories_burned': activity['calories_burned'],
            'heart_rate': activity['avg_heart_rate'],
            'duration': activity['duration_minutes'],
            'effectiveness': activity['calories_burned'] / activity['duration_minutes'] if activity['duration_minutes'] > 0 else 0
        })
    return recommendations

def assess_health_risk(user_data):
    risk_score = (
        (user_data.get('bmi', 24) - 22) ** 2 * 0.3 +
        (7 - user_data.get('sleep_hours', 7)) ** 2 * 0.2 +
        user_data.get('stress_level', 5) * 0.2 +
        (10000 - user_data.get('daily_steps', 8000)) * 0.0001 +
        (2.5 - user_data.get('hydration_level', 2.5)) ** 2 * 0.1
    )
    if risk_score > 50:
        risk_level = "High"
    elif risk_score > 20:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    recommendations = []
    if risk_level in ['High']:
        recommendations.extend([
            "Consider consulting a healthcare provider for a comprehensive health assessment",
            "Focus on stress management techniques like meditation or yoga",
            "Aim for 7-9 hours of quality sleep per night",
            "Increase daily physical activity gradually",
            "Monitor blood pressure regularly"
        ])
    elif risk_level == 'Medium':
        recommendations.extend([
            "Maintain current healthy habits",
            "Consider adding more physical activity to your routine",
            "Focus on stress reduction techniques",
            "Ensure adequate sleep and hydration"
        ])
    else:
        recommendations.extend([
            "Continue maintaining your healthy lifestyle",
            "Regular check-ups are still important",
            "Consider preventive health measures"
        ])
    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'recommendations': recommendations
    }

def show_health_analysis(data_dict=None):
    # Load custom CSS
    load_custom_css()
    
    st.markdown('<h2 class="sub-header">🧬 Health Analysis</h2>', unsafe_allow_html=True)

    # Introduction section
    st.markdown("""
    <div class="info-box">
        <p>Provides a personalized health analysis based on your profile. Enter your details to receive tailored insights on your BMI, exercise, diet, and health risk.</p>
    </div>
    """, unsafe_allow_html=True)
    # Use loaded data if data_dict is not provided
    _health_data = health_data
    _food_data = food_data
    _sleep_data = sleep_data

    with st.form("user_info_health_analysis_main"):
        st.markdown('<h4 class="sub-header">👤 Personal & Health Information</h4>', unsafe_allow_html=True)

        # First row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col3:
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        with col4:
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

        # Second row
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            activity_level = st.selectbox("Activity Level", list(activity_map.keys()))
        with col6:
            fitness_goal = st.selectbox("Fitness Goal", ["Weight Loss", "Weight Gain", "Maintenance", "Muscle Building", "General Health"])
        with col7:
            dietary_preference = st.selectbox("Dietary Preference", list(dietary_map.keys()))
        with col8:
            disease = st.multiselect("Disease/Health Condition", ["None", "Diabetes", "Hypertension", "Heart Disease", "Obesity", "Asthma", "Kidney Disease", "Other"])

        # Third row
        col9, col10, col11, col12 = st.columns(4)
        with col9:
            stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
        with col10:
            sleep_hours = st.slider("Hours of Sleep", min_value=4, max_value=12, value=7)
        with col11:
            daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=8000)
        with col12:
            hydration_level = st.number_input("Hydration Level (L)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)

        # Fourth row
        col13, col14, col15 = st.columns(3)
        with col13:
            resting_heart_rate = st.number_input("Resting Heart Rate", min_value=30, max_value=200, value=70)
        with col14:
            blood_pressure_systolic = st.number_input("Blood Pressure Systolic", min_value=80, max_value=200, value=120)
        with col15:
            blood_pressure_diastolic = st.number_input("Blood Pressure Diastolic", min_value=40, max_value=130, value=80)

        submitted = st.form_submit_button("🔍 Analyze My Health Profile")

    if submitted:
        bmi = calculate_bmi(weight, height)
        bmi_category = get_bmi_category(bmi)
        user_data = {
            'age': age,
            'gender': gender,
            'height': height,
            'weight': weight,
            'activity_level': activity_level,
            'fitness_goal': fitness_goal,
            'dietary_preference': dietary_preference,
            'disease': [d for d in disease if d != "None"],
            'stress_level': stress_level,
            'sleep_hours': sleep_hours,
            'daily_steps': daily_steps,
            'hydration_level': hydration_level,
            'resting_heart_rate': resting_heart_rate,
            'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic,
            'bmi': bmi,
            'bmi_category': bmi_category
        }
        user_input = {
            'Activity Level': activity_map[activity_level],
            'Dietary Preference': dietary_map[dietary_preference],
            'Disease': [d for d in disease if d != "None"]
        }

        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.markdown('<h4 class="sub-header">📈 Your Health Profile Overview</h4>', unsafe_allow_html=True)
        
        # Enhanced metrics display
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">BMI</div>
                <div class="metric-value">{bmi:.1f}</div>
                <div style="color: {'#00d2d3' if bmi_category == 'Normal' else '#ffab00' if bmi_category in ['Underweight', 'Overweight'] else '#ff6b6b'};">
                    {bmi_category}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with metric_cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Age</div>
                <div class="metric-value">{age}</div>
                <div style="color: var(--text-secondary);">years</div>
            </div>
            """, unsafe_allow_html=True)
            
        with metric_cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Weight</div>
                <div class="metric-value">{weight}</div>
                <div style="color: var(--text-secondary);">kg</div>
            </div>
            """, unsafe_allow_html=True)
            
        with metric_cols[3]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Height</div>
                <div class="metric-value">{height}</div>
                <div style="color: var(--text-secondary);">cm</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Enhanced status cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h5 style="color: var(--text-primary); margin-bottom: 1rem;">📊 BMI Analysis</h5>', unsafe_allow_html=True)
            bmi_status_class = {
                "Underweight": "status-underweight",
                "Normal": "status-normal", 
                "Overweight": "status-overweight",
                "Obese": "status-obese"
            }[bmi_category]
            
            bmi_messages = {
                "Underweight": "You are underweight. Consider increasing caloric intake and incorporating strength training to build healthy muscle mass.",
                "Normal": "Excellent! Your BMI is in the healthy range. Continue maintaining your current lifestyle and regular exercise routine.",
                "Overweight": "You are overweight. Consider implementing a balanced diet and regular exercise routine to achieve a healthier weight.",
                "Obese": "You are in the obese category. We recommend consulting with a healthcare provider for personalized guidance and support."
            }
            
            st.markdown(f"""
            <div class="status-card {bmi_status_class}">
                <strong>BMI Status: {bmi_category}</strong><br>
                {bmi_messages[bmi_category]}
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown('<h5 style="color: var(--text-primary); margin-bottom: 1rem;">💤 Sleep Analysis</h5>', unsafe_allow_html=True)
            if sleep_hours < 6:
                sleep_class = "status-card" + " " + "status-obese"
                sleep_message = "⚠️ Insufficient sleep detected. Aim for 7-9 hours nightly for optimal health, immune function, and mental clarity."
            elif sleep_hours <= 9:
                sleep_class = "status-card" + " " + "status-normal"
                sleep_message = "✅ Excellent sleep duration! You're getting the recommended amount of rest for optimal health and recovery."
            else:
                sleep_class = "status-card" + " " + "status-overweight"
                sleep_message = "💤 Good sleep duration, though consider if this amount works best for your energy levels and daily routine."
                
            st.markdown(f"""
            <div class="{sleep_class}">
                <strong>Sleep Status: {sleep_hours} hours</strong><br>
                {sleep_message}
            </div>
            """, unsafe_allow_html=True)

        # Enhanced BMI Gauge Chart
        st.markdown('<h5 style="color: var(--text-primary); margin: 2rem 0 1rem 0;">📈 BMI Visualization</h5>', unsafe_allow_html=True)
        
        bmi_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=bmi,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Body Mass Index (BMI)", 'font': {'size': 24, 'color': '#ffffff'}},
            delta={'reference': 22, 'increasing': {'color': "#ff6b6b"}, 'decreasing': {'color': "#00d2d3"}},
            gauge={
                'axis': {'range': [10, 40], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                'bar': {'color': "#647dee", 'thickness': 0.3},
                'bgcolor': "#1a1f2e",
                'borderwidth': 2,
                'bordercolor': "#2d3142",
                'steps': [
                    {'range': [10, 18.5], 'color': "rgba(241, 196, 15, 0.3)"},
                    {'range': [18.5, 25], 'color': "rgba(0, 210, 211, 0.3)"},
                    {'range': [25, 30], 'color': "rgba(255, 171, 0, 0.3)"},
                    {'range': [30, 40], 'color': "rgba(255, 107, 107, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': bmi
                }
            }
        ))
        
        bmi_gauge.update_layout(
            height=300,
            paper_bgcolor='#23242b',
            plot_bgcolor='#23242b',
            font={'color': '#ffffff', 'family': 'Inter'}
        )
        
        st.plotly_chart(bmi_gauge, use_container_width=True)

        # Enhanced Tabbed Results
        st.markdown('<h5 class="sub-header">📋 Detailed Health Recommendations</h5>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs([
            "🍽️ Nutrition & Meal Plans",
            "🏃‍♀️ Exercise Recommendations", 
            "⚕️ Health Risk Assessment"
        ])

        with tab1:
            try:
                meal_df = recommend_meals(user_input)
                if not meal_df.empty:
                    st.markdown('<h4 class="sub-header">Top 3 Meal Recommendation</h4>', unsafe_allow_html=True)
                    for i, (_, row) in enumerate(meal_df.iterrows(), 1):
                        st.markdown(f'<h4 style="color: var(--accent-color); margin-bottom: 1.5rem;">Meal Recommendation {i}</h4>', unsafe_allow_html=True)
                    
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            <div class="meal-item">
                                <div class="meal-label">🌅 BREAKFAST</div>
                                <div class="meal-description">{row['Breakfast Suggestion']}</div>
                            </div>
                            <div class="meal-item">
                                <div class="meal-label">🌞 LUNCH</div>
                                <div class="meal-description">{row['Lunch Suggestion']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class="meal-item">
                                <div class="meal-label">🌙 DINNER</div>
                                <div class="meal-description">{row['Dinner Suggestion']}</div>
                            </div>
                            <div class="meal-item">
                                <div class="meal-label">🥜 SNACK</div>
                                <div class="meal-description">{row['Snack Suggestion']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        # Place the nutrition info below the columns, still inside the card
                        st.markdown(f"""
                        <div class="nutrition-info">
                            <b>Nutritional Profile:</b> 
                            <b>Calories:</b> {row['Calories']} kcal | 
                            <b>Protein:</b> {row['Protein']}g | 
                            <b>Carbs:</b> {row['Carbohydrates']}g | 
                            <b>Fat:</b> {row['Fat']}g | 
                            <b>Fiber:</b> {row['Fiber']}g | 
                            <b>Sugar:</b> {row['Sugar']}g | 
                            <b>Sodium:</b> {row['Sodium']}mg
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="tab-content-card">
                        <h4>🔍 No specific meal recommendations found</h4>
                        <p>Based on your current profile, we couldn't find specific meal recommendations. Consider:</p>
                        <ul>
                            <li>Consulting with a registered dietitian</li>
                            <li>Exploring general healthy eating guidelines</li>
                            <li>Adjusting your dietary preferences or activity level</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="tab-content-card">
                    <h4>⚠️ Error generating meal recommendations</h4>
                    <p>We encountered an issue: {str(e)}</p>
                    <p>Please try again or contact support if the problem persists.</p>
                </div>
                """, unsafe_allow_html=True)
                
        with tab2:
            try:
                exercise_recs = get_exercise_recommendations(user_data)
                if exercise_recs:
                    st.markdown('<h4 style="color: var(--accent-color); margin-bottom: 1.5rem;">🏃‍♀️ Tailored Exercise Program</h4>', unsafe_allow_html=True)
                    
                    for rec in exercise_recs:
                        effectiveness_score = min(rec['effectiveness'] * 10, 10)  # Scale to 10
                        effectiveness_color = "#00d2d3" if effectiveness_score >= 7 else "#ffab00" if effectiveness_score >= 5 else "#ff6b6b"
                        
                        st.markdown(f"""
                        <div class="exercise-card">
                            <div class="exercise-name">{rec['activity']}</div>
                            <div class="exercise-stats">
                                <div class="exercise-stat">⏱️ {rec['duration']:.0f} min</div>
                                <div class="exercise-stat">🔥 {rec['calories_burned']:.0f} kcal</div>
                                <div class="exercise-stat">❤️ {rec['heart_rate']:.0f} bpm</div>
                                <div class="exercise-stat" style="color: {effectiveness_color};">
                                    📈 Effectiveness: {effectiveness_score:.1f}/10
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="tab-content-card">
                        <h4>🔍 No exercise recommendations available</h4>
                        <p>We couldn't generate specific exercise recommendations. Consider general activities like:</p>
                        <ul>
                            <li>30 minutes of brisk walking daily</li>
                            <li>Swimming or water aerobics</li>
                            <li>Bodyweight exercises (push-ups, squats)</li>
                            <li>Yoga or stretching routines</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="tab-content-card">
                    <h4>⚠️ Error generating exercise recommendations</h4>
                    <p>We encountered an issue: {str(e)}</p>
                    <p>Please try again or contact support if the problem persists.</p>
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            try:
                risk = assess_health_risk(user_data)
                tab3_content = f"""
                <b>Health Risk Level:</b> {risk['risk_level']} (Score: {risk['risk_score']:.2f})<br>
                <b>Recommendations:</b>
                <ul style='margin-top:0.7em; padding-left:1.2em;'>
                {''.join([f'<li style="margin-bottom:0.5em;">{r}</li>' for r in risk['recommendations']])}
                </ul>
                """
                st.markdown(f'<div class="tab-content-card">{tab3_content}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="tab-content-card">Error assessing health risk: {e}</div>', unsafe_allow_html=True)


                
            except Exception as e:
                st.markdown(f"""
                <div class="tab-content-card">
                    <h4>⚠️ Error assessing health risk</h4>
                    <p>We encountered an issue: {str(e)}</p>
                    <p>Please try again or contact support if the problem persists.</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # Close fade-in div
        
        # Additional Health Insights
        st.markdown('<h2 class="sub-header">💡 Additional Health Insights</h2>', unsafe_allow_html=True)
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("""
            <div class="tab-content-card">
                <h4>🎯 Goal Achievement Tips</h4>
                <ul>
                    <li>Set realistic, measurable goals</li>
                    <li>Track your progress regularly</li>
                    <li>Stay consistent with healthy habits</li>
                    <li>Celebrate small victories</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with insights_col2:
            st.markdown("""
            <div class="tab-content-card">
                <h4>📱 Health Monitoring</h4>
                <ul>
                    <li>Regular check-ups with healthcare providers</li>
                    <li>Monitor vital signs at home</li>
                    <li>Keep a health journal</li>
                    <li>Use fitness tracking apps</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    show_health_analysis()