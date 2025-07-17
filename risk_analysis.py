import streamlit as st

def show_risk_analysis(data_dict):
    """Show combined risk analysis for Heart Health and Diabetes Risk with tabs."""
    st.markdown('<h2 class="sub-header">🩺 Risk Analysis</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p>Assess your risk for heart disease and diabetes using the calculators below. Enter your health parameters and view your results in the respective tabs.</p>
    </div>
    """, unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["❤️ Heart Health", "🩸 Diabetes Risk"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 20, 100, 50, key="heart_age")
            sex = st.selectbox("Sex", ["Male", "Female"], key="heart_sex")
            cp = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"], key="heart_cp")
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120, key="heart_trestbps")
            chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200, key="heart_chol")
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], key="heart_fbs")
            restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"], key="heart_restecg")
            thalach = st.slider("Maximum Heart Rate", 70, 202, 150, key="heart_thalach")
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"], key="heart_exang")
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1, key="heart_oldpeak")
        if st.button("❤️ Assess Heart Health", type="primary", key="heart_btn"):
            risk_score = 0
            if age > 65:
                risk_score += 2
            elif age > 45:
                risk_score += 1
            if sex == "Male":
                risk_score += 1
            if trestbps > 140:
                risk_score += 2
            elif trestbps > 120:
                risk_score += 1
            if chol > 300:
                risk_score += 2
            elif chol > 200:
                risk_score += 1
            if fbs == "Yes":
                risk_score += 1
            if exang == "Yes":
                risk_score += 2
            max_risk = 10
            risk_percentage = (risk_score / max_risk) * 100
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
                import plotly.graph_objects as go
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
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.slider("Number of Pregnancies", 0, 20, 1, key="diab_preg")
            glucose = st.slider("Glucose Level", 50, 300, 120, key="diab_glucose")
            bp = st.slider("Blood Pressure (mm Hg)", 40, 140, 70, key="diab_bp")
            skin = st.slider("Skin Thickness (mm)", 7, 99, 20, key="diab_skin")
            insulin = st.slider("Insulin Level (mu U/ml)", 15, 846, 80, key="diab_insulin")
        with col2:
            bmi = st.slider("BMI", 15, 60, 25, key="diab_bmi")
            dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01, key="diab_dpf")
            age = st.slider("Age", 18, 100, 35, key="diab_age")
        if st.button("🩸 Assess Diabetes Risk", type="primary", key="diab_btn"):
            risk_score = 0
            if glucose > 140:
                risk_score += 2
            elif glucose > 110:
                risk_score += 1
            if bmi > 30:
                risk_score += 2
            elif bmi > 25:
                risk_score += 1
            if age > 60:
                risk_score += 2
            elif age > 45:
                risk_score += 1
            if pregnancies > 5:
                risk_score += 1
            if bp > 90:
                risk_score += 1
            max_risk = 8
            risk_percentage = (risk_score / max_risk) * 100
            st.markdown("### 📊 Diabetes Risk Assessment Results")
            col1, col2 = st.columns(2)
            with col1:
                if risk_percentage < 30:
                    st.success(f"🟢 Low Risk: {risk_percentage:.1f}%")
                    st.write("Your diabetes risk is low. Continue maintaining a healthy lifestyle!")
                elif risk_percentage < 60:
                    st.warning(f"🟡 Moderate Risk: {risk_percentage:.1f}%")
                    st.write("You have moderate risk factors. Consider lifestyle changes and regular check-ups.")
                else:
                    st.error(f"🔴 High Risk: {risk_percentage:.1f}%")
                    st.write("You have high risk factors. Please consult a healthcare professional.")
            with col2:
                import plotly.graph_objects as go
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
            st.markdown("### 💡 Recommendations")
            if risk_percentage < 30:
                st.markdown("""
                - Maintain a healthy weight
                - Eat a balanced, low-sugar diet
                - Exercise regularly
                - Get regular blood sugar checks
                """)
            elif risk_percentage < 60:
                st.markdown("""
                - Monitor blood sugar more frequently
                - Reduce intake of refined carbs and sugars
                - Increase fiber intake
                - Consult a dietitian if needed
                - Schedule regular medical check-ups
                """)
            else:
                st.markdown("""
                - **Immediate medical consultation recommended**
                - Strict dietary modifications
                - Medication compliance if prescribed
                - Frequent monitoring of blood sugar
                - Lifestyle changes under medical supervision
                """)
    
    st.markdown("""
    <style>
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #1a1f2e;
        border-radius: 12px;
        padding: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #2d3142;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #b3b8c5;
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
        background: #23242b;
        color: #ffffff;
        transform: translateY(-1px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7f53ac 0%, #647dee 100%); !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(100, 125, 222, 0.3);
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background: none !important;
    }
    
    /* Tab content styling */
    .tab-content-card {
        background: #23242b;
        border: 1px solid #2d3142;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #ffffff;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        line-height: 1.6;
    }
    
    .tab-content-card ul {
        margin: 0.5rem 0;
        padding-left: 1.2rem;
    }
    
    .tab-content-card li {
        margin-bottom: 0.5rem;
        color: #b3b8c5;
    }
    
    .tab-content-card b {
        color: #647de;
    }
    </style>
     """,unsafe_allow_html=True)
