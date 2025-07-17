import streamlit as st

def show_about_us(data_dict):
    """Show About Us page in a simple, modern, and clear format."""
    st.markdown('<h2 class="sub-header">About Us</h2>', unsafe_allow_html=True)
    
    # Key metrics at the top
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
    
    st.markdown("""
    ##  Application Overview
    This application is designed to empower individuals with personalized health insights using artificial intelligence and machine learning. It provides users with data-driven recommendations, risk assessments, and educational resources to support better health decisions.
    
    ---
    
    ##  How to Use This App
    
    ### 🔍 Symptom Checker
    - Go to the **Symptom Checker** page from the sidebar.
    - Enter your personal details (age, gender, medical history, etc.).
    - Select your symptoms from the categorized list or search bar.
    - Click **Analyze My Symptoms** to receive AI-based disease predictions.
    - View the predicted condition, confidence score, and detailed recommendations.
    
    ### 🧬 Health Analysis
    - Navigate to the **Health Analysis** page.
    - Enter your health and lifestyle information (age, height, weight, activity, sleep, etc.).
    - Submit to receive a comprehensive health profile, including BMI, exercise, and diet recommendations.
    
    ### 🧎 Risk Analysis
    - Go to the **Risk Analysis** page.
    - Enter the required health parameters for heart disease or diabetes risk.
    - View your risk assessment and personalized advice.
    
    ### 💊 Disease Information
    - Select the **Disease Information** page.
    - Choose a disease from the dropdown to view its description, medications, diet, precautions, and recommended specialist.
    
    ### 🤖 AI Chatbot
    - Open the **AI Chatbot** page.
    - Type your health-related question in the chat box.
    - Receive instant, AI-generated responses for general health queries.
    
    ---
    
    ## 💻 Technical Details
    This application integrates several machine learning and data analysis tools:
    - **Disease Prediction:** Naive Bayes classifier trained on symptom-disease datasets.
    - **Health & Risk Analysis:** Data-driven calculators for BMI, heart disease, and diabetes risk.
    - **Diet & Exercise Recommendations:** Analysis based on user profile and curated datasets.
    - **Conversational AI:** Integrated chatbot for health education and support.
    - **Visualization:** Interactive charts and metrics powered by Plotly and Streamlit.
    
    ---
    
    ## 👥 Team Members
    - **USHA KIRAN PARUCHURI**  
      📧 22BQ1A42B1@vvit.net
    - **JAKKA CHARISHMA**  
      📧 22BQ1A4262@vvit.net
    - **KAKUMANU RAVI CHANDRA**  
      📧 22BQ1A4267@vvit.net
    - **MANDADAPU PRABHAS**  
      📧 22BQ1A4290@vvit.net
    
    ---
    
    **Thank You**  
    """)
