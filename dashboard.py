import streamlit as st

def show_dashboard(data_dict):
    """Show main dashboard with dark theme styling"""
    
    # Custom CSS for dark theme
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        color: #B3E5FC;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .section-header {
        color: #81C784;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .info-box {
        background-color: #1B2631;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        color: #EAECEE;
    }
    
    .intro-box p {
        color: #E0F2FE;
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 0;
        text-align: center;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #262626 0%, #1F2937 100%);
        padding: 1.8rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        border: 1px solid #374151;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.5);
        border-color: #60A5FA;
    }
    
    .feature-card h4 {
        color: #60A5FA;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .feature-card p {
        color: #D1D5DB;
        font-size: 1rem;
        line-height: 1.5;
        margin: 0;
    }
    
    .image-container {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    
    .hero-image {
        max-width: 900px;
        max-height: 320px;
        width: 100%;
        object-fit: cover;
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border: 3px solid #3B82F6;
    }
    
    /* Custom scrollbar for dark theme */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1F2937;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4B5563;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6B7280;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">Welcome to Your Personal Health Advisor</h1>', unsafe_allow_html=True)
    
    # Introduction section
    st.markdown("""
    <div class="intro-box">
        <p>The Personalized Health Advice App is a virtual assistant that provides tailored health recommendations using user inputs and a trained machine learning model. It helps users assess health risks and receive lifestyle and medical suggestions based on their profile.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # # Hero image
    # st.markdown("""
    # <div class="image-container">
    #     <img src="https://images.pexels.com/photos/40568/medical-appointment-doctor-healthcare-40568.jpeg?auto=compress&cs=tinysrgb&w=800"
    #          alt="Medical Appointment"
    #          class="hero-image">
    # </div>
    # """, unsafe_allow_html=True)
    
    # Quick access features section
    st.markdown('<h3 class="section-header">🚀 Quick Access Features</h3>', unsafe_allow_html=True)
    
    # Create two columns for features
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🧬 Health Analysis</h4>
            <p>Provides a personalized health analysis based on your profile. Enter your details to receive tailored insights on your BMI, exercise, diet, and health risk assessment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>💊 Disease Information</h4>
            <p>Browse comprehensive information about diseases, including detailed descriptions, medications, dietary recommendations, and important precautions to follow.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🔍 Symptom Checker</h4>
            <p>Select your symptoms and get instant disease predictions with detailed information about treatments, medications, and precautions for better health management.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🩺 Risk Analysis</h4>
            <p>Assess your risk for heart disease and diabetes using advanced calculators. Enter your health parameters and view comprehensive results in the respective tabs.</p>
        </div>
        """, unsafe_allow_html=True)