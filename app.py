import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from file_loader import load_data
from dashboard import show_dashboard
from health_analysis import show_health_analysis
from symptom_checker import show_symptom_checker, predict_disease, format_symptom_key, get_symptom_description
from disease_info import show_disease_info_page, get_disease_info
from risk_analysis import show_risk_analysis
from about import show_about_us
from health_assistant import show_ai_chatbot


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

def main():
    # Move the main header to the sidebar
    st.sidebar.markdown('<h1 class="main-header" style="text-align:left; font-size:2.2rem; margin-bottom:1rem;">🏥 Personalised Health Advisor</h1>', unsafe_allow_html=True)
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard"
    if st.sidebar.button("🏠 Dashboard", use_container_width=True):
        st.session_state.current_page = "🏠 Dashboard"
    if st.sidebar.button("🧬 Health Analysis", use_container_width=True):
        st.session_state.current_page = "🧬 Health Analysis"
    if st.sidebar.button("🔍 Symptom Checker", use_container_width=True):
        st.session_state.current_page = "🔍 Symptom Checker"
    if st.sidebar.button("💊 Disease Information", use_container_width=True):
        st.session_state.current_page = "💊 Disease Information"
    if st.sidebar.button("🩺 Risk Analysis", use_container_width=True):
        st.session_state.current_page = "🩺 Risk Analysis"
    if st.sidebar.button("🤖 AI Chatbot", use_container_width=True):
        st.session_state.current_page = "🤖 AI Chatbot"
    if st.sidebar.button("ℹ️ About Us", use_container_width=True):
        st.session_state.current_page = "ℹ️ About Us"
    st.sidebar.markdown("""
    <div style='background-color:#2E2913; border-left:4px solid #ffc107; padding:1rem; margin:1.5rem 0 0 0; border-radius:5px; color:#EAECEE; font-size:0.95rem;'>
        <strong>Disclaimer:</strong> This tool is for educational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)
    data_dict = load_data()
    if data_dict is None:
        st.error("Failed to load data. Please check your data files.")
        return
    page = st.session_state.current_page
    if page == "🏠 Dashboard":
        show_dashboard(data_dict)
    elif page == "🧬 Health Analysis":
        show_health_analysis(data_dict)
    elif page == "🔍 Symptom Checker":
        show_symptom_checker(data_dict)
    elif page == "💊 Disease Information":
        show_disease_info_page(data_dict)
    elif page == "🩺 Risk Analysis":
        show_risk_analysis(data_dict)
    elif page == "ℹ️ About Us":
        show_about_us(data_dict)
    elif page == "🤖 AI Chatbot":
        show_ai_chatbot()

if __name__ == "__main__":
    main() 