import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import warnings

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
            'doctor_consult': doctor_consult_df,
            
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
