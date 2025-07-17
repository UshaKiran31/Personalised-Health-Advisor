import streamlit as st
import numpy as np
import pandas as pd
from disease_info import get_disease_info

# Load symptom descriptions from CSV
@st.cache_data
def load_symptom_descriptions():
    df = pd.read_csv('datasets/symptoms/symptom_descriptions.csv')
    # Ensure 'Symptom' is a pandas Series for string operations
    symptoms_series = pd.Series(df['Symptom'])
    symptoms_norm = symptoms_series.astype(str).str.lower().str.replace(r'[_ ]', '', regex=True)
    df['norm_symptom'] = symptoms_norm
    df = df.drop_duplicates(subset='norm_symptom')
    return df[['Symptom', 'Description']]

symptom_desc_df = load_symptom_descriptions()

# Build a mapping for display and description
symptom_display_dict = {str(row['Symptom']): str(row['Symptom']).replace('_', ' ').title() for _, row in symptom_desc_df.iterrows()}
symptom_description_dict = {str(row['Symptom']): row['Description'] for _, row in symptom_desc_df.iterrows()}

# Categorize symptoms (using the same logic as before, but only for these symptoms)
def categorize_symptoms(symptom_list):
    categories = {
        "General Symptoms": [],
        "Pain & Discomfort": [],
        "Digestive": [],
        "Respiratory": [],
        "Neurological": [],
        "Skin & External": [],
        # New subcategories for 'Other'
        "Urinary": [],
        "Eye & Vision": [],
        "Mental Health": [],
        "Reproductive": [],
        "Musculoskeletal": [],
        "Miscellaneous": []
    }
    for s in symptom_list:
        sl = s.lower()
        if any(word in sl for word in ['fever', 'fatigue', 'weakness', 'weight', 'appetite']):
            categories["General Symptoms"].append(s)
        elif any(word in sl for word in ['pain', 'ache', 'cramp', 'burning', 'stiff']):
            categories["Pain & Discomfort"].append(s)
        elif any(word in sl for word in ['nausea', 'vomit', 'diarrhea', 'constipation', 'stomach', 'abdominal']):
            categories["Digestive"].append(s)
        elif any(word in sl for word in ['cough', 'breathe', 'chest', 'throat', 'sinus']):
            categories["Respiratory"].append(s)
        elif any(word in sl for word in ['headache', 'dizziness', 'confusion', 'memory', 'seizure']):
            categories["Neurological"].append(s)
        elif any(word in sl for word in ['rash', 'skin', 'itch', 'swelling', 'bruising']):
            categories["Skin & External"].append(s)
        elif any(word in sl for word in ['urine', 'urination', 'bladder', 'kidney', 'renal']):
            categories["Urinary"].append(s)
        elif any(word in sl for word in ['eye', 'vision', 'sight', 'blur', 'pupil', 'retina', 'cornea']):
            categories["Eye & Vision"].append(s)
        elif any(word in sl for word in ['anxiety', 'depression', 'mood', 'mental', 'psychosis', 'delusion', 'hallucination', 'memory', 'confusion', 'cognitive', 'behavior', 'irritability']):
            categories["Mental Health"].append(s)
        elif any(word in sl for word in ['menstruation', 'period', 'pregnant', 'vaginal', 'uterine', 'breast', 'testicular', 'fertility', 'reproductive', 'nipple', 'genital', 'penis']):
            categories["Reproductive"].append(s)
        elif any(word in sl for word in ['muscle', 'joint', 'bone', 'spasm', 'cramp', 'limb', 'back', 'neck', 'arm', 'leg', 'hip', 'knee', 'wrist', 'elbow', 'shoulder']):
            categories["Musculoskeletal"].append(s)
        else:
            categories["Miscellaneous"].append(s)
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def show_symptom_checker(data_dict):
    """Show enhanced symptom checker interface with modern UI"""
    st.markdown('<h2 class="sub-header">🔍 Symptom Checker</h2>', unsafe_allow_html=True)
    
    # Introduction section
    st.markdown("""
    <div class="info-box">
        <p>Our AI-powered diagnostic tool analyzes your symptoms along with personal health factors to provide accurate health insights. 
        Please provide your information below for a comprehensive assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Symptom Selection Section
    st.markdown("### Symptom Selection")
    
    # Only use symptoms from the CSV
    all_symptoms = symptom_desc_df['Symptom'].tolist()
    # Build categories
    symptom_categories = categorize_symptoms(all_symptoms)

    selected_symptoms = []
    # Option 1: Quick symptom search
    search_term = st.text_input("Search for symptoms...", placeholder="Type to search symptoms")
    if search_term:
        filtered_symptoms = [s for s in all_symptoms if search_term.lower() in s.lower()]
        if filtered_symptoms:
            # Create search results with tooltips
            for symptom in filtered_symptoms:
                description = symptom_description_dict[symptom]
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

    st.markdown("#### 📋 Browse by Category")
    # Tabs for symptom categories
    tabs = st.tabs(list(symptom_categories.keys()))
    for i, (category, symptoms) in enumerate(symptom_categories.items()):
        with tabs[i]:
            if symptoms:
                cols = st.columns(2)
                for j, symptom in enumerate(symptoms):
                    with cols[j % 2]:
                        description = symptom_description_dict[symptom]
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
            description = symptom_description_dict[symptom]
            with cols[i % 4]:
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
