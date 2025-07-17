import streamlit as st

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
                    st.markdown("""<h2 style="margin: 0; color: #3498db; font-size: 1.5rem;"> Description </h2>""", unsafe_allow_html=True)
                    st.write(disease_info['description'])
                
                if 'medications' in disease_info:
                    st.markdown("""<h2 style="margin: 0; color: #3498db; font-size: 1.5rem;"> Medications </h2>""", unsafe_allow_html=True)
                    st.write(disease_info['medications'])
                
                if 'diet' in disease_info:
                    st.markdown("""<h2 style="margin: 0; color: #3498db; font-size: 1.5rem;"> Dietary Recommendations </h2>""", unsafe_allow_html=True)
                    st.write(disease_info['diet'])
                
                if 'precautions' in disease_info:
                    st.markdown("""<h2 style="margin: 0; color: #3498db; font-size: 1.5rem;"> Precautions </h2>""", unsafe_allow_html=True)  
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
                st.markdown("### 🧑‍⚕️ Recommended Specialist")
                st.markdown(f"""
                <div style="background: #1B2631; 
                           color: white; padding: 1rem; border-radius: 8px; 
                           border-left: 4px solid #3498db; margin: 1rem 0;">
                    <h4 style="margin: 0; color: #3498db; font-size: 1rem;">{disease_info['doctor']}</h4>
                </div>
                """, unsafe_allow_html=True)

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
