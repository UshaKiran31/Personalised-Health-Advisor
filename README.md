# 🏥 Personalised Health Advisor

A comprehensive AI-powered health advisory application built with Streamlit that provides disease prediction, health risk assessment, and medical information based on symptoms and health parameters.

## 🌟 Features

### 🔍 **Symptom Checker**
- Select from 607+ symptoms
- AI-powered disease prediction using Naive Bayes model
- Detailed disease information including:
  - Disease descriptions
  - Recommended medications
  - Dietary recommendations
  - Precautions and preventive measures
- Confidence scoring with visual indicators

### 📈 **Health Analysis**
- Comprehensive health analysis based on user parameters
- Calculates BMI and categorizes health status
- Provides personalized meal and exercise recommendations
- Assesses overall health risk and suggests improvements
- Visualizes health metrics with interactive charts

### ❤️ **Heart Health Assessment**
- Comprehensive heart disease risk evaluation
- 14 medical parameters analysis
- Visual risk assessment with gauge charts
- Personalized recommendations based on risk level

### 🩸 **Diabetes Risk Calculator**
- Multi-parameter diabetes risk assessment
- BMI, glucose, blood pressure analysis
- Risk stratification with actionable recommendations

### 📊 **Health Analytics Dashboard**
- Disease category distribution analysis
- Symptom severity analysis
- Training data statistics
- Interactive visualizations

### 💊 **Disease Information Center**
- Comprehensive database of 287+ diseases
- Detailed medical information
- Treatment and prevention guidelines

### 📈 **Data Insights**
- Dataset overview and statistics
- Disease-symptom relationship analysis
- Heart disease and diabetes dataset insights
- Interactive charts and visualizations

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd VHA
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

## 📁 Project Structure

```
VHA/
├── app.py                          # Main Streamlit application
├── health_analysis.py              # Health analysis module: BMI, risk, recommendations
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore rules
├── models/                        # Trained machine learning models
│   ├── NaiveBayes.pkl            # Naive Bayes classifier model
│   └── label_encoder.pkl         # Label encoder for disease mapping
└── datasets/                      # Health datasets
    ├── symptoms/                  # Symptom and disease datasets
    │   ├── unique_diseases.csv   # List of unique diseases
    │   ├── unique_symptoms.csv   # List of unique symptoms
    │   ├── description.csv       # Disease descriptions
    │   ├── medications.csv       # Disease medications
    │   ├── diets.csv            # Dietary recommendations
    │   ├── precautions_df.csv   # Disease precautions
    │   ├── dis_symp_dict.txt    # Disease-symptom dictionary
    │   ├── Main_Dataset.csv     # Main training dataset
    │   ├── Original_Dataset.csv # Original dataset
    │   ├── refined data/        # Processed training data
    │   │   ├── Train Data.csv   # Training dataset
    │   │   └── Symptom-severity.csv # Symptom severity data
    │   └── workout_df.csv       # Exercise recommendations
    ├── heart/                    # Heart disease dataset
    │   └── heart.csv            # Heart disease risk data
    └── diabetes/                 # Diabetes dataset
        └── diabetes.csv         # Diabetes risk data
```

## 🎯 How to Use

### 1. **Dashboard Overview**
- Start at the main dashboard to see key metrics and quick access to all features
- Navigate using the sidebar menu

### 2. **Symptom Checker**
- Select symptoms from the comprehensive list
- Click "Analyze Symptoms" to get AI predictions
- Review detailed disease information and recommendations
- Check confidence scores and disclaimers

### 3. **Health Risk Assessments**
- **Heart Health**: Enter age, sex, blood pressure, cholesterol, and other parameters
- **Diabetes Risk**: Input glucose levels, BMI, blood pressure, and other health indicators
- Get personalized risk assessments with visual indicators

### 4. **Analytics & Insights**
- Explore disease distributions and symptom severity
- View training data statistics
- Analyze heart disease and diabetes dataset patterns

## 🔧 Technical Details

### **Health Analysis Module**
- **health_analysis.py**: Provides BMI calculation, health risk assessment, and personalized recommendations for nutrition and exercise. Integrates interactive visualizations for user health metrics.

### **AI Model**
- **Algorithm**: Naive Bayes Classifier
- **Training Data**: 4,922 records with 607 symptom features
- **Disease Coverage**: 287 unique diseases
- **Accuracy**: Estimated ~85%

### **Datasets**
- **Symptoms-Disease Dataset**: Primary dataset for disease prediction
- **Heart Disease Dataset**: 14 parameters for cardiovascular risk assessment
- **Diabetes Dataset**: 8 parameters for diabetes risk evaluation

### **Technologies Used**
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Machine Learning**: Scikit-learn
- **Model Persistence**: Pickle

## ⚠️ Important Disclaimers

### **Medical Disclaimer**
- This application is for **educational and informational purposes only**
- **NOT a substitute for professional medical advice, diagnosis, or treatment**
- Always consult qualified healthcare professionals for medical decisions
- The AI predictions are estimates and should not be used for self-diagnosis

### **Data Limitations**
- The model is trained on specific datasets and may not cover all medical conditions
- Individual health factors may vary significantly
- Regular medical check-ups are essential for proper health monitoring



**Remember**: This tool is designed to complement, not replace, professional medical care. Always consult healthcare professionals for medical decisions. 