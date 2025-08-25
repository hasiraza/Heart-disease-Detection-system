import streamlit as st
import pandas as pd
import pickle
import PyPDF2
import io
import re
from sklearn.preprocessing import StandardScaler


# Load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    """Load the model and scaler with caching"""
    try:
        with open("Model/Heart disease detection model.pkl", "rb") as file:
            model = pickle.load(file)

        # Try to load a saved scaler first
        try:
            with open("Model/scaler.pkl", "rb") as file:
                scaler = pickle.load(file)
        except FileNotFoundError:
            # If no saved scaler, create and fit a new one with typical ranges
            # This is a fallback - ideally you should save your fitted scaler during training
            scaler = StandardScaler()
            # Fit with approximate ranges based on typical medical data
            # You should replace this with your actual training data statistics
            dummy_data = pd.DataFrame({
                'male': [0, 1, 0, 1],
                'age': [30, 70, 45, 55],
                'currentSmoker': [0, 1, 0, 1],
                'cigsPerDay': [0, 20, 5, 10],
                'BPMeds': [0, 1, 0, 1],
                'prevalentStroke': [0, 1, 0, 0],
                'prevalentHyp': [0, 1, 0, 1],
                'diabetes': [0, 1, 0, 0],
                'totChol': [150, 300, 200, 250],
                'sysBP': [100, 180, 120, 140],
                'diaBP': [60, 110, 80, 90],
                'BMI': [18, 40, 25, 30],
                'heartRate': [50, 120, 70, 80],
                'glucose': [70, 200, 100, 120]
            })
            scaler.fit(dummy_data)
            st.warning("Using default scaler parameters. For better accuracy, save your trained scaler.")

        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


model, scaler = load_model_and_scaler()


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


def extract_text_from_txt(txt_file):
    """Extract text from uploaded TXT file"""
    try:
        text = txt_file.read().decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error reading TXT file: {str(e)}")
        return ""


def extract_medical_data(text):
    """Extract medical data from text using regex patterns"""
    extracted_data = {}

    # Define patterns for extracting medical data
    patterns = {
        'age': r'age[:\s]*(\d+)',
        'BMI': r'BMI[:\s]*(\d+\.?\d*)',
        'heartRate': r'heart\s*rate[:\s]*(\d+)',
        'glucose': r'glucose[:\s]*(\d+\.?\d*)',
        'totChol': r'cholesterol[:\s]*(\d+\.?\d*)',
        'sysBP': r'systolic[:\s]*(\d+\.?\d*)',
        'diaBP': r'diastolic[:\s]*(\d+\.?\d*)',
        'cigsPerDay': r'cigarettes[:\s]*(\d+\.?\d*)',
        'BPMeds': r'BP\s*medications?[:\s]*(\d+\.?\d*)'
    }

    # Boolean patterns
    bool_patterns = {
        'currentSmoker': r'smoker[:\s]*(yes|no|true|false)',
        'prevalentStroke': r'stroke[:\s]*(yes|no|true|false)',
        'prevalentHyp': r'hypertension[:\s]*(yes|no|true|false)',
        'diabetes': r'diabetes[:\s]*(yes|no|true|false)'
    }

    text_lower = text.lower()

    # Extract numeric values
    for key, pattern in patterns.items():
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            try:
                extracted_data[key] = float(match.group(1))
            except ValueError:
                pass

    # Extract boolean values
    for key, pattern in bool_patterns.items():
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            if value in ['yes', 'true']:
                extracted_data[key] = 1
            elif value in ['no', 'false']:
                extracted_data[key] = 0

    # Check for gender
    if 'male' in text_lower and 'female' not in text_lower:
        extracted_data['male'] = 1
    elif 'female' in text_lower and 'male' not in text_lower:
        extracted_data['male'] = 0

    return extracted_data


def prediction(male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP,
               diaBP, BMI, heartRate, glucose):
    """Make prediction using the loaded model"""
    if model is None or scaler is None:
        return "Model or scaler not loaded", None

    data = {
        'male': [male],
        'age': [age],
        'currentSmoker': [currentSmoker],
        'cigsPerDay': [cigsPerDay],
        'BPMeds': [BPMeds],
        'prevalentStroke': [prevalentStroke],
        'prevalentHyp': [prevalentHyp],
        'diabetes': [diabetes],
        'totChol': [totChol],
        'sysBP': [sysBP],
        'diaBP': [diaBP],
        'BMI': [BMI],
        'heartRate': [heartRate],
        'glucose': [glucose]
    }

    df = pd.DataFrame(data)

    try:
        # Scale the data using the fitted scaler
        df_scaled = scaler.transform(df)
        result = model.predict(df_scaled)
        probability = model.predict_proba(df_scaled)[0][1] if hasattr(model, 'predict_proba') else None
        return result[0], probability
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None


# Main app
st.title("🫀 Heart Disease Detection Model")
st.markdown("---")

# Display model status
if model is not None and scaler is not None:
    st.success("✅ Model and scaler loaded successfully!")
else:
    st.error("❌ Error loading model or scaler. Please check your files.")

# Sidebar for file upload
st.sidebar.header("📁 File Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload medical report (TXT or PDF)",
    type=['txt', 'pdf'],
    help="Upload a medical report containing patient data"
)

# Process uploaded file
extracted_data = {}
if uploaded_file is not None:
    with st.sidebar:
        st.success(f"File uploaded: {uploaded_file.name}")

        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_txt(uploaded_file)

        if text:
            extracted_data = extract_medical_data(text)
            if extracted_data:
                st.write("**Extracted Data:**")
                for key, value in extracted_data.items():
                    st.write(f"- {key}: {value}")
            else:
                st.warning("No medical data patterns found in the file.")

# Input section
st.header("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Enter your name", value="")
    gender = st.selectbox("Select your gender", ["Male", "Female"])
    age = st.number_input("Enter your age",
                          min_value=1, max_value=120,
                          value=int(extracted_data.get('age', 30)))

    currentSmoker = st.selectbox("Are you a current smoker?", ["No", "Yes"])
    cigsPerDay = st.number_input("Cigarettes per day",
                                 min_value=0.0, max_value=100.0,
                                 value=float(extracted_data.get('cigsPerDay', 0.0)))

    BPMeds = st.number_input("BP Medications (0 or 1)",
                             min_value=0, max_value=1,
                             value=int(extracted_data.get('BPMeds', 0)))

    prevalentStroke = st.selectbox("Previous stroke?", ["No", "Yes"])

with col2:
    prevalentHyp = st.selectbox("Hypertension?", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes?", ["No", "Yes"])

    totChol = st.number_input("Total Cholesterol",
                              min_value=0.0, max_value=600.0,
                              value=float(extracted_data.get('totChol', 200.0)))

    sysBP = st.number_input("Systolic BP",
                            min_value=70.0, max_value=250.0,
                            value=float(extracted_data.get('sysBP', 120.0)))

    diaBP = st.number_input("Diastolic BP",
                            min_value=40.0, max_value=150.0,
                            value=float(extracted_data.get('diaBP', 80.0)))

    BMI = st.number_input("BMI",
                          min_value=10.0, max_value=60.0,
                          value=float(extracted_data.get('BMI', 25.0)))

    heartRate = st.number_input("Heart Rate",
                                min_value=40, max_value=200,
                                value=int(extracted_data.get('heartRate', 70)))

    glucose = st.number_input("Glucose Level",
                              min_value=50.0, max_value=500.0,
                              value=float(extracted_data.get('glucose', 100.0)))

# Convert inputs to appropriate format
male = 1 if gender == "Male" else 0
currentSmoker_val = 1 if currentSmoker == "Yes" else 0
prevalentStroke_val = 1 if prevalentStroke == "Yes" else 0
prevalentHyp_val = 1 if prevalentHyp == "Yes" else 0
diabetes_val = 1 if diabetes == "Yes" else 0

# Prediction button and results
st.header("🔬 Prediction")
if st.button("🚀 Predict Heart Disease Risk", type="primary"):
    # Display data table when predict button is clicked
    st.header("📊 Patient Data Summary")
    data_dict = {
        'Parameter': ['Name', 'Gender', 'Age', 'Current Smoker', 'Cigarettes/Day', 'BP Medications',
                      'Previous Stroke', 'Hypertension', 'Diabetes', 'Total Cholesterol',
                      'Systolic BP', 'Diastolic BP', 'BMI', 'Heart Rate', 'Glucose'],
        'Value': [name, gender, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke,
                  prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]
    }

    df_display = pd.DataFrame(data_dict)
    st.dataframe(df_display, use_container_width=True)

    st.markdown("---")

    # Make prediction
    if model is not None and scaler is not None:
        result, probability = prediction(male, age, currentSmoker_val, cigsPerDay, BPMeds,
                                         prevalentStroke_val, prevalentHyp_val, diabetes_val,
                                         totChol, sysBP, diaBP, BMI, heartRate, glucose)

        if result is not None:
            st.subheader("🔬 Prediction Results")
            col1, col2 = st.columns(2)

            with col1:
                if result == 1:
                    st.error("⚠️ **High Risk** - Heart disease risk detected")
                else:
                    st.success("✅ **Low Risk** - No immediate heart disease risk detected")

            with col2:
                if probability is not None:
                    st.metric("Risk Probability", f"{probability:.2%}")

            # Additional information
            st.info(
                "⚠️ **Disclaimer**: This prediction is for educational purposes only. Always consult with healthcare professionals for proper medical diagnosis and treatment.")
        else:
            st.error("Could not make prediction. Please check your input data.")
    else:
        st.error("Model or scaler could not be loaded. Please check your model files.")

# Instructions for file upload
st.sidebar.markdown("---")
st.sidebar.header("📝 File Format Instructions")
st.sidebar.markdown("""
**For TXT files, include data like:**
```
Age: 45
Gender: Male
BMI: 28.5
Heart Rate: 75
Glucose: 110
Cholesterol: 220
Systolic: 130
Diastolic: 85
Smoker: Yes
Cigarettes: 10
Hypertension: No
Diabetes: No
Stroke: No
BP Medications: 0
```

**For PDF files:**
Upload medical reports containing similar information.
""")