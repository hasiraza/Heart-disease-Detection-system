# 🫀 Heart Disease Detection Model

A machine learning-powered web application built with Streamlit that predicts heart disease risk based on patient medical data. The application supports both manual data entry and automated extraction from medical reports (PDF/TXT files).

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **Interactive Web Interface**: User-friendly Streamlit web application
- **Manual Data Entry**: Input patient information through intuitive forms
- **Document Processing**: Extract medical data from PDF and TXT files automatically
- **Real-time Predictions**: Get instant heart disease risk assessments
- **Risk Probability**: View prediction confidence scores
- **Data Visualization**: Display patient data in organized tables
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Demo

![Heart Disease Detection Demo]([https://via.placeholder.com/800x400/FF6B6B/FFFFFF?text=Heart+Disease+Detection+App+Demo](https://heart-disease-detection-system-fvh28thvufgqwyxc95zi6n.streamlit.app/))

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Input Parameters](#input-parameters)
- [File Upload Format](#file-upload-format)
- [Model Information](#model-information)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-disease-detection.git
   cd heart-disease-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv heart_disease_env
   
   # Activate virtual environment
   # On Windows:
   heart_disease_env\Scripts\activate
   # On macOS/Linux:
   source heart_disease_env/bin/activate
   ```

3. **Install required packages**
   ```bash
   # Install minimal requirements
   pip install -r requirements-minimal.txt
   
   # Or install full requirements with additional features
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   - Open your web browser and navigate to `http://localhost:8501`

## 📖 Usage

### Manual Data Entry

1. **Launch the application** using `streamlit run app.py`
2. **Enter patient information** in the input forms:
   - Personal details (name, gender, age)
   - Medical parameters (BP, cholesterol, glucose, etc.)
   - Lifestyle factors (smoking, medications)
3. **Click "Predict Heart Disease Risk"** to get results
4. **View the prediction** and risk probability

### File Upload Method

1. **Prepare your medical report** in TXT or PDF format
2. **Upload the file** using the sidebar file uploader
3. **Review extracted data** that appears in the sidebar
4. **Modify any incorrect values** in the input forms
5. **Get prediction** by clicking the predict button

## 📊 Input Parameters

The model requires the following 14 parameters:

| Parameter | Description | Range/Type |
|-----------|-------------|------------|
| **Gender** | Patient gender | Male/Female |
| **Age** | Patient age in years | 1-120 |
| **Current Smoker** | Smoking status | Yes/No |
| **Cigarettes per Day** | Number of cigarettes smoked daily | 0-100 |
| **BP Medications** | Taking blood pressure medications | 0/1 |
| **Previous Stroke** | History of stroke | Yes/No |
| **Hypertension** | High blood pressure condition | Yes/No |
| **Diabetes** | Diabetes condition | Yes/No |
| **Total Cholesterol** | Cholesterol level (mg/dL) | 0-600 |
| **Systolic BP** | Systolic blood pressure | 70-250 |
| **Diastolic BP** | Diastolic blood pressure | 40-150 |
| **BMI** | Body Mass Index | 10-60 |
| **Heart Rate** | Heart rate (BPM) | 40-200 |
| **Glucose** | Glucose level (mg/dL) | 50-500 |

## 📄 File Upload Format

### TXT File Format
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

### PDF Files
- Upload medical reports containing patient information
- The app will automatically extract relevant medical data
- Review and correct any misextracted information

## 🧠 Model Information

- **Algorithm**: Machine Learning Classification Model (specifics depend on your trained model)
- **Features**: 14 medical and lifestyle parameters
- **Output**: Binary classification (High Risk / Low Risk) with probability score
- **Preprocessing**: StandardScaler for feature normalization

### Model Performance
- **Accuracy**: [Add your model's accuracy]
- **Precision**: [Add your model's precision]
- **Recall**: [Add your model's recall]
- **F1-Score**: [Add your model's F1-score]

## 📁 Project Structure

```
heart-disease-detection/
│
├── app.py                          # Main Streamlit application
├── Model/
│   ├── Heart disease detection model.pkl  # Trained ML model
│   └── scaler.pkl                  # Fitted StandardScaler (optional)
├── requirements.txt                # Full dependencies
├── requirements-minimal.txt        # Minimal dependencies
├── .gitignore                     # Git ignore file
├── README.md                      # Project documentation
└── assets/                        # Images and other assets (optional)
    └── demo_image.png
```

## 🔮 Future Enhancements

- [ ] **Enhanced Visualizations**: Add charts and graphs for better data representation
- [ ] **Model Comparison**: Implement multiple ML algorithms and compare results
- [ ] **Historical Data**: Store and track patient data over time
- [ ] **Export Reports**: Generate PDF reports of predictions
- [ ] **API Integration**: RESTful API for external integrations
- [ ] **Mobile App**: React Native or Flutter mobile application
- [ ] **Multi-language Support**: Support for multiple languages

## ⚠️ Disclaimer

**Important**: This application is for educational and research purposes only. The predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/heart-disease-detection.git
cd heart-disease-detection

# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
pytest

# Run the app in development mode
streamlit run app.py --server.runOnSave true
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Muhammad Haseeb Raza**

- 📧 Email: [hasiraza511@gmail.com](mailto:hasiraza511@gmail.com)
- 💼 LinkedIn: [Muhammad Haseeb Raza](https://www.linkedin.com/in/muhammad-haseeb-raza-71987a366/)
- 🐙 GitHub: [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing libraries
- Streamlit team for the fantastic web app framework
- Scikit-learn contributors for machine learning tools
- Healthcare data providers for making medical datasets available

## 📈 Statistics

![GitHub stars](https://img.shields.io/github/stars/yourusername/heart-disease-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/heart-disease-detection?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/heart-disease-detection)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/heart-disease-detection)

---

⭐ **Star this repository if you found it helpful!**
