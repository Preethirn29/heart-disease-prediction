## 🚀 HEART DISEASE PREDICTION SYSTEM 🚀
Welcome to a data-driven healthcare initiative!
This heart disease prediction system uses advanced ML algorithms, symptom-based analysis, and a conversational AI assistant to empower users and healthcare providers with early diagnostic capabilities.
# Table of Contents
- [Description](#description)
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [LLM Assistant](#llm-assistant)
- [How We Built It](#how-we-built-it)
- [Future Work](#future-work)
- [Contributing](#contributing)
## 📝 Description
This project focuses on building an intelligent system to predict heart disease using machine learning techniques. Users can input their symptoms via a Streamlit web interface, and the model will determine the likelihood of heart disease. To enhance accessibility and explainability, the system integrates a local LLM assistant that answers health-related questions.
## ❗Problem Statement
Cardiovascular diseases are a leading cause of mortality worldwide. Timely detection is critical, yet access to diagnostic tools is limited in many areas. This project aims to provide:  
• An accurate heart disease prediction model  
• A user-friendly interface for non-technical users  
• AI-driven explanations for better health awareness  
## Project Structure
heart-disease-prediction/  
├── dataset.csv                  
├── main.py                 
├── cleaned_data.csv        
├── model.pkl               
├── scaler.pkl              
├── streamlit_app.py                
├── requirements.txt       
└── README.md               
## ⚙️ Installation
To install the necessary dependencies run:  
```bash  
pip install -r requirements.txt
```
## 🤖 LLM Assistance
This project integrates a local LLM (Mistral via Ollama) to act as an intelligent assistant.  
You can:  
	•	Ask questions about symptoms, health terms, or model outputs  
	•	Get explanations for risk factors  
	•	Learn more about heart conditions directly in the app   
If Ollama isn’t available, you can modify the app to use OpenAI’s API instead.  
To use the AI assistant:  
```bash
ollama run mistral
```
## 🔨 How We Built It

This project was developed using a combination of powerful tools and technologies to ensure accuracy, usability, and explainability.

### 🧰 Tools & Technologies Used

| Category              | Tools / Libraries                             |
|-----------------------|-----------------------------------------------|
| Programming Language  | Python                                        |
| Data Processing       | pandas, numpy                                 |
| Visualization         | matplotlib, seaborn                           |
| Machine Learning      | scikit-learn (RandomForest), imbalanced-learn |
| Model Persistence     | joblib                                        |
| Web Framework         | Streamlit                                     |
| AI Assistant (LLM)    | Ollama (Mistral model)                        |
| System Requirements   | Ollama (Optional), Python 3.8+, pip           |

### 🛠 Implementation Steps  

1. Data Preprocessing 
   - Cleaned and removed nulls/outliers  
   - Saved cleaned dataset to cleaned_data.csv

2. Model Training
   - Applied SMOTE to handle class imbalance  
   - Trained Random Forest Classifier  
   - Saved model as model.pkl and scaler as scaler.pkl

3. Data Visualization 
   - Created heatmaps and boxplots for symptom analysis  

4. Web Interface 
   - Built a Streamlit web app (streamlit_app.py)  
   - Allows users to input symptoms and get predictions

5. LLM Integration
   - Added a text area to query a local LLM using Ollama  
   - Used requests to communicate with the local Mistral model
## 🔮 Future Work
- Improve UI for better user experience  
- Add multilingual support  
- Integrate real-time API for medical record input  
- Expand to include other cardiovascular diseases  
- Deploy to Hugging Face Spaces or Render


