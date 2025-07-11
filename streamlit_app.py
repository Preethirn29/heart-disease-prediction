import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from PIL import Image

# Load trained model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("Heart Disease Prediction App")

st.markdown("Enter Patient Symptoms:")

# Input form
with st.form("symptom_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", value=150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced", value=1.0)
    slope = st.selectbox("Slope of ST segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)", [0, 1, 2])

    submit = st.form_submit_button("Predict")

# Predict
if submit:
    input_data = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(f"Likely to Have Heart Disease")
    else:
        st.success(f"Unlikely to Have Heart Disease")
# LLM Section - Ollama
st.markdown("---")
st.markdown("## 🤖 Local AI Assistant (Ollama)")
user_input = st.text_area("Ask anything about symptoms or analysis:")

if st.button("Ask Ollama"):
    if user_input.strip() != "":
        with st.spinner("Thinking..."):
            try:
                response = requests.post("http://localhost:11434/api/generate", json={
                    "model": "mistral",
                    "prompt": user_input
                })
                result = response.json()['response']
                st.write(result)
            except Exception as e:
                st.error("⚠ Could not connect to Ollama. Make sure Ollama is running locally.")