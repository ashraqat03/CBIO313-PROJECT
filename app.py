# --- app.py ---
import streamlit as st
import numpy as np
import joblib
import zipfile
import os

# Extract ZIP if needed
@st.cache_resource
def extract_model():
    if not os.path.exists("best_model.pkl"):
        with zipfile.ZipFile("best_model.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

extract_model()

# Load model and label encoder only
@st.cache_resource
def load_files():
    try:
        model = joblib.load('best_model.pkl')
        le_diag = joblib.load('label_encoder_diag.pkl')
        return model, le_diag
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.stop()

model, le_diag = load_files()

# App UI
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧠 Alzheimer's Stage Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter values below to predict Alzheimer's stage.</p>", unsafe_allow_html=True)

# Gene Expression Features (PCA already applied → 50 components)
st.subheader("🧬 Enter PCA-Reduced Gene Expression Values")

gene_pca_input = []
for i in range(50):
    val = st.slider(f"PC{i+1}", 0.0, 10.0, 5.0, key=f"pc_{i}")
    gene_pca_input.append(val)

# Clinical Features
st.subheader("🧑‍⚕️ Enter Clinical Data")
age = st.slider("Age", 50, 90, 70)
mmse = st.slider("MMSE Score", 0, 30, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
apoe4 = st.selectbox("APOE4 Status", ["0", "1", "2"])
education = st.slider("Years of Education", 5, 20, 12)
cdr_global = st.slider("CDGLOBAL", 0.0, 3.0, 0.5)
faq_total = st.slider("FAQTOTAL", 0, 30, 10)
gd_total = st.slider("GDTOTAL", 0, 10, 3)
viscode = st.slider("VISCODE", 0.0, 5.0, 1.0)

# Convert categorical inputs
gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# Combine into 59 features
input_data = np.array([gene_pca_input + [age, mmse, gender_num, apoe4_num, education, cdr_global, faq_total, gd_total, viscode]])

# Predict
try:
    prediction_encoded = model.predict(input_data)
    prediction = le_diag.inverse_transform(prediction_encoded)
except Exception as e:
    st.error(f"🚨 Prediction error: {e}")

# Show result
st.markdown("<h3 style='text-align: center; color: green;'>📊 Prediction Result</h3>", unsafe_allow_html=True)
st.markdown(f"<h2 style='text-align: center; color: #0066cc;'>Predicted Diagnosis: <strong>{prediction[0]}</strong></h2>", unsafe_allow_html=True)
