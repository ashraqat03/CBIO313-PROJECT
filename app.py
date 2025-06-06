# --- app.py ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved transformers and model
@st.cache_resource
def load_files():
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    model = joblib.load('best_model.pkl')
st.title("🧠 Alzheimer's Disease Stage Classifier")
st.markdown("Classify Alzheimer's stages (CU, MCI, AD) using clinical data and gene expression")

# Section: Gene Expression Input
st.subheader("🧬 Enter Gene Expression Values (Top 50 Genes)")
gene_input = []
for i in range(50):
    val = st.slider(f"Gene {i+1} Expression", min_value=0.0, max_value=10.0, value=5.0, key=f"gene_{i}")
    gene_input.append(val)

# Section: Clinical Features
st.subheader("Enter Clinical Data")
age = st.slider("Age", min_value=50, max_value=90, value=70)
mmse = st.slider("MMSE Score", min_value=0, max_value=30, value=25)
gender = st.selectbox("Gender", options=["Male", "Female"])
apoe4 = st.selectbox("APOE4 Status", options=["0", "1", "2"])

# Convert categorical features to numerical (match training format)
gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# Combine clinical + gene expression inputs
input_data = np.array([gene_input + [age, mmse, gender_num, apoe4_num]])

# Apply Scaler and PCA
scaled_data = scaler.transform(input_data)
pca_data = pca.transform(scaled_data)

# Predict
prediction_encoded = model.predict(pca_data)
prediction = le_diag.inverse_transform(prediction_encoded)

# Show Result
st.subheader("📊 Prediction Result")
st.success(f"Predicted Diagnosis: **{prediction[0]}**")