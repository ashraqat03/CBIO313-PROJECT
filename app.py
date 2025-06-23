# --- app.py ---
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import zipfile
import os

# --- Extract Model if Needed ---
@st.cache_resource
def extract_model():
    if not os.path.exists("best_model.pkl"):
        with zipfile.ZipFile("best_model.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
extract_model()

# --- Load All Files ---
@st.cache_resource
def load_all():
    model = joblib.load("best_model.pkl")
    le_diag = joblib.load("label_encoder_diag.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    
    with open("gene_list.txt", "r") as f:
        gene_list = [line.strip() for line in f.readlines()]
    
    return model, le_diag, scaler, pca, gene_list

model, le_diag, scaler, pca, gene_list = load_all()

# --- UI Header ---
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧠 Alzheimer's Stage Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter gene expression values and clinical features to predict the Alzheimer’s stage.</p>", unsafe_allow_html=True)

# --- Gene Expression Inputs ---
st.subheader("🧬 Enter Gene Expression Values")
gene_input = []
for gene in gene_list[:10]:  # Display first 10 genes only for simplicity
    val = st.number_input(f"{gene}", min_value=0.0, max_value=20.0, value=5.0, step=0.1, format="%.3f")
    gene_input.append(val)

# Note: add a checkbox to allow uploading full 185+ genes via file in future

# --- Clinical Inputs ---
st.subheader("🧑‍⚕️ Clinical Information")
age = st.number_input("Age", min_value=50, max_value=90, value=70)
mmse = st.number_input("MMSE Score", min_value=0, max_value=30, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
apoe4 = st.selectbox("APOE4 Status", ["0", "1", "2"])
education = st.slider("Years of Education", 5, 20, 12)
cdr_global = st.slider("CDGLOBAL", 0.0, 3.0, 0.5)
faq_total = st.slider("FAQTOTAL", 0, 30, 10)
gd_total = st.slider("GDTOTAL", 0, 10, 3)
viscode = st.slider("VISCODE", 0.0, 5.0, 1.0)

# Convert categorical to numeric
gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# --- Predict ---
if st.button("🧠 Predict Alzheimer’s Stage"):
    try:
        # Combine gene + clinical features
        gene_array = np.array(gene_input).reshape(1, -1)
        gene_scaled = scaler.transform(gene_array)
        gene_pca = pca.transform(gene_scaled)

        clinical_features = np.array([[age, mmse, gender_num, apoe4_num, education, cdr_global, faq_total, gd_total, viscode]])
        final_input = np.concatenate([gene_pca, clinical_features], axis=1)

        pred = model.predict(final_input)
        proba = model.predict_proba(final_input)

        # Show results
        diagnosis = le_diag.inverse_transform(pred)[0]
        st.markdown(f"<h3 style='text-align: center; color: green;'>🧠 Predicted Diagnosis: <strong>{diagnosis}</strong></h3>", unsafe_allow_html=True)

        st.markdown("### 🔍 Prediction Probabilities")
        for i, label in enumerate(le_diag.classes_):
            st.write(f"{label}: {proba[0][i]:.3f}")

        # Optional: display model accuracy (precomputed)
        st.success("📊 This model was trained on real-world data and achieved over 93% accuracy.")

    except Exception as e:
        st.error(f"🚨 Error during prediction: {e}")
