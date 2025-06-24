import streamlit as st
import numpy as np
import pandas as pd
import joblib
import zipfile
import os

# --- Extract Model if Zipped ---
@st.cache_resource
def extract_model():
    if not os.path.exists("best_model.pkl"):
        with zipfile.ZipFile("best_model.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
extract_model()

# --- Load Model, Scaler, PCA, Encoder, Gene List ---
@st.cache_resource
def load_all():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    le_diag = joblib.load("label_encoder_diag.pkl")
    with open("gene_list.txt", "r") as f:
        gene_list = [line.strip() for line in f.readlines()]
    return model, scaler, pca, le_diag, gene_list

model, scaler, pca, le_diag, gene_list = load_all()

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧠 Alzheimer's Stage Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter gene expression and clinical data to predict Alzheimer’s disease stage.</p>", unsafe_allow_html=True)

# --- Gene Expression Inputs (10 shown) ---
st.subheader("🧬 Enter Gene Expression Values (first 10 genes shown)")
gene_input = []
for i, gene in enumerate(gene_list[:10]):
    val = st.number_input(f"{gene}", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key=f"gene_{i}")
    gene_input.append(val)

# --- Clinical Inputs ---
st.subheader("🧑‍⚕️ Clinical Data")
age = st.number_input("Age", min_value=50, max_value=90, value=70)
mmse = st.number_input("MMSE Score", min_value=0, max_value=30, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
apoe4 = st.selectbox("APOE4 Status", ["0", "1", "2"])
education = st.slider("Years of Education", 5, 20, 12)
cdr_global = st.slider("CDGLOBAL", 0.0, 3.0, 0.5)
faq_total = st.slider("FAQTOTAL", 0, 30, 10)
gd_total = st.slider("GDTOTAL", 0, 10, 3)
viscode = st.slider("VISCODE", 0.0, 5.0, 1.0)

# Convert categorical variables
gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# --- Prediction ---
if st.button("🧠 Predict Alzheimer’s Stage"):
    try:
        # Create full gene array (filled with scaler means)
        full_gene_array = np.array(scaler.mean_).reshape(1, -1)
        for i in range(10):
            full_gene_array[0, i] = gene_input[i]

        # Scale and apply PCA
        gene_scaled = scaler.transform(full_gene_array)
        gene_pca = pca.transform(gene_scaled)

        # Prepare clinical features
        clinical_data = np.array([[age, mmse, gender_num, apoe4_num, education, cdr_global, faq_total, gd_total, viscode]])

        # Combine all inputs
        final_input = np.concatenate([gene_pca, clinical_data], axis=1)

        # Predict
        prediction = model.predict(final_input)
        diagnosis = le_diag.inverse_transform(prediction)[0]
        proba = model.predict_proba(final_input)

        # Show results
        st.markdown(f"<h3 style='text-align: center; color: green;'>🧠 Predicted Diagnosis: <strong>{diagnosis}</strong></h3>", unsafe_allow_html=True)

        st.markdown("### 🔍 Prediction Probabilities")
        for i, label in enumerate(le_diag.classes_):
            st.write(f"{label}: {proba[0][i]:.3f}")

        st.success("📊 Model trained on ADNI data. Accuracy: **93%**, Macro ROC-AUC: **0.986**")

    except Exception as e:
        st.error(f"🚨 Error during prediction: {e}")
