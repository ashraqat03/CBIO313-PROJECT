import streamlit as st
import numpy as np
import joblib
import zipfile
import os

# --- Load Model and Transformers ---
@st.cache_resource
def load_all():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")  # expects 59 features
    pca = joblib.load("pca.pkl")        # trained on 189 gene features
    le_diag = joblib.load("label_encoder_diag.pkl")
    with open("gene_list.txt", "r") as f:
        gene_list = [line.strip() for line in f.readlines()]
    return model, scaler, pca, le_diag, gene_list

model, scaler, pca, le_diag, gene_list = load_all()

# --- Means filler ---
gene_means = np.ones(189) * 5.0  # can be replaced by actual means later

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧠 Alzheimer's Stage Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter gene expression and clinical data to predict Alzheimer’s disease stage.</p>", unsafe_allow_html=True)

# --- Gene Inputs (first 10 only) ---
st.subheader("🧬 Gene Expression (first 10 of 189 genes)")
gene_input = []
for i, gene in enumerate(gene_list[:10]):
    val = st.number_input(f"{gene}", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key=f"gene_{i}")
    gene_input.append(val)

# --- Clinical Inputs ---
st.subheader("🧑‍⚕️ Clinical Features")
age = st.number_input("Age", 50, 90, 70)
mmse = st.number_input("MMSE Score", 0, 30, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
apoe4 = st.selectbox("APOE4 Status", ["0", "1", "2"])
education = st.slider("Years of Education", 5, 20, 12)
cdr_global = st.slider("CDR Global", 0.0, 3.0, 0.5)
faq_total = st.slider("FAQ Total", 0, 30, 10)
gd_total = st.slider("GDS Total", 0, 10, 3)
viscode = st.slider("VISCODE", 0.0, 5.0, 1.0)

gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# --- Predict ---
if st.button("🧠 Predict Alzheimer’s Stage"):
    try:
        # Step 1: Build full gene array (189)
        gene_full = gene_means.copy().reshape(1, -1)
        for i in range(10):
            gene_full[0, i] = gene_input[i]

        # Step 2: Apply PCA (→ shape: 1 × 50)
        gene_pca = pca.transform(gene_full)

        # Step 3: Prepare clinical data
        clinical_input = np.array([[age, mmse, gender_num, apoe4_num,
                                    education, cdr_global, faq_total, gd_total, viscode]])

        # Step 4: Merge → shape (1, 59)
        full_input = np.concatenate([gene_pca, clinical_input], axis=1)

        # Step 5: Scale full input
        full_input_scaled = scaler.transform(full_input)

        # Step 6: Predict
        prediction = model.predict(full_input_scaled)
        diagnosis = le_diag.inverse_transform(prediction)[0]
        proba = model.predict_proba(full_input_scaled)

        # --- Output ---
        st.success(f"🧠 Predicted Stage: {diagnosis}")
        st.markdown("### 🔍 Prediction Probabilities")
        for i, label in enumerate(le_diag.classes_):
            st.write(f"{label}: {proba[0][i]:.3f}")
        st.info("📊 Model Accuracy: 93%  |  ROC-AUC: 0.986")

    except Exception as e:
        st.error(f"🚨 Error during prediction: {e}")
