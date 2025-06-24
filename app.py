import streamlit as st
import numpy as np
import joblib
import zipfile
import os

# --- Load files ---
@st.cache_resource
def load_all():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")  # trained on final 59 features
    pca = joblib.load("pca.pkl")        # trained on 189 gene expressions
    le_diag = joblib.load("label_encoder_diag.pkl")
    with open("gene_list.txt", "r") as f:
        gene_list = [line.strip() for line in f.readlines()]
    return model, scaler, pca, le_diag, gene_list

model, scaler, pca, le_diag, gene_list = load_all()

# --- Gene filler means ---
gene_means = np.ones(189) * 5.0  # You can use real means later

# --- Title ---
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧠 Alzheimer's Stage Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter gene expression and clinical data to predict Alzheimer’s stage.</p>", unsafe_allow_html=True)

# --- Gene Inputs (10 only for simplicity) ---
st.subheader("🧬 Gene Expression (first 10 of 189)")
gene_input = []
for i, gene in enumerate(gene_list[:10]):
    value = st.number_input(f"{gene}", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key=f"gene_{i}")
    gene_input.append(value)

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

# Convert gender and APOE4
gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# --- Predict ---
if st.button("🧠 Predict Alzheimer’s Stage"):
    try:
        # Step 1: Fill gene array
        full_gene_input = gene_means.copy().reshape(1, -1)
        for i in range(10):
            full_gene_input[0, i] = gene_input[i]

        # Step 2: PCA only (no scaling here!)
        gene_pca = pca.transform(full_gene_input)  # shape: (1, 50)

        # Step 3: Clinical → (1, 9)
        clinical = np.array([[age, mmse, gender_num, apoe4_num,
                              education, cdr_global, faq_total, gd_total, viscode]])

        # Step 4: Final merge → (1, 59)
        final_input = np.concatenate([gene_pca, clinical], axis=1)

        # Step 5: Now apply scaler (since it was trained on full 59)
        final_input_scaled = scaler.transform(final_input)

        # Step 6: Predict
        pred = model.predict(final_input_scaled)
        proba = model.predict_proba(final_input_scaled)
        diagnosis = le_diag.inverse_transform(pred)[0]

        # --- Display ---
        st.success(f"🧠 Predicted Diagnosis: {diagnosis}")
        st.markdown("### 🔍 Prediction Probabilities")
        for i, label in enumerate(le_diag.classes_):
            st.write(f"{label}: {proba[0][i]:.3f}")
        st.info("📊 Model Accuracy: 93%  |  ROC-AUC: 0.986")

    except Exception as e:
        st.error(f"🚨 Error during prediction: {e}")
