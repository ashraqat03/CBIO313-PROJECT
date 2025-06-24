import streamlit as st
import numpy as np
import joblib
import zipfile
import os

# --- Extract if needed ---
@st.cache_resource
def extract_model():
    if not os.path.exists("best_model.pkl"):
        with zipfile.ZipFile("best_model.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
extract_model()

# --- Load all components ---
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

# --- Title ---
st.title("🧠 Alzheimer's Stage Classifier")
st.markdown("Enter gene expression values + clinical features to predict Alzheimer’s disease stage.")

# --- Gene Expression (first 10 genes) ---
st.subheader("🧬 Enter Expression Values for First 10 Genes")
gene_input = []
for i, gene in enumerate(gene_list[:10]):
    value = st.number_input(f"{gene}", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key=f"gene_{i}")
    gene_input.append(value)

# --- Clinical Inputs ---
st.subheader("🧑‍⚕️ Clinical Data")
age = st.number_input("Age", 50, 90, 70)
mmse = st.number_input("MMSE Score", 0, 30, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
apoe4 = st.selectbox("APOE4 Status", ["0", "1", "2"])
education = st.slider("Years of Education", 5, 20, 12)
cdr_global = st.slider("CDGLOBAL", 0.0, 3.0, 0.5)
faq_total = st.slider("FAQTOTAL", 0, 30, 10)
gd_total = st.slider("GDTOTAL", 0, 10, 3)
viscode = st.slider("VISCODE", 0.0, 5.0, 1.0)

gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# --- Predict ---
if st.button("🧠 Predict"):
    try:
        # STEP 1: Construct full 189 gene vector (10 user + 179 scaler.mean_)
        full_gene_vector = np.array(scaler.mean_).reshape(1, -1)
        for i in range(10):
            full_gene_vector[0, i] = gene_input[i]

        st.write(f"🧬 Gene input shape: {full_gene_vector.shape}")  # should be (1, 189)

        # STEP 2: Scale + PCA only on gene part
        gene_scaled = scaler.transform(full_gene_vector)
        st.write(f"🔬 Scaled shape: {gene_scaled.shape}")  # should be (1, 189)

        gene_pca = pca.transform(gene_scaled)
        st.write(f"📉 PCA output shape: {gene_pca.shape}")  # should be (1, 50)

        # STEP 3: Clinical array
        clinical = np.array([[age, mmse, gender_num, apoe4_num,
                              education, cdr_global, faq_total, gd_total, viscode]])  # (1, 9)

        # STEP 4: Combine → (1, 59)
        final_input = np.concatenate([gene_pca, clinical], axis=1)
        st.write(f"✅ Final input shape: {final_input.shape}")

        # STEP 5: Predict
        pred = model.predict(final_input)
        proba = model.predict_proba(final_input)
        diagnosis = le_diag.inverse_transform(pred)[0]

        # STEP 6: Show result
        st.success(f"🧠 Predicted Diagnosis: {diagnosis}")
        st.markdown("### 🔍 Prediction Probabilities")
        for i, label in enumerate(le_diag.classes_):
            st.write(f"{label}: {proba[0][i]:.3f}")

        st.info("✅ Model Accuracy: 93%, ROC-AUC: 0.986")

    except Exception as e:
        st.error(f"🚨 Error during prediction: {e}")
