# --- app.py ---
import streamlit as st
import numpy as np
import joblib
import zipfile
import os

# --- Step 1: Extract ZIP file if needed ---
@st.cache_resource
def extract_model():
    if not os.path.exists("best_model.pkl"):
        try:
            with zipfile.ZipFile("best_model.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            st.success("✅ Model extracted successfully.")
        except FileNotFoundError:
            st.error("🚨 File not found: best_model.zip is missing in the repo.")
            st.stop()
        except Exception as e:
            st.error(f"🚨 Extraction failed: {e}")
            st.stop()

extract_model()

# --- Step 2: Load All Required Files ---
@st.cache_resource
def load_files():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")         # StandardScaler trained on raw genes
        pca = joblib.load("pca.pkl")               # PCA(n_components=50)
        le_diag = joblib.load("label_encoder_diag.pkl")

        with open("gene_list.txt", "r") as f:
            gene_list = [line.strip() for line in f.readlines()]

        return model, scaler, pca, le_diag, gene_list

model, scaler, pca, le_diag, gene_list = load_files()

# --- Gene filler means (for missing genes) ---
gene_means = np.ones(len(gene_list)) * 5.0  # Replace with real means if available

# --- UI Section ---
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧠 Alzheimer's Stage Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter gene expression and clinical data to predict diagnosis.</p>", unsafe_allow_html=True)

# --- Gene Expression Input (First 10 of 189) ---
st.subheader("🧬 Enter Gene Expression Values (First 10)")
gene_input = []
for i, gene in enumerate(gene_list[:10]):
    value = st.number_input(f"{gene}", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key=f"gene_{i}")
    gene_input.append(value)

# --- Clinical Features ---
st.subheader("🧑‍⚕️ Enter Clinical Data")
age = st.number_input("Age", 50, 90, 70)
mmse = st.number_input("MMSE Score", 0, 30, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
apoe4 = st.selectbox("APOE4 Status", ["0", "1", "2"])
education = st.slider("Years of Education", 5, 20, 12)
cdr_global = st.slider("CDR Global", 0.0, 3.0, 0.5)
faq_total = st.slider("FAQ Total", 0, 30, 10)
gd_total = st.slider("GDS Total", 0, 10, 3)
viscode = st.slider("VISCODE", 0.0, 5.0, 1.0)

# Convert categorical inputs
gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# --- Prediction Button ---
if st.button("🧠 Predict Diagnosis"):
    try:
        # Step 1: Fill user input into gene array
        full_gene_array = gene_means.copy().reshape(1, -1)
        for i in range(10):
            full_gene_array[0, i] = gene_input[i]

        # Step 2: Apply Scaler trained on raw gene features
        scaled_genes = scaler.transform(full_gene_array)  # shape: (1, 189)

        # Step 3: Apply PCA to reduce to 50 components
        gene_pca = pca.transform(scaled_genes)  # shape: (1, 50)

        # Step 4: Combine with clinical features
        clinical_data = np.array([[age, mmse, gender_num, apoe4_num,
                                  education, cdr_global, faq_total, gd_total, viscode]])

        final_input = np.hstack([gene_pca, clinical_data])  # shape: (1, 59)

        # Step 5: Predict
        prediction_encoded = model.predict(final_input)
        prediction_probs = model.predict_proba(final_input)
        diagnosis = le_diag.inverse_transform(prediction_encoded)[0]

        # --- Display Results ---
        st.success(f"🧠 Predicted Diagnosis: **{diagnosis}**")
        st.markdown("### 🔍 Prediction Probabilities")
        for i, label in enumerate(le_diag.classes_):
            st.write(f"{label}: {prediction_probs[0][i]:.3f}")

        st.info("📊 Model Accuracy: 93% | ROC-AUC: 0.986")

    except Exception as e:
        st.error(f"🚨 Prediction failed: {e}")
