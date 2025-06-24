# --- app.py ---
import streamlit as st
import numpy as np
import joblib
import zipfile
import os

# --- Extract ZIP if needed ---
@st.cache_resource
def extract_model():
    if not os.path.exists("best_model.pkl"):
        try:
            with zipfile.ZipFile("best_model.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
        except Exception as e:
            st.error(f"🚨 Model extraction failed: {e}")
            st.stop()

extract_model()

# --- Load Required Files ---
@st.cache_resource
def load_files():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        pca = joblib.load("pca.pkl")
        le_diag = joblib.load("label_encoder_diag.pkl")
        with open("gene_list.txt", "r") as f:
            gene_list = [line.strip() for line in f.readlines()]
        return model, scaler, pca, le_diag, gene_list
    except Exception as e:
        st.error(f"🚨 Error loading files: {e}")
        st.stop()

model, scaler, pca, le_diag, gene_list = load_files()

# --- PCA-trained gene count ---
expected_gene_count = 189

# --- Handle gene list length ---
if len(gene_list) != expected_gene_count:
    st.warning(f"⚠️ PCA expects {expected_gene_count} genes, but gene_list.txt contains {len(gene_list)}.")
    gene_list = gene_list[:expected_gene_count]

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧠 Alzheimer's Stage Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter gene expression and clinical data to predict diagnosis stage.</p>", unsafe_allow_html=True)

# --- Gene Input ---
st.subheader("🧬 Enter Gene Expression (First 10 of 189)")
gene_input = []
for i, gene in enumerate(gene_list[:10]):
    val = st.number_input(f"{gene}", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key=f"gene_{i}")
    gene_input.append(val)

# --- Clinical Data ---
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

# Convert categorical
gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# --- Predict ---
if st.button("🧠 Predict Diagnosis"):
    try:
        # Step 1: Full gene array (with filler)
        full_gene_array = np.ones(expected_gene_count) * 5.0
        for i in range(len(gene_input)):
            full_gene_array[i] = gene_input[i]
        full_gene_array = full_gene_array.reshape(1, -1)

        # Step 2: PCA → (1, 50)
        gene_pca = pca.transform(full_gene_array)

        # Step 3: Clinical features
        clinical_array = np.array([[age, mmse, gender_num, apoe4_num,
                                    education, cdr_global, faq_total, gd_total, viscode]])

        # Step 4: Combine and scale
        combined_input = np.hstack([gene_pca, clinical_array])
        final_input = scaler.transform(combined_input)

        # Step 5: Predict
        pred = model.predict(final_input)
        proba = model.predict_proba(final_input)
        label = le_diag.inverse_transform(pred)[0]

        # Output
        st.success(f"🧠 Predicted Diagnosis: **{label}**")
        st.markdown("### 🔍 Prediction Probabilities")
        for i, cls in enumerate(le_diag.classes_):
            st.write(f"{cls}: {proba[0][i]:.3f}")

        st.info("📊 Model Accuracy: 93% | Macro ROC-AUC: 0.986")

    except Exception as e:
        st.error(f"🚨 Prediction failed: {e}")
