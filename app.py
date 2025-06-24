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
        scaler = joblib.load("scaler.pkl")         # Trained on full 59-feature input
        pca = joblib.load("pca.pkl")               # PCA trained on 189 genes
        le_diag = joblib.load("label_encoder_diag.pkl")
        with open("gene_list.txt", "r") as f:
            gene_list = [line.strip() for line in f.readlines()]
        return model, scaler, pca, le_diag, gene_list
    except Exception as e:
        st.error(f"🚨 Error loading files: {e}")
        st.stop()

model, scaler, pca, le_diag, gene_list = load_files()

# --- Set expected gene count (based on PCA training) ---
expected_gene_count = 189
if len(gene_list) != expected_gene_count:
    st.error(f"🚨 gene_list.txt has {len(gene_list)} genes, but PCA expects {expected_gene_count}. Please fix.")
    st.stop()

# --- Gene input UI ---
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧠 Alzheimer's Stage Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter gene expression and clinical data to predict diagnosis stage.</p>", unsafe_allow_html=True)

st.subheader("🧬 Enter Gene Expression (First 10 Only)")
gene_input = []
for i, gene in enumerate(gene_list[:10]):
    val = st.number_input(f"{gene}", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key=f"gene_{i}")
    gene_input.append(val)

# --- Clinical Input UI ---
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

# --- Convert Categorical ---
gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# --- Predict ---
if st.button("🧠 Predict Diagnosis"):
    try:
        # Fill gene vector: 10 inputs + 179 mean fillers
        gene_array = np.ones(expected_gene_count) * 5.0
        for i in range(len(gene_input)):
            gene_array[i] = gene_input[i]

        gene_array = gene_array.reshape(1, -1)  # shape: (1, 189)

        # PCA → 1×50
        gene_pca = pca.transform(gene_array)

        # Combine with clinical → 1×59
        clinical_array = np.array([[age, mmse, gender_num, apoe4_num,
                                    education, cdr_global, faq_total, gd_total, viscode]])
        final_input = np.hstack([gene_pca, clinical_array])

        # Scale → 1×59
        final_scaled = scaler.transform(final_input)

        # Predict
        pred = model.predict(final_scaled)
        proba = model.predict_proba(final_scaled)
        label = le_diag.inverse_transform(pred)[0]

        # --- Display ---
        st.success(f"🧠 Predicted Diagnosis: **{label}**")
        st.markdown("### 🔍 Class Probabilities")
        for i, cls in enumerate(le_diag.classes_):
            st.write(f"{cls}: {proba[0][i]:.3f}")

        st.info("📊 Model Accuracy: 93% | Macro ROC-AUC: 0.986")

    except Exception as e:
        st.error(f"🚨 Prediction failed: {e}")
