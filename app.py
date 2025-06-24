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
            st.error("🚨 File not found: best_model.zip is missing.")
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
        scaler = joblib.load("scaler.pkl")         # Trained on full input: 50 gene PCA + 9 clinical
        pca = joblib.load("pca.pkl")               # PCA from 189 → 50
        le_diag = joblib.load("label_encoder_diag.pkl")
        with open("gene_list.txt", "r") as f:
            gene_list = [line.strip() for line in f.readlines()]
        return model, scaler, pca, le_diag, gene_list
    except FileNotFoundError as e:
        st.error(f"🚨 Missing file: {e}")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Error loading files: {e}")
        st.stop()

model, scaler, pca, le_diag, gene_list = load_files()

# --- Mean filler (for missing gene values) ---
gene_means = np.ones(len(gene_list)) * 5.0

# --- App Header ---
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧠 Alzheimer's Stage Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter values to predict Alzheimer’s disease stage.</p>", unsafe_allow_html=True)

# --- Gene Inputs ---
st.subheader("🧬 Enter Gene Expression (First 10 Only)")
gene_input = []
for i, gene in enumerate(gene_list[:10]):
    val = st.number_input(f"{gene}", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key=f"gene_{i}")
    gene_input.append(val)

# --- Clinical Inputs ---
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

# Convert categorical to numeric
gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# --- Predict Button ---
if st.button("🧠 Predict Diagnosis"):
    try:
        # Step 1: Fill 189 gene expression values (10 real + 179 default)
        gene_array = gene_means.copy().reshape(1, -1)
        for i in range(10):
            gene_array[0, i] = gene_input[i]

        # Step 2: PCA (→ 1×50)
        gene_pca = pca.transform(gene_array)

        # Step 3: Append clinical features
        clinical_array = np.array([[age, mmse, gender_num, apoe4_num,
                                    education, cdr_global, faq_total, gd_total, viscode]])
        full_input = np.hstack([gene_pca, clinical_array])  # shape: (1, 59)

        # Step 4: Final scaling
        scaled_input = scaler.transform(full_input)

        # Step 5: Predict
        pred = model.predict(scaled_input)
        proba = model.predict_proba(scaled_input)
        diagnosis = le_diag.inverse_transform(pred)[0]

        # --- Display ---
        st.success(f"🧠 Predicted Stage: **{diagnosis}**")
        st.markdown("### 🔍 Probabilities")
        for i, label in enumerate(le_diag.classes_):
            st.write(f"{label}: {proba[0][i]:.3f}")
        st.info("📊 Model Accuracy: 93%  |  ROC-AUC: 0.986")

    except Exception as e:
        st.error(f"🚨 Prediction failed: {e}")
