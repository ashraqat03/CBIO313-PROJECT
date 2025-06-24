import streamlit as st
import numpy as np
import joblib
import zipfile
import os

# --- Extract model if zipped ---
@st.cache_resource
def extract_model():
    if not os.path.exists("best_model.pkl"):
        with zipfile.ZipFile("best_model.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
extract_model()

# --- Load all necessary components ---
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

# --- UI Title ---
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧠 Alzheimer's Stage Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter gene expression and clinical data to predict Alzheimer’s stage.</p>", unsafe_allow_html=True)

# --- Gene expression inputs (first 10 only) ---
st.subheader("🧬 Enter Gene Expression Values (first 10 genes shown)")
gene_input = []
for i, gene in enumerate(gene_list[:10]):
    val = st.number_input(f"{gene}", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key=f"gene_{i}")
    gene_input.append(val)

# --- Clinical inputs ---
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

# --- Convert categorical variables ---
gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# --- Predict button ---
if st.button("🧠 Predict Alzheimer’s Stage"):
    try:
        # Step 1: Prepare full gene vector (189 total)
        full_gene_array = np.array(scaler.mean_).reshape(1, -1)  # start with mean values
        for i in range(10):
            full_gene_array[0, i] = gene_input[i]

        # Step 2: Apply scaler and PCA on gene data only
        gene_scaled = scaler.transform(full_gene_array)          # shape: [1, 189]
        gene_pca = pca.transform(gene_scaled)                    # shape: [1, 50]

        # Step 3: Prepare clinical features
        clinical_array = np.array([[age, mmse, gender_num, apoe4_num,
                                    education, cdr_global, faq_total, gd_total, viscode]])  # shape: [1, 9]

        # Step 4: Combine PCA + clinical → [1, 59]
        final_input = np.concatenate([gene_pca, clinical_array], axis=1)

        # Step 5: Predict
        prediction = model.predict(final_input)
        prediction_label = le_diag.inverse_transform(prediction)[0]
        proba = model.predict_proba(final_input)

        # Step 6: Output
        st.markdown(f"<h3 style='text-align: center; color: green;'>🧠 Predicted Diagnosis: <strong>{prediction_label}</strong></h3>", unsafe_allow_html=True)
        st.markdown("### 🔍 Prediction Probabilities")
        for i, label in enumerate(le_diag.classes_):
            st.write(f"{label}: {proba[0][i]:.3f}")

        st.success("📊 Model trained on ADNI data.\n\n✅ Accuracy: **93%**\n✅ ROC-AUC: **0.986**")

    except Exception as e:
        st.error(f"🚨 Error during prediction: {e}")
