# --- app.py ---
import streamlit as st
import numpy as np
import joblib
import zipfile
import os

# Extract ZIP if needed
@st.cache_resource
def extract_model():
    if not os.path.exists("best_model.pkl"):
        try:
            with zipfile.ZipFile("best_model.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
        except Exception as e:
            st.error(f"Error extracting model: {e}")

extract_model()

# Load files
@st.cache_resource
def load_files():
    try:
        scaler = joblib.load('scaler.pkl')         # Should expect 59 features
        pca = joblib.load('pca.pkl')               # Should expect 185 gene features
        model = joblib.load('best_model.pkl')
        le_diag = joblib.load('label_encoder_diag.pkl')
        return scaler, pca, model, le_diag
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.stop()

scaler, pca, model, le_diag = load_files()

# Streamlit UI
st.title("🧠 Alzheimer's Stage Predictor")
st.markdown("Enter clinical and gene expression values to predict diagnosis")

# Gene Expression Input (must match what PCA expects)
st.subheader("🧬 Enter Gene Expression Values")
num_gene_features = pca.n_components_  # Must be 50
gene_input = [st.slider(f"Gene {i+1}", 0.0, 10.0, 5.0, key=f"gene_{i}") for i in range(185)]  # All genes needed for PCA
gene_array = np.array([gene_input])

# Clinical Features
st.subheader("🧑‍⚕️ Enter Clinical Data")
age = st.slider("Age", 50, 90, 70)
mmse = st.slider("MMSE Score", 0, 30, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
apoe4 = st.selectbox("APOE4 Status", ["0", "1", "2"])
education = st.slider("Years of Education", 5, 20, 12)
cdr_global = st.slider("CDGLOBAL", 0.0, 3.0, 0.5)
faq_total = st.slider("FAQTOTAL", 0, 30, 10)
gd_total = st.slider("GDTOTAL", 0, 10, 3)
viscode = 1.0  # Optional or dummy

# Convert categorical inputs
gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

clinical_input = np.array([[age, mmse, gender_num, apoe4_num, education, cdr_global, faq_total, gd_total, viscode]])

# Apply PCA to gene features
try:
    scaled_genes = scaler.transform(gene_array)   # Scales all genes before PCA
    pca_genes = pca.transform(scaled_genes)      # Converts 185 → 50 PC
    input_data = np.hstack([pca_genes, clinical_input])  # Combine PCA + clinical
except ValueError as e:
    st.error(f"🚨 Input shape mismatch: {e}")
    st.write("Expected gene input shape:", pca.n_features_in_)
    st.write("Your gene input shape:", gene_array.shape[1])
    st.stop()

# Predict
prediction_encoded = model.predict(input_data)
prediction = le_diag.inverse_transform(prediction_encoded)

# Show result
st.subheader("📊 Prediction Result")
st.success(f"Predicted Diagnosis: **{prediction[0]}**")
