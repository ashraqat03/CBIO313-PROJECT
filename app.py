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

# Load transformers and model
@st.cache_resource
def load_files():
    try:
        scaler = joblib.load('scaler.pkl')
        pca = joblib.load('pca.pkl')
        model = joblib.load('best_model.pkl')
        le_diag = joblib.load('label_encoder_diag.pkl')
        return scaler, pca, model, le_diag
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.stop()

scaler, pca, model, le_diag = load_files()

# Streamlit UI
st.title("🧠 Alzheimer's Stage Predictor")
st.markdown("Enter gene expression and clinical data to predict Alzheimer’s stage (`CU`, `MCI`, `AD`)")

# Gene Expression Input (must match what PCA was trained on)
st.subheader("🧬 Enter Gene Expression Values")
gene_input = [st.slider(f"Gene {i+1}", 0.0, 10.0, 5.0, key=f"gene_{i}") for i in range(50)]

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

# Convert categorical inputs
gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# Combine all clinical features (should be 9)
clinical_input = [age, mmse, gender_num, apoe4_num, education, cdr_global, faq_total, gd_total, 1.0]  # Replace last one with real feature if available

# Combine gene + clinical
input_data = np.array([gene_input + clinical_input])

# Apply transformations
try:
    scaled_data = scaler.transform(input_data)
    pca_data = pca.transform(scaled_data)
except ValueError as e:
    st.error(f"🚨 Input shape mismatch: {e}")
    st.write("Expected input shape:", scaler.n_features_in_)
    st.write("Your input shape:", input_data.shape[1])
    st.stop()

# Predict
prediction_encoded = model.predict(pca_data)
prediction = le_diag.inverse_transform(prediction_encoded)

# Show result
st.subheader("📊 Prediction Result")
st.success(f"Predicted Diagnosis: **{prediction[0]}**")
