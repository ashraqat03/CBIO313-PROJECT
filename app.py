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
            st.error(f"🚨 Error extracting model: {e}")

extract_model()

# Load transformers and model
@st.cache_resource
def load_files():
    try:
        scaler = joblib.load('scaler.pkl')             # Should expect 189 gene features
        pca = joblib.load('pca.pkl')                   # Should reduce them to 50 components
        model = joblib.load('best_model.pkl')           # Your best classifier
        le_diag = joblib.load('label_encoder_diag.pkl') # Diagnosis label encoder
        return scaler, pca, model, le_diag
    except FileNotFoundError as e:
        st.error(f"📂 Missing file: {e}")
        st.stop()

scaler, pca, model, le_diag = load_files()

# App Title & Description
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧠 Alzheimer's Stage Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Classify Alzheimer’s stages using clinical data and gene expression values.</p>", unsafe_allow_html=True)

# Gene Expression Input Section
st.subheader("🧬 Enter Gene Expression Values")

# Make sure this matches what your PCA was trained on (probably 189 genes)
num_gene_features = pca.n_features_in_  # Should be 189
gene_input = []

for i in range(num_gene_features):
    val = st.slider(f"Gene {i+1}", min_value=0.0, max_value=10.0, value=5.0, key=f"gene_{i}")
    gene_input.append(val)

gene_array = np.array([gene_input])

# Clinical Features
st.subheader("🧑‍⚕️ Enter Clinical Data")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 50, 90, 70)
    mmse = st.slider("MMSE Score", 0, 30, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    apoe4 = st.selectbox("APOE4 Status", ["0", "1", "2"])

with col2:
    education = st.slider("Years of Education", 5, 20, 12)
    cdr_global = st.slider("CDGLOBAL", 0.0, 3.0, 0.5)
    faq_total = st.slider("FAQTOTAL", 0, 30, 10)
    gd_total = st.slider("GDTOTAL", 0, 10, 3)

# Convert categorical inputs
gender_num = 0 if gender == "Male" else 1
apoe4_num = int(apoe4)

# Combine clinical features into an array
clinical_input = np.array([[age, mmse, gender_num, apoe4_num, education, cdr_global, faq_total, gd_total]])

# Apply Scaler + PCA to gene data
try:
    scaled_genes = scaler.transform(gene_array)   # Scale raw gene features
    pca_genes = pca.transform(scaled_genes)      # Reduce to 50 PCs
    input_data = np.hstack([pca_genes, clinical_input])  # Combine PCA + clinical = 50 + 8 = 58? Or 50 + 9 = 59?
except ValueError as e:
    st.error(f"🚨 Input shape mismatch: {e}")
    st.write("Expected gene input shape:", num_gene_features)
    st.write("Your gene input shape:", gene_array.shape[1])
    st.stop()

# Predict
prediction_encoded = model.predict(input_data)
prediction = le_diag.inverse_transform(prediction_encoded)

# Show result
st.markdown("<h3 style='text-align: center; color: green;'>📊 Prediction Result</h3>", unsafe_allow_html=True)
st.markdown(f"<h2 style='text-align: center; color: #0066cc;'>Predicted Diagnosis: <strong>{prediction[0]}</strong></h2>", unsafe_allow_html=True)
