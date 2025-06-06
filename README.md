# CBIO313-PROJECT
Alzheimer's disease stage prediction Using multimodal machine learnig

# Alzheimer's Disease Stage Classification Project

This project aims to classify Alzheimer's disease stages (Cognitively Unimpaired [CU], Mild Cognitive Impairment [MCI], and Alzheimer's Disease [AD]) using clinical and gene expression data from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** database.

The goal is to build and compare multiple machine learning models to determine whether we can accurately predict Alzheimer's progression using multi-modal data. The final model was deployed as a web application using Streamlit for demonstration purposes.

PRESENTATION VIDEO LINK: 
---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset Description](#dataset-description)  
3. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)  
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
5. [Feature Engineering and Selection](#feature-engineering-and-selection)  
6. [Modeling and Evaluation](#modeling-and-evaluation)  
7. [Deployment](#deployment)  
8. [Requirements](#requirements)  
9. [Folder Structure](#folder-structure)  
10. [Future Improvements](#future-improvements)

---

## Project Overview

The objective of this project is to:
- Determine whether Alzheimer's disease stages can be classified using both clinical and gene expression data.
- Compare the performance of multiple classification algorithms.
- Identify which features (clinical or genetic) contribute most to prediction accuracy.
- Deploy a functional web app that allows users to input patient data and receive a predicted diagnosis.

This work aligns with real-world applications in bioinformatics and health informatics, where early detection of neurodegenerative diseases like Alzheimer's is critical for timely intervention.

Source of data: [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/) 

---

## Dataset Description

### Source:
- **Alzheimer's Disease Neuroimaging Initiative (ADNI)**
- Contains longitudinal clinical, imaging, genetic, and biomarker data collected from participants across different diagnostic groups.

### Size:
- Raw dataset: ~15,000 samples × ~1,500 features
- Cleaned and processed dataset: ~15,000 samples × 59 features

### Features Used:
- **Gene Expression Features**: Reduced to 50 Principal Components using PCA
- **Clinical Features** (9):
  - Age
  - MMSE Score (Mini-Mental State Examination)
  - Gender
  - APOE4 Status
  - Years of Education
  - CDRSB (Clinical Dementia Rating Sum of Boxes)
  - CDGLOBAL (Global Clinical Dementia Rating)
  - FAQTOTAL (Functional Assessment Questionnaire Total)
  - GDTOTAL (Geriatric Depression Scale Total)

### Target Variable:
- Diagnosis_Label: {CU, MCI, AD}

The dataset required significant preprocessing before modeling, including imputation, encoding, scaling, and feature reduction via PCA.

---

## Data Cleaning and Preprocessing

### Steps Taken:
1. **Data Inspection**:
   - Checked for missing values
   - Verified data types and consistency

2. **Missing Value Handling**:
   - Imputed missing values using median imputation for numerical features
   - Dropped features with >70% missing values

3. **Encoding**:
   - Applied label encoding to the target variable (`Diagnosis_Label`)
   - One-hot encoded categorical variables if used

4. **Scaling**:
   - StandardScaler applied to gene expression and selected clinical features

5. **Dimensionality Reduction**:
   - PCA was applied to gene expression features to reduce them from ~185 to 50 components

6. **Final Feature Set**:
   - 50 PCA-transformed gene features
   - 9 clinical features
   - Final shape: `(n_samples, 59)` features

All transformations were documented and saved for use in deployment.

---

## Exploratory Data Analysis (EDA)

The EDA phase included:
- **Univariate analysis** of key clinical and gene expression features
- **Bivariate analysis** comparing feature distributions across diagnosis classes
- **Multivariate analysis** using correlation matrices and pairplots

### Visualizations Included:
1. Histograms for continuous features (e.g., MMSCORE, SEMA4C)
2. Boxplots comparing feature values across CU, MCI, AD
3. Countplot showing class distribution
4. Correlation heatmap between clinical features
5. ROC-AUC curves for multi-class classification
6. Confusion matrices for all models

These visualizations helped guide feature selection and model choice.

---

## Feature Engineering and Selection

### Feature Engineering:
- Gene expression features were standardized and reduced using PCA to retain the most informative components while minimizing overfitting.

### Feature Selection:
- Clinical features were selected based on biological relevance and availability in the ADNI dataset.
- PCA automatically selects features that capture maximum variance in gene expression data.

Final feature set consists of:
- 50 principal components from gene expression
- 9 selected clinical features

---

## Modeling and Evaluation

Multiple classifiers were trained and evaluated:

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|----------|
| Logistic Regression | 0.86     | 0.86      | 0.86   | 0.86     | 0.950    |
| Random Forest       | 0.93     | 0.93      | 0.93   | 0.93     | 0.950    |
| XGBoost             | 0.93     | 0.93      | 0.93   | 0.93     | 0.986    |
| KNN                 | 0.83     | 0.84      | 0.83   | 0.83     | 0.890    |
| SVM                 | 0.86     | 0.86      | 0.86   | 0.86     | 0.950    |
| Decision Tree       | 0.90     | 0.90      | 0.89   | 0.90     | 0.960    |
| Voting Classifier   | 0.93     | 0.93      | 0.93   | 0.93     | 0.986    |
| Stacking Classifier | 0.93     | 0.94      | 0.93   | 0.93     | 0.986    |

### Best Performing Model:
- **StackingClassifier** with base estimators:
  - LogisticRegression
  - RandomForestClassifier
  - XGBClassifier
- Final estimator: LogisticRegression
- Achieved **ROC-AUC score of 0.986**

All models were evaluated using:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- ROC-AUC (multi-class, one-vs-rest)

Cross-validation and hyperparameter tuning were performed using `GridSearchCV`.

---

## Deployment

A web-based interface was created using **Streamlit** to allow users to enter clinical and gene expression data and receive a predicted diagnosis.

Live App: [https://cbio313-project-dxewgbdnhafygjmklnlvyy.streamlit.app/#predicted-diagnosis-mci]

The app uses:
- `scaler.pkl`: StandardScaler trained on gene expression features
- `pca.pkl`: PCA transformer for dimensionality reduction
- `best_model.pkl`: Trained stacking classifier
- `label_encoder_diag.pkl`: For decoding predicted labels back to original diagnosis names

---

## Requirements

To run this project locally, install the following packages:
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib

---

## Folder Structure

CBIO313-PROJECT/
│
├── app.py                     # Web app code
├── best_model.pkl             # Final stacking classifier
├── scaler.pkl                 # StandardScaler trained on gene features
├── pca.pkl                    # PCA transformer
├── label_encoder_diag.pkl     # Diagnosis label encoder
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── ML_PROJECT.ipynb  # Colab notebook with full analysis

---
## Future Improvements

  Use SHAP values for better model interpretation
  Add batch prediction support (CSV upload)
  Explore deep learning approaches for integrating multi-modal data
  Improve UI/UX of the web app
  Apply cross-dataset validation for generalization
     

