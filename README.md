# KneeAI: Deep Learning System for Knee Osteoarthritis Diagnosis 🩺

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

This repository contains the official implementation of the paper:
**"A Web-Based Deep Learning System for Automatic Severity Classification of Knee Osteoarthritis Using Kellgren-Lawrence Grading"**

## 📋 Overview
**KneeAI** is a clinical decision-support tool powered by an **EfficientNetB3** Convolutional Neural Network, optimized via Bayesian Hyperparameter Search (Optuna). The system automatically classifies knee radiographs into three clinically relevant severity levels, providing instant diagnostic suggestions and visual explainability to assist medical professionals.

### Key Capabilities:
- **Automated Triage:** Classifies images into **Healthy** (KL 0-1), **Mild-Moderate** (KL 2-3), or **Severe** (KL 4) with >83% accuracy.
- **Visual Explainability:** Generates real-time **Grad-CAM heatmaps** to highlight the anatomical regions (e.g., joint space) guiding the AI's decision.
- **Uncertainty Estimation:** Calculates predictive entropy to flag low-confidence predictions.
- **Clinical Risk Profile:** Visualizes class probabilities using a radar chart.

## 📥 Download Model Files (Required)
Due to GitHub's file size limits, the trained model files are hosted externally.
**You must download BOTH files below and place/extract them in the root folder of this project for the app to work.**

### 1. Model Weights (For Grad-CAM)
- **File:** `KneeOA_temp_weights.h5`
- **Action:** Download and place it directly in the `KneeAI-Project` folder.
- **[Download .h5 Weights Here](https://drive.google.com/file/d/1E1nCcaS49FVDamO7spmA__CpxRpGX4q3/view?usp=sharing)**

### 2. Inference Model (For Predictions)
- **File:** `KneeOA_GradCAM.zip`
- **Action:** Download, **unzip/extract** it, and ensure the resulting folder is named `KneeOA_GradCAM` (it should contain a `saved_model.pb` file inside). Place the folder in the project root.
- **[Download SavedModel Zip Here](https://drive.google.com/file/d/1V73MEJOIW2_Qz4XLKlbLmo2ezBW-oNrn/view?usp=sharing)**

## 🧠 Model Architecture
- **Backbone:** EfficientNetB3 (Pre-trained on ImageNet).
- **Optimization:** Hyperparameters (Dropout=0.35, L2=1e-4) tuned using the TPE algorithm (Optuna).
- **Training Strategy:** Two-stage Transfer Learning (Warm-up + Fine-Tuning).
- **Input Resolution:** 300 x 300 pixels.

## 📂 Repository Structure
¡Por supuesto! Aquí tienes el archivo README.md completo y definitivo.

He integrado todas las correcciones, incluyendo la sección crítica para descargar ambos archivos (los pesos .h5 y la carpeta del modelo .zip) desde tu Drive, explicando claramente dónde debe ponerlos el usuario.

Copia todo el bloque de abajo y pégalo en tu archivo README.md.

Markdown

# KneeAI: Deep Learning System for Knee Osteoarthritis Diagnosis 🩺

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

This repository contains the official implementation of the paper:
**"A Web-Based Deep Learning System for Automatic Severity Classification of Knee Osteoarthritis Using Kellgren-Lawrence Grading"**

## 📋 Overview
**KneeAI** is a clinical decision-support tool powered by an **EfficientNetB3** Convolutional Neural Network, optimized via Bayesian Hyperparameter Search (Optuna). The system automatically classifies knee radiographs into three clinically relevant severity levels, providing instant diagnostic suggestions and visual explainability to assist medical professionals.

### Key Capabilities:
- **Automated Triage:** Classifies images into **Healthy** (KL 0-1), **Mild-Moderate** (KL 2-3), or **Severe** (KL 4) with >83% accuracy.
- **Visual Explainability:** Generates real-time **Grad-CAM heatmaps** to highlight the anatomical regions (e.g., joint space) guiding the AI's decision.
- **Uncertainty Estimation:** Calculates predictive entropy to flag low-confidence predictions.
- **Clinical Risk Profile:** Visualizes class probabilities using a radar chart.

## 📥 Download Model Files (Required)
Due to GitHub's file size limits, the trained model files are hosted externally.
**You must download BOTH files below and place/extract them in the root folder of this project for the app to work.**

### 1. Model Weights (For Grad-CAM)
- **File:** `KneeOA_temp_weights.h5`
- **Action:** Download and place it directly in the `KneeAI-Project` folder.
- **[Download .h5 Weights Here](PEGAR_AQUI_EL_LINK_DEL_ARCHIVO_H5)**

### 2. Inference Model (For Predictions)
- **File:** `KneeOA_GradCAM.zip`
- **Action:** Download, **unzip/extract** it, and ensure the resulting folder is named `KneeOA_GradCAM` (it should contain a `saved_model.pb` file inside). Place the folder in the project root.
- **[Download SavedModel Zip Here](PEGAR_AQUI_EL_LINK_DEL_ZIP)**

## 🧠 Model Architecture
- **Backbone:** EfficientNetB3 (Pre-trained on ImageNet).
- **Optimization:** Hyperparameters (Dropout=0.35, L2=1e-4) tuned using the TPE algorithm (Optuna).
- **Training Strategy:** Two-stage Transfer Learning (Warm-up + Fine-Tuning).
- **Input Resolution:** 300 x 300 pixels.

## 📂 Repository Structure
KneeAI-Project/ ├── app.py # Main Streamlit application ├── requirements.txt # Python dependencies ├── KneeOA_temp_weights.h5 # (Download externally) Grad-CAM weights ├── KneeOA_GradCAM/ # (Download & Unzip externally) Inference Model │ ├── saved_model.pb │ └── variables/ └── README.md # Project documentation
## 🛠️ Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/kevinyepez-1409/KneeAI-Project.git](https://github.com/TU_USUARIO/KneeAI-Project.git)
cd KneeAI-Project
