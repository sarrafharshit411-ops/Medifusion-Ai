# MediFusion AI 🧬
### *Multimodal Clinical Decision-Support & Disease-Assessment System*

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-185a9d.svg?style=flat)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/Explainability-Grad--CAM%20%7C%20SHAP-blueviolet.svg?style=flat)](https://github.com/slundberg/shap)
[![License](https://img.shields.io/badge/License-Educational%20Use%20Only-orange.svg?style=flat)](#license)

---

## ⚠️ Important Mentions & Medical Disclaimers

> [!CAUTION]
> **EDUCATIONAL & RESEARCH PROTOTYPE ONLY**  
> MediFusion AI is an academic and research proof-of-concept designed to explore multimodal deep learning, explainability, and clinical decision support.  
> - **NOT A MEDICAL DEVICE:** This system is **NOT** an FDA/CE-cleared diagnostic medical device.
> - **NOT A SUBSTITUTE FOR PROFESSIONAL JUDGMENT:** It should **NEVER** be used as a definitive diagnostic tool or as a replacement for clinical consultation with a qualified, licensed healthcare professional.
> - **EMERGENCY NOTICE:** In the event of a medical emergency, immediately seek urgent medical attention from emergency services.

### Dataset Acknowledgement & Architecture Disclosures
- **Cohort Notice:** The imaging dataset (Chest X-Rays) and symptom datasets originate from independent open-source repositories and are **not patient-paired** (i.e., they do not share a common patient ID or longitudinal electronic health record).
- **Synthetic Physiological Features:** The clinical vitals modality (blood pressure, oxygen saturation, temperature, respiratory rate, etc.) integrates synthetic parameters to demonstrate how multimodal gating handles tabular clinical vitals alongside unstructured imaging.
- **Multimodal Fusion Demonstration:** The fusion network evaluates architectural feasibility (Gated Attention Fusion with modality masking for handling missing clinical data) rather than an FDA-validated clinical pipeline.

---

## 🌟 Key Highlights & Capabilities

MediFusion AI is built on a modular tri-stream deep learning architecture that fuses unstructured radiologic imagery with structured clinical biomarkers and differential symptom indicators:

1. **Radiological Assessment (ResNet-18 + Grad-CAM)**
   - High-throughput chest radiograph classification (Normal vs. Pneumonia) fine-tuned on clinical X-rays.
   - Built-in **Radiograph Validation Filter** (`image_validator.py`) to detect and reject invalid/non-radiologic uploads, corrupted files, and inverted contrast errors.
   - **Grad-CAM (Gradient-weighted Class Activation Mapping)** generates interpretable spatial heatmaps highlighting the exact pulmonary regions driving model inference.

2. **General Disease 41-Class Diagnostic Ensemble**
   - Comprehensive multi-class diagnostic engine covering **41 systemic diseases** and **132 distinct symptoms**.
   - Generates confidence-ranked differential diagnoses, matched symptom breakdowns, and probability distributions.

3. **Integrated Clinical Pharmacology & Knowledge Base**
   - Actionable clinical pharmacology recommendations, disease overviews, lifestyle precautions, and standard treatment context extracted from clinical knowledge bases.

4. **Gated Attention Multimodal Fusion**
   - Learnable cross-modality attention mechanism fusing imaging embeddings (512-d), symptom vectors (15-d), and clinical vitals (22-d).
   - Robust **Modality Masking**: dynamically handles partial or missing clinical data (e.g., assessing a patient when only symptoms or only imaging is available).

5. **SHAP Tree Interpretability**
   - SHAP (SHapley Additive exPlanations) values decompose tabular clinical models to attribute exact credit and risk factors to specific symptoms and physiological vitals.

6. **Interactive Multi-Page Web Platform**
   - Modern medical dashboard built with Vanilla CSS/JS and Chart.js, featuring real-time diagnostic reporting, interactive symptom pickers, and visual heatmaps.

---

## 🏗️ System Architecture

```
                                  ┌────────────────────────┐
                                  │   Patient Evaluation   │
                                  └───────────┬────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         │                                    │                                    │
         ▼                                    ▼                                    ▼
┌──────────────────┐               ┌──────────────────────┐             ┌──────────────────────┐
│   Chest X-Ray    │               │     132 Symptoms     │             │ 7 Physiological Vitals│
│ (DICOM/PNG/JPEG) │               │  (Differential Match)│             │ (BP, O2, HR, Temp, RR)│
└────────┬─────────┘               └──────────┬───────────┘             └──────────┬───────────┘
         │                                    │                                    │
    Validation                           Preprocessing                         Standard
    & ResNet18                            & Embedding                           Scaling
         │                                    │                                    │
   512-d Embedding                      15-d Embedding                       22-d Embedding
         │                                    │                                    │
         └────────────────────────┬───────────┴────────────────────────────────────┘
                                  │
                                  ▼
                     ┌───────────────────────────┐
                     │   Gated Attention Fusion  │  ◄── Modality Availability Mask
                     │  (Cross-Modal Synthesis)  │
                     └────────────┬──────────────┘
                                  │
                                  ▼
                     ┌───────────────────────────┐
                     │   Differential Diagnosis   │
                     │  & Clinical Risk Profile  │
                     └────────────┬──────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
    ┌───────────────────────────┐     ┌───────────────────────────┐
    │     Grad-CAM Heatmap      │     │      SHAP Risk Credit     │
    │  (Radiology Localization) │     │  (Feature-Level Impact)   │
    └───────────────────────────┘     └───────────────────────────┘
```

---

## 💻 Tech Stack

| Domain | Technology | Details |
|--------|-----------|---------|
| **Deep Learning** | PyTorch & Torchvision | ResNet-18 Transfer Learning & Custom Gated Attention |
| **Machine Learning** | XGBoost & Scikit-Learn | Calibrated multi-class classification and scalers |
| **Model Explainability**| Grad-CAM & SHAP | Spatial activations and Shapley feature attribution |
| **Backend API** | FastAPI, Uvicorn, Pydantic | Asynchronous, validated RESTful endpoints |
| **Data Processing** | NumPy, Pandas, Joblib | Efficient tensor manipulation & pipeline serialization |
| **Frontend UI** | HTML5, Modern Vanilla CSS, JS | Multi-page responsive UI with Chart.js visualization |
| **Containerization** | Docker & Docker Compose | Multi-platform containerized deployment |

---

## 📁 Repository Structure

```
.
├── api/
│   ├── main.py                      # FastAPI application entry point
│   ├── model_loader.py              # Singleton model registry & cache
│   ├── general_disease.py           # 41-class general disease API routes
│   └── schemas.py                   # Pydantic request/response schemas
├── configs/
│   └── config.yaml                  # Global training and model hyperparameters
├── data/
│   ├── Training.csv                 # 41-disease training dataset
│   ├── Testing.csv                  # Multi-class evaluation dataset
│   ├── disease_medicine_recommendation.csv # Pharmacology guidance data
│   ├── disease_symptom.csv          # Symptom-to-disease mappings
│   └── medications.csv              # Medication reference catalog
├── frontend/
│   ├── index.html                   # MediFusion landing page & system overview
│   ├── image-analysis.html          # Radiology Grad-CAM workspace
│   ├── symptom-analysis.html        # 132-symptom diagnosis & pharmacology hub
│   ├── multimodal.html              # Multi-stream gated fusion workspace
│   ├── insights.html                # Model performance & SHAP analytics
│   ├── about.html                   # Research documentation & ethics
│   ├── css/                         # Modular CSS design system
│   └── js/                          # Client-side API connectors & chart rendering
├── models/
│   ├── image_model.pth              # Trained ResNet-18 radiograph weights
│   ├── symptom_model.joblib         # XGBoost symptom classifier
│   ├── clinical_model.joblib        # XGBoost clinical vitals classifier
│   ├── fusion_gated.pth             # Gated Attention Multimodal network
│   ├── general_disease_ensemble.joblib # 41-disease ensemble model
│   ├── disease_knowledge_base.json  # Comprehensive clinical knowledge base
│   └── symptoms_132_metadata.json   # Symptom definitions and metadata
├── src/
│   ├── data/                        # Dataset loaders and transformers
│   ├── explainability/              # Grad-CAM and SHAP implementations
│   ├── models/                      # PyTorch and XGBoost architecture definitions
│   ├── training/                    # Training routines and metrics computation
│   └── utils/                       # Radiograph validator and helpers
├── Dockerfile                       # Production container specification
├── docker-compose.yml               # Service orchestration
├── requirements.txt                 # Project dependencies
└── run_training.py                  # End-to-end model training orchestrator
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/sarrafharshit411-ops/Medifusion-Ai.git
cd Medifusion-Ai
```

### 2. Set Up Environment & Install Dependencies
It is recommended to use Python 3.10+ in a virtual environment:
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Web Application
Start the FastAPI server:
```bash
uvicorn api.main:app --reload --port 8000
```

Once running, launch your browser and navigate to:
- **Web Dashboard:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc API Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔌 API Endpoints Reference

| Route | Method | Description |
|---|---|---|
| `/api/health` | `GET` | Health status and verification of loaded model weights |
| `/api/predict/image` | `POST` | Chest X-Ray inference (Normal vs. Pneumonia) with radiograph validation |
| `/api/predict/symptoms` | `POST` | Core symptom-based multi-class probability assessment |
| `/api/predict/clinical` | `POST` | Disease prediction combining symptoms with physiological vitals |
| `/api/predict/multimodal` | `POST` | Unified multimodal inference via Gated Attention Fusion |
| `/api/explain/image` | `POST` | Generates Grad-CAM spatial activation heatmaps |
| `/api/explain/clinical` | `POST` | Computes SHAP values and feature attribution charts |
| `/api/general-disease/predict` | `POST` | 41-class differential diagnosis from 132 symptoms + pharmacology |
| `/api/general-disease/symptoms`| `GET` | Comprehensive list of recognized symptoms and categorical metadata |

---

## 🐳 Docker Deployment

To launch the full system inside an isolated container:

```bash
docker-compose up --build
```
The application will be accessible at `http://localhost:8000`.

---

## 📊 Model Evaluation & Metrics

All evaluation reports and metrics are strictly calculated from validated test partitions:
- **Radiology Stream (ResNet-18):** Macro F1 > 0.88, ROC-AUC > 0.94 on held-out test radiography.
- **Symptom Classifier:** Multi-class log-loss minimization across binary indicators.
- **Multimodal Fusion:** Gated cross-modality synthesis improves classification resilience when individual modalities suffer from noise or absence.

---

## 👤 Author & Acknowledgements

- **Developer:** [Harshit Raj](https://github.com/sarrafharshit411-ops)
- **Repository:** [https://github.com/sarrafharshit411-ops/Medifusion-Ai](https://github.com/sarrafharshit411-ops/Medifusion-Ai)
- **Datasets:**
  - *Chest X-Ray Images (Pneumonia)* by Paul Mooney (Kaggle)
  - *Disease Symptom Prediction Dataset* (Kaggle)

---

## 📄 License

This project is licensed strictly for **Educational and Academic Research Purposes**. It is not licensed or approved for clinical diagnosis or direct medical intervention.

