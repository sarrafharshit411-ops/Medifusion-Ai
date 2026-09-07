# MediFusion AI Datasets

## Overview

This project uses two Kaggle datasets. **They are NOT patient-paired multimodal data.**

## 1. Chest X-Ray Images (Pneumonia)

- **Source**: [Kaggle — Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Structure**: ImageFolder with `train/`, `test/`, `val/` splits
- **Classes**: `NORMAL`, `PNEUMONIA`
- **Counts**: train (1341 NORMAL, 3875 PNEUMONIA), test (234 NORMAL, 390 PNEUMONIA), val (8+8)
- **Task**: Binary classification

## 2. Symptom-Based Disease Prediction

- **Source**: [Kaggle — MiltonMacGyver](https://www.kaggle.com/datasets/miltonmacgyver/symptom-based-disease-prediction-dataset)
- **Structure**: Single CSV (`disease_prediction.csv`)
- **Features**: 15 binary symptom columns: fever, cough, headache, nausea, vomiting, fatigue, sore_throat, chills, body_pain, loss_of_appetite, abdominal_pain, diarrhea, sweating, rapid_breathing, dizziness
- **Target**: `label` column
- **Classes**: Malaria, Pneumonia, Typhoid (each 1666 samples — perfectly balanced)
- **No missing values**

## Important Note

> There is no shared patient ID linking a chest X-ray to a symptom record.
> The multimodal fusion demonstrates the *architecture* for combining modalities,
> not a clinically validated patient-level pipeline.

## Clinical Data

The "clinical data" modality uses **synthetic features** generated from symptom correlations
for educational/demonstration purposes. In a real system, actual patient clinical records
would be used.
