"""
MediFusion AI — General Disease Classifier & Pharmacological Knowledge Base Trainer
Trains a high-performance ensemble (Random Forest + XGBoost) on the 41-disease, 132-symptom dataset.
Compiles a clinical knowledge base mapping all 41 diseases to OTC/prescription medicines,
symptom relief, lifestyle/diet self-care, emergency warnings, and recommended specialists.
Categorizes all 132 symptoms into 8 anatomical body systems for intuitive clinical navigation.
"""

import os
import sys
import json
import re
import csv
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── 1. ANATOMICAL BODY SYSTEM CATEGORIZATION FOR 132 SYMPTOMS ───
BODY_SYSTEM_CATEGORIES = {
    "Respiratory": [
        "continuous_sneezing", "cough", "breathlessness", "phlegm", 
        "throat_irritation", "sinus_pressure", "runny_nose", "congestion", 
        "chest_pain", "mucoid_sputum", "rusty_sputum", "blood_in_sputum", 
        "patches_in_throat"
    ],
    "Gastrointestinal & Abdominal": [
        "stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", 
        "indigestion", "nausea", "loss_of_appetite", "constipation", 
        "abdominal_pain", "diarrhoea", "passage_of_gases", "belly_pain", 
        "stomach_bleeding", "distention_of_abdomen", "swelling_of_stomach",
        "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", 
        "irritation_in_anus"
    ],
    "Skin, Hair & Nails": [
        "itching", "skin_rash", "nodal_skin_eruptions", "yellowish_skin", 
        "internal_itching", "red_spots_over_body", "dischromic _patches", 
        "pus_filled_pimples", "blackheads", "scurring", "skin_peeling", 
        "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", 
        "blister", "red_sore_around_nose", "yellow_crust_ooze", "brittle_nails",
        "bruising"
    ],
    "Neurological & Psychological": [
        "headache", "dizziness", "loss_of_balance", "unsteadiness", 
        "weakness_of_one_body_side", "loss_of_smell", "spinning_movements", 
        "altered_sensorium", "lack_of_concentration", "visual_disturbances", 
        "blurred_and_distorted_vision", "slurred_speech", "anxiety", 
        "mood_swings", "restlessness", "lethargy", "depression", 
        "irritability", "coma"
    ],
    "Musculoskeletal & Joints": [
        "joint_pain", "muscle_wasting", "back_pain", "weakness_in_limbs", 
        "neck_pain", "cramps", "knee_pain", "hip_joint_pain", 
        "muscle_weakness", "stiff_neck", "swelling_joints", 
        "movement_stiffness", "painful_walking", "muscle_pain"
    ],
    "Cardiovascular & Circulatory": [
        "fast_heart_rate", "palpitations", "cold_hands_and_feets", 
        "prominent_veins_on_calf", "swollen_blood_vessels"
    ],
    "Endocrine, Systemic & Vital": [
        "shivering", "chills", "fatigue", "weight_gain", "weight_loss", 
        "high_fever", "sweating", "dehydration", "mild_fever", "malaise", 
        "obesity", "swollen_legs", "puffy_face_and_eyes", "enlarged_thyroid", 
        "swollen_extremeties", "excessive_hunger", "drying_and_tingling_lips", 
        "toxic_look_(typhos)", "increased_appetite", "polyuria", "sunken_eyes", 
        "swelled_lymph_nodes", "fluid_overload", "acute_liver_failure"
    ],
    "Urinary, Sensory & Other": [
        "burning_micturition", "spotting_ urination", "dark_urine", 
        "yellow_urine", "yellowing_of_eyes", "redness_of_eyes", 
        "pain_behind_the_eyes", "watering_from_eyes", "bladder_discomfort", 
        "foul_smell_of urine", "continuous_feel_of_urine", "abnormal_menstruation", 
        "extra_marital_contacts", "family_history", "receiving_blood_transfusion", 
        "receiving_unsterile_injections", "history_of_alcohol_consumption", 
        "irregular_sugar_level"
    ]
}

def format_symptom_label(col_name: str) -> str:
    """Format raw column name into a clean, human-readable medical symptom label."""
    clean = col_name.replace("_", " ").strip()
    clean = re.sub(r"\s+", " ", clean)
    return clean.title()

def build_symptom_metadata(feature_columns):
    """Build a rich searchable directory for all symptoms."""
    feature_set = set(feature_columns)
    categorized = {}
    assigned_features = set()

    for category, symptoms in BODY_SYSTEM_CATEGORIES.items():
        matched = []
        for sym in symptoms:
            if sym in feature_set:
                matched.append({
                    "id": sym,
                    "label": format_symptom_label(sym),
                    "category": category
                })
                assigned_features.add(sym)
        categorized[category] = matched

    # Assign any remaining features to 'General / Other'
    remaining = feature_set - assigned_features
    if remaining:
        categorized["General & Other"] = [
            {"id": sym, "label": format_symptom_label(sym), "category": "General & Other"}
            for sym in sorted(remaining)
        ]

    # Flat lookup list
    flat_list = []
    for cat, items in categorized.items():
        flat_list.extend(items)

    return {
        "categories": categorized,
        "all_symptoms": flat_list,
        "feature_columns": feature_columns
    }

# ─── 2. SPECIALIST MAPPING PER DISEASE ───
SPECIALIST_MAP = {
    "Fungal infection": "Dermatologist",
    "Allergy": "Allergist / Immunologist",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Hepatologist / Gastroenterologist",
    "Drug Reaction": "Dermatologist / Allergist",
    "Peptic ulcer diseae": "Gastroenterologist",
    "AIDS": "Infectious Disease Specialist / Immunologist",
    "Diabetes ": "Endocrinologist / Diabetologist",
    "Gastroenteritis": "Gastroenterologist / General Physician",
    "Bronchial Asthma": "Pulmonologist / Allergist",
    "Hypertension ": "Cardiologist / Internal Medicine",
    "Migraine": "Neurologist",
    "Cervical spondylosis": "Orthopedic Surgeon / Neurologist",
    "Paralysis (brain hemorrhage)": "Neurologist / Neurosurgeon (EMERGENCY)",
    "Jaundice": "Hepatologist / Gastroenterologist",
    "Malaria": "Infectious Disease Specialist / General Physician",
    "Chicken pox": "General Physician / Pediatrician",
    "Dengue": "Infectious Disease Specialist / Hematologist",
    "Typhoid": "Infectious Disease Specialist / General Physician",
    "hepatitis A": "Hepatologist / Gastroenterologist",
    "Hepatitis B": "Hepatologist / Gastroenterologist",
    "Hepatitis C": "Hepatologist / Gastroenterologist",
    "Hepatitis D": "Hepatologist / Gastroenterologist",
    "Hepatitis E": "Hepatologist / Gastroenterologist",
    "Alcoholic hepatitis": "Hepatologist / Addiction Specialist",
    "Tuberculosis": "Pulmonologist / Infectious Disease Specialist",
    "Common Cold": "General Physician / Primary Care",
    "Pneumonia": "Pulmonologist / Critical Care Specialist",
    "Dimorphic hemmorhoids(piles)": "Colorectal Surgeon / Proctologist",
    "Heart attack": "Cardiologist / Emergency Medicine (IMMEDIATE 911)",
    "Varicose veins": "Vascular Surgeon / Phlebologist",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist / Emergency Physician",
    "Osteoarthristis": "Orthopedic Specialist / Rheumatologist",
    "Arthritis": "Rheumatologist",
    "(vertigo) Paroymsal  Positional Vertigo": "ENT Specialist / Neurologist",
    "Acne": "Dermatologist",
    "Urinary tract infection": "Urologist / Nephrologist",
    "Psoriasis": "Dermatologist",
    "Impetigo": "Dermatologist / Pediatrician"
}

# ─── 3. SEVERITY / TRIAGE MAPPING PER DISEASE ───
TRIAGE_LEVELS = {
    "Heart attack": {"level": "CRITICAL EMERGENCY", "color": "#ef4444", "action": "Call Emergency Medical Services (911) immediately. Do not drive."},
    "Paralysis (brain hemorrhage)": {"level": "CRITICAL EMERGENCY", "color": "#ef4444", "action": "Urgent emergency stroke center evaluation required immediately."},
    "Dengue": {"level": "HIGH / MONITOR", "color": "#f97316", "action": "Immediate medical consultation and serial platelet monitoring required. AVOID NSAIDs/Aspirin."},
    "Pneumonia": {"level": "HIGH", "color": "#f97316", "action": "Chest radiography, oxygen saturation monitoring, and prompt antibiotic/antiviral treatment needed."},
    "Tuberculosis": {"level": "HIGH", "color": "#f97316", "action": "Chest X-ray, sputum AFB culture, and isolation precautions. Directly Observed Therapy required."},
    "AIDS": {"level": "CHRONIC SERIOUS", "color": "#a855f7", "action": "Lifelong antiretroviral therapy (ART) and CD4/viral load monitoring under specialist care."},
    "Typhoid": {"level": "MODERATE / HIGH", "color": "#eab308", "action": "Blood culture, hydration, and targeted antibiotic therapy."},
    "Malaria": {"level": "MODERATE / HIGH", "color": "#eab308", "action": "Blood smear / rapid test, artemisinin-based combination therapy, hydration."},
    "Hypoglycemia": {"level": "URGENT", "color": "#f97316", "action": "Administer 15-20g fast-acting glucose immediately. If unaroused, call emergency services."},
    "Common Cold": {"level": "MILD / SELF-LIMITING", "color": "#10b981", "action": "Supportive care, hydration, rest, OTC symptom relief."},
    "Allergy": {"level": "MILD TO MODERATE", "color": "#10b981", "action": "Antihistamines, trigger avoidance. Seek urgent care if throat tightness occurs."}
}

def compile_knowledge_base(prognosis_list):
    """Compile comprehensive clinical & medication advice for all 41 diseases."""
    # Load medications.csv
    med_df = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
    med_map = {}
    for _, row in med_df.iterrows():
        disease = str(row["Disease"]).strip()
        sugg = str(row["Suggestion"]).strip(' "')
        med_map[disease.lower()] = sugg

    # Load disease_medicine_recommendation.csv if available
    rec_path = os.path.join(DATA_DIR, "disease_medicine_recommendation.csv")
    rec_map = {}
    if os.path.exists(rec_path):
        try:
            with open(rec_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        d_name = row[0].strip().lower()
                        rec_map[d_name] = {
                            "immediate_relief": row[1].strip() if len(row) > 1 else "",
                            "alternative_relief": row[2].strip() if len(row) > 2 else "",
                            "self_care": row[3].strip() if len(row) > 3 else ""
                        }
        except Exception as e:
            print("Warning loading recommendation CSV:", e)

    knowledge_base = {}

    for d in prognosis_list:
        d_clean = d.strip()
        d_lower = d_clean.lower()
        d_norm = re.sub(r"[^a-z0-9]", "", d_lower)

        # Match medication suggestion
        suggestion = ""
        for k, v in med_map.items():
            k_norm = re.sub(r"[^a-z0-9]", "", k)
            if k_norm in d_norm or d_norm in k_norm:
                suggestion = v
                break

        # Match recommendation details
        immediate_relief = ""
        alt_relief = ""
        self_care = ""
        for k, v in rec_map.items():
            k_norm = re.sub(r"[^a-z0-9]", "", k)
            if k_norm in d_norm or d_norm in k_norm:
                immediate_relief = v.get("immediate_relief", "")
                alt_relief = v.get("alternative_relief", "")
                self_care = v.get("self_care", "")
                break

        # Fill sensible defaults if missing
        if not suggestion:
            suggestion = f"Consult a qualified {SPECIALIST_MAP.get(d, 'Physician')} for a comprehensive diagnostic and medical management plan."
        if not immediate_relief:
            immediate_relief = suggestion
        if not self_care:
            self_care = "Ensure adequate rest, hydration, balanced nutrition, and symptom tracking. Avoid self-prescribed antibiotics."

        # Specific safety contraindications
        warnings = []
        if "dengue" in d_lower:
            warnings.append("ABSOLUTELY AVOID NSAIDs (Ibuprofen, Aspirin, Naproxen) as they dramatically increase internal hemorrhage/bleeding risk.")
        elif "heart attack" in d_lower or "infarction" in d_lower:
            warnings.append("CRITICAL MEDICAL EMERGENCY: Chew one adult aspirin (325mg) if not allergic, and call 911 / emergency services immediately.")
        elif "paralysis" in d_lower or "stroke" in d_lower:
            warnings.append("TIME-CRITICAL EMERGENCY: Do not give food or fluids. Seek emergency hospital care within the golden window (<3-4.5 hours).")
        elif "tuberculosis" in d_lower:
            warnings.append("Infectious transmission risk: Wear an N95 mask, isolate in well-ventilated space until cleared non-infectious by a physician.")
        elif "hepatitis" in d_lower:
            warnings.append("Strictly avoid all alcohol and hepatotoxic substances. Avoid excessive acetaminophen/paracetamol.")
        elif "peptic ulcer" in d_lower:
            warnings.append("Avoid NSAIDs, aspirin, caffeine, and smoking which severely irritate the gastrointestinal mucosal lining.")

        triage = TRIAGE_LEVELS.get(d_clean, {
            "level": "MODERATE", 
            "color": "#38bdf8", 
            "action": f"Schedule an in-person clinical consultation with a {SPECIALIST_MAP.get(d_clean, 'Medical Specialist')}."
        })

        knowledge_base[d_clean] = {
            "disease": d_clean,
            "specialist": SPECIALIST_MAP.get(d_clean, "General Physician / Internist"),
            "triage_level": triage["level"],
            "triage_color": triage["color"],
            "triage_action": triage["action"],
            "primary_medications": suggestion,
            "immediate_symptom_relief": immediate_relief,
            "alternative_relief": alt_relief,
            "self_care_lifestyle": self_care,
            "contraindications_warnings": warnings
        }

    return knowledge_base

sys.path.insert(0, PROJECT_ROOT)
from src.models.general_disease_model import GeneralDiseaseEnsemble

def train():
    print("=" * 60)
    print("MediFusion AI: Training General Disease 41-Class Ensemble")
    print("=" * 60)

    train_path = os.path.join(DATA_DIR, "Training.csv")
    test_path = os.path.join(DATA_DIR, "Testing.csv")

    df_train = pd.read_csv(train_path).drop(columns=["Unnamed: 133"], errors="ignore")
    df_test = pd.read_csv(test_path).drop(columns=["Unnamed: 133"], errors="ignore")

    feature_cols = [c for c in df_train.columns if c != "prognosis"]
    print(f"Loaded {len(df_train)} training records across {len(feature_cols)} symptom features.")
    print(f"Loaded {len(df_test)} test evaluation records.")

    X_train = df_train[feature_cols]
    y_train = df_train["prognosis"].str.strip()

    X_test = df_test[feature_cols]
    y_test = df_test["prognosis"].str.strip()

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    classes = list(le.classes_)
    print(f"Identified {len(classes)} unique disease classes.")

    # 1. Train Random Forest
    print("\nTraining Random Forest (150 estimators, balanced bootstrap)...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_enc)
    rf_acc = accuracy_score(y_test_enc, rf.predict(X_test))
    print(f"[OK] Random Forest Test Accuracy: {rf_acc * 100:.2f}%")

    # 2. Train XGBoost
    print("Training XGBoost Classifier (max_depth=5, lr=0.08)...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=5,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train_enc)
    xgb_acc = accuracy_score(y_test_enc, xgb_clf.predict(X_test))
    print(f"[OK] XGBoost Test Accuracy: {xgb_acc * 100:.2f}%")

    # 3. Create Ensemble
    ensemble = GeneralDiseaseEnsemble(rf, xgb_clf, le, feature_cols)
    ens_preds = ensemble.predict(X_test)
    ens_acc = accuracy_score(y_test, ens_preds)
    print(f"\n[+] Calibrated Ensemble Test Accuracy: {ens_acc * 100:.2f}%")

    # Top-3 Accuracy
    top3_correct = 0
    for i in range(len(X_test)):
        row_vec = X_test.iloc[i].values
        top3 = [r["disease"] for r in ensemble.predict_top_k(row_vec, k=3)]
        if y_test.iloc[i] in top3:
            top3_correct += 1
    print(f"[+] Ensemble Top-3 Diagnostic Accuracy: {top3_correct / len(X_test) * 100:.2f}%")

    # 4. Compile Knowledge Base
    print("\nCompiling Pharmacological Knowledge Base & Clinical Recommendations...")
    knowledge_base = compile_knowledge_base(classes)
    kb_path = os.path.join(MODELS_DIR, "disease_knowledge_base.json")
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, indent=2)
    print(f"[OK] Saved disease knowledge base to {kb_path}")

    # 5. Build Symptom Metadata
    print("Structuring 132 symptoms into 8 anatomical systems...")
    symptom_meta = build_symptom_metadata(feature_cols)
    symptom_meta_path = os.path.join(MODELS_DIR, "symptoms_132_metadata.json")
    with open(symptom_meta_path, "w", encoding="utf-8") as f:
        json.dump(symptom_meta, f, indent=2)
    print(f"[OK] Saved symptom metadata to {symptom_meta_path}")

    # 6. Save Model Package
    model_save_path = os.path.join(MODELS_DIR, "general_disease_ensemble.joblib")
    joblib.dump(ensemble, model_save_path)
    print(f"[OK] Saved trained ensemble model to {model_save_path}")

    print("\nTraining and asset compilation complete!")

if __name__ == "__main__":
    train()
