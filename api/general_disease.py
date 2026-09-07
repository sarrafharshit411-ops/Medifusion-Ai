"""
MediFusion AI — General Disease Diagnostic & Pharmacological Recommendation Router
Endpoints for 41-disease diagnosis, 132-symptom inventory, NLP narrative extraction,
and clinical medication & self-care recommendations.
"""

import re
import json
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from api.model_loader import registry
from src.models.general_disease_model import GeneralDiseaseEnsemble

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/general-disease", tags=["General Disease Diagnosis & Medication"])


# ─── Pydantic Request & Response Schemas ───
class GeneralSymptomPredictRequest(BaseModel):
    symptoms: List[str] = Field(
        ..., 
        description="List of symptom feature identifiers selected from the 132-symptom catalog",
        example=["continuous_sneezing", "chills", "fatigue", "cough", "high_fever"]
    )
    top_k: Optional[int] = Field(default=5, ge=1, le=10, description="Number of differential diagnoses to return")


class DifferentialDiagnosis(BaseModel):
    disease: str
    confidence: float
    probability: float
    triage_level: str
    triage_color: str
    triage_action: str
    specialist: str
    primary_medications: str
    immediate_symptom_relief: str
    alternative_relief: str
    self_care_lifestyle: str
    contraindications_warnings: List[str]


class GeneralDiseasePredictionResponse(BaseModel):
    status: str
    selected_symptoms_count: int
    selected_symptoms: List[str]
    primary_diagnosis: DifferentialDiagnosis
    differential_diagnoses: List[DifferentialDiagnosis]
    critical_warning: Optional[str] = None


class NarrativeExtractRequest(BaseModel):
    text: str = Field(
        ..., 
        description="Patient's subjective description of symptoms in plain natural language",
        example="I have been suffering from high fever, shaking chills, and a severe headache with nausea for two days."
    )


class ExtractedSymptomItem(BaseModel):
    id: str
    label: str
    category: str


class NarrativeExtractResponse(BaseModel):
    raw_text: str
    extracted_symptoms: List[ExtractedSymptomItem]
    count: int


# ─── Synonyms & Keyword Mapping for NLP Extraction ───
COMMON_SYNONYMS: Dict[str, List[str]] = {
    "itching": ["itch", "itchy", "scratching", "pruritus"],
    "skin_rash": ["rash", "skin eruption", "red bumps on skin", "skin redness"],
    "continuous_sneezing": ["sneezing", "sneeze", "frequent sneezing"],
    "shivering": ["shivering", "trembling", "shivers", "rigors"],
    "chills": ["chills", "feeling cold", "cold shivers"],
    "joint_pain": ["joint pain", "joints hurt", "arthralgia", "knee pain", "elbow pain", "wrist pain"],
    "stomach_pain": ["stomach pain", "stomach ache", "belly pain", "abdominal pain", "tummy ache"],
    "acidity": ["acidity", "acid reflux", "heartburn", "acidic burp"],
    "ulcers_on_tongue": ["tongue ulcer", "mouth ulcer", "canker sore"],
    "vomiting": ["vomiting", "throwing up", "vomit", "emesis"],
    "burning_micturition": ["burning urine", "pain when peeing", "dysuria", "burning pee"],
    "fatigue": ["fatigue", "exhausted", "tired", "tiredness", "weakness", "lethargic", "no energy"],
    "weight_loss": ["losing weight", "weight loss", "unintended weight drop"],
    "restlessness": ["restless", "restlessness", "agitated"],
    "lethargy": ["lethargic", "sluggish", "drowsy"],
    "cough": ["cough", "coughing", "dry cough", "hacking cough"],
    "high_fever": ["high fever", "burning up", "severe fever", "high temp", "high temperature"],
    "mild_fever": ["mild fever", "slight fever", "low fever", "warm body"],
    "breathlessness": ["breathless", "short of breath", "difficulty breathing", "dyspnea", "gasping"],
    "sweating": ["sweating", "sweats", "night sweats", "perspiring heavily"],
    "headache": ["headache", "head ache", "head throbbing", "migraine", "severe head pain"],
    "yellowish_skin": ["yellow skin", "jaundice skin", "yellowish"],
    "dark_urine": ["dark urine", "brown urine", "tea colored urine"],
    "nausea": ["nausea", "nauseous", "feeling sick", "queasy"],
    "loss_of_appetite": ["loss of appetite", "no appetite", "not hungry", "cannot eat"],
    "constipation": ["constipated", "constipation", "hard stools"],
    "diarrhoea": ["diarrhea", "diarrhoea", "loose motions", "loose stools", "watery stools"],
    "chest_pain": ["chest pain", "pain in chest", "chest tightness", "chest pressure"],
    "fast_heart_rate": ["fast heart rate", "heart racing", "palpitations", "rapid heartbeat"],
    "dizziness": ["dizzy", "dizziness", "lightheaded", "lightheadedness", "vertigo", "spinning"],
    "blurred_and_distorted_vision": ["blurred vision", "blurry vision", "distorted vision"],
    "phlegm": ["phlegm", "mucus", "sputum"],
    "throat_irritation": ["sore throat", "throat irritation", "scratchy throat", "throat pain"],
    "runny_nose": ["runny nose", "running nose", "nasal discharge"],
    "congestion": ["congested", "congestion", "stuffy nose", "blocked nose"],
    "neck_pain": ["neck pain", "stiff neck", "neck ache"],
    "cramps": ["muscle cramps", "cramps", "cramping"],
    "stiff_neck": ["stiff neck", "nuchal rigidity", "cannot turn neck"],
    "loss_of_smell": ["loss of smell", "cannot smell", "anosmia"],
    "foul_smell_of urine": ["smelly urine", "foul smelling urine"],
    "muscle_pain": ["muscle pain", "body ache", "body pain", "myalgia", "sore muscles"],
    "red_spots_over_body": ["red spots", "red dots on skin", "petechiae"],
    "swelled_lymph_nodes": ["swollen lymph nodes", "swollen glands", "lumps in neck"]
}


# ─── Endpoints ───

@router.get("/symptoms")
async def get_all_symptoms() -> Dict[str, Any]:
    """Retrieve all 132 symptoms grouped by anatomical body system for UI browsing."""
    meta = registry.general_disease_symptoms
    if not meta:
        raise HTTPException(status_code=503, detail="General disease symptom metadata not loaded.")
    return meta


@router.post("/nlp-extract", response_model=NarrativeExtractResponse)
async def extract_symptoms_from_narrative(req: NarrativeExtractRequest):
    """
    Extract matching symptom IDs from patient narrative text using keyword & synonym analysis.
    """
    text = req.text.lower().strip()
    meta = registry.general_disease_symptoms
    if not meta:
        raise HTTPException(status_code=503, detail="Symptom metadata not loaded.")

    all_symptoms = meta.get("all_symptoms", [])
    symptom_lookup = {item["id"]: item for item in all_symptoms}

    matched_ids = set()

    # 1. Check synonym dictionary
    for sym_id, patterns in COMMON_SYNONYMS.items():
        if sym_id in symptom_lookup:
            for pattern in patterns:
                if re.search(r"\b" + re.escape(pattern) + r"\b", text):
                    matched_ids.add(sym_id)
                    break

    # 2. Check direct formatted symptom labels and raw IDs
    for item in all_symptoms:
        sym_id = item["id"]
        label = item["label"].lower()
        cleaned_id = sym_id.replace("_", " ")

        if re.search(r"\b" + re.escape(label) + r"\b", text) or re.search(r"\b" + re.escape(cleaned_id) + r"\b", text):
            matched_ids.add(sym_id)

    # Convert to response objects
    extracted_items = [
        ExtractedSymptomItem(
            id=sym_id,
            label=symptom_lookup[sym_id]["label"],
            category=symptom_lookup[sym_id]["category"]
        )
        for sym_id in sorted(matched_ids)
    ]

    return NarrativeExtractResponse(
        raw_text=req.text,
        extracted_symptoms=extracted_items,
        count=len(extracted_items)
    )


@router.post("/predict", response_model=GeneralDiseasePredictionResponse)
async def predict_disease_and_medications(req: GeneralSymptomPredictRequest):
    """
    Run 41-class ensemble inference on selected symptoms and return differential diagnoses,
    confidence metrics, triage urgency, medication guidelines, and contraindication alerts.
    """
    model: GeneralDiseaseEnsemble = registry.general_disease_model
    kb: Dict[str, Any] = registry.general_disease_kb
    meta: Dict[str, Any] = registry.general_disease_symptoms

    if not model or not kb or not meta:
        raise HTTPException(status_code=503, detail="General disease diagnostic engine is not loaded.")

    feature_names = model.feature_names
    selected_set = set(req.symptoms)

    if not selected_set:
        raise HTTPException(status_code=400, detail="Please select at least one symptom.")

    # Construct binary feature vector
    vec = [1 if feat in selected_set else 0 for feat in feature_names]

    # Predict top-k
    top_k = min(max(req.top_k, 1), 10)
    predictions = model.predict_top_k(vec, k=top_k)

    # Format diagnoses
    diff_diagnoses: List[DifferentialDiagnosis] = []
    critical_warning = None

    for p in predictions:
        d_name = p["disease"]
        card = kb.get(d_name, {})

        diag = DifferentialDiagnosis(
            disease=d_name,
            confidence=p["confidence"],
            probability=p["probability"],
            triage_level=card.get("triage_level", "MODERATE"),
            triage_color=card.get("triage_color", "#38bdf8"),
            triage_action=card.get("triage_action", "Consult a physician for clinical confirmation."),
            specialist=card.get("specialist", "General Physician / Internist"),
            primary_medications=card.get("primary_medications", "Consult your physician."),
            immediate_symptom_relief=card.get("immediate_symptom_relief", "Adequate rest, hydration, and monitoring."),
            alternative_relief=card.get("alternative_relief", "Supportive care."),
            self_care_lifestyle=card.get("self_care_lifestyle", "Rest and balanced nutrition."),
            contraindications_warnings=card.get("contraindications_warnings", [])
        )
        diff_diagnoses.append(diag)

    primary = diff_diagnoses[0]

    # Check for critical emergency triage
    if primary.triage_level == "CRITICAL EMERGENCY":
        critical_warning = f"CRITICAL: Symptoms align with potential {primary.disease.upper()}. {primary.triage_action}"
    elif primary.contraindications_warnings:
        critical_warning = primary.contraindications_warnings[0]

    return GeneralDiseasePredictionResponse(
        status="success",
        selected_symptoms_count=len(selected_set),
        selected_symptoms=list(selected_set),
        primary_diagnosis=primary,
        differential_diagnoses=diff_diagnoses,
        critical_warning=critical_warning
    )


@router.get("/knowledge/{disease_name}")
async def get_disease_knowledge(disease_name: str) -> Dict[str, Any]:
    """Retrieve the pharmacology and medical guidance profile for a given disease."""
    kb: Dict[str, Any] = registry.general_disease_kb
    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded.")

    # Match normalized
    d_norm = disease_name.strip().lower()
    for name, data in kb.items():
        if name.lower() == d_norm or d_norm in name.lower():
            return data

    raise HTTPException(status_code=404, detail=f"No knowledge base entry found for disease: {disease_name}")
