/**
 * MediFusion AI — Shared Clinical Symptom Ontology Data (36 Symptoms)
 */

window.MediFusionSymptoms = (function () {
    "use strict";

    const CATALOG = [
        // ─── Respiratory (7) ───
        { id: "cough", label: "Persistent Cough", category: "respiratory", core: "cough", keywords: ["cough", "coughing", "throat clearing"] },
        { id: "rapid_breathing", label: "Rapid / Short Breathing (Dyspnea)", category: "respiratory", core: "rapid_breathing", keywords: ["breathing", "breathless", "shortness of breath", "dyspnea", "gasping"] },
        { id: "sore_throat", label: "Sore Throat / Pharyngitis", category: "respiratory", core: "sore_throat", keywords: ["sore throat", "throat pain", "pharyngitis", "scratchy throat"] },
        { id: "chest_tightness", label: "Chest Tightness / Heaviness", category: "respiratory", core: "rapid_breathing", keywords: ["chest tightness", "tight chest", "chest pressure"] },
        { id: "wheezing", label: "Wheezing / Stridor", category: "respiratory", core: "rapid_breathing", keywords: ["wheeze", "wheezing", "stridor", "whistling breath"] },
        { id: "productive_sputum", label: "Productive Sputum / Phlegm", category: "respiratory", core: "cough", keywords: ["sputum", "phlegm", "mucus", "green phlegm", "yellow sputum"] },
        { id: "nasal_congestion", label: "Nasal Congestion / Rhinorrhea", category: "respiratory", core: "sore_throat", keywords: ["nasal", "runny nose", "congestion", "rhinorrhea"] },

        // ─── General & Systemic (8) ───
        { id: "fever", label: "Fever / Elevated Temperature", category: "general", core: "fever", keywords: ["fever", "pyrexia", "hot", "high temperature", "feverish"] },
        { id: "chills", label: "Chills / Rigors / Shivering", category: "general", core: "chills", keywords: ["chills", "shivering", "cold", "rigors", "trembling"] },
        { id: "fatigue", label: "Fatigue / Severe Malaise", category: "general", core: "fatigue", keywords: ["fatigue", "tired", "exhausted", "weakness", "lethargic", "malaise"] },
        { id: "sweating", label: "Profuse Sweating / Diaphoresis", category: "general", core: "sweating", keywords: ["sweat", "sweating", "diaphoresis", "night sweats", "clammy"] },
        { id: "dehydration", label: "Dehydration / Parched Mouth", category: "general", core: "fatigue", keywords: ["dehydration", "thirsty", "dry mouth", "parched"] },
        { id: "weight_loss", label: "Unintended Weight Loss", category: "general", core: "loss_of_appetite", keywords: ["weight loss", "losing weight", "emaciated"] },
        { id: "restlessness", label: "Restlessness / Agitation", category: "general", core: "dizziness", keywords: ["restless", "agitated", "uneasy"] },
        { id: "insomnia", label: "Sleep Disturbance / Insomnia", category: "general", core: "fatigue", keywords: ["sleep", "insomnia", "cannot sleep"] },

        // ─── Neurological & Sensory (6) ───
        { id: "headache", label: "Severe Headache / Cephalea", category: "neurological", core: "headache", keywords: ["headache", "migraine", "head pain", "throbbing head"] },
        { id: "dizziness", label: "Dizziness / Lightheadedness", category: "neurological", core: "dizziness", keywords: ["dizzy", "dizziness", "lightheaded", "vertigo", "woozy"] },
        { id: "confusion", label: "Confusion / Altered Sensorium", category: "neurological", core: "dizziness", keywords: ["confusion", "disoriented", "delirium", "foggy brain"] },
        { id: "stiff_neck", label: "Stiff Neck / Nuchal Rigidity", category: "neurological", core: "headache", keywords: ["stiff neck", "neck stiffness", "nuchal"] },
        { id: "photophobia", label: "Photophobia / Light Sensitivity", category: "neurological", core: "headache", keywords: ["photophobia", "light sensitive", "eyes hurt light"] },
        { id: "loss_taste_smell", label: "Anosmia / Loss of Smell or Taste", category: "neurological", core: "sore_throat", keywords: ["smell", "taste", "anosmia", "ageusia"] },

        // ─── Gastrointestinal (6) ───
        { id: "nausea", label: "Nausea / Queasiness", category: "gastrointestinal", core: "nausea", keywords: ["nausea", "queasy", "sick to stomach", "upset stomach"] },
        { id: "vomiting", label: "Vomiting / Emesis", category: "gastrointestinal", core: "vomiting", keywords: ["vomit", "vomiting", "throwing up", "emesis"] },
        { id: "loss_of_appetite", label: "Loss of Appetite / Anorexia", category: "gastrointestinal", core: "loss_of_appetite", keywords: ["appetite", "not hungry", "anorexia", "loss of appetite"] },
        { id: "abdominal_pain", label: "Abdominal Cramps / Belly Pain", category: "gastrointestinal", core: "abdominal_pain", keywords: ["stomach ache", "belly pain", "abdominal", "stomach pain", "cramps"] },
        { id: "diarrhea", label: "Diarrhea / Watery Stool", category: "gastrointestinal", core: "diarrhea", keywords: ["diarrhea", "loose motion", "watery stool", "frequent stool"] },
        { id: "bloating", label: "Abdominal Bloating / Gas", category: "gastrointestinal", core: "abdominal_pain", keywords: ["bloating", "distended", "flatulence", "gas"] },

        // ─── Musculoskeletal & Pain (5) ───
        { id: "body_pain", label: "Generalized Body Ache / Myalgia", category: "pain", core: "body_pain", keywords: ["body pain", "muscle ache", "myalgia", "sore muscles"] },
        { id: "joint_pain", label: "Joint Pain / Arthralgia", category: "pain", core: "body_pain", keywords: ["joint pain", "arthralgia", "knees hurt", "wrist pain"] },
        { id: "pleuritic_chest_pain", label: "Pleuritic Sharp Chest Pain", category: "pain", core: "rapid_breathing", keywords: ["pleuritic", "sharp chest pain", "pain on deep breath"] },
        { id: "lower_back_pain", label: "Lower Back / Flank Pain", category: "pain", core: "body_pain", keywords: ["back pain", "lower back", "flank pain"] },
        { id: "muscle_weakness", label: "Profound Muscle Weakness", category: "pain", core: "fatigue", keywords: ["muscle weakness", "heavy limbs", "cannot stand"] },

        // ─── Cardiovascular & Hemodynamic (4) ───
        { id: "palpitations", label: "Palpitations / Tachycardia", category: "cardiovascular", core: "rapid_breathing", keywords: ["palpitations", "racing heart", "heart pounding", "tachycardia"] },
        { id: "cyanosis", label: "Cyanosis (Bluish Lips / Fingertips)", category: "cardiovascular", core: "rapid_breathing", keywords: ["cyanosis", "blue lips", "blue fingers", "hypoxemia"] },
        { id: "cold_extremities", label: "Cold Hands & Feet (Hypoperfusion)", category: "cardiovascular", core: "chills", keywords: ["cold hands", "cold feet", "chilly extremities"] },
        { id: "orthostatic_dizziness", label: "Postural / Orthostatic Lightheadedness", category: "cardiovascular", core: "dizziness", keywords: ["standing up dizzy", "orthostatic", "blacking out standing"] },
    ];

    return {
        CATALOG,
        /**
         * Maps a set of selected symptom IDs to the 15 core XGBoost features
         */
        toCoreFeaturesDict: function (selectedSet) {
            const CORE_15 = [
                "fever", "cough", "headache", "nausea", "vomiting",
                "fatigue", "sore_throat", "chills", "body_pain",
                "loss_of_appetite", "abdominal_pain", "diarrhea",
                "sweating", "rapid_breathing", "dizziness"
            ];
            const dict = {};
            CORE_15.forEach((feat) => { dict[feat] = 0; });

            if (selectedSet) {
                selectedSet.forEach((symId) => {
                    const item = CATALOG.find((s) => s.id === symId);
                    if (item && item.core && dict.hasOwnProperty(item.core)) {
                        dict[item.core] = 1;
                    }
                });
            }

            return dict;
        },
    };
})();
