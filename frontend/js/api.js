/**
 * MediFusion AI — Centralized API Client Module
 * Communicates with FastAPI backend endpoints with real-time error dispatching.
 */

const MediFusionAPI = (function () {
    "use strict";

    const API_BASE = window.location.origin;

    /**
     * Safe JSON fetch with timeout
     */
    async function request(endpoint, options = {}, timeoutMs = 30000) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

        try {
            const url = `${API_BASE}${endpoint}`;
            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            let data;
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                data = await response.json();
            } else {
                data = await response.text();
            }

            if (!response.ok) {
                const errorMsg = (typeof data === "object" && data.detail) 
                    ? data.detail 
                    : `Server returned status ${response.status}`;
                throw new Error(errorMsg);
            }

            return data;
        } catch (err) {
            clearTimeout(timeoutId);
            if (err.name === "AbortError") {
                throw new Error("Request timed out. The server took too long to respond.");
            }
            throw err;
        }
    }

    // ── Endpoint-to-model-key mapping ──
    const IMAGE_MODEL_ENDPOINTS = {
        chest_xray: "/api/predict/image",
        brain_tumor: "/api/predict/image/brain-tumor",
        skin_cancer: "/api/predict/image/skin-cancer",
        retinopathy: "/api/predict/image/retinopathy",
        blood_cell: "/api/predict/image/blood-cell",
    };

    return {
        /**
         * System Health & Model Telemetry
         */
        getHealth: async function () {
            return request("/api/health", { method: "GET" }, 8000);
        },

        /**
         * Image Analysis (ResNet18) — Original Chest X-Ray endpoint
         */
        predictImage: async function (file) {
            const formData = new FormData();
            formData.append("file", file);
            return request("/api/predict/image", {
                method: "POST",
                body: formData,
            });
        },

        /**
         * Advanced Image Analysis — Route to specific model endpoint
         * @param {string} modelKey - e.g. "brain_tumor", "skin_cancer", etc.
         * @param {File} file - Image file to analyze
         */
        predictImageAdvanced: async function (modelKey, file) {
            const endpoint = IMAGE_MODEL_ENDPOINTS[modelKey];
            if (!endpoint) {
                throw new Error(`Unknown image model key: ${modelKey}`);
            }
            const formData = new FormData();
            formData.append("file", file);
            return request(endpoint, {
                method: "POST",
                body: formData,
            }, 60000);
        },

        /**
         * Grad-CAM Explainability — Original Chest X-Ray
         */
        explainImage: async function (file) {
            const formData = new FormData();
            formData.append("file", file);
            return request("/api/explain/image", {
                method: "POST",
                body: formData,
            });
        },

        /**
         * Advanced Grad-CAM — For any loaded image model
         * @param {string} modelKey - e.g. "brain_tumor", "skin_cancer", etc.
         * @param {File} file - Image file for explanation
         */
        explainImageAdvanced: async function (modelKey, file) {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("model_key", modelKey);
            return request("/api/explain/image/advanced", {
                method: "POST",
                body: formData,
            }, 60000);
        },

        /**
         * Get status of all image models (loaded/not loaded)
         */
        getImageModelsStatus: async function () {
            return request("/api/image-models/status", { method: "GET" }, 8000);
        },

        /**
         * Get clinical metadata for all image models
         */
        getImageModelsMetadata: async function () {
            return request("/api/image-models/metadata", { method: "GET" }, 8000);
        },

        /**
         * Symptom Intelligence (XGBoost)
         */
        predictSymptoms: async function (symptomsDict) {
            return request("/api/predict/symptoms", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ symptoms: symptomsDict }),
            });
        },

        /**
         * Clinical Signal Intelligence (XGBoost)
         */
        predictClinical: async function (symptomsDict, clinicalData) {
            return request("/api/predict/clinical", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    symptoms: symptomsDict,
                    clinical_data: clinicalData,
                }),
            });
        },

        /**
         * SHAP Explainability for Clinical/Symptom Models
         */
        explainClinical: async function (symptomsDict, clinicalData) {
            return request("/api/explain/clinical", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    symptoms: symptomsDict,
                    clinical_data: clinicalData,
                }),
            });
        },

        /**
         * Flagship Multimodal Fusion (Gated Attention / Concat)
         */
        predictMultimodal: async function (arg1, arg2, arg3) {
            let file = null;
            let symptoms = null;
            let clinicalData = null;

            if (arg1 && typeof arg1 === "object" && !(arg1 instanceof File) && !(arg1 instanceof Blob) && ("file" in arg1 || "symptoms" in arg1 || "clinicalData" in arg1)) {
                file = arg1.file || null;
                symptoms = arg1.symptoms || null;
                clinicalData = arg1.clinicalData || null;
            } else {
                file = arg1 || null;
                symptoms = arg2 || null;
                clinicalData = arg3 || null;
            }

            const formData = new FormData();
            if (file) formData.append("file", file);
            if (symptoms) formData.append("symptoms", JSON.stringify(symptoms));
            if (clinicalData) formData.append("clinical_data", JSON.stringify(clinicalData));

            return request("/api/predict/multimodal", {
                method: "POST",
                body: formData,
            });
        },
    };
})();

// Export globally
window.MediFusionAPI = MediFusionAPI;

