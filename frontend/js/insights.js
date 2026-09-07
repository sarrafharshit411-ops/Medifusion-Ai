/**
 * MediFusion AI — Model Insights & Explainability Module
 */

(function () {
    "use strict";

    document.addEventListener("DOMContentLoaded", () => {
        setupInteractivePipeline();
        loadLiveExplainabilityPreview();
    });

    function setupInteractivePipeline() {
        const nodes = document.querySelectorAll(".pipeline-node-card");
        const detailBox = document.getElementById("pipelineNodeDetails");

        const descriptions = {
            vision: {
                title: "ResNet18 Vision Backbone",
                desc: "Processes 224x224 chest radiographs through deep convolutional residual blocks. Frozen early layers maintain robust low-level spatial features while layer3/4 are fine-tuned for thoracic opacities. Extracts a dense 512-dimensional latent embedding.",
                tech: "PyTorch • Transfer Learning • Grad-CAM Hooked",
            },
            symptoms: {
                title: "XGBoost Symptom Intelligence",
                desc: "Evaluates multi-label binary symptom vectors (15 dimensions) across gradient-boosted decision trees. Calculates conditional leaf distribution and provides direct interpretable SHAP game-theoretic feature contributions.",
                tech: "XGBoost 3.0 • Multi-Softprob • 15 Feature Space",
            },
            clinical: {
                title: "Standardized Clinical Signal Encoder",
                desc: "Scales continuous vital signs (SpO2, heart rate, body temperature, blood pressures, respiratory rate) and joins them with patient symptom profiles to gauge acute physiological distress.",
                tech: "StandardScaler • XGBoost • SHAP TreeExplainer",
            },
            fusion: {
                title: "Gated Multi-Head Cross-Attention",
                desc: "Projects heterogenous embeddings into a shared 128-dimensional latent space. Learns dynamic per-modality importance gates and cross-attends across tokens. Gracefully supports missing modalities via attention key padding masks.",
                tech: "PyTorch MultiheadAttention • Modality Masking • Dynamic Gating",
            },
        };

        nodes.forEach((node) => {
            node.addEventListener("click", () => {
                nodes.forEach((n) => n.classList.remove("active"));
                node.classList.add("active");
                const key = node.dataset.node;
                const info = descriptions[key];
                if (detailBox && info) {
                    detailBox.innerHTML = `
                        <div class="glass-card" style="padding: 1.5rem; border-color: var(--border-accent); animation: fade-in-up 0.3s ease;">
                            <span class="badge badge-teal" style="margin-bottom:0.75rem;">${info.tech}</span>
                            <h3 style="font-size:1.25rem; font-weight:700; margin-bottom:0.5rem; color:var(--text-highlight);">${info.title}</h3>
                            <p style="font-size:0.9rem; color:var(--text-secondary); line-height:1.6;">${info.desc}</p>
                        </div>
                    `;
                }
            });
        });
    }

    async function loadLiveExplainabilityPreview() {
        const shapContainer = document.getElementById("liveShapContainer");
        if (!shapContainer) return;

        try {
            const data = await MediFusionAPI.explainClinical(
                { fever: 1, cough: 1, rapid_breathing: 1, fatigue: 1, chills: 1 },
                { age: 55, temperature: 39.2, heart_rate: 108, bp_systolic: 130, bp_diastolic: 85, respiratory_rate: 26, oxygen_saturation: 91 }
            );

            if (data && data.plot_base64) {
                shapContainer.innerHTML = `
                    <img src="data:image/png;base64,${data.plot_base64}" alt="SHAP Feature Importance" style="width:100%; border-radius:var(--radius-md); border:1px solid var(--border-normal);">
                `;
            }
        } catch (e) {
            console.error("Could not load SHAP preview:", e);
        }
    }
})();
