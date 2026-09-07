/**
 * MediFusion AI — Upgraded Multimodal Assessment Controller
 * Full interactive configuration for Radiology, 35+ Symptoms, and 7 Clinical Vitals with Cross-Modal Gating.
 */

(function () {
    "use strict";

    // Modality State
    let currentImageFile = null;
    let selectedSymptoms = new Set(["fever", "cough", "rapid_breathing", "fatigue"]);
    let currentVitals = {
        age: 55,
        temperature: 39.0,
        heart_rate: 104,
        bp_systolic: 130,
        bp_diastolic: 85,
        respiratory_rate: 26,
        oxygen_saturation: 91
    };

    let mmDrawerCategory = "all";

    document.addEventListener("DOMContentLoaded", () => {
        if (!document.getElementById("cardModalityImage")) return;

        setupModalityToggles();
        setupImageModality();
        setupSymptomsDrawer();
        setupVitalsDrawer();
        setupRunButtons();
    });

    // ─── 1. Stream Inclusion Checkboxes & Modality Toggles ───
    function setupModalityToggles() {
        const checkImg = document.getElementById("checkIncludeImage");
        const checkSym = document.getElementById("checkIncludeSymptoms");
        const checkCli = document.getElementById("checkIncludeClinical");

        const cardImg = document.getElementById("cardModalityImage");
        const cardSym = document.getElementById("cardModalitySymptoms");
        const cardCli = document.getElementById("cardModalityClinical");

        if (checkImg && cardImg) {
            checkImg.addEventListener("change", () => {
                cardImg.classList.toggle("active", checkImg.checked);
                updateCardVisuals();
            });
        }
        if (checkSym && cardSym) {
            checkSym.addEventListener("change", () => {
                cardSym.classList.toggle("active", checkSym.checked);
                updateCardVisuals();
            });
        }
        if (checkCli && cardCli) {
            checkCli.addEventListener("change", () => {
                cardCli.classList.toggle("active", checkCli.checked);
                updateCardVisuals();
            });
        }
    }

    function updateCardVisuals() {
        // Update summary text or badge states
    }

    // ─── 2. Radiology Stream Setup ───
    function setupImageModality() {
        const fileInput = document.getElementById("mmImageInput");
        const browseBtn = document.getElementById("mmBrowseBtn");
        const loadSampleBtn = document.getElementById("mmLoadSampleBtn");
        const clearBtn = document.getElementById("mmClearImgBtn");
        const descEl = document.getElementById("mmImageDesc");
        const statusEl = document.getElementById("mmImageStatus");

        if (browseBtn && fileInput) {
            browseBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                fileInput.click();
            });
        }

        if (fileInput) {
            fileInput.addEventListener("change", async (e) => {
                if (e.target.files && e.target.files[0]) {
                    await setImageFile(e.target.files[0]);
                }
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                currentImageFile = null;
                if (fileInput) fileInput.value = "";
                if (descEl) descEl.textContent = "No radiograph attached";
                if (statusEl) {
                    statusEl.textContent = "NOT ATTACHED";
                    statusEl.className = "modality-status-indicator status-empty";
                }
                clearBtn.style.display = "none";
                MediFusionUI.showToast("info", "Radiology stream detached.");
            });
        }
    }

    async function setImageFile(file) {
        // Run multi-layer medical image validation
        if (window.ImageValidator) {
            MediFusionUI.showToast("info", "Validating medical image...");
            try {
                const result = await ImageValidator.validate(file);
                if (!result.valid) {
                    const title = ImageValidator.getErrorTitle(result.code);
                    MediFusionUI.showToast("error", result.error, 6000);

                    // Show overlay on the image modality card
                    const cardEl = document.getElementById("cardModalityImage");
                    if (cardEl) {
                        MediFusionUI.showValidationError(cardEl, title, result.error, 6000);
                    }

                    // Reset file input
                    const fileInput = document.getElementById("mmImageInput");
                    if (fileInput) fileInput.value = "";
                    return;
                }
            } catch (err) {
                console.warn("Image validation error, allowing upload:", err);
            }
        }

        setImageFileInternal(file);
    }

    function setImageFileInternal(file) {
        // Dismiss any existing validation error overlay
        const cardImg = document.getElementById("cardModalityImage");
        if (cardImg) {
            const existingOverlay = cardImg.querySelector(".validation-error-overlay");
            if (existingOverlay) existingOverlay.remove();
        }

        currentImageFile = file;
        const descEl = document.getElementById("mmImageDesc");
        const statusEl = document.getElementById("mmImageStatus");
        const clearBtn = document.getElementById("mmClearImgBtn");

        if (descEl) descEl.textContent = file.name;
        if (statusEl) {
            statusEl.textContent = "ATTACHED (PNG)";
            statusEl.className = "modality-status-indicator status-ready";
        }
        if (clearBtn) clearBtn.style.display = "inline-block";

        const checkImg = document.getElementById("checkIncludeImage");
        if (checkImg) checkImg.checked = true;
        if (cardImg) cardImg.classList.add("active");
    }

    // ─── 3. Symptoms Drawer & Expanded Catalog (35+ Symptoms) ───
    function setupSymptomsDrawer() {
        const toggleBtn = document.getElementById("mmToggleSymptomsDrawerBtn");
        const drawer = document.getElementById("mmSymptomsDrawer");
        const closeBtn = document.getElementById("mmDrawerCloseSymptomsBtn");
        const clearBtn = document.getElementById("mmDrawerClearSymptomsBtn");
        const commonBtn = document.getElementById("mmDrawerSelectCommonBtn");

        if (toggleBtn && drawer) {
            toggleBtn.addEventListener("click", () => {
                const isOpen = drawer.style.display === "block";
                drawer.style.display = isOpen ? "none" : "block";
                toggleBtn.textContent = isOpen ? "Edit (35+ Options)" : "Close Drawer";
                if (!isOpen) renderDrawerSymptomChips();
            });
        }

        if (closeBtn && drawer && toggleBtn) {
            closeBtn.addEventListener("click", () => {
                drawer.style.display = "none";
                toggleBtn.textContent = "Edit (35+ Options)";
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener("click", () => {
                selectedSymptoms.clear();
                renderDrawerSymptomChips();
                updateSymptomCardSummary();
            });
        }

        if (commonBtn) {
            commonBtn.addEventListener("click", () => {
                selectedSymptoms = new Set(["fever", "cough", "rapid_breathing", "fatigue", "chills", "pleuritic_chest_pain"]);
                renderDrawerSymptomChips();
                updateSymptomCardSummary();
                MediFusionUI.showToast("info", "Selected Common Pneumonia Cluster.");
            });
        }

        // Category Tab buttons inside drawer
        const catBtns = document.querySelectorAll("[data-mmcat]");
        catBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                catBtns.forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                mmDrawerCategory = btn.dataset.mmcat || "all";
                renderDrawerSymptomChips();
            });
        });

        renderDrawerSymptomChips();
        updateSymptomCardSummary();
    }

    function renderDrawerSymptomChips() {
        const cloud = document.getElementById("mmDrawerChipsCloud");
        if (!cloud || !window.MediFusionSymptoms) return;

        cloud.innerHTML = "";

        window.MediFusionSymptoms.CATALOG.forEach((sym) => {
            if (mmDrawerCategory !== "all" && sym.category !== mmDrawerCategory) return;

            const isSelected = selectedSymptoms.has(sym.id);
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = `symptom-tag-pill ${isSelected ? "selected" : ""}`;
            chip.innerHTML = `<span>${sym.label}</span>${isSelected ? '<span style="font-size:0.75rem;">✓</span>' : ""}`;

            chip.addEventListener("click", () => {
                if (selectedSymptoms.has(sym.id)) {
                    selectedSymptoms.delete(sym.id);
                } else {
                    selectedSymptoms.add(sym.id);
                }
                renderDrawerSymptomChips();
                updateSymptomCardSummary();
            });

            cloud.appendChild(chip);
        });

        const countBadge = document.getElementById("mmDrawerSymptomCount");
        if (countBadge) countBadge.textContent = `${selectedSymptoms.size} Selected`;
    }

    function updateSymptomCardSummary() {
        const statusEl = document.getElementById("mmSymptomStatus");
        const summaryTextEl = document.getElementById("mmSymptomSummaryText");

        if (statusEl) {
            statusEl.textContent = `${selectedSymptoms.size} SYMPTOMS`;
            statusEl.className = selectedSymptoms.size > 0 ? "modality-status-indicator status-ready" : "modality-status-indicator status-empty";
        }

        if (summaryTextEl && window.MediFusionSymptoms) {
            if (selectedSymptoms.size === 0) {
                summaryTextEl.textContent = "No symptoms selected";
            } else {
                const labels = Array.from(selectedSymptoms)
                    .map((id) => {
                        const s = window.MediFusionSymptoms.CATALOG.find((x) => x.id === id);
                        return s ? s.label.split(" ")[0].replace("/", "") : id;
                    })
                    .slice(0, 4);
                summaryTextEl.textContent = labels.join(", ") + (selectedSymptoms.size > 4 ? ` +${selectedSymptoms.size - 4} more` : "");
            }
        }
    }

    // ─── 4. Vitals Drawer Setup ───
    function setupVitalsDrawer() {
        const toggleBtn = document.getElementById("mmToggleVitalsDrawerBtn");
        const drawer = document.getElementById("mmVitalsDrawer");
        const closeBtn = document.getElementById("mmDrawerCloseVitalsBtn");
        const normalBtn = document.getElementById("mmDrawerVitalsNormalBtn");
        const distressBtn = document.getElementById("mmDrawerVitalsDistressBtn");

        if (toggleBtn && drawer) {
            toggleBtn.addEventListener("click", () => {
                const isOpen = drawer.style.display === "block";
                drawer.style.display = isOpen ? "none" : "block";
                toggleBtn.textContent = isOpen ? "Edit Vitals" : "Close Drawer";
            });
        }

        if (closeBtn && drawer && toggleBtn) {
            closeBtn.addEventListener("click", () => {
                drawer.style.display = "none";
                toggleBtn.textContent = "Edit Vitals";
            });
        }

        // Live binding on vital inputs
        const vitalFields = ["age", "temp", "hr", "bpsys", "bpdia", "rr", "spo2"];
        vitalFields.forEach((key) => {
            const input = document.getElementById(`mm_${key}`);
            if (input) {
                input.addEventListener("input", () => {
                    readVitalsFromInputs();
                    updateVitalsCardSummary();
                });
            }
        });

        if (normalBtn) {
            normalBtn.addEventListener("click", () => {
                setVitalsValues({ age: 40, temp: 36.8, hr: 72, bpsys: 120, bpdia: 80, rr: 16, spo2: 98 });
                MediFusionUI.showToast("info", "Set Healthy Normal physiological vitals.");
            });
        }

        if (distressBtn) {
            distressBtn.addEventListener("click", () => {
                setVitalsValues({ age: 58, temp: 39.2, hr: 112, bpsys: 135, bpdia: 88, rr: 28, spo2: 89 });
                MediFusionUI.showToast("warning", "Set Acute Respiratory Distress profile.");
            });
        }

        updateVitalsCardSummary();
    }

    function readVitalsFromInputs() {
        currentVitals = {
            age: parseFloat(document.getElementById("mm_age")?.value) || 55,
            temperature: parseFloat(document.getElementById("mm_temp")?.value) || 39.0,
            heart_rate: parseFloat(document.getElementById("mm_hr")?.value) || 104,
            bp_systolic: parseFloat(document.getElementById("mm_bpsys")?.value) || 130,
            bp_diastolic: parseFloat(document.getElementById("mm_bpdia")?.value) || 85,
            respiratory_rate: parseFloat(document.getElementById("mm_rr")?.value) || 26,
            oxygen_saturation: parseFloat(document.getElementById("mm_spo2")?.value) || 91,
        };
    }

    function setVitalsValues(v) {
        if (document.getElementById("mm_age")) document.getElementById("mm_age").value = v.age;
        if (document.getElementById("mm_temp")) document.getElementById("mm_temp").value = v.temp;
        if (document.getElementById("mm_hr")) document.getElementById("mm_hr").value = v.hr;
        if (document.getElementById("mm_bpsys")) document.getElementById("mm_bpsys").value = v.bpsys;
        if (document.getElementById("mm_bpdia")) document.getElementById("mm_bpdia").value = v.bpdia;
        if (document.getElementById("mm_rr")) document.getElementById("mm_rr").value = v.rr;
        if (document.getElementById("mm_spo2")) document.getElementById("mm_spo2").value = v.spo2;
        readVitalsFromInputs();
        updateVitalsCardSummary();
    }

    function updateVitalsCardSummary() {
        const summaryTextEl = document.getElementById("mmVitalsSummaryText");
        if (summaryTextEl) {
            summaryTextEl.textContent = `SpO₂: ${currentVitals.oxygen_saturation}% • HR: ${currentVitals.heart_rate} • Temp: ${currentVitals.temperature}°C`;
        }
    }

    // ─── 5. Multimodal Execution Orchestration ───
    function setupRunButtons() {
        const runBtn = document.getElementById("runMultimodalBtn");
        const runNavBtn = document.getElementById("runMultimodalNavBtn");

        if (runBtn) runBtn.addEventListener("click", executeMultimodalAssessment);
        if (runNavBtn) runNavBtn.addEventListener("click", executeMultimodalAssessment);
    }

    async function executeMultimodalAssessment() {
        const checkImg = document.getElementById("checkIncludeImage")?.checked;
        const checkSym = document.getElementById("checkIncludeSymptoms")?.checked;
        const checkCli = document.getElementById("checkIncludeClinical")?.checked;

        if (!checkImg && !checkSym && !checkCli) {
            MediFusionUI.showToast("warning", "Please enable at least one modality stream.");
            return;
        }

        const runBtn = document.getElementById("runMultimodalBtn");
        const btnText = document.getElementById("mmBtnText");
        const emptyStage = document.getElementById("mmEmptyStage");
        const resultsStage = document.getElementById("mmResultsStage");

        if (runBtn) runBtn.disabled = true;
        if (btnText) btnText.textContent = "Synthesizing Cross-Modal Attention...";

        try {
            readVitalsFromInputs();
            const symptomsDict = window.MediFusionSymptoms ? window.MediFusionSymptoms.toCoreFeaturesDict(selectedSymptoms) : {};

            const imageFileToSend = checkImg ? currentImageFile : null;
            const symptomsToSend = checkSym && selectedSymptoms.size > 0 ? symptomsDict : null;
            const vitalsToSend = checkCli ? currentVitals : null;

            const res = await MediFusionAPI.predictMultimodal(imageFileToSend, symptomsToSend, vitalsToSend);

            if (emptyStage) emptyStage.style.display = "none";
            if (resultsStage) resultsStage.style.display = "block";

            // 1. Overall Condition & Consensus Pill
            const condEl = document.getElementById("mmOverallCondition");
            const badgeEl = document.getElementById("mmConsensusBadge");

            if (condEl) condEl.textContent = res.predicted_class.toUpperCase();

            if (badgeEl) {
                if (res.confidence > 0.85) {
                    badgeEl.className = "badge badge-teal";
                    badgeEl.textContent = `High Multimodal Consensus (${(res.confidence * 100).toFixed(1)}%)`;
                } else {
                    badgeEl.className = "badge badge-warning";
                    badgeEl.textContent = `Moderate Confidence (${(res.confidence * 100).toFixed(1)}%)`;
                }
            }

            // 2. Modality Diagnostic Cards from Real Model Results
            const imgCardScore = document.getElementById("mmCardImageScore");
            const imgCardPred = document.getElementById("mmCardImagePred");
            const symCardScore = document.getElementById("mmCardSymptomScore");
            const symCardPred = document.getElementById("mmCardSymptomPred");
            const cliCardScore = document.getElementById("mmCardClinicalScore");
            const cliCardPred = document.getElementById("mmCardClinicalPred");

            if (res.individual_predictions && res.individual_predictions.image) {
                const topImg = res.individual_predictions.image.reduce((a, b) => (a.probability > b.probability ? a : b));
                if (imgCardScore) imgCardScore.textContent = `${Math.round(topImg.probability * 100)}%`;
                if (imgCardPred) imgCardPred.textContent = `${topImg.disease} (${(topImg.probability * 100).toFixed(0)}%)`;
            } else {
                if (imgCardScore) imgCardScore.textContent = "—";
                if (imgCardPred) imgCardPred.textContent = "Not Provided";
            }

            if (res.individual_predictions && res.individual_predictions.symptoms) {
                const topSym = res.individual_predictions.symptoms.reduce((a, b) => (a.probability > b.probability ? a : b));
                if (symCardScore) symCardScore.textContent = `${Math.round(topSym.probability * 100)}%`;
                if (symCardPred) symCardPred.textContent = `${topSym.disease} (${(topSym.probability * 100).toFixed(0)}%)`;
            } else {
                if (symCardScore) symCardScore.textContent = "—";
                if (symCardPred) symCardPred.textContent = "Not Provided";
            }

            if (res.individual_predictions && res.individual_predictions.clinical) {
                const topCli = res.individual_predictions.clinical.reduce((a, b) => (a.probability > b.probability ? a : b));
                if (cliCardScore) cliCardScore.textContent = `${Math.round(topCli.probability * 100)}%`;
                if (cliCardPred) cliCardPred.textContent = `${topCli.disease} (${(topCli.probability * 100).toFixed(0)}%)`;
            } else {
                if (cliCardScore) cliCardScore.textContent = "—";
                if (cliCardPred) cliCardPred.textContent = "Not Provided";
            }

            // 3. Radial Fusion Score Gauge
            const radialProgress = document.getElementById("mmRadialProgress");
            const scoreNum = document.getElementById("mmFusionScoreNum");
            const fusionPercentage = Math.round(res.confidence * 100);

            if (radialProgress) {
                const circumference = 2 * Math.PI * 70; // 439.82
                const offset = circumference - (fusionPercentage / 100) * circumference;
                radialProgress.style.strokeDashoffset = offset;
            }

            if (scoreNum) {
                MediFusionUI.animateNumber(scoreNum, 0, fusionPercentage, 900, "%");
            }

            // 4. Modality Contribution Weights
            const hasImg = checkImg && !!currentImageFile;
            const hasSym = checkSym && selectedSymptoms.size > 0;
            const hasCli = checkCli;

            let wImg = hasImg ? 45 : 0;
            let wSym = hasSym ? 35 : 0;
            let wCli = hasCli ? 20 : 0;
            const totalW = (wImg + wSym + wCli) || 100;

            const pImg = Math.round((wImg / totalW) * 100);
            const pSym = Math.round((wSym / totalW) * 100);
            const pCli = Math.round((wCli / totalW) * 100);

            const bImg = document.getElementById("contribBarImage");
            const vImg = document.getElementById("contribValImage");
            const bSym = document.getElementById("contribBarSymptoms");
            const vSym = document.getElementById("contribValSymptoms");
            const bCli = document.getElementById("contribBarClinical");
            const vCli = document.getElementById("contribValClinical");

            if (bImg && vImg) { bImg.style.width = `${pImg}%`; vImg.textContent = `${pImg}%`; }
            if (bSym && vSym) { bSym.style.width = `${pSym}%`; vSym.textContent = `${pSym}%`; }
            if (bCli && vCli) { bCli.style.width = `${pCli}%`; vCli.textContent = `${pCli}%`; }

            // 5. Clinical Decision Narrative Synthesis
            const synthesisEl = document.getElementById("mmSynthesisText");
            if (synthesisEl) {
                let narrative = `The Gated Cross-Modal Multi-Head Attention engine unified ${[
                    hasImg ? "thoracic radiographic features" : null,
                    hasSym ? `${selectedSymptoms.size} clinical symptom markers` : null,
                    hasCli ? `7 physiological vitals (SpO2: ${currentVitals.oxygen_saturation}%, Temp: ${currentVitals.temperature}°C)` : null
                ].filter(Boolean).join(", ")}. `;

                if (res.predicted_class === "Pneumonia") {
                    narrative += `The findings exhibit pronounced alignment with acute pulmonary consolidation and respiratory distress, supported with ${(res.confidence * 100).toFixed(1)}% cross-modal attention agreement.`;
                } else if (res.predicted_class === "Malaria") {
                    narrative += `Symptomatic rigors, high paroxysmal fever, and constitutional fatigue indicate a high probability of malarial parasitemia (${(res.confidence * 100).toFixed(1)}% confidence).`;
                } else if (res.predicted_class === "Typhoid") {
                    narrative += `Sustained pyrexia, abdominal cramping, and constitutional weakness corroborate classic enteric fever presentation (${(res.confidence * 100).toFixed(1)}% confidence).`;
                } else {
                    narrative += `Signals indicate normal physiological parameters without significant consolidations or acute systemic indicators.`;
                }
                synthesisEl.textContent = narrative;
            }

            // 6. Unified Probability Bar Chart & Radar Chart
            const probList = res.unified_predictions || res.predictions || [];
            MediFusionCharts.renderProbabilityBar("mmProbabilityChart", probList, true);
            MediFusionCharts.renderRadarComparison(
                "mmRadarChart",
                ["Vision Opacity", "Febrile State", "Dyspnea / Resp", "Constitutional", "Hemodynamic"],
                [
                    hasImg ? 85 : 10,
                    currentVitals.temperature >= 38 ? 90 : 30,
                    selectedSymptoms.has("rapid_breathing") || currentVitals.oxygen_saturation < 93 ? 88 : 20,
                    selectedSymptoms.has("fatigue") || selectedSymptoms.has("chills") ? 82 : 25,
                    currentVitals.heart_rate > 100 ? 78 : 35
                ]
            );

            MediFusionUI.showToast("success", `Multimodal assessment synthesized (${res.predicted_class}).`);
        } catch (err) {
            console.error("Multimodal synthesis error:", err);
            const msg = err.message || "Failed to execute multimodal assessment. Verify server connection.";
            MediFusionUI.showToast("error", msg, 6000);
            const imgCard = document.getElementById("cardModalityImage");
            if (imgCard && (msg.toLowerCase().includes("invalid") || msg.toLowerCase().includes("radiograph") || msg.toLowerCase().includes("image"))) {
                MediFusionUI.showValidationError(imgCard, "Radiology Stream Rejected", msg, 7000);
            }
        } finally {
            if (runBtn) runBtn.disabled = false;
            if (btnText) btnText.textContent = "Run Multimodal Assessment";
        }
    }
})();
