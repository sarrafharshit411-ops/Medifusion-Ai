/**
 * MediFusion AI — General Disease Diagnostic & Pharmacological Intelligence Controller
 * Powered by Calibrated 41-Class Ensemble (Random Forest + XGBoost) across 132 Symptoms.
 * Features Natural Language NLP extraction, 8 body system filters, clinical presets,
 * top-5 differential diagnoses, and comprehensive medication & self-care recommendations.
 */

(function () {
    "use strict";

    // State management
    const state = {
        allSymptoms: [],
        categories: {},
        selectedSymptoms: new Set(),
        currentCategory: "all",
        searchQuery: "",
        differentialChart: null,
        currentResult: null
    };

    // Clinical Scenario Presets
    const CLINICAL_PRESETS = {
        cold_flu: {
            name: "Cold & Influenza",
            symptoms: ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever", "headache", "throat_irritation", "runny_nose"]
        },
        dengue: {
            name: "Dengue Warning",
            symptoms: ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "pain_behind_the_eyes", "back_pain", "red_spots_over_body"]
        },
        malaria: {
            name: "Malaria Profile",
            symptoms: ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain"]
        },
        gastro: {
            name: "Gastroenteritis / GI",
            symptoms: ["vomiting", "sunken_eyes", "dehydration", "diarrhoea", "abdominal_pain"]
        },
        migraine: {
            name: "Migraine & Neuro",
            symptoms: ["headache", "acidity", "visual_disturbances", "blurred_and_distorted_vision", "stiff_neck", "depression"]
        },
        allergy: {
            name: "Allergic Flare",
            symptoms: ["itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering"]
        },
        arthritis: {
            name: "Joint & Arthritis",
            symptoms: ["joint_pain", "neck_pain", "knee_pain", "hip_joint_pain", "swelling_joints", "painful_walking"]
        }
    };

    // Initialize on DOM ready
    document.addEventListener("DOMContentLoaded", () => {
        init();
    });

    async function init() {
        setupEventListeners();
        await loadSymptomCatalog();
        renderPresetButtons();
    }

    // ─── 1. Load 132 Symptoms from Backend API ───
    async function loadSymptomCatalog() {
        const grid = document.getElementById("symptomCatalogGrid");
        if (grid) {
            grid.innerHTML = '<div style="grid-column: 1/-1; text-align:center; padding: 2rem; color:var(--text-muted);"><span class="loading-spinner"></span> Loading 132-symptom clinical catalog...</div>';
        }

        try {
            const res = await fetch("/api/general-disease/symptoms");
            if (!res.ok) throw new Error("Failed to fetch symptoms");
            const data = await res.json();
            
            state.categories = data.categories || {};
            state.allSymptoms = data.all_symptoms || [];

            renderCategoryTabs();
            renderSymptomCards();
            updateSelectedTray();
        } catch (err) {
            console.error("Error loading symptoms from API:", err);
            // Fallback notification
            if (grid) {
                grid.innerHTML = '<div style="grid-column: 1/-1; text-align:center; padding: 2rem; color:var(--danger);">Error connecting to symptom service. Please ensure the server is running.</div>';
            }
        }
    }

    // ─── 2. Setup Category Tabs ───
    function renderCategoryTabs() {
        const container = document.getElementById("categoryTabsContainer");
        if (!container) return;

        let html = `<button class="cat-filter-btn ${state.currentCategory === 'all' ? 'active' : ''}" data-cat="all">All (132)</button>`;
        
        for (const [catName, items] of Object.entries(state.categories)) {
            const shortName = catName.split("&")[0].trim();
            html += `<button class="cat-filter-btn ${state.currentCategory === catName ? 'active' : ''}" data-cat="${catName}">${shortName} (${items.length})</button>`;
        }

        container.innerHTML = html;

        container.querySelectorAll(".cat-filter-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                container.querySelectorAll(".cat-filter-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                state.currentCategory = btn.getAttribute("data-cat");
                renderSymptomCards();
            });
        });
    }

    // ─── 3. Render Symptom Cards Grid ───
    function renderSymptomCards() {
        const grid = document.getElementById("symptomCatalogGrid");
        if (!grid) return;

        let filtered = state.allSymptoms;

        // Filter by category
        if (state.currentCategory !== "all") {
            filtered = filtered.filter(item => item.category === state.currentCategory);
        }

        // Filter by search query
        if (state.searchQuery.trim()) {
            const q = state.searchQuery.toLowerCase().trim();
            filtered = filtered.filter(item => 
                item.label.toLowerCase().includes(q) || 
                item.id.toLowerCase().includes(q) ||
                item.category.toLowerCase().includes(q)
            );
        }

        if (filtered.length === 0) {
            grid.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding: 2.5rem 1rem; color:var(--text-muted); font-size:0.88rem;">No symptoms found matching "<strong>${escapeHtml(state.searchQuery)}</strong>". Try clearing your search.</div>`;
            return;
        }

        let html = "";
        filtered.forEach(item => {
            const isSelected = state.selectedSymptoms.has(item.id);
            html += `
                <div class="symptom-card-item ${isSelected ? 'selected' : ''}" data-id="${item.id}">
                    <div class="symptom-checkbox-box">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.5"><polyline points="20 6 9 17 4 12"/></svg>
                    </div>
                    <div class="symptom-text-col">
                        <span class="symptom-item-title" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</span>
                        <span class="symptom-item-category">${escapeHtml(item.category)}</span>
                    </div>
                </div>
            `;
        });

        grid.innerHTML = html;

        // Attach click listener to each card
        grid.querySelectorAll(".symptom-card-item").forEach(card => {
            card.addEventListener("click", () => {
                const symId = card.getAttribute("data-id");
                toggleSymptom(symId);
            });
        });
    }

    // ─── 4. Toggle Symptom Selection ───
    function toggleSymptom(symId) {
        if (state.selectedSymptoms.has(symId)) {
            state.selectedSymptoms.delete(symId);
        } else {
            state.selectedSymptoms.add(symId);
        }
        updateSelectedTray();
        renderSymptomCards();
    }

    // ─── 5. Update Selected Tray & Counter ───
    function updateSelectedTray() {
        const countBadge = document.getElementById("selectedCountBadge");
        const countText = document.getElementById("selectedCountText");
        const chipsContainer = document.getElementById("selectedChipsContainer");
        const tray = document.getElementById("selectedSymptomsTray");
        const analyzeBtn = document.getElementById("runGeneralAnalysisBtn");

        const count = state.selectedSymptoms.size;

        if (countBadge) countBadge.textContent = `${count} Selected`;
        if (countText) countText.textContent = `${count} Selected`;

        if (analyzeBtn) {
            analyzeBtn.disabled = count === 0;
            if (count > 0) {
                analyzeBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                    <span>Run Clinical Diagnosis (${count} Symptoms)</span>
                `;
            } else {
                analyzeBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    <span>Select Symptoms to Begin</span>
                `;
            }
        }

        if (!chipsContainer || !tray) return;

        if (count === 0) {
            tray.style.display = "none";
            chipsContainer.innerHTML = "";
            return;
        }

        tray.style.display = "block";
        let html = "";
        state.selectedSymptoms.forEach(symId => {
            const sym = state.allSymptoms.find(s => s.id === symId);
            const label = sym ? sym.label : symId.replace(/_/g, " ");
            html += `
                <span class="symptom-chip" data-id="${symId}">
                    <span>${escapeHtml(label)}</span>
                    <span class="symptom-chip-remove" title="Remove">&times;</span>
                </span>
            `;
        });
        chipsContainer.innerHTML = html;

        chipsContainer.querySelectorAll(".symptom-chip-remove").forEach(btn => {
            btn.addEventListener("click", (e) => {
                e.stopPropagation();
                const symId = btn.closest(".symptom-chip").getAttribute("data-id");
                toggleSymptom(symId);
            });
        });
    }

    // ─── 6. Render Clinical Presets ───
    function renderPresetButtons() {
        const container = document.getElementById("presetButtonsContainer");
        if (!container) return;

        let html = `<span class="preset-title">Quick Presets:</span>`;
        for (const [key, preset] of Object.entries(CLINICAL_PRESETS)) {
            html += `<button class="preset-chip" data-preset="${key}">${preset.name}</button>`;
        }
        container.innerHTML = html;

        container.querySelectorAll(".preset-chip").forEach(btn => {
            btn.addEventListener("click", () => {
                const key = btn.getAttribute("data-preset");
                applyPreset(key);
            });
        });
    }

    function applyPreset(key) {
        const preset = CLINICAL_PRESETS[key];
        if (!preset) return;

        state.selectedSymptoms.clear();
        preset.symptoms.forEach(sym => {
            state.selectedSymptoms.add(sym);
        });

        updateSelectedTray();
        renderSymptomCards();
        showToast(`Applied preset: ${preset.name} (${preset.symptoms.length} symptoms)`);
    }

    // ─── 7. Natural Language Narrative Parser ───
    async function parseNarrative() {
        const textarea = document.getElementById("naturalLanguageInput");
        const btn = document.getElementById("parseNlBtn");
        if (!textarea) return;

        const text = textarea.value.trim();
        if (!text) {
            showToast("Please type a description of your symptoms first.", "warning");
            textarea.focus();
            return;
        }

        const originalBtnText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = `<span class="loading-spinner"></span> Extracting Symptoms...`;

        try {
            const res = await fetch("/api/general-disease/nlp-extract", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });

            if (!res.ok) throw new Error("NLP extraction failed");
            const data = await res.json();

            if (data.extracted_symptoms && data.extracted_symptoms.length > 0) {
                let addedCount = 0;
                data.extracted_symptoms.forEach(item => {
                    if (!state.selectedSymptoms.has(item.id)) {
                        state.selectedSymptoms.add(item.id);
                        addedCount++;
                    }
                });

                updateSelectedTray();
                renderSymptomCards();

                showToast(`Extracted and checked ${data.extracted_symptoms.length} symptoms (${addedCount} new added)!`, "success");
            } else {
                showToast("No specific clinical symptoms recognized in description. Try using common medical terms.", "warning");
            }
        } catch (err) {
            console.error("NLP error:", err);
            showToast("Failed to connect to NLP extractor.", "danger");
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalBtnText;
        }
    }

    // ─── 8. Run General Disease Diagnosis ───
    async function runGeneralDiagnosis() {
        if (state.selectedSymptoms.size === 0) {
            showToast("Please select at least one symptom to run diagnosis.", "warning");
            return;
        }

        const btn = document.getElementById("runGeneralAnalysisBtn");
        const resultsContainer = document.getElementById("generalResultsContainer");
        const emptyState = document.getElementById("generalEmptyState");

        const originalBtnContent = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = `<span class="loading-spinner"></span> Synthesizing Differential Diagnosis...`;

        try {
            const payload = {
                symptoms: Array.from(state.selectedSymptoms),
                top_k: 5
            };

            const res = await fetch("/api/general-disease/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || "Diagnosis computation failed");
            }

            const data = await res.json();
            state.currentResult = data;

            if (emptyState) emptyState.style.display = "none";
            if (resultsContainer) {
                resultsContainer.style.display = "block";
                renderResults(data);
                resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
            }

            showToast(`Diagnosis generated: Primary ${data.primary_diagnosis.disease} (${data.primary_diagnosis.confidence}%)`, "success");
        } catch (err) {
            console.error("Diagnosis error:", err);
            showToast(err.message || "Failed to run diagnosis.", "danger");
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalBtnContent;
        }
    }

    // ─── 9. Render Diagnostic Results & Medication Guide ───
    function renderResults(data) {
        const primary = data.primary_diagnosis;

        // 1. Triage Badge & Headline
        const triageBadge = document.getElementById("resultTriageBadge");
        if (triageBadge) {
            triageBadge.textContent = primary.triage_level;
            triageBadge.style.backgroundColor = primary.triage_color + "25";
            triageBadge.style.color = primary.triage_color;
            triageBadge.style.border = `1px solid ${primary.triage_color}60`;
        }

        const diseaseTitle = document.getElementById("resultDiseaseTitle");
        if (diseaseTitle) {
            diseaseTitle.textContent = primary.disease;
        }

        const confidenceVal = document.getElementById("resultConfidenceVal");
        if (confidenceVal) {
            confidenceVal.textContent = `${primary.confidence}%`;
        }

        // 2. Specialist Consultation Banner
        const specialistBanner = document.getElementById("resultSpecialistBanner");
        if (specialistBanner) {
            specialistBanner.innerHTML = `
                <div>
                    <div class="specialist-label">Recommended Clinical Specialist</div>
                    <div class="specialist-name">${escapeHtml(primary.specialist)}</div>
                </div>
                <div style="text-align:right;">
                    <span class="triage-badge-pill" style="background:${primary.triage_color}25; color:${primary.triage_color}; border:1px solid ${primary.triage_color}60;">
                        ${escapeHtml(primary.triage_level)}
                    </span>
                </div>
            `;
        }

        // 3. Triage Action Directive
        const actionDirective = document.getElementById("resultActionDirective");
        if (actionDirective) {
            actionDirective.textContent = primary.triage_action;
        }

        // 4. Critical Warning Callout Box
        const warningBox = document.getElementById("resultCriticalWarningBox");
        if (warningBox) {
            if (data.critical_warning) {
                warningBox.style.display = "flex";
                warningBox.innerHTML = `
                    <div class="contraindication-icon">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    </div>
                    <div>
                        <div class="contraindication-title">Clinical Contraindication & Safety Warning</div>
                        <div class="contraindication-text">${escapeHtml(data.critical_warning)}</div>
                    </div>
                `;
            } else {
                warningBox.style.display = "none";
            }
        }

        // 5. Differential Diagnoses Probability Chart
        renderDifferentialChart(data.differential_diagnoses);

        // 6. Pharmacological Treatment & Medication Section
        const medContainer = document.getElementById("resultMedicationGuideContainer");
        if (medContainer) {
            medContainer.innerHTML = `
                <!-- Primary Medications -->
                <div class="medication-block">
                    <div class="med-block-header">
                        <div class="med-block-icon">💊</div>
                        <div class="med-block-title">Primary Prescription & Therapeutic Regimen</div>
                    </div>
                    <div class="med-block-body">
                        ${escapeHtml(primary.primary_medications)}
                    </div>
                </div>

                <!-- Immediate Symptom Relief (OTC) -->
                <div class="medication-block">
                    <div class="med-block-header">
                        <div class="med-block-icon" style="background:rgba(16,185,129,0.15); color:var(--success);">⚡</div>
                        <div class="med-block-title">Immediate Symptom Relief (Over-the-Counter)</div>
                    </div>
                    <div class="med-block-body">
                        ${escapeHtml(primary.immediate_symptom_relief)}
                    </div>
                </div>

                <!-- Alternative OTC & Supportive Therapies -->
                ${primary.alternative_relief && primary.alternative_relief !== "N/A" ? `
                <div class="medication-block">
                    <div class="med-block-header">
                        <div class="med-block-icon" style="background:rgba(139,92,246,0.15); color:var(--accent-violet);">🌿</div>
                        <div class="med-block-title">Alternative OTC & Supportive Therapy</div>
                    </div>
                    <div class="med-block-body">
                        ${escapeHtml(primary.alternative_relief)}
                    </div>
                </div>` : ''}

                <!-- Lifestyle, Diet & Self-Care Recovery -->
                <div class="medication-block">
                    <div class="med-block-header">
                        <div class="med-block-icon" style="background:rgba(245,158,11,0.15); color:var(--warning);">🥗</div>
                        <div class="med-block-title">Self-Care, Dietary & Recovery Protocol</div>
                    </div>
                    <div class="med-block-body">
                        ${escapeHtml(primary.self_care_lifestyle)}
                    </div>
                </div>
            `;
        }
    }

    // ─── 10. Differential Diagnosis Horizontal Bar Chart ───
    function renderDifferentialChart(diffs) {
        const canvas = document.getElementById("differentialProbChart");
        if (!canvas) return;

        if (state.differentialChart) {
            state.differentialChart.destroy();
        }

        const labels = diffs.map(d => d.disease);
        const dataVals = diffs.map(d => d.confidence);
        const backgroundColors = diffs.map((d, idx) => {
            if (idx === 0) return "#0ea5e9";
            if (idx === 1) return "#0284c7";
            return "rgba(14, 165, 233, 0.4)";
        });

        if (window.Chart) {
            Chart.defaults.color = "#0f172a";
            Chart.defaults.font.family = "'Plus Jakarta Sans', 'Inter', sans-serif";
        }

        const ctx = canvas.getContext("2d");
        state.differentialChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Diagnostic Confidence (%)",
                    data: dataVals,
                    backgroundColor: backgroundColors,
                    borderRadius: 6,
                    borderSkipped: false
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "#ffffff",
                        titleColor: "#0f172a",
                        bodyColor: "#0f172a",
                        borderColor: "#94a3b8",
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label: function (ctx) {
                                return ` Probability: ${ctx.parsed.x}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: "rgba(148, 163, 184, 0.25)" },
                        ticks: {
                            color: "#0f172a",
                            font: { weight: "700", size: 12, family: "'Plus Jakarta Sans', sans-serif" },
                            callback: function (val) { return val + "%"; }
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: {
                            color: "#0f172a",
                            font: { weight: "700", size: 13, family: "'Plus Jakarta Sans', sans-serif" }
                        }
                    }
                }
            }
        });
    }

    // ─── 11. Event Listeners Setup ───
    function setupEventListeners() {
        // Search Input
        const searchInput = document.getElementById("symptomSearchInput");
        if (searchInput) {
            searchInput.addEventListener("input", (e) => {
                state.searchQuery = e.target.value;
                renderSymptomCards();
            });
        }

        // Clear button
        const clearBtn = document.getElementById("clearSymptomsBtn");
        if (clearBtn) {
            clearBtn.addEventListener("click", () => {
                state.selectedSymptoms.clear();
                updateSelectedTray();
                renderSymptomCards();
                showToast("Cleared selected symptoms.");
            });
        }

        // Natural language button
        const parseBtn = document.getElementById("parseNlBtn");
        if (parseBtn) {
            parseBtn.addEventListener("click", parseNarrative);
        }

        // Run diagnosis button
        const runBtn = document.getElementById("runGeneralAnalysisBtn");
        if (runBtn) {
            runBtn.addEventListener("click", runGeneralDiagnosis);
        }

        // Print Report button
        const printBtn = document.getElementById("printReportBtn");
        if (printBtn) {
            printBtn.addEventListener("click", printClinicalReport);
        }
    }

    // ─── 12. Printable Clinical Summary ───
    function printClinicalReport() {
        if (!state.currentResult) {
            showToast("No active diagnostic results to print.", "warning");
            return;
        }

        const primary = state.currentResult.primary_diagnosis;
        const printWindow = window.open("", "_blank");
        if (!printWindow) {
            showToast("Please allow popups to generate printable report.", "warning");
            return;
        }

        const now = new Date().toLocaleString();
        const symptomsList = state.currentResult.selected_symptoms.map(s => s.replace(/_/g, " ")).join(", ");

        const html = `
            <!DOCTYPE html>
            <html>
            <head>
                <title>Clinical Assessment Report — MediFusion AI</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 40px; color: #1e293b; line-height: 1.6; }
                    .report-header { border-bottom: 2px solid #0284c7; padding-bottom: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: flex-end; }
                    .brand { font-size: 24px; font-weight: 800; color: #0284c7; }
                    .date { font-size: 12px; color: #64748b; }
                    .section { margin-bottom: 20px; }
                    .section-title { font-size: 14px; font-weight: 700; text-transform: uppercase; color: #0284c7; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px; margin-bottom: 10px; }
                    .diag-box { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
                    .diag-name { font-size: 20px; font-weight: 800; color: #0369a1; }
                    .conf { font-size: 14px; color: #0284c7; font-weight: 600; }
                    .med-item { margin-bottom: 12px; }
                    .med-title { font-weight: 700; font-size: 13px; color: #334155; }
                    .warning { background: #fff1f2; border: 1px solid #fecdd3; padding: 12px; border-radius: 6px; color: #9f1239; font-size: 13px; margin-bottom: 20px; }
                    .disclaimer { font-size: 11px; color: #94a3b8; margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 15px; }
                </style>
            </head>
            <body>
                <div class="report-header">
                    <div>
                        <div class="brand">MediFusion AI Clinical Decision-Support</div>
                        <div>General Disease Diagnostic & Pharmacological Triage</div>
                    </div>
                    <div class="date">Generated: ${now}</div>
                </div>

                <div class="diag-box">
                    <div class="diag-name">${primary.disease}</div>
                    <div class="conf">Diagnostic Confidence: ${primary.confidence}% • Triage: ${primary.triage_level}</div>
                    <div style="margin-top:8px; font-size:13px;"><strong>Recommended Specialist:</strong> ${primary.specialist}</div>
                    <div style="font-size:13px;"><strong>Action Directive:</strong> ${primary.triage_action}</div>
                </div>

                ${state.currentResult.critical_warning ? `
                <div class="warning">
                    <strong>CRITICAL PRECAUTION:</strong> ${state.currentResult.critical_warning}
                </div>` : ''}

                <div class="section">
                    <div class="section-title">Reported Patient Symptoms (${state.currentResult.selected_symptoms_count})</div>
                    <p style="font-size:13px;">${symptomsList}</p>
                </div>

                <div class="section">
                    <div class="section-title">Pharmacological Guidance & Medications</div>
                    <div class="med-item">
                        <div class="med-title">Primary Therapeutics & Prescription Regimen:</div>
                        <div style="font-size:13px;">${primary.primary_medications}</div>
                    </div>
                    <div class="med-item">
                        <div class="med-title">Immediate Symptom Relief (OTC):</div>
                        <div style="font-size:13px;">${primary.immediate_symptom_relief}</div>
                    </div>
                    ${primary.alternative_relief ? `
                    <div class="med-item">
                        <div class="med-title">Alternative Supportive Care:</div>
                        <div style="font-size:13px;">${primary.alternative_relief}</div>
                    </div>` : ''}
                    <div class="med-item">
                        <div class="med-title">Self-Care & Recovery Protocol:</div>
                        <div style="font-size:13px;">${primary.self_care_lifestyle}</div>
                    </div>
                </div>

                <div class="disclaimer">
                    EDUCATIONAL/RESEARCH CLINICAL DECISION-SUPPORT PROTOTYPE. This document is generated for informational and clinical triage evaluation. It does not replace independent professional medical examination, formal laboratory testing, or physician diagnosis.
                </div>

                <script>
                    window.onload = function() { window.print(); }
                </script>
            </body>
            </html>
        `;

        printWindow.document.open();
        printWindow.document.write(html);
        printWindow.document.close();
    }

    // ─── Utility Helpers ───
    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function showToast(msg, type = "info") {
        let toast = document.getElementById("clinicalToast");
        if (!toast) {
            toast = document.createElement("div");
            toast.id = "clinicalToast";
            toast.style.position = "fixed";
            toast.style.bottom = "24px";
            toast.style.right = "24px";
            toast.style.zIndex = "9999";
            toast.style.padding = "0.75rem 1.25rem";
            toast.style.borderRadius = "8px";
            toast.style.fontSize = "0.85rem";
            toast.style.fontWeight = "600";
            toast.style.boxShadow = "0 10px 25px rgba(0,0,0,0.5)";
            toast.style.transition = "all 0.3s ease";
            document.body.appendChild(toast);
        }

        if (type === "success") {
            toast.style.background = "#10b981";
            toast.style.color = "#ffffff";
        } else if (type === "warning") {
            toast.style.background = "#f59e0b";
            toast.style.color = "#ffffff";
        } else if (type === "danger") {
            toast.style.background = "#f43f5e";
            toast.style.color = "#ffffff";
        } else {
            toast.style.background = "#0284c7";
            toast.style.color = "#ffffff";
        }

        toast.textContent = msg;
        toast.style.opacity = "1";
        toast.style.transform = "translateY(0)";

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateY(10px)";
        }, 3500);
    }

})();
