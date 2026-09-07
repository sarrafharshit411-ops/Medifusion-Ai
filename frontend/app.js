/**
 * MediFusion AI — Frontend Application
 * Vanilla JavaScript dashboard for multimodal healthcare assessment.
 */

(function () {
    "use strict";

    // ──────────────────────────────────────────────────────────
    // Configuration
    // ──────────────────────────────────────────────────────────
    const API_BASE = window.location.origin;

    const SYMPTOMS = [
        { id: "fever", label: "Fever" },
        { id: "cough", label: "Cough" },
        { id: "headache", label: "Headache" },
        { id: "nausea", label: "Nausea" },
        { id: "vomiting", label: "Vomiting" },
        { id: "fatigue", label: "Fatigue" },
        { id: "sore_throat", label: "Sore Throat" },
        { id: "chills", label: "Chills" },
        { id: "body_pain", label: "Body Pain" },
        { id: "loss_of_appetite", label: "Loss of Appetite" },
        { id: "abdominal_pain", label: "Abdominal Pain" },
        { id: "diarrhea", label: "Diarrhea" },
        { id: "sweating", label: "Sweating" },
        { id: "rapid_breathing", label: "Rapid Breathing" },
        { id: "dizziness", label: "Dizziness" },
    ];

    const CHART_COLORS = {
        teal: "rgba(0, 212, 170, 0.8)",
        tealBg: "rgba(0, 212, 170, 0.15)",
        violet: "rgba(124, 58, 237, 0.8)",
        violetBg: "rgba(124, 58, 237, 0.15)",
        sky: "rgba(56, 189, 248, 0.8)",
        skyBg: "rgba(56, 189, 248, 0.15)",
        rose: "rgba(244, 63, 94, 0.8)",
        roseBg: "rgba(244, 63, 94, 0.15)",
        amber: "rgba(245, 158, 11, 0.8)",
        amberBg: "rgba(245, 158, 11, 0.15)",
    };

    const BAR_COLORS = [CHART_COLORS.teal, CHART_COLORS.violet, CHART_COLORS.sky, CHART_COLORS.rose];
    const BAR_BG_COLORS = [CHART_COLORS.tealBg, CHART_COLORS.violetBg, CHART_COLORS.skyBg, CHART_COLORS.roseBg];

    // ──────────────────────────────────────────────────────────
    // State
    // ──────────────────────────────────────────────────────────
    let selectedFile = null;
    const selectedSymptoms = new Set();
    const charts = {};

    // ──────────────────────────────────────────────────────────
    // DOM Elements
    // ──────────────────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const dropzone = $("#dropzone");
    const dropzoneContent = $("#dropzoneContent");
    const fileInput = $("#fileInput");
    const imagePreview = $("#imagePreview");
    const clearImageBtn = $("#clearImage");
    const symptomGrid = $("#symptomGrid");
    const symptomSearch = $("#symptomSearch");
    const selectedCountEl = $("#selectedCount");
    const analyzeBtn = $("#analyzeBtn");
    const btnLoader = $("#btnLoader");
    const resultsSection = $("#resultsSection");
    const statusDot = $("#statusDot");
    const statusText = $("#statusText");

    // ──────────────────────────────────────────────────────────
    // Initialize
    // ──────────────────────────────────────────────────────────
    function init() {
        renderSymptomChips();
        setupDropzone();
        setupSymptomSearch();
        setupAnalyzeButton();
        setupDisclaimer();
        checkHealth();

        // Set up Chart.js defaults
        Chart.defaults.color = "#94a3b8";
        Chart.defaults.borderColor = "rgba(100, 116, 139, 0.15)";
        Chart.defaults.font.family = "'Inter', sans-serif";
    }

    // ──────────────────────────────────────────────────────────
    // Health Check
    // ──────────────────────────────────────────────────────────
    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/api/health`);
            if (res.ok) {
                const data = await res.json();
                statusDot.classList.add("online");
                statusDot.classList.remove("error");
                const loaded = Object.values(data.models_loaded).filter(Boolean).length;
                statusText.textContent = `Online • ${loaded}/4 models`;
            } else {
                throw new Error("API not healthy");
            }
        } catch {
            statusDot.classList.add("error");
            statusDot.classList.remove("online");
            statusText.textContent = "Offline";
        }
    }

    // ──────────────────────────────────────────────────────────
    // Disclaimer
    // ──────────────────────────────────────────────────────────
    function setupDisclaimer() {
        const closeBtn = $("#disclaimerClose");
        const banner = $("#disclaimerBanner");
        if (closeBtn && banner) {
            closeBtn.addEventListener("click", () => {
                banner.style.animation = "slideDown 0.3s ease reverse forwards";
                setTimeout(() => banner.classList.add("hidden"), 300);
            });
        }
    }

    // ──────────────────────────────────────────────────────────
    // Dropzone / Image Upload
    // ──────────────────────────────────────────────────────────
    function setupDropzone() {
        dropzone.addEventListener("click", () => fileInput.click());
        dropzone.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") fileInput.click();
        });

        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.classList.add("drag-over");
        });

        dropzone.addEventListener("dragleave", () => {
            dropzone.classList.remove("drag-over");
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("drag-over");
            const files = e.dataTransfer.files;
            if (files.length > 0) handleFile(files[0]);
        });

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) handleFile(e.target.files[0]);
        });

        clearImageBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            clearImage();
        });
    }

    function handleFile(file) {
        if (!file.type.startsWith("image/")) return;
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove("hidden");
            clearImageBtn.classList.remove("hidden");
            dropzoneContent.classList.add("hidden");
        };
        reader.readAsDataURL(file);
        updateAnalyzeButton();
    }

    function clearImage() {
        selectedFile = null;
        imagePreview.src = "";
        imagePreview.classList.add("hidden");
        clearImageBtn.classList.add("hidden");
        dropzoneContent.classList.remove("hidden");
        fileInput.value = "";
        updateAnalyzeButton();
    }

    // ──────────────────────────────────────────────────────────
    // Symptom Chips
    // ──────────────────────────────────────────────────────────
    function renderSymptomChips(filter = "") {
        symptomGrid.innerHTML = "";
        const lower = filter.toLowerCase();
        SYMPTOMS.forEach((sym) => {
            if (lower && !sym.label.toLowerCase().includes(lower) && !sym.id.includes(lower)) return;

            const chip = document.createElement("button");
            chip.className = "symptom-chip" + (selectedSymptoms.has(sym.id) ? " active" : "");
            chip.textContent = sym.label;
            chip.dataset.id = sym.id;
            chip.setAttribute("role", "checkbox");
            chip.setAttribute("aria-checked", selectedSymptoms.has(sym.id));

            chip.addEventListener("click", () => toggleSymptom(sym.id, chip));
            symptomGrid.appendChild(chip);
        });
    }

    function toggleSymptom(id, chipEl) {
        if (selectedSymptoms.has(id)) {
            selectedSymptoms.delete(id);
            chipEl.classList.remove("active");
            chipEl.setAttribute("aria-checked", "false");
        } else {
            selectedSymptoms.add(id);
            chipEl.classList.add("active");
            chipEl.setAttribute("aria-checked", "true");
        }
        selectedCountEl.textContent = selectedSymptoms.size;
        updateAnalyzeButton();
    }

    function setupSymptomSearch() {
        symptomSearch.addEventListener("input", (e) => {
            renderSymptomChips(e.target.value);
        });
    }

    // ──────────────────────────────────────────────────────────
    // Clinical Data
    // ──────────────────────────────────────────────────────────
    function getClinicalData() {
        const fields = ["age", "temperature", "heart_rate", "bp_systolic", "bp_diastolic", "respiratory_rate", "oxygen_saturation"];
        const data = {};
        let hasAny = false;
        fields.forEach((f) => {
            const el = document.getElementById(f);
            if (el && el.value !== "") {
                data[f] = parseFloat(el.value);
                hasAny = true;
            }
        });
        return hasAny ? data : null;
    }

    function getSymptomsDict() {
        const dict = {};
        SYMPTOMS.forEach((s) => {
            dict[s.id] = selectedSymptoms.has(s.id) ? 1 : 0;
        });
        return dict;
    }

    // ──────────────────────────────────────────────────────────
    // Analyze Button
    // ──────────────────────────────────────────────────────────
    function updateAnalyzeButton() {
        const hasData = selectedFile || selectedSymptoms.size > 0;
        analyzeBtn.disabled = !hasData;
    }

    function setupAnalyzeButton() {
        analyzeBtn.addEventListener("click", runAnalysis);
        // Listen for clinical input changes too
        $$(".clinical-input").forEach((el) => {
            el.addEventListener("input", updateAnalyzeButton);
        });
    }

    // ──────────────────────────────────────────────────────────
    // Run Analysis
    // ──────────────────────────────────────────────────────────
    async function runAnalysis() {
        analyzeBtn.disabled = true;
        btnLoader.classList.remove("hidden");
        $(".btn-text").textContent = "Analyzing...";

        // Show results section
        resultsSection.classList.remove("hidden");
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

        const hasImage = selectedFile !== null;
        const hasSymptoms = selectedSymptoms.size > 0;
        const clinicalData = getClinicalData();
        const hasClinical = clinicalData !== null && hasSymptoms;

        const promises = [];

        // Individual predictions
        if (hasImage) promises.push(predictImage());
        if (hasSymptoms) promises.push(predictSymptoms());
        if (hasClinical) promises.push(predictClinical(clinicalData));

        // Multimodal
        promises.push(predictMultimodal(clinicalData));

        // Explainability
        if (hasImage) promises.push(explainImage());
        if (hasClinical) promises.push(explainClinical(clinicalData));

        try {
            await Promise.allSettled(promises);
        } catch (err) {
            console.error("Analysis error:", err);
        }

        // Build comparison if we have multiple
        buildComparison(hasImage, hasSymptoms, hasClinical);

        btnLoader.classList.add("hidden");
        $(".btn-text").textContent = "Analyze";
        analyzeBtn.disabled = false;
    }

    // ──────────────────────────────────────────────────────────
    // API Calls
    // ──────────────────────────────────────────────────────────
    async function predictImage() {
        const card = $("#imageResultCard");
        try {
            const formData = new FormData();
            formData.append("file", selectedFile);
            const res = await fetch(`${API_BASE}/api/predict/image`, { method: "POST", body: formData });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Image prediction failed");

            card.classList.remove("hidden");
            $("#imagePrediction").textContent = data.predicted_class;
            $("#imageConfidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
            renderChart("imageChart", data.predictions, "Image Predictions");
            return data;
        } catch (err) {
            console.error("Image prediction error:", err);
            card.classList.remove("hidden");
            $("#imagePrediction").textContent = "Error: " + err.message;
        }
    }

    async function predictSymptoms() {
        const card = $("#symptomResultCard");
        try {
            const res = await fetch(`${API_BASE}/api/predict/symptoms`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ symptoms: getSymptomsDict() }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Symptom prediction failed");

            card.classList.remove("hidden");
            $("#symptomPrediction").textContent = data.predicted_class;
            $("#symptomConfidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
            renderChart("symptomChart", data.predictions, "Symptom Predictions");
            return data;
        } catch (err) {
            console.error("Symptom prediction error:", err);
            card.classList.remove("hidden");
            $("#symptomPrediction").textContent = "Error: " + err.message;
        }
    }

    async function predictClinical(clinicalData) {
        const card = $("#clinicalResultCard");
        try {
            const res = await fetch(`${API_BASE}/api/predict/clinical`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    symptoms: getSymptomsDict(),
                    clinical_data: clinicalData,
                }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Clinical prediction failed");

            card.classList.remove("hidden");
            $("#clinicalPrediction").textContent = data.predicted_class;
            $("#clinicalConfidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
            renderChart("clinicalChart", data.predictions, "Clinical Risk");
            return data;
        } catch (err) {
            console.error("Clinical prediction error:", err);
            card.classList.remove("hidden");
            $("#clinicalPrediction").textContent = "Error: " + err.message;
        }
    }

    async function predictMultimodal(clinicalData) {
        const card = $("#fusionCard");
        try {
            const formData = new FormData();
            if (selectedFile) formData.append("file", selectedFile);
            if (selectedSymptoms.size > 0) formData.append("symptoms", JSON.stringify(getSymptomsDict()));
            if (clinicalData && selectedSymptoms.size > 0) formData.append("clinical_data", JSON.stringify(clinicalData));

            const res = await fetch(`${API_BASE}/api/predict/multimodal`, { method: "POST", body: formData });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Multimodal prediction failed");

            card.classList.remove("hidden");
            $("#fusionPrediction").textContent = data.predicted_class;
            $("#fusionConfidence").textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

            // Modalities badges
            const used = $("#modalitiesUsed");
            used.innerHTML = "";
            data.modalities_used.forEach((m) => {
                const badge = document.createElement("span");
                badge.className = `modality-badge badge-${m === "image" ? "image" : m === "symptoms" ? "symptom" : "clinical"}`;
                badge.textContent = m.charAt(0).toUpperCase() + m.slice(1);
                used.appendChild(badge);
            });

            renderChart("fusionChart", data.unified_predictions, "Unified Assessment", true);
            return data;
        } catch (err) {
            console.error("Multimodal prediction error:", err);
            card.classList.remove("hidden");
            $("#fusionPrediction").textContent = "Error: " + err.message;
        }
    }

    async function explainImage() {
        const card = $("#gradcamCard");
        try {
            const formData = new FormData();
            formData.append("file", selectedFile);
            const res = await fetch(`${API_BASE}/api/explain/image`, { method: "POST", body: formData });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Grad-CAM failed");

            card.classList.remove("hidden");
            $("#gradcamImage").src = `data:image/png;base64,${data.gradcam_overlay_base64}`;
        } catch (err) {
            console.error("Grad-CAM error:", err);
        }
    }

    async function explainClinical(clinicalData) {
        const card = $("#shapCard");
        try {
            const res = await fetch(`${API_BASE}/api/explain/clinical`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    symptoms: getSymptomsDict(),
                    clinical_data: clinicalData,
                }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "SHAP failed");

            card.classList.remove("hidden");
            $("#shapImage").src = `data:image/png;base64,${data.plot_base64}`;
        } catch (err) {
            console.error("SHAP error:", err);
        }
    }

    // ──────────────────────────────────────────────────────────
    // Chart Rendering
    // ──────────────────────────────────────────────────────────
    function renderChart(canvasId, predictions, title, horizontal = false) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        // Destroy old chart if exists
        if (charts[canvasId]) {
            charts[canvasId].destroy();
        }

        const labels = predictions.map((p) => p.disease);
        const data = predictions.map((p) => (p.probability * 100).toFixed(1));
        const colors = labels.map((_, i) => BAR_COLORS[i % BAR_COLORS.length]);
        const bgColors = labels.map((_, i) => BAR_BG_COLORS[i % BAR_BG_COLORS.length]);

        charts[canvasId] = new Chart(canvas, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Probability (%)",
                        data: data,
                        backgroundColor: bgColors,
                        borderColor: colors,
                        borderWidth: 2,
                        borderRadius: 6,
                        barPercentage: 0.7,
                    },
                ],
            },
            options: {
                indexAxis: horizontal ? "y" : "x",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: false,
                    },
                    tooltip: {
                        backgroundColor: "rgba(17, 22, 56, 0.95)",
                        titleColor: "#e8ecf4",
                        bodyColor: "#94a3b8",
                        borderColor: "rgba(100, 116, 139, 0.2)",
                        borderWidth: 1,
                        cornerRadius: 8,
                        padding: 10,
                        callbacks: {
                            label: (ctx) => `${ctx.parsed[horizontal ? "x" : "y"]}%`,
                        },
                    },
                },
                scales: {
                    [horizontal ? "x" : "y"]: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: "rgba(100, 116, 139, 0.08)" },
                        ticks: {
                            callback: (v) => v + "%",
                            font: { size: 11 },
                        },
                    },
                    [horizontal ? "y" : "x"]: {
                        grid: { display: false },
                        ticks: { font: { size: 11, weight: "500" } },
                    },
                },
                animation: {
                    duration: 800,
                    easing: "easeOutQuart",
                },
            },
        });
    }

    function buildComparison(hasImage, hasSymptoms, hasClinical) {
        const card = $("#comparisonCard");
        const count = [hasImage, hasSymptoms, hasClinical].filter(Boolean).length;
        if (count < 2) return;

        card.classList.remove("hidden");

        const canvas = document.getElementById("comparisonChart");
        if (charts["comparisonChart"]) charts["comparisonChart"].destroy();

        // Collect confidence from each modality
        const labels = [];
        const confidences = [];
        const colors = [];

        if (hasImage) {
            const conf = $("#imageConfidence").textContent;
            labels.push("Image");
            confidences.push(parseFloat(conf) || 0);
            colors.push(CHART_COLORS.teal);
        }
        if (hasSymptoms) {
            const conf = $("#symptomConfidence").textContent;
            labels.push("Symptoms");
            confidences.push(parseFloat(conf) || 0);
            colors.push(CHART_COLORS.violet);
        }
        if (hasClinical) {
            const conf = $("#clinicalConfidence").textContent;
            labels.push("Clinical");
            confidences.push(parseFloat(conf) || 0);
            colors.push(CHART_COLORS.sky);
        }

        // Add fusion
        const fusionConf = $("#fusionConfidence").textContent;
        const fusionVal = parseFloat(fusionConf.replace(/[^0-9.]/g, "")) || 0;
        labels.push("Fusion");
        confidences.push(fusionVal);
        colors.push(CHART_COLORS.amber);

        charts["comparisonChart"] = new Chart(canvas, {
            type: "radar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Confidence %",
                        data: confidences,
                        backgroundColor: "rgba(0, 212, 170, 0.1)",
                        borderColor: CHART_COLORS.teal,
                        borderWidth: 2,
                        pointBackgroundColor: colors,
                        pointBorderColor: "#fff",
                        pointBorderWidth: 1,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: "rgba(100, 116, 139, 0.12)" },
                        angleLines: { color: "rgba(100, 116, 139, 0.12)" },
                        pointLabels: {
                            font: { size: 13, weight: "600", family: "'Outfit', sans-serif" },
                            color: "#e8ecf4",
                        },
                        ticks: {
                            display: false,
                            stepSize: 25,
                        },
                    },
                },
                animation: { duration: 1000, easing: "easeOutQuart" },
            },
        });
    }

    // ──────────────────────────────────────────────────────────
    // Boot
    // ──────────────────────────────────────────────────────────
    document.addEventListener("DOMContentLoaded", init);
})();
