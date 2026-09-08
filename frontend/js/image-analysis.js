/**
 * MediFusion AI — Advanced Multi-Model Image Analysis Module
 * Handles model selection, drag/drop, multi-endpoint inference,
 * clinical context display, and Grad-CAM inspection.
 */

(function () {
    "use strict";

    // ── State ──
    let currentFile = null;
    let originalDataUrl = null;
    let gradcamOverlayBase64 = null;
    let gradcamHeatmapBase64 = null;
    let zoomLevel = 1;
    let selectedModelKey = null;
    let modelsMetadata = {};
    let modelsStatus = {};

    // ── Model Definitions (fallback if API unavailable) ──
    const MODEL_DEFS = [
        {
            key: "chest_xray",
            title: "Chest X-Ray",
            icon: "🫁",
            classes: "2-class",
            imageType: "grayscale",
            description: "Pneumonia Detection",
        },
        {
            key: "brain_tumor",
            title: "Brain Tumor MRI",
            icon: "🧠",
            classes: "4-class",
            imageType: "grayscale",
            description: "Intracranial Neoplasm",
        },
        {
            key: "skin_cancer",
            title: "Skin Cancer",
            icon: "🔬",
            classes: "7-class",
            imageType: "color",
            description: "Dermatological Lesion",
        },
        {
            key: "retinopathy",
            title: "Retinopathy",
            icon: "👁",
            classes: "5-class",
            imageType: "color",
            description: "DR Severity Grading",
        },
        {
            key: "blood_cell",
            title: "Blood Cell",
            icon: "🩸",
            classes: "4-class",
            imageType: "color",
            description: "Hematological Microscopy",
        },
    ];

    document.addEventListener("DOMContentLoaded", () => {
        initModelSelector();
        setupUpload();
        setupControls();
        loadModelsStatus();
    });

    // ── Model Selector ──

    async function loadModelsStatus() {
        try {
            const statusData = await MediFusionAPI.getImageModelsStatus();
            if (statusData && statusData.models) {
                statusData.models.forEach(m => {
                    modelsStatus[m.model_key] = m.loaded;
                });
                updateModelCards();
            }
        } catch (e) {
            console.warn("Could not load image model status:", e);
        }

        try {
            const metaData = await MediFusionAPI.getImageModelsMetadata();
            if (metaData) {
                modelsMetadata = metaData;
            }
        } catch (e) {
            console.warn("Could not load image model metadata:", e);
        }
    }

    function initModelSelector() {
        const grid = document.getElementById("modelSelectorGrid");
        if (!grid) return;

        grid.innerHTML = "";

        MODEL_DEFS.forEach(def => {
            const card = document.createElement("div");
            card.className = "model-card";
            card.dataset.modelKey = def.key;

            card.innerHTML = `
                <div class="model-card-status not-loaded" id="status-${def.key}"></div>
                <div class="model-card-icon">${def.icon}</div>
                <div class="model-card-title">${def.title}</div>
                <div class="model-card-classes">${def.classes} • ${def.description}</div>
            `;

            card.addEventListener("click", () => selectModel(def.key));
            grid.appendChild(card);
        });
    }

    function updateModelCards() {
        MODEL_DEFS.forEach(def => {
            const statusDot = document.getElementById(`status-${def.key}`);
            if (statusDot) {
                const loaded = modelsStatus[def.key] || false;
                statusDot.className = `model-card-status ${loaded ? "loaded" : "not-loaded"}`;
            }
        });
    }

    function selectModel(modelKey) {
        selectedModelKey = modelKey;

        // Update card selection
        document.querySelectorAll(".model-card").forEach(card => {
            card.classList.toggle("selected", card.dataset.modelKey === modelKey);
        });

        // Update dropzone text
        const def = MODEL_DEFS.find(d => d.key === modelKey);
        const dropzoneMain = document.getElementById("dropzoneMainText");
        const dropzoneSub = document.getElementById("dropzoneSubText");
        const dropzoneNotice = document.getElementById("dropzoneNotice");
        const analyzeBtn = document.getElementById("runImageAnalysisBtn");
        const btnText = document.getElementById("imageBtnText");
        const backboneTag = document.getElementById("modelBackboneTag");

        if (dropzoneMain) dropzoneMain.textContent = `Drag & drop ${def ? def.title : "medical"} image`;
        if (dropzoneSub) dropzoneSub.textContent = `${def?.imageType === "grayscale" ? "Grayscale" : "Color"} JPEG, PNG • Max 15MB`;
        if (dropzoneNotice) {
            dropzoneNotice.innerHTML = `
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                <span>${def ? def.description : "Analysis"} pipeline selected</span>
            `;
        }

        if (btnText) {
            if (currentFile) {
                btnText.textContent = `Analyze with ${def?.title || "Model"}`;
                if (analyzeBtn) analyzeBtn.disabled = false;
            } else {
                btnText.textContent = `Upload Image for ${def?.title || "Analysis"}`;
            }
        }

        if (backboneTag) backboneTag.textContent = "ResNet18";

        // Update model info box
        const infoBox = document.getElementById("modelInfoBox");
        if (infoBox && def) {
            const meta = modelsMetadata[modelKey] || {};
            infoBox.innerHTML = `
                <strong style="color:var(--text-secondary);">${def.title}:</strong>
                ${meta.subtitle || def.description}. ${def.classes} classification using ${meta.backbone || "ResNet-18"} transfer learning
                ${meta.dataset_source ? `<br><span style="color:var(--text-muted); font-size:0.72rem;">Dataset: ${meta.dataset_source}</span>` : ""}.
            `;
        }
    }

    // ── Upload & Image Handling ──

    function setupUpload() {
        const dropzone = document.getElementById("imageDropzone");
        const fileInput = document.getElementById("imageFileInput");
        const removeBtn = document.getElementById("removeImageBtn");
        const analyzeBtn = document.getElementById("runImageAnalysisBtn");

        if (!dropzone || !fileInput) return;

        dropzone.addEventListener("click", () => fileInput.click());

        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.classList.add("drag-active");
        });

        dropzone.addEventListener("dragleave", () => {
            dropzone.classList.remove("drag-active");
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("drag-active");
            if (e.dataTransfer.files.length > 0) {
                loadFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                loadFile(e.target.files[0]);
            }
        });

        if (removeBtn) {
            removeBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                clearImage();
            });
        }

        if (analyzeBtn) {
            analyzeBtn.addEventListener("click", runImageInference);
        }
    }

    async function loadFile(file) {
        if (!file.type.startsWith("image/")) {
            MediFusionUI.showToast("error", "Please select a valid image file (PNG, JPG, JPEG).");
            return;
        }

        // Only run strict radiograph validation for chest_xray mode
        if (selectedModelKey === "chest_xray" && window.ImageValidator) {
            const dropzone = document.getElementById("imageDropzone");
            MediFusionUI.showToast("info", "Validating radiograph structure...");

            try {
                const result = await ImageValidator.validate(file);
                if (!result.valid) {
                    const title = ImageValidator.getErrorTitle(result.code);
                    MediFusionUI.showToast("error", result.error, 6000);
                    if (dropzone) {
                        MediFusionUI.showValidationError(dropzone, title, result.error, 6000);
                    }
                    const fileInput = document.getElementById("imageFileInput");
                    if (fileInput) fileInput.value = "";
                    return;
                }
            } catch (err) {
                console.warn("Image validation error, allowing upload:", err);
            }
        }

        loadFileDirectly(file);
    }

    function loadFileDirectly(file) {
        const dropzone = document.getElementById("imageDropzone");
        if (dropzone) {
            const existingOverlay = dropzone.querySelector(".validation-error-overlay");
            if (existingOverlay) existingOverlay.remove();
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            originalDataUrl = e.target.result;
            const previewImg = document.getElementById("previewImg");
            const dropzoneInitial = document.getElementById("dropzoneInitial");
            const previewContainer = document.getElementById("previewContainer");
            const analyzeBtn = document.getElementById("runImageAnalysisBtn");
            const fileMeta = document.getElementById("fileMetadataPill");
            const btnText = document.getElementById("imageBtnText");

            if (previewImg) previewImg.src = originalDataUrl;
            if (dropzoneInitial) dropzoneInitial.style.display = "none";
            if (previewContainer) previewContainer.style.display = "flex";

            if (selectedModelKey) {
                if (analyzeBtn) analyzeBtn.disabled = false;
                const def = MODEL_DEFS.find(d => d.key === selectedModelKey);
                if (btnText) btnText.textContent = `Analyze with ${def?.title || "Model"}`;
            } else {
                if (btnText) btnText.textContent = "Select a Model First";
            }

            if (fileMeta) {
                const sizeKb = (file.size / 1024).toFixed(1);
                fileMeta.textContent = `${file.name} • ${sizeKb} KB`;
            }
        };
        reader.readAsDataURL(file);
    }

    function clearImage() {
        currentFile = null;
        originalDataUrl = null;
        gradcamOverlayBase64 = null;
        gradcamHeatmapBase64 = null;

        const dropzoneInitial = document.getElementById("dropzoneInitial");
        const previewContainer = document.getElementById("previewContainer");
        const fileInput = document.getElementById("imageFileInput");
        const analyzeBtn = document.getElementById("runImageAnalysisBtn");
        const btnText = document.getElementById("imageBtnText");

        if (dropzoneInitial) dropzoneInitial.style.display = "block";
        if (previewContainer) previewContainer.style.display = "none";
        if (fileInput) fileInput.value = "";
        if (analyzeBtn) analyzeBtn.disabled = true;
        if (btnText) btnText.textContent = selectedModelKey ? "Upload an Image" : "Select a Model to Begin";

        const emptyStage = document.getElementById("emptyStage");
        const resultsStage = document.getElementById("imageResultsStage");
        if (emptyStage) emptyStage.style.display = "flex";
        if (resultsStage) resultsStage.style.display = "none";
    }

    // ── Inference ──

    async function runImageInference() {
        if (!currentFile) {
            MediFusionUI.showToast("error", "Please upload an image first.");
            return;
        }
        if (!selectedModelKey) {
            MediFusionUI.showToast("error", "Please select an imaging model first.");
            return;
        }

        const analyzeBtn = document.getElementById("runImageAnalysisBtn");
        const btnText = document.getElementById("imageBtnText");
        const scannerBeam = document.getElementById("imageScannerBeam");
        const emptyStage = document.getElementById("emptyStage");
        const resultsStage = document.getElementById("imageResultsStage");

        if (analyzeBtn) analyzeBtn.disabled = true;
        if (btnText) btnText.textContent = "Analyzing...";
        if (scannerBeam) scannerBeam.style.display = "block";

        try {
            let predData, explainData;

            if (selectedModelKey === "chest_xray") {
                // Use original endpoints for backward compatibility
                [predData, explainData] = await Promise.all([
                    MediFusionAPI.predictImage(currentFile),
                    MediFusionAPI.explainImage(currentFile),
                ]);
                // Normalize chest_xray response to match advanced format
                predData = normalizeChestXrayResponse(predData);
            } else {
                // Advanced model endpoints
                [predData, explainData] = await Promise.all([
                    MediFusionAPI.predictImageAdvanced(selectedModelKey, currentFile),
                    MediFusionAPI.explainImageAdvanced(selectedModelKey, currentFile).catch(err => {
                        console.warn("Grad-CAM not available for this model:", err);
                        return null;
                    }),
                ]);
            }

            if (emptyStage) emptyStage.style.display = "none";
            if (resultsStage) resultsStage.style.display = "flex";

            renderResults(predData);
            renderGradCAM(explainData);

            MediFusionUI.showToast("success", "Analysis complete with clinical context.");
        } catch (err) {
            console.error("Image analysis error:", err);
            const msg = err.message || "Unable to complete image analysis.";
            MediFusionUI.showToast("error", msg, 6000);
        } finally {
            if (analyzeBtn) analyzeBtn.disabled = false;
            const def = MODEL_DEFS.find(d => d.key === selectedModelKey);
            if (btnText) btnText.textContent = `Analyze with ${def?.title || "Model"}`;
            if (scannerBeam) scannerBeam.style.display = "none";
        }
    }

    function normalizeChestXrayResponse(data) {
        // Convert original chest_xray response to AdvancedImagePredictionResponse format
        const meta = modelsMetadata["chest_xray"] || {};
        const classesMeta = meta.classes || {};

        const predictions = (data.predictions || []).map(p => {
            const cmeta = classesMeta[p.disease] || {};
            return {
                class_id: p.disease,
                display_name: cmeta.display_name || p.disease,
                probability: p.probability,
                severity: cmeta.severity || (p.disease === "PNEUMONIA" ? "high" : "none"),
                color: cmeta.color || "#3b82f6",
                description: cmeta.description || "",
                action: cmeta.action || "",
            };
        });

        predictions.sort((a, b) => b.probability - a.probability);

        const primary = predictions[0] || {};
        return {
            model_key: "chest_xray",
            model_title: meta.title || "Chest X-Ray Analysis",
            backbone: "ResNet-18",
            predicted_class: data.predicted_class,
            predicted_display_name: primary.display_name || data.predicted_class,
            confidence: data.confidence,
            severity: primary.severity || "none",
            predictions: predictions,
            clinical_context: primary.description || "",
            recommended_action: primary.action || "",
        };
    }

    // ── Result Rendering ──

    function renderResults(data) {
        // Model label
        const modelLabel = document.getElementById("resultModelLabel");
        if (modelLabel) modelLabel.textContent = data.model_title || data.model_key;

        // Predicted condition
        const conditionEl = document.getElementById("imagePredCondition");
        if (conditionEl) conditionEl.textContent = data.predicted_display_name || data.predicted_class;

        // Confidence
        const confEl = document.getElementById("imagePredConfidence");
        if (confEl) {
            MediFusionUI.animateNumber(confEl, 0, data.confidence * 100, 800, "%");
        }

        // Severity badge
        const severityBadge = document.getElementById("imageSeverityBadge");
        if (severityBadge) {
            const sev = data.severity || "none";
            severityBadge.className = `severity-badge severity-${sev}`;
            const labels = {
                critical: "⚠ Critical",
                high: "⚠ High Severity",
                moderate: "Moderate",
                low: "Low Risk",
                none: "✓ Normal",
                info: "ℹ Informational",
            };
            severityBadge.textContent = labels[sev] || sev;
        }

        // Predictions list
        const listContainer = document.getElementById("predictionsListContainer");
        if (listContainer && data.predictions) {
            listContainer.innerHTML = "";
            data.predictions.forEach((p, idx) => {
                const isPrimary = idx === 0;
                const pct = (p.probability * 100).toFixed(1);

                const row = document.createElement("div");
                row.className = `pred-row${isPrimary ? " primary" : ""}`;
                row.innerHTML = `
                    <div class="pred-color-dot" style="background:${p.color || "#3b82f6"};"></div>
                    <div class="pred-label">${p.display_name || p.class_id}</div>
                    <div class="pred-bar-container">
                        <div class="pred-bar-fill" style="width:${pct}%; background:${p.color || "#3b82f6"};"></div>
                    </div>
                    <div class="pred-percent">${pct}%</div>
                `;
                listContainer.appendChild(row);
            });
        }

        // Clinical context
        const contextBox = document.getElementById("clinicalContextBox");
        const contextText = document.getElementById("clinicalContextText");
        const actionBox = document.getElementById("actionBox");
        const actionText = document.getElementById("actionText");

        if (data.clinical_context && contextBox) {
            contextBox.style.display = "block";
            if (contextText) contextText.textContent = data.clinical_context;

            if (data.recommended_action && actionBox) {
                actionBox.style.display = "block";
                if (actionText) actionText.textContent = data.recommended_action;
            } else if (actionBox) {
                actionBox.style.display = "none";
            }
        } else if (contextBox) {
            contextBox.style.display = "none";
        }
    }

    function renderGradCAM(explainData) {
        const origLayer = document.getElementById("viewerOrigImg");
        const heatLayer = document.getElementById("viewerHeatmapImg");
        const overlayLayer = document.getElementById("viewerOverlayImg");

        if (origLayer) origLayer.src = originalDataUrl || "";

        if (explainData && explainData.gradcam_overlay_base64) {
            gradcamOverlayBase64 = explainData.gradcam_overlay_base64;
            gradcamHeatmapBase64 = explainData.gradcam_heatmap_base64;

            if (heatLayer) heatLayer.src = `data:image/png;base64,${gradcamHeatmapBase64}`;
            if (overlayLayer) overlayLayer.src = `data:image/png;base64,${gradcamOverlayBase64}`;
        } else {
            // No Grad-CAM available — show original image
            if (heatLayer) heatLayer.src = originalDataUrl || "";
            if (overlayLayer) overlayLayer.src = originalDataUrl || "";
        }

        setViewerMode("overlay");
    }

    // ── Viewer Controls ──

    function setupControls() {
        const tabBtns = document.querySelectorAll(".viewer-tab-btn");
        tabBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                tabBtns.forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                setViewerMode(btn.dataset.mode);
            });
        });

        const opacitySlider = document.getElementById("gradcamOpacitySlider");
        const opacityVal = document.getElementById("opacityValLabel");
        if (opacitySlider) {
            opacitySlider.addEventListener("input", (e) => {
                const val = e.target.value;
                if (opacityVal) opacityVal.textContent = `${val}%`;
                const heatLayer = document.getElementById("viewerHeatmapImg");
                if (heatLayer) heatLayer.style.opacity = val / 100;
            });
        }

        const zoomIn = document.getElementById("zoomInBtn");
        const zoomOut = document.getElementById("zoomOutBtn");
        const zoomReset = document.getElementById("zoomResetBtn");
        const stageImages = document.querySelectorAll(".stage-layer img");

        if (zoomIn) {
            zoomIn.addEventListener("click", () => {
                zoomLevel = Math.min(zoomLevel + 0.25, 2.5);
                applyZoom(stageImages);
            });
        }
        if (zoomOut) {
            zoomOut.addEventListener("click", () => {
                zoomLevel = Math.max(zoomLevel - 0.25, 0.75);
                applyZoom(stageImages);
            });
        }
        if (zoomReset) {
            zoomReset.addEventListener("click", () => {
                zoomLevel = 1;
                applyZoom(stageImages);
            });
        }
    }

    function setViewerMode(mode) {
        const origLayer = document.getElementById("viewerOrigImg");
        const heatLayer = document.getElementById("viewerHeatmapImg");
        const overlayLayer = document.getElementById("viewerOverlayImg");
        const opacityRow = document.getElementById("opacitySliderRow");

        if (!origLayer || !heatLayer || !overlayLayer) return;

        if (mode === "original") {
            origLayer.style.display = "block";
            heatLayer.style.display = "none";
            overlayLayer.style.display = "none";
            if (opacityRow) opacityRow.style.display = "none";
        } else if (mode === "heatmap") {
            origLayer.style.display = "none";
            heatLayer.style.display = "block";
            heatLayer.style.opacity = "1";
            overlayLayer.style.display = "none";
            if (opacityRow) opacityRow.style.display = "none";
        } else if (mode === "overlay") {
            origLayer.style.display = "none";
            heatLayer.style.display = "none";
            overlayLayer.style.display = "block";
            if (opacityRow) opacityRow.style.display = "none";
        }
    }

    function applyZoom(images) {
        images.forEach((img) => {
            img.style.transform = `scale(${zoomLevel})`;
        });
    }
})();
