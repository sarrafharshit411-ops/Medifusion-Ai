/**
 * MediFusion AI — Dedicated Image Analysis Module
 * Handles drag/drop, ResNet18 inference, and interactive Grad-CAM inspection.
 */

(function () {
    "use strict";

    let currentFile = null;
    let originalDataUrl = null;
    let gradcamOverlayBase64 = null;
    let gradcamHeatmapBase64 = null;
    let zoomLevel = 1;

    document.addEventListener("DOMContentLoaded", () => {
        setupUpload();
        setupControls();
    });

    function setupUpload() {
        const dropzone = document.getElementById("imageDropzone");
        const fileInput = document.getElementById("imageFileInput");
        const dropzoneContent = document.getElementById("dropzoneInitial");
        const previewContainer = document.getElementById("previewContainer");
        const previewImg = document.getElementById("previewImg");
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

        // Run multi-layer medical image validation
        if (window.ImageValidator) {
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
                    // Reset file input
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
        // Dismiss any existing validation error overlay
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

            if (previewImg) previewImg.src = originalDataUrl;
            if (dropzoneInitial) dropzoneInitial.style.display = "none";
            if (previewContainer) previewContainer.style.display = "flex";
            if (analyzeBtn) analyzeBtn.disabled = false;

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

        if (dropzoneInitial) dropzoneInitial.style.display = "block";
        if (previewContainer) previewContainer.style.display = "none";
        if (fileInput) fileInput.value = "";
        if (analyzeBtn) analyzeBtn.disabled = true;

        const emptyStage = document.getElementById("emptyStage");
        const resultsStage = document.getElementById("imageResultsStage");
        if (emptyStage) emptyStage.style.display = "flex";
        if (resultsStage) resultsStage.style.display = "none";
    }

    async function runImageInference() {
        if (!currentFile) return;

        const analyzeBtn = document.getElementById("runImageAnalysisBtn");
        const btnText = document.getElementById("imageBtnText");
        const scannerBeam = document.getElementById("imageScannerBeam");
        const emptyStage = document.getElementById("emptyStage");
        const resultsStage = document.getElementById("imageResultsStage");

        if (analyzeBtn) analyzeBtn.disabled = true;
        if (btnText) btnText.textContent = "Analyzing Radiograph...";
        if (scannerBeam) scannerBeam.style.display = "block";

        try {
            // Run prediction and Grad-CAM in parallel
            const [predData, explainData] = await Promise.all([
                MediFusionAPI.predictImage(currentFile),
                MediFusionAPI.explainImage(currentFile),
            ]);

            if (emptyStage) emptyStage.style.display = "none";
            if (resultsStage) resultsStage.style.display = "flex";

            // Update primary card values
            const conditionEl = document.getElementById("imagePredCondition");
            const confEl = document.getElementById("imagePredConfidence");
            const badgeEl = document.getElementById("imageStatusBadge");

            if (conditionEl) conditionEl.textContent = predData.predicted_class;
            if (confEl) {
                MediFusionUI.animateNumber(confEl, 0, predData.confidence * 100, 800, "%");
            }

            if (badgeEl) {
                if (predData.predicted_class === "PNEUMONIA") {
                    badgeEl.className = "badge badge-warning";
                    badgeEl.textContent = "Pathology Detected";
                } else {
                    badgeEl.className = "badge badge-success";
                    badgeEl.textContent = "Normal Clearance";
                }
            }

            // Render Chart
            MediFusionCharts.renderProbabilityBar("imageProbabilityChart", predData.predictions, true);

            // Store explainability outputs
            gradcamOverlayBase64 = explainData.gradcam_overlay_base64;
            gradcamHeatmapBase64 = explainData.gradcam_heatmap_base64;

            // Update Viewer Stage Layers
            const origLayer = document.getElementById("viewerOrigImg");
            const heatLayer = document.getElementById("viewerHeatmapImg");
            const overlayLayer = document.getElementById("viewerOverlayImg");

            if (origLayer) origLayer.src = originalDataUrl;
            if (heatLayer) heatLayer.src = `data:image/png;base64,${gradcamHeatmapBase64}`;
            if (overlayLayer) overlayLayer.src = `data:image/png;base64,${gradcamOverlayBase64}`;

            setViewerMode("overlay");

            MediFusionUI.showToast("success", "Inference & Grad-CAM attention map generated.");
        } catch (err) {
            console.error("Image analysis error:", err);
            const msg = err.message || "Unable to complete image analysis. Please verify server connection.";
            MediFusionUI.showToast("error", msg, 6000);
            const dropzone = document.getElementById("imageDropzone");
            if (dropzone && (msg.toLowerCase().includes("invalid") || msg.toLowerCase().includes("radiograph") || msg.toLowerCase().includes("screenshot"))) {
                MediFusionUI.showValidationError(dropzone, "Medical Image Rejected", msg, 7000);
            }
        } finally {
            if (analyzeBtn) analyzeBtn.disabled = false;
            if (btnText) btnText.textContent = "Analyze Image";
            if (scannerBeam) scannerBeam.style.display = "none";
        }
    }

    function setupControls() {
        // Mode Tabs
        const tabBtns = document.querySelectorAll(".viewer-tab-btn");
        tabBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                tabBtns.forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                const mode = btn.dataset.mode;
                setViewerMode(mode);
            });
        });

        // Opacity Slider
        const opacitySlider = document.getElementById("gradcamOpacitySlider");
        const opacityVal = document.getElementById("opacityValLabel");
        if (opacitySlider) {
            opacitySlider.addEventListener("input", (e) => {
                const val = e.target.value;
                if (opacityVal) opacityVal.textContent = `${val}%`;
                const heatLayer = document.getElementById("viewerHeatmapImg");
                if (heatLayer) {
                    heatLayer.style.opacity = val / 100;
                }
            });
        }

        // Zoom Controls
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
