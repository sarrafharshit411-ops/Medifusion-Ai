/**
 * MediFusion AI — Medical Image Validator Module (Strict Clinical Edition)
 * Multi-layer client-side validation ensuring ONLY legitimate chest radiographs are processed.
 * Detects & rejects: desktop screenshots, software windows, documents, selfies, photos, memes, graphics.
 */

const ImageValidator = (function () {
    "use strict";

    const CONFIG = {
        maxFileSizeMB: 15,
        minWidth: 100,
        minHeight: 100,
        minAspectRatio: 0.60,  // Chest X-rays are typically square or portrait/slight landscape
        maxAspectRatio: 1.55,  // Rejects 16:9 (1.78), 16:10 (1.6), banners, and ultra-widescreen screenshots
        maxColorSaturation: 0.22,  // Medical X-rays are grayscale (saturation < 0.22 allows minor JPEG noise)
        maxChannelDelta: 14,       // Max difference between R, G, B channels
        maxTop3BinRatio: 0.72,     // Screenshots have flat palette (>72% pixels in 3 discrete bins)
        colorSampleSize: 3600,     // Pixel sample count for analysis
        allowedTypes: ["image/png", "image/jpeg", "image/jpg"],
    };

    /**
     * Main validation entry point.
     * Returns a Promise that resolves to { valid: boolean, error: string|null, code: string|null }
     */
    async function validate(file) {
        // Layer 1: File type check
        if (!file || !file.type) {
            return fail("INVALID_TYPE", "No file provided. Please select a chest X-ray image.");
        }

        const mime = file.type.toLowerCase();
        if (!CONFIG.allowedTypes.includes(mime)) {
            return fail("INVALID_TYPE",
                "Unsupported file format. Please upload a chest radiograph in JPEG or PNG format.");
        }

        // Layer 2: File size check
        const fileSizeMB = file.size / (1024 * 1024);
        if (fileSizeMB > CONFIG.maxFileSizeMB) {
            return fail("FILE_TOO_LARGE",
                `File size (${fileSizeMB.toFixed(1)} MB) exceeds the ${CONFIG.maxFileSizeMB} MB limit.`);
        }

        if (file.size < 1024) {
            return fail("FILE_TOO_SMALL",
                "File is too small to contain diagnostic radiograph data.");
        }

        // Layer 3-7: Deep canvas & pixel-level structural analysis
        try {
            const imgData = await loadImageData(file);

            // Layer 3: Minimum resolution check
            if (imgData.naturalWidth < CONFIG.minWidth || imgData.naturalHeight < CONFIG.minHeight) {
                return fail("TOO_SMALL",
                    `Image resolution (${imgData.naturalWidth}×${imgData.naturalHeight}) is too low for radiological analysis (minimum 100×100 required).`);
            }

            // Layer 4: Strict Aspect Ratio Check
            const aspectRatio = imgData.naturalWidth / imgData.naturalHeight;
            if (aspectRatio < CONFIG.minAspectRatio || aspectRatio > CONFIG.maxAspectRatio) {
                return fail("BAD_ASPECT_RATIO",
                    `Invalid aspect ratio (${aspectRatio.toFixed(2)}:1). Chest X-rays are typically square or standard clinical proportions (0.60 to 1.55). Desktop and phone screenshots are not supported.`);
            }

            // Layer 5: Color & Chromaticity Analysis (Strict Grayscale Check)
            const colorMetrics = analyzeColorAndChroma(imgData);
            if (colorMetrics.isColorImage) {
                return fail("TOO_COLORFUL",
                    `Color photograph or graphic detected (chroma delta: ${colorMetrics.avgChannelDelta.toFixed(1)}). Chest radiographs are strictly greyscale medical scans.`);
            }

            // Layer 6: Discrete Palette & Synthetic UI Screenshot Check
            const histMetrics = analyzeHistogramDistribution(imgData);
            if (histMetrics.isSyntheticFlat) {
                return fail("SYNTHETIC_SCREENSHOT",
                    "Software window or document screenshot detected. Diagnostic X-rays contain continuous tissue density gradients, not flat UI panels or text windows.");
            }

            // Layer 7: Geometric Straight Orthogonal Lines (UI Tables, Windows, Dialogs)
            const edgeMetrics = analyzeOrthogonalEdges(imgData);
            if (edgeMetrics.isUIOrDocument) {
                return fail("UI_OR_DOCUMENT",
                    "Computer interface, table, or document detected. Medical radiographs contain organic anatomical contours (ribs, cardiac silhouette, diaphragm), not rectangular UI boxes.");
            }

            // Layer 8: Blank or Extreme Exposure
            if (histMetrics.isExtremeExposure) {
                return fail("EXTREME_EXPOSURE",
                    "The image is overexposed or pitch black. Please upload a clear diagnostic radiograph.");
            }

            return { valid: true, error: null, code: null };

        } catch (err) {
            console.warn("ImageValidator: client analysis warning:", err);
            // If browser canvas analysis encounters an error, pass through to let server-side validator verify
            return { valid: true, error: null, code: null };
        }
    }

    /**
     * Load image into canvas for sub-pixel inspection
     */
    function loadImageData(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    try {
                        const sampleSize = 256;
                        const canvas = document.createElement("canvas");
                        canvas.width = sampleSize;
                        canvas.height = sampleSize;
                        const ctx = canvas.getContext("2d", { willReadFrequently: true });
                        ctx.drawImage(img, 0, 0, sampleSize, sampleSize);
                        const pixelData = ctx.getImageData(0, 0, sampleSize, sampleSize);
                        
                        resolve({
                            naturalWidth: img.naturalWidth,
                            naturalHeight: img.naturalHeight,
                            sampleSize: sampleSize,
                            pixels: pixelData.data,
                        });
                    } catch (err) {
                        reject(err);
                    }
                };
                img.onerror = () => reject(new Error("Image decode failure"));
                img.src = e.target.result;
            };
            reader.onerror = () => reject(new Error("File read failure"));
            reader.readAsDataURL(file);
        });
    }

    /**
     * Measure channel differences and color saturation
     */
    function analyzeColorAndChroma(imgData) {
        const pixels = imgData.pixels;
        let totalSaturation = 0;
        let totalChannelDelta = 0;
        let sampleCount = 0;
        const total = pixels.length;

        // Sample step
        for (let i = 0; i < total; i += 16) {
            const r = pixels[i];
            const g = pixels[i + 1];
            const b = pixels[i + 2];

            // Channel delta (r-g, g-b, b-r)
            const delta = (Math.abs(r - g) + Math.abs(g - b) + Math.abs(b - r)) / 3.0;
            totalChannelDelta += delta;

            // HSL saturation
            const max = Math.max(r, g, b) / 255;
            const min = Math.min(r, g, b) / 255;
            const l = (max + min) / 2;
            let s = 0;
            if (max !== min) {
                s = l > 0.5 ? (max - min) / (2 - max - min) : (max - min) / (max + min);
            }
            totalSaturation += s;
            sampleCount++;
        }

        const avgSaturation = totalSaturation / sampleCount;
        const avgChannelDelta = totalChannelDelta / sampleCount;

        return {
            avgSaturation,
            avgChannelDelta,
            isColorImage: avgSaturation > CONFIG.maxColorSaturation || avgChannelDelta > CONFIG.maxChannelDelta,
        };
    }

    /**
     * Analyze luminance histogram to detect synthetic flat palettes (screenshots)
     */
    function analyzeHistogramDistribution(imgData) {
        const pixels = imgData.pixels;
        const hist64 = new Array(64).fill(0);
        let totalBrightness = 0;
        let sampleCount = 0;

        for (let i = 0; i < pixels.length; i += 4) {
            const r = pixels[i];
            const g = pixels[i + 1];
            const b = pixels[i + 2];
            // Luminance
            const lum = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
            totalBrightness += lum;

            const bin = Math.min(63, Math.floor(lum / 4));
            hist64[bin]++;
            sampleCount++;
        }

        const avgBrightness = totalBrightness / sampleCount;

        // Sort bins
        const sortedBins = [...hist64].sort((a, b) => b - a);
        const top2Ratio = (sortedBins[0] + sortedBins[1]) / sampleCount;
        const top3Ratio = (sortedBins[0] + sortedBins[1] + sortedBins[2]) / sampleCount;

        return {
            avgBrightness,
            isSyntheticFlat: top2Ratio > 0.55 || top3Ratio > CONFIG.maxTop3BinRatio,
            isExtremeExposure: avgBrightness < 12 || avgBrightness > 242,
        };
    }

    /**
     * Detect straight horizontal & vertical UI grid lines / text blocks
     */
    function analyzeOrthogonalEdges(imgData) {
        const size = imgData.sampleSize;
        const pixels = imgData.pixels;
        
        // Convert to 2D grayscale array
        const gray = new Uint8Array(size * size);
        for (let i = 0; i < pixels.length; i += 4) {
            gray[i / 4] = Math.round(0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2]);
        }

        let horizDiffs = 0;
        let vertDiffs = 0;
        let strongStepEdges = 0;

        for (let y = 1; y < size - 1; y++) {
            for (let x = 1; x < size - 1; x++) {
                const idx = y * size + x;
                const gx = Math.abs(gray[idx + 1] - gray[idx - 1]);
                const gy = Math.abs(gray[idx + size] - gray[idx - size]);

                if (gx > 50 && gy < 12) horizDiffs++; // Strong vertical edge
                if (gy > 50 && gx < 12) vertDiffs++; // Strong horizontal edge
                if (gx > 80 || gy > 80) strongStepEdges++;
            }
        }

        const totalAnalyzed = (size - 2) * (size - 2);
        const pureOrthoRatio = (horizDiffs + vertDiffs) / totalAnalyzed;
        const sharpStepRatio = strongStepEdges / totalAnalyzed;

        // UI Screenshots with text windows and sharp borders have large pure orthogonal ratios
        const isUIOrDocument = pureOrthoRatio > 0.08 || (pureOrthoRatio > 0.05 && sharpStepRatio > 0.12);

        return {
            pureOrthoRatio,
            sharpStepRatio,
            isUIOrDocument,
        };
    }

    function fail(code, error) {
        return { valid: false, error, code };
    }

    /**
     * User-friendly title for modal/toast notification
     */
    function getErrorTitle(code) {
        const titles = {
            INVALID_TYPE: "Unsupported File Format",
            FILE_TOO_LARGE: "File Size Exceeded",
            FILE_TOO_SMALL: "File Too Small",
            TOO_SMALL: "Resolution Too Low",
            BAD_ASPECT_RATIO: "Invalid Aspect Ratio (Screenshot Detected)",
            TOO_COLORFUL: "Color Image Detected",
            SYNTHETIC_SCREENSHOT: "UI Screenshot / Document Detected",
            UI_OR_DOCUMENT: "Software Window / Text Detected",
            EXTREME_EXPOSURE: "Invalid Exposure",
        };
        return titles[code] || "Invalid Medical Image";
    }

    return {
        validate,
        getErrorTitle,
        CONFIG,
    };
})();

window.ImageValidator = ImageValidator;
