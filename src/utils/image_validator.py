"""
MediFusion AI - Medical Image Validation Module
Strict server-side validation to ensure only legitimate medical chest radiographs are processed.
Rejects non-medical images, screenshots, software windows, documents, selfies, and photos.
"""

import io
import logging
from typing import Tuple
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


def validate_chest_radiograph(image_bytes: bytes) -> Tuple[bool, str, str]:
    """
    Validate whether the uploaded image is a valid chest radiograph.
    
    Returns:
        (is_valid, error_code, error_message)
    """
    try:
        # 1. Decode image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return False, "INVALID_FORMAT", "Unable to decode image. Please upload a valid PNG or JPEG image."
            
        h, w, c = img.shape
        
        # 2. Minimum resolution
        if h < 100 or w < 100:
            return False, "LOW_RESOLUTION", f"Image resolution ({w}x{h}) is too low for diagnostic radiograph analysis (minimum 100x100 required)."
            
        # 3. Aspect Ratio Check (Chest radiographs are square or portrait/slight landscape)
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.60 or aspect_ratio > 1.55:
            return False, "INVALID_ASPECT_RATIO", (
                f"Image proportions ({aspect_ratio:.2f}:1) do not match clinical chest radiographs. "
                "Chest X-rays are typically square or slightly rectangular (0.65:1 to 1.55:1). "
                "Please upload a proper radiograph, not a desktop/phone screenshot."
            )
            
        # 4. Monochromatic / Chromaticity Check (X-rays are strictly grayscale)
        b, g, r = cv2.split(img)
        diff_rg = np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32)))
        diff_gb = np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32)))
        diff_br = np.mean(np.abs(b.astype(np.float32) - r.astype(np.float32)))
        mean_chroma_diff = (diff_rg + diff_gb + diff_br) / 3.0
        max_chroma_diff = max(diff_rg, diff_gb, diff_br)
        
        if mean_chroma_diff > 8.0 or max_chroma_diff > 12.0:
            return False, "COLOR_IMAGE", (
                "Color photograph or non-medical graphic detected. "
                "Chest radiographs are greyscale anatomical imaging studies. "
                "Please upload a valid chest X-ray."
            )
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 5. Flat Palette / Synthetic UI Screenshot Check
        # Natural analog/digital X-rays have a smooth continuous gradient spectrum.
        # Screenshots and synthetic UI windows have massive peaks at discrete values (e.g. background, window fill).
        hist, _ = np.histogram(gray, bins=64, range=(0, 256))
        hist_norm = hist / float(hist.sum())
        sorted_hist = np.sort(hist_norm)
        top2_ratio = sorted_hist[-2:].sum()
        top3_ratio = sorted_hist[-3:].sum()
        
        # If > 60% of all pixels fall into only 2 histogram bins, it's a flat graphic/screenshot
        if top2_ratio > 0.55 or top3_ratio > 0.68:
            return False, "SYNTHETIC_SCREENSHOT", (
                "Non-radiological graphic or UI screenshot detected. "
                "The image contains flat synthetic color blocks rather than anatomical tissue density. "
                "Please upload a valid clinical radiograph."
            )
            
        # 6. Orthogonal Line & Manhattan Edge Detection (Detects UI windows, text, tables, dialogs)
        # Resize to standard analysis size
        analysis_size = 256
        resized_gray = cv2.resize(gray, (analysis_size, analysis_size))
        edges = cv2.Canny(resized_gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=25, maxLineGap=5)
        
        if lines is not None and len(lines) >= 6:
            ortho_count = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                # Check horizontal or vertical within ~3 degrees
                if dx > 0 and (dy / float(dx)) < 0.06:
                    ortho_count += 1
                elif dy > 0 and (dx / float(dy)) < 0.06:
                    ortho_count += 1
            
            ortho_ratio = ortho_count / float(len(lines))
            # If large proportion of detected lines are strictly horizontal/vertical, it's a window/document/table
            if len(lines) >= 8 and ortho_ratio > 0.60:
                return False, "UI_OR_DOCUMENT", (
                    "Document, table, or software interface detected. "
                    "Diagnostic X-rays exhibit organic anatomical curves (ribs, cardiac silhouette, diaphragm) "
                    "rather than straight grid lines or text boxes."
                )
                
        # 7. Extreme Exposure / Blank Canvas Check
        mean_brightness = np.mean(gray)
        if mean_brightness < 12.0:
            return False, "TOO_DARK", "The uploaded image is almost entirely black. Please upload a clear chest radiograph."
        if mean_brightness > 240.0:
            return False, "TOO_BRIGHT", "The uploaded image is completely overexposed/white. Please upload a clear chest radiograph."
            
        # 8. Gradient Variance / Texture Check (Detect solid / empty images)
        laplacian_var = cv2.Laplacian(resized_gray, cv2.CV_64F).var()
        if laplacian_var < 15.0:
            return False, "BLANK_IMAGE", "Image has insufficient structural contrast or anatomical texture to be a radiograph."
            
        return True, "VALID", "Valid chest radiograph"
        
    except Exception as e:
        logger.warning(f"Error during image validation check: {e}", exc_info=True)
        # If validator encounters an unexpected processing error, fail safe
        return False, "VALIDATION_ERROR", f"Could not validate image structure: {str(e)}"
