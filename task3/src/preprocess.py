"""
Image Preprocessing Pipeline for Industrial Equipment Nameplate OCR.

Key Enhancements:
1. Grayscale conversion.
2. Contrast-Limited Adaptive Histogram Equalization (CLAHE) to neutralize glare and uneven lighting.
3. Edge-preserving Bilateral Filtering (d=5, sigma=50) to smooth metallic grain without blurring font strokes.
4. Automatic skew detection and orientation rectification.
5. High-frequency unsharp masking for crisp text edges.
"""

import cv2
import numpy as np


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to single-channel grayscale."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Equalize localized contrast across metallic nameplates."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def denoise_bilateral(gray: np.ndarray, d: int = 5, sigma_color: int = 50, sigma_space: int = 50) -> np.ndarray:
    """Smooth metallic surface noise while preserving sharp font character edges."""
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def deskew_image(gray: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Detect skew angle via Otsu horizontal gradient projection and rotate."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in contours:
        if cv2.contourArea(c) > 400:
            rect = cv2.minAreaRect(c)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            if abs(angle) <= max_angle:
                angles.append(angle)
    
    if not angles:
        return gray
    
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.2:
        return gray
    
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def sharpen_image(gray: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Unsharp masking to boost character definition against metallic background."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.5)
    sharpened = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_nameplate(img: np.ndarray) -> np.ndarray:
    """
    Standard preprocessing workflow optimized for Tesseract LSTM:
    Gray -> CLAHE -> Bilateral Denoise -> Deskew -> Subtle Sharpen.
    """
    gray = convert_to_grayscale(img)
    clahe_img = apply_clahe(gray, clip_limit=2.0, tile_size=8)
    denoised = denoise_bilateral(clahe_img, d=5, sigma_color=50, sigma_space=50)
    deskewed = deskew_image(denoised)
    sharpened = sharpen_image(deskewed, amount=0.4)
    return sharpened
