"""Stages 1-2: preprocessing and restoration (first half of notebook cell 3).
"""
import cv2
import numpy as np  # noqa: F401

from crackvision.config import CFG
from crackvision.image_ops import (
    apply_frequency_filter,
    butterworth_lowpass_kernel,
    high_boost_filter,
    homomorphic_correct,
)

# =============================================================================
# Cell 4A: Preprocessing, Feature Extraction & Ridge Fusion (Stages 1 - 3)
# =============================================================================

# --- Stage 1 & 2: Preprocessing & Restoration ---
def preprocess_image(gray, cfg=CFG):
    median = cv2.medianBlur(gray, cfg.median_k if cfg.median_k % 2 != 0 else cfg.median_k + 1)
    bilateral = cv2.bilateralFilter(median, cfg.bilateral_d, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space)

    restored = homomorphic_correct(bilateral, cfg) if cfg.use_homomorphic else bilateral.copy()

    if cfg.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        enhanced = clahe.apply(restored)
    else:
        enhanced = restored.copy()

    H_butter = butterworth_lowpass_kernel(enhanced.shape, cfg.butter_d0, cfg.butter_n)
    smooth = apply_frequency_filter(enhanced, H_butter)
    bg_k = cfg.bg_k if cfg.bg_k % 2 != 0 else cfg.bg_k + 1
    background = cv2.medianBlur(enhanced, bg_k).astype(np.float32)
    bg_sub = np.clip(background - smooth, 0, 255)

    sharpened = high_boost_filter(bg_sub, cfg.sharpen_sigma, cfg.sharpen_k) if cfg.sharpen_mode == "highboost" else bg_sub

    return {
        "median": median, "bilateral": bilateral, "restored": restored,
        "enhanced": enhanced, "smooth": smooth, "background": background,
        "bg_sub": bg_sub, "sharpened": sharpened
    }
