"""Pipeline configuration (notebook cell 1).

Defines the ``Config`` dataclass, the shared ``CFG`` instance used as the default
argument throughout the package, and seeds the RNGs exactly as the notebook did.
"""
import os  # noqa: F401  (carried over verbatim from notebook cell 0)
import sys  # noqa: F401  (carried over verbatim from notebook cell 0)
import random
from dataclasses import dataclass

import numpy as np

@dataclass
class Config:
    #Preprocessing & Denoising
    median_k: int = 5
    bilateral_d: int = 7
    bilateral_sigma_color: float = 40.0
    bilateral_sigma_space: float = 40.0

    #Restoration & Frequency Filters
    use_homomorphic: bool = True
    homo_gamma_l: float = 0.5
    homo_gamma_h: float = 1.5
    homo_c: float = 1.0
    homo_d0: float = 30.0
    butter_d0: float = 60.0
    butter_n: int = 2
    bg_k: int = 31

    #Contrast Enhancement & Sharpening
    use_clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    sharpen_mode: str = "highboost"  # Options: "highboost", "none"
    sharpen_sigma: float = 1.5
    sharpen_k: float = 1.0

    #Feature Extraction & Multi-Scale Ridge Fusion
    sobel_t: float = 40.0
    canny_lo: int = 15
    canny_hi: int = 45
    lap_sigma: float = 1.0
    min_contrast: float = 10.0

    use_ridge_fusion: bool = True
    ridge_sigmas: tuple = (1.0, 2.0, 3.0)
    blackhat_k: int = 15
    gabor_ksize: int = 21
    gabor_sigma: float = 4.0
    gabor_lambda: float = 10.0
    gabor_orients: int = 8
    hyst_lo: float = 0.25
    hyst_hi: float = 0.50

    #Morphological Cleaning & Component Filtering
    link_len: int = 9
    min_len: int = 20           # Minimum skeleton pixels per crack segment
    sure_len: int = 60          # Long components skip shape filtering
    min_elong: float = 3.0      # Major/minor axis ratio (rejects round stains/pores)
    detect_len: int = 25        # Skeleton pixels required to flag image as cracked

    #Calibration & Output Settings
    mm_per_px: float = 1.0      # Physical calibration scale factor (1.0 = pixels)
    is_calibrated: bool = False # Set True only when mm_per_px is calibrated
    severity_bins: tuple = (0.3, 1.0) # Bins: (hairline < 0.3mm, moderate < 1.0mm, severe >= 1.0mm)

    random_seed: int = 42
    output_dir: str = "crackvision_outputs"
    save_outputs: bool = True

CFG = Config()
random.seed(CFG.random_seed)
np.random.seed(CFG.random_seed)

print("Configuration initialized successfully!")
