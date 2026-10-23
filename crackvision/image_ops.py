"""Image I/O and low-level image operators (notebook cell 2).

Includes image loading, uint8 normalisation, cached frequency-domain kernels
(Butterworth, homomorphic), FFT filtering and high-boost sharpening.
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from crackvision.config import CFG

# =============================================================================
# Cell 3: Image I/O & Low-Level Image Operators
# =============================================================================
def load_image(image_path):
    """Safely load an image from disk."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path.resolve()}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image at: {path.resolve()}")
    return bgr

def normalize_u8(image):
    """Safely convert a float map to displayable uint8 [0, 255]."""
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(image, 0, 255).astype(np.uint8)

@lru_cache(maxsize=16)
def distance_grid(shape):
    """Frequency-domain distance grid centered with np.fft.fftshift."""
    h, w = shape
    y = np.arange(h)[:, None] - h // 2
    x = np.arange(w)[None, :] - w // 2
    return np.hypot(x, y)

@lru_cache(maxsize=16)
def butterworth_lowpass_kernel(shape, d0, n):
    """Generate Butterworth low-pass filter mask."""
    d = distance_grid(shape)
    d0 = max(float(d0), 1e-6)
    return (1.0 / (1.0 + (d / d0) ** (2 * n))).astype(np.float32)

@lru_cache(maxsize=16)
def homomorphic_kernel(shape, gl, gh, c, d0):
    """Generate Homomorphic filter mask for shadow & illumination correction."""
    d = distance_grid(shape)
    d0 = max(float(d0), 1e-6)
    return ((gh - gl) * (1.0 - np.exp(-c * (d ** 2) / (d0 ** 2))) + gl).astype(np.float32)

def apply_frequency_filter(img, H):
    """Apply a 2D frequency domain filter mask H using FFT."""
    img_f = img.astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(img_f))
    res = np.fft.ifft2(np.fft.ifftshift(F * H))
    return np.real(res).astype(np.float32)

def homomorphic_correct(gray, cfg=CFG):
    """Correct non-uniform illumination and shadows via Homomorphic filtering."""
    H = homomorphic_kernel(gray.shape, cfg.homo_gamma_l, cfg.homo_gamma_h, cfg.homo_c, cfg.homo_d0)
    z = apply_frequency_filter(np.log1p(np.maximum(gray.astype(np.float32), 0)), H)
    out = np.expm1(z)
    mean_scale = float(gray.mean()) / max(float(out.mean()), 1e-6)
    out *= mean_scale
    return normalize_u8(out)

def high_boost_filter(img, sigma=1.5, k=1.0):
    """Spatial High-Boost sharpening."""
    img_f = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigma)
    mask = img_f - blurred
    return np.clip(img_f + k * mask, 0, 255).astype(np.float32)

print("Utility functions ready!")
