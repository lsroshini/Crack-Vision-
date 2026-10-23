"""Stage 3: feature extraction and multi-scale ridge fusion (second half of notebook cell 3).
"""
import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold, sato

from crackvision.config import CFG
from crackvision.image_ops import normalize_u8

# --- Stage 3: Feature Extraction & Ridge Fusion ---
def gabor_response(img, cfg=CFG):
    img_f = img.astype(np.float32)
    resp = np.zeros_like(img_f)
    for theta in np.linspace(0, np.pi, cfg.gabor_orients, endpoint=False):
        k = cv2.getGaborKernel((cfg.gabor_ksize, cfg.gabor_ksize), cfg.gabor_sigma, theta, cfg.gabor_lambda, 0.5, 0, ktype=cv2.CV_32F)
        k -= k.mean()
        filtered = cv2.filter2D(img_f, cv2.CV_32F, k)
        resp = np.maximum(resp, filtered)
    return resp

def ridge_score_map(pre_img, enh_img, cfg=CFG):
    sato_r = sato(enh_img, sigmas=cfg.ridge_sigmas, black_ridges=False)
    gabor_r = gabor_response(enh_img, cfg)
    ksize = cfg.blackhat_k if cfg.blackhat_k % 2 != 0 else cfg.blackhat_k + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    blackhat = cv2.morphologyEx(pre_img, cv2.MORPH_BLACKHAT, se).astype(np.float32)

    def norm01(x):
        hi = np.percentile(x, 99.5)
        return np.clip(x / hi, 0, 1).astype(np.float32) if hi > 0 else np.zeros_like(x, np.float32)

    return (norm01(sato_r) + norm01(gabor_r) + norm01(blackhat)) / 3.0

def extract_features(prep_dict, cfg=CFG):
    enh = prep_dict["sharpened"]
    pre = prep_dict["enhanced"]
    smooth = prep_dict["smooth"]
    enh8 = normalize_u8(enh)

    sobel_x = cv2.Sobel(enh, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enh, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.hypot(sobel_x, sobel_y)
    canny = cv2.Canny(enh8, cfg.canny_lo, cfg.canny_hi)
    laplacian = cv2.Laplacian(cv2.GaussianBlur(smooth, (0, 0), cfg.lap_sigma), cv2.CV_32F)

    otsu_t, _ = cv2.threshold(enh8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(float(otsu_t), cfg.min_contrast)
    edges = ((canny > 0) | (sobel > cfg.sobel_t)).astype(np.uint8)
    near_edge = cv2.dilate(edges, np.ones((5, 5), np.uint8)).astype(bool)
    candidate = (enh > t) & (laplacian > 0) & near_edge

    ridge_score = ridge_score_map(pre, enh, cfg)
    ridge_mask = apply_hysteresis_threshold(ridge_score, cfg.hyst_lo, cfg.hyst_hi)
    final_candidate = candidate | (ridge_mask & (enh > cfg.min_contrast))

    return {
        "sobel": sobel, "canny": canny, "laplacian": laplacian,
        "ridge_score": ridge_score, "ridge_mask": ridge_mask,
        "candidate": final_candidate
    }

print("Stages 1-3 functions loaded!")
