"""Stage 6: crack quantification (second half of notebook cell 5).
"""
import cv2
import numpy as np
from skimage.measure import label as label_components

from crackvision.config import CFG

# --- Stage 6: Crack Quantification ---
def measure_cracks(refined_mask, skeleton, cfg=CFG):
    dist_transform = cv2.distanceTransform(refined_mask.astype(np.uint8), cv2.DIST_L2, 3)
    widths_px = 2.0 * dist_transform[skeleton]

    length_px = int(skeleton.sum())
    mean_width_px = float(widths_px.mean()) if widths_px.size > 0 else 0.0
    max_width_px = float(widths_px.max()) if widths_px.size > 0 else 0.0

    length_val = length_px * cfg.mm_per_px
    mean_width_val = mean_width_px * cfg.mm_per_px
    max_width_val = max_width_px * cfg.mm_per_px

    ys, xs = np.nonzero(skeleton)
    if xs.size >= 2:
        cov = np.cov(np.vstack([xs, ys]))
        _, vecs = np.linalg.eigh(cov)
        vx, vy = vecs[:, -1]
        angle_deg = float(np.degrees(np.arctan2(-vy, vx)) % 180)
    else:
        angle_deg = float("nan")

    if np.isnan(angle_deg): orientation = "None"
    elif angle_deg < 22.5 or angle_deg >= 157.5: orientation = "Horizontal"
    elif 67.5 <= angle_deg < 112.5: orientation = "Vertical"
    else: orientation = "Diagonal"

    if max_width_val <= 0: severity = "None"
    elif max_width_val < cfg.severity_bins[0]: severity = "Hairline"
    elif max_width_val < cfg.severity_bins[1]: severity = "Moderate"
    else: severity = "Severe"

    num_components = int(label_components(refined_mask, connectivity=2).max())
    crack_percentage = float((refined_mask.sum() / refined_mask.size) * 100.0)
    is_cracked = bool(length_px >= cfg.detect_len)

    return {
        "length_px": length_px, "mean_width_px": mean_width_px, "max_width_px": max_width_px,
        "length_val": length_val, "mean_width_val": mean_width_val, "max_width_val": max_width_val,
        "num_components": num_components, "crack_percentage": crack_percentage,
        "orientation": orientation, "angle_deg": angle_deg, "severity": severity,
        "is_cracked": is_cracked, "dist_transform": dist_transform
    }

print("Stages 4-6 functions loaded!")
