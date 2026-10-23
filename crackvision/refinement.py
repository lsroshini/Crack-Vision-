"""Stages 4-5: mask refinement and skeletonisation (first half of notebook cell 5).
"""
from functools import lru_cache

import cv2
import numpy as np
from skimage.measure import label as label_components
from skimage.measure import regionprops
from skimage.morphology import skeletonize

from crackvision.config import CFG

# =============================================================================
# Cell 4B: Mask Refinement, Skeletonization & Measurements (Stages 4 - 6)
# =============================================================================

# --- Stage 4 & 5: Mask Refinement & Skeletonization ---
@lru_cache(maxsize=32)
def _line_kernel(length, angle):
    kernel = np.zeros((length, length), np.uint8)
    c = length // 2
    rad = np.deg2rad(angle)
    dx, dy = np.cos(rad) * c, np.sin(rad) * c
    cv2.line(kernel, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, 1)
    return kernel

def refine_crack_mask(binary_candidate, cfg=CFG):
    linked = np.zeros_like(binary_candidate, dtype=np.uint8)
    cand_u8 = binary_candidate.astype(np.uint8)
    for ang in (0, 45, 90, 135):
        linked |= cv2.morphologyEx(cand_u8, cv2.MORPH_CLOSE, _line_kernel(cfg.link_len, ang))

    labels = label_components(linked, connectivity=2)
    skel_temp = skeletonize(linked.astype(bool), method="zhang")

    n_labels = labels.max() + 1
    skel_lengths = np.bincount(labels[skel_temp], minlength=n_labels)
    keep_lookup = skel_lengths >= cfg.min_len

    for prop in regionprops(labels):
        lbl = prop.label
        if not keep_lookup[lbl] or skel_lengths[lbl] >= cfg.sure_len:
            continue
        elongation = prop.axis_major_length / max(prop.axis_minor_length, 1.0)
        if elongation < cfg.min_elong:
            keep_lookup[lbl] = False

    keep_lookup[0] = False
    refined_mask = keep_lookup[labels].astype(np.uint8)
    return linked, refined_mask

def compute_skeleton(refined_mask):
    return skeletonize(refined_mask.astype(bool), method="zhang").astype(bool)
