"""Master detector: runs the full six-stage CrackVision pipeline (notebook cell 6).
"""
import cv2
import numpy as np

from crackvision.config import CFG
from crackvision.features import extract_features
from crackvision.measurement import measure_cracks
from crackvision.preprocessing import preprocess_image
from crackvision.refinement import compute_skeleton, refine_crack_mask

# =============================================================================
# Cell 5: Master Detector Integration
# =============================================================================
def detect_cracks(bgr_image, cfg=CFG):
    """Run full 6-stage CrackVision pipeline on a BGR image."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    prep_dict = preprocess_image(gray, cfg)
    feat_dict = extract_features(prep_dict, cfg)
    linked_mask, refined_mask = refine_crack_mask(feat_dict["candidate"], cfg)
    skeleton = compute_skeleton(refined_mask)
    measures = measure_cracks(refined_mask, skeleton, cfg)

    # Generate Overlay
    overlay = bgr_image.copy()
    overlay[refined_mask.astype(bool)] = (
        0.5 * overlay[refined_mask.astype(bool)] + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)
    skel_dilated = cv2.dilate(skeleton.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    overlay[skel_dilated] = (0, 0, 255)  # Red skeleton line

    return {
        "gray": gray, **prep_dict, **feat_dict,
        "linked_mask": linked_mask, "refined_mask": refined_mask,
        "skeleton": skeleton, "overlay": overlay, **measures
    }

print("Master detector loaded!")
