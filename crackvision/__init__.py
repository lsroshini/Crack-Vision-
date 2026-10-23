"""CrackVision: classical computer-vision pipeline for surface crack detection and quantification."""

from crackvision.config import CFG, Config
from crackvision.image_ops import (
    apply_frequency_filter,
    butterworth_lowpass_kernel,
    distance_grid,
    high_boost_filter,
    homomorphic_correct,
    homomorphic_kernel,
    load_image,
    normalize_u8,
)
from crackvision.preprocessing import preprocess_image
from crackvision.features import extract_features, gabor_response, ridge_score_map
from crackvision.refinement import compute_skeleton, refine_crack_mask
from crackvision.measurement import measure_cracks
from crackvision.detector import detect_cracks
from crackvision.visualization import display_results, print_console_summary
from crackvision.evaluation import (
    display_evaluation_report,
    evaluate_dataset_classification,
    evaluate_pixel_segmentation,
)

__version__ = "1.0.0"

__all__ = [
    "CFG", "Config",
    "load_image", "normalize_u8", "distance_grid", "butterworth_lowpass_kernel",
    "homomorphic_kernel", "apply_frequency_filter", "homomorphic_correct", "high_boost_filter",
    "preprocess_image", "gabor_response", "ridge_score_map", "extract_features",
    "refine_crack_mask", "compute_skeleton", "measure_cracks", "detect_cracks",
    "display_results", "print_console_summary",
    "evaluate_pixel_segmentation", "display_evaluation_report", "evaluate_dataset_classification",
]
