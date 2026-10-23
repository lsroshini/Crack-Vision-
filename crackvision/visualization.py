"""Visualisation and console reporting utilities (notebook cell 7).
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from crackvision.config import CFG
from crackvision.image_ops import normalize_u8

# =============================================================================
# Cell 6: Visualization & Reporting Utilities
# =============================================================================
def display_results(bgr_image, results, title_prefix="CrackVision Analysis", cfg=CFG):
    """Renders a clean 13-stage image subplot grid."""
    unit_str = "mm" if cfg.is_calibrated else "px"

    visual_stages = [
        ("1. Original RGB Image", cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)),
        ("2. Grayscale Input", results["gray"]),
        ("3. Preprocessed & Denoised", results["enhanced"]),
        ("4. Background Subtraction", results["bg_sub"]),
        ("5. Sobel Gradient Magnitude", normalize_u8(results["sobel"])),
        ("6. Canny Edge Detector", results["canny"]),
        ("7. Laplacian Edge Map", normalize_u8(np.abs(results["laplacian"]))),
        ("8. Fused Ridge Score Map", normalize_u8(results["ridge_score"] * 255)),
        ("9. Candidate Binary Mask", results["candidate"].astype(np.uint8) * 255),
        ("10. Final Refined Crack Mask", results["refined_mask"] * 255),
        ("11. Zhang-Suen Skeleton", results["skeleton"].astype(np.uint8) * 255),
        ("12. Distance Transform Map", normalize_u8(cv2.normalize(results["dist_transform"], None, 0, 255, cv2.NORM_MINMAX))),
        ("13. Detection Overlay (Red=Skel)", cv2.cvtColor(results["overlay"], cv2.COLOR_BGR2RGB))
    ]

    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    axes = axes.ravel()

    for idx, (title, img) in enumerate(visual_stages):
        if img.ndim == 3:
            axes[idx].imshow(img)
        else:
            axes[idx].imshow(img, cmap="gray" if "Distance" not in title else "jet")
        axes[idx].set_title(title, fontsize=10, fontweight="bold")
        axes[idx].axis("off")

    for idx in range(len(visual_stages), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"{title_prefix} | Detected: {'CRACK' if results['is_cracked'] else 'NO CRACK'} | "
        f"Length: {results['length_val']:.1f}{unit_str} | Max Width: {results['max_width_val']:.2f}{unit_str}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

def print_console_summary(image_name, results, cfg=CFG):
    """Outputs text summary to the console."""
    unit = "mm" if cfg.is_calibrated else "px"
    print("\n" + "=" * 60)
    print(f"   CRACKVISION ANALYSIS SUMMARY: {image_name}")
    print("=" * 60)
    print(f"Status               : {'CRACK DETECTED' if results['is_cracked'] else 'NO CRACK DETECTED'}")
    print(f"Total Skeleton Length: {results['length_val']:.2f} {unit} ({results['length_px']} pixels)")
    print(f"Average Crack Width  : {results['mean_width_val']:.2f} {unit} ({results['mean_width_px']:.2f} pixels)")
    print(f"Maximum Crack Width  : {results['max_width_val']:.2f} {unit} ({results['max_width_px']:.2f} pixels)")
    print(f"Connected Components : {results['num_components']}")
    print(f"Crack Pixel Density  : {results['crack_percentage']:.3f}% of total area")
    print(f"Dominant Orientation : {results['orientation']} ({results['angle_deg']:.1f}°)")
    print(f"Severity Bin         : {results['severity']}")
    print("=" * 60 + "\n")

print("Visualization functions loaded!")
