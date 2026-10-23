"""End-to-end benchmark on the Kaggle *Surface Crack Detection* dataset.

Runs notebook cells 9 -> 10 -> 11 in their original order. The three blocks share
state (``positive_images``, ``negative_images``, ``test_results_log``) exactly as
they did in the notebook, so they are kept as one top-level script.

Usage:
    pip install -e .
    python scripts/run_benchmark.py
"""
import random
from pathlib import Path

import cv2
import kagglehub
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tqdm.auto import tqdm

from crackvision.config import CFG
from crackvision.detector import detect_cracks

# ----- Notebook cell 9 --------------------------------------------------------
# =============================================================================
# Cell A: Dataset Ingestion & Discovery
# =============================================================================
print("Step 1: Ingesting Dataset...")

try:
    root_path = Path(kagglehub.dataset_download("arunrk7/surface-crack-detection"))
    positive_images = sorted(root_path.rglob("Positive/*.jpg"))
    negative_images = sorted(root_path.rglob("Negative/*.jpg"))

    print(f"\nDataset Root Directory : {root_path.resolve()}")
    print(f"Total Positive Images  : {len(positive_images):,}")
    print(f"Total Negative Images  : {len(negative_images):,}")

    if not positive_images or not negative_images:
        raise FileNotFoundError("Could not locate Positive or Negative image subfolders.")

except Exception as e:
    print(f"KaggleHub download skipped or failed: {e}")
    positive_images, negative_images = [], []

print("\nDataset ready for testing!")

# ----- Notebook cell 10 -------------------------------------------------------
SAMPLE_LIMIT_PER_CLASS = 1000

if positive_images and negative_images:
    pos_samples = [(p, 1) for p in random.sample(positive_images, min(SAMPLE_LIMIT_PER_CLASS, len(positive_images)))]
    neg_samples = [(p, 0) for p in random.sample(negative_images, min(SAMPLE_LIMIT_PER_CLASS, len(negative_images)))]

    test_queue = pos_samples + neg_samples
    random.shuffle(test_queue)  # Shuffle order for unbiased execution

    test_results_log = []

    for img_path, true_label in tqdm(test_queue, desc="Testing Files"):
        bgr_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr_img is None:
            continue

        # Run detection pipeline
        res = detect_cracks(bgr_img, CFG)
        pred_label = int(res["is_cracked"])

        # Record file-by-file log
        test_results_log.append({
            "path": img_path,
            "filename": img_path.name,
            "true_label": true_label,            # 1 = Positive (Cracked), 0 = Negative (Uncracked)
            "predicted_label": pred_label,      # 1 = Positive (Cracked), 0 = Negative (Uncracked)
            "is_correct": (true_label == pred_label),
            "length_px": res["length_px"],
            "max_width_px": res["max_width_px"],
            "severity": res["severity"]
        })

    print(f"\nCompleted testing on {len(test_results_log)} images!")
else:
    print("No images loaded. Please execute Cell A first.")

# ----- Notebook cell 11 -------------------------------------------------------
# =============================================================================
# Cell C: Classification Evaluation Report
# =============================================================================
print("Step 3: Generating Classification Evaluation Report...\n")

if 'test_results_log' in locals() and test_results_log:
    y_true = [item["true_label"] for item in test_results_log]
    y_pred = [item["predicted_label"] for item in test_results_log]

    # Calculate Confusion Matrix: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Calculate Class-Specific Scores
    total_positive = tp + fn
    total_negative = tn + fp

    pos_accuracy = (tp / total_positive * 100.0) if total_positive > 0 else 0.0
    neg_accuracy = (tn / total_negative * 100.0) if total_negative > 0 else 0.0
    overall_accuracy = ((tp + tn) / len(y_true) * 100.0) if len(y_true) > 0 else 0.0

    print("=" * 65)
    print("        CRACKVISION CLASSIFICATION EVALUATION REPORT")
    print("=" * 65)
    print(f" Total Images Evaluated     : {len(y_true)}")
    print(f" Overall Accuracy           : {overall_accuracy:.2f}%\n")

    print("-" * 65)
    print(" 1. POSITIVE IMAGES SCORE (Ground Truth: Cracked)")
    print("-" * 65)
    print(f"   * Total Positive Images  : {total_positive}")
    print(f"   * Correctly Classified   : {tp} (True Positives)")
    print(f"   * Incorrectly Classified : {fn} (False Negatives / Missed Cracks)")
    print(f"   * Positive Class Score   : {pos_accuracy:.2f}% Correct\n")

    print("-" * 65)
    print(" 2. NEGATIVE IMAGES SCORE (Ground Truth: Non-Cracked)")
    print("-" * 65)
    print(f"   * Total Negative Images  : {total_negative}")
    print(f"   * Correctly Classified   : {tn} (True Negatives)")
    print(f"   * Incorrectly Classified : {fp} (False Positives / False Alarms)")
    print(f"   * Negative Class Score   : {neg_accuracy:.2f}% Correct\n")

    print("=" * 65)
    print(" SUMMARY METRICS")
    print("=" * 65)
    print(f" Precision (Exactness)    : {precision_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
    print(f" Recall / Sensitivity     : {recall_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
    print(f" F1-Score (Balance)       : {f1_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
    print("=" * 65)

    # Display Misclassified File Details for Error Inspection
    # misclassified = [item for item in test_results_log if not item["is_correct"]]
    # if misclassified:
    #     print(f"\nMisclassified Files Inspection ({len(misclassified)} files):")
    #     for item in misclassified:
    #         gt_str = "Positive (Cracked)" if item["true_label"] == 1 else "Negative (Non-Cracked)"
    #         pred_str = "Positive (Cracked)" if item["predicted_label"] == 1 else "Negative (Non-Cracked)"
            #print(f" - {item['filename']:<20} | True: {gt_str:<20} | Pred: {pred_str}")

else:
    print("No test results found. Please run Cell B first.")
