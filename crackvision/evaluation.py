"""Evaluation metrics: pixel-level segmentation and folder-level classification (notebook cell 8).
"""
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (  # noqa: F401  (classification_report carried over verbatim)
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm.auto import tqdm

from crackvision.config import CFG
from crackvision.detector import detect_cracks

# =============================================================================
# Cell 7: Evaluation Metrics Modules
# =============================================================================
def evaluate_pixel_segmentation(pred_mask, gt_mask):
    """Compute pixel-level segmentation metrics against a ground-truth binary mask."""
    pred_bin = (pred_mask > 0).astype(np.uint8).ravel()
    gt_bin = (gt_mask > 0).astype(np.uint8).ravel()

    if pred_bin.shape != gt_bin.shape:
        raise ValueError(f"Shape mismatch: pred_mask {pred_bin.shape} vs gt_mask {gt_bin.shape}")

    tn, fp, fn, tp = confusion_matrix(gt_bin, pred_bin, labels=[0, 1]).ravel()

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    spec = tn / max(tn + fp, 1e-9)
    f1 = 2 * (prec * rec) / max(prec + rec, 1e-9)
    iou = tp / max(tp + fp + fn, 1e-9)

    return {
        "Accuracy": acc * 100.0, "Precision": prec * 100.0, "Recall": rec * 100.0,
        "Specificity": spec * 100.0, "F1-Score": f1 * 100.0, "IoU": iou * 100.0,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }

def display_evaluation_report(metrics_dict):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("      PIXEL-LEVEL SEGMENTATION EVALUATION REPORT")
    print("=" * 60)
    print(f" Accuracy    : {metrics_dict['Accuracy']:.2f}%")
    print(f" Precision   : {metrics_dict['Precision']:.2f}%")
    print(f" Recall (Sens): {metrics_dict['Recall']:.2f}%")
    print(f" Specificity : {metrics_dict['Specificity']:.2f}%")
    print(f" F1-Score    : {metrics_dict['F1-Score']:.2f}%")
    print(f" IoU Score   : {metrics_dict['IoU']:.2f}%")
    print("-" * 60)
    print(" Confusion Matrix:")
    print(f"   [ Actual BG  (0) ] -> TN: {metrics_dict['TN']:<8} | FP: {metrics_dict['FP']:<8}")
    print(f"   [ Actual Crack(1) ] -> FN: {metrics_dict['FN']:<8} | TP: {metrics_dict['TP']:<8}")
    print("=" * 60 + "\n")

def evaluate_dataset_classification(positive_paths, negative_paths, max_images=100, cfg=CFG):
    """Evaluate image classification statistics on Kaggle Positive/Negative splits."""
    samples = [(p, 1) for p in positive_paths] + [(p, 0) for p in negative_paths]
    random.shuffle(samples)
    if max_images is not None:
        samples = samples[:max_images]

    y_true, y_pred, records = [], [], []
    t0 = time.time()

    for path, label in tqdm(samples, desc="Benchmarking Dataset"):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None: continue

        res = detect_cracks(bgr, cfg)
        pred = int(res["is_cracked"])

        y_true.append(label)
        y_pred.append(pred)
        records.append({
            "path": str(path), "true_label": label, "predicted_label": pred,
            "length_px": res["length_px"], "max_width_px": res["max_width_px"],
            "severity": res["severity"]
        })

    elapsed = time.time() - t0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print("\n" + "=" * 60)
    print("   FOLDER-LEVEL IMAGE CLASSIFICATION BENCHMARK")
    print("=" * 60)
    print(f" Images Evaluated : {len(y_true)} ({elapsed:.1f}s total | {1000 * elapsed / len(y_true):.0f} ms/img)")
    print(f" Accuracy         : {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f" Precision        : {precision_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
    print(f" Recall           : {recall_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
    print(f" F1-Score         : {f1_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
    print(f" Confusion Matrix : [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
    print("=" * 60 + "\n")

    if cfg.save_outputs:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "classification_benchmark.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"Saved dataset classification log to: {csv_path.resolve()}")

print("Evaluation engine ready!")
