# Notebook → Module Mapping

Source: `notebooks/original/CV_PACKAGE_FINAL.ipynb` (12 cells, SHA-256 in `SHA256SUM`).

| Notebook cell | Content | Destination |
|---|---|---|
| 0 | Imports | Redistributed to the top of each module that uses them |
| 1 | `Config`, `CFG`, RNG seeding | `crackvision/config.py` |
| 2 | I/O + low-level operators | `crackvision/image_ops.py` |
| 3 (Stages 1–2) | `preprocess_image` | `crackvision/preprocessing.py` |
| 3 (Stage 3) | `gabor_response`, `ridge_score_map`, `extract_features` | `crackvision/features.py` |
| 4 | Markdown note ("Pratho Starts here") | Not code; remains in the preserved notebook |
| 5 (Stages 4–5) | `_line_kernel`, `refine_crack_mask`, `compute_skeleton` | `crackvision/refinement.py` |
| 5 (Stage 6) | `measure_cracks` | `crackvision/measurement.py` |
| 6 | `detect_cracks` | `crackvision/detector.py` |
| 7 | Visualisation / console summary | `crackvision/visualization.py` |
| 8 | Evaluation metrics | `crackvision/evaluation.py` |
| 9, 10, 11 | Dataset ingestion → batch run → report | `scripts/run_benchmark.py` (one script, original order) |

## Dependency and execution-order notes

- **Shared `CFG`.** Every function uses `cfg=CFG` as a default argument, bound at definition time.
  All modules import the single instance from `config.py`, so mutating `CFG` affects the whole package
  exactly as it did in the notebook.
- **RNG seeding** happens when `config.py` is first imported, as in cell 1. Verified: the resulting
  `random` / `numpy` streams are identical to notebook order.
- **Cells 9–11 share state** (`positive_images`, `test_results_log`, and a `'test_results_log' in locals()`
  check), so they stay top-level in one script rather than being wrapped in functions.

## Things deliberately left as-is (per the "do not modify" rule)

- Unused imports `os`, `sys` (in `config.py`) and `classification_report` (in `evaluation.py`) are kept
  and marked `# noqa: F401`. They are safe to delete later.
- Module-level `print()` status messages are preserved; each split cell's trailing print lives in the
  last module derived from that cell (`features.py`, `measurement.py`).
- `display_results`, `print_console_summary`, `evaluate_pixel_segmentation`, `display_evaluation_report`
  and `evaluate_dataset_classification` are never called in the notebook; they are preserved and exported.
- Commented-out misclassification inspection code in cell 11 is retained verbatim.
- Original cell-header comments (e.g. `# Cell 4A: ...`) are retained, so they refer to notebook cell labels.
