"""Executes the ORIGINAL notebook cells and compares every output against the package."""
import json
from pathlib import Path

import numpy as np
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "original" / "CV_PACKAGE_FINAL.ipynb"


@pytest.fixture(scope="module")
def nb_ns():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns = {"__name__": "__nb__"}
    for i in range(0, 9):  # imports, config, and all function definitions
        src = "".join(nb["cells"][i]["source"])
        if nb["cells"][i]["cell_type"] == "code":
            exec(compile(src, f"<cell {i}>", "exec"), ns)
    return ns


def test_detect_cracks_matches_notebook(nb_ns, synthetic_bgr):
    from crackvision import detect_cracks

    expected = nb_ns["detect_cracks"](synthetic_bgr, nb_ns["CFG"])
    actual = detect_cracks(synthetic_bgr)
    assert expected.keys() == actual.keys()
    for key in expected:
        np.testing.assert_array_equal(np.asarray(expected[key]), np.asarray(actual[key]), err_msg=key)


def test_pipeline_detects_synthetic_crack(synthetic_bgr):
    from crackvision import detect_cracks

    assert detect_cracks(synthetic_bgr)["is_cracked"] is True


def test_evaluate_pixel_segmentation_matches_notebook(nb_ns):
    from crackvision import evaluate_pixel_segmentation

    rng = np.random.RandomState(1)
    pred = rng.randint(0, 2, (32, 32))
    gt = rng.randint(0, 2, (32, 32))
    assert nb_ns["evaluate_pixel_segmentation"](pred, gt) == evaluate_pixel_segmentation(pred, gt)
