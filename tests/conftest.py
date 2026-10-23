import numpy as np
import pytest


@pytest.fixture(scope="session")
def synthetic_bgr():
    """Deterministic 128x128 textured surface with one dark diagonal crack."""
    rng = np.random.RandomState(0)
    gray = (140 + rng.normal(0, 6, (128, 128))).clip(0, 255).astype(np.uint8)
    import cv2
    cv2.line(gray, (10, 20), (115, 105), 40, 2)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
