"""benchmarks/_real.py — loader for the real-image slot.

Reads ``benchmarks/real_images/manifest.json`` (if present) and yields
``(image_gray, config, expected_ids)`` tuples for robustness.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import numpy as np

_MANIFEST_NAME = "manifest.json"
_CONFIGS = {"FRACTAL_2L_6", "FRACTAL_3L_6", "FRACTAL_4L_6", "FRACTAL_5L_6"}


def _load_cv2():
    try:
        import cv2

        return cv2
    except ImportError:
        return None


def iter_real_images(
    real_dir: str | Path,
) -> Generator[tuple[np.ndarray, str, list[int]], None, None]:
    """Yield ``(gray_image, config, expected_ids)`` from *real_dir/manifest.json*.

    Silently skips entries whose image file is missing.
    Raises ``FileNotFoundError`` if ``manifest.json`` itself is absent.
    """
    real_dir = Path(real_dir)
    manifest_path = real_dir / _MANIFEST_NAME

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found at {manifest_path}.\n"
            f"Copy manifest.example.json → manifest.json and populate it."
        )

    cv2 = _load_cv2()
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for the real-image loader.\n"
            "Install with: pip install nanofractal[bench]"
        )

    entries = json.loads(manifest_path.read_text())

    for entry in entries:
        img_path = real_dir / entry["path"]
        if not img_path.exists():
            print(f"  [real] WARNING: image not found: {img_path} — skipping")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [real] WARNING: could not read: {img_path} — skipping")
            continue

        config: str = entry.get("config", "FRACTAL_5L_6")
        if config not in _CONFIGS:
            print(f"  [real] WARNING: unknown config {config!r} — skipping")
            continue

        expected_ids: list[int] = entry.get("expected_ids", [])
        yield img, config, expected_ids
