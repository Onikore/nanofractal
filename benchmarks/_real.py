"""benchmarks/_real.py — loader for the real-image slot.

Reads ``benchmarks/real_images/manifest.json`` (if present) and yields
``(image_bgr, nf_dict, expected_ids)`` tuples for robustness.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import numpy as np

import nanofractal as nf

_MANIFEST_NAME = "manifest.json"


def _load_cv2():
    try:
        import cv2

        return cv2
    except ImportError:
        return None


def iter_real_images(
    real_dir: str | Path,
) -> Generator[tuple[np.ndarray, nf.Dict, list[int]], None, None]:
    """Yield ``(gray_image, nf_dict, expected_ids)`` from *real_dir/manifest.json*.

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

        dict_name: str = entry.get("dictionary", "DICT_4X4_50")
        try:
            nf_dict = nf.Dict[dict_name]
        except KeyError:
            print(f"  [real] WARNING: unknown dict {dict_name!r} — skipping")
            continue

        expected_ids: list[int] = entry.get("expected_ids", [])
        yield img, nf_dict, expected_ids
