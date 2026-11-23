"""Foreground masks against the OpenScan's black background.

The holder (stick, putty, label) is kept in the mask on purpose: it rotates with the
prop and gives the feature matcher texture that the plain base coat lacks. It is cut
away later, on the mesh, using the STL.
"""
from __future__ import annotations

import cv2
import numpy as np


def foreground_mask(rgb: np.ndarray, cfg: dict) -> np.ndarray:
    """uint8 mask, 255 = object, 0 = background."""
    m = (rgb.max(axis=2) > cfg["threshold"]).astype(np.uint8)
    if cfg["close_px"]:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["close_px"],) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_area = cfg["min_area_frac"] * m.size
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    m = keep[lab]
    # fill holes (dark paint inside the silhouette)
    inv = (1 - m).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = m.shape
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if x > 0 and y > 0 and x + bw < w and y + bh < h:
            m[lab == i] = 1
    if cfg["dilate_px"]:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["dilate_px"],) * 2)
        m = cv2.dilate(m, k)
    return m * 255


def read_rgb(path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
