"""Edge spread (10-90 % rise width) on the strongest edges of an image.

Profiles are taken across the strongest luminance edges, along the gradient direction.
Reported as the median over edges with enough contrast. Used on the baked texture
(in texels) and on source photos (in pixels) to see how much the baking loses.

On a plain base coat the strongest edges are shading/holder/label edges, not painted
strokes; the number is a proxy until a highlighted scan is available.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates


def edge_spread(rgb: np.ndarray, valid: np.ndarray | None = None, top_frac: float = 0.003,
                half: int = 8, min_contrast: float = 40.0, max_edges: int = 4000) -> dict:
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ok = np.ones(lum.shape, bool) if valid is None else valid.copy()
    # stay away from chart borders / image borders
    ok = cv2.erode(ok.astype(np.uint8), np.ones((2 * half + 3,) * 2, np.uint8)).astype(bool)
    # local maxima only (one sample per edge crossing)
    ok &= mag >= cv2.dilate(mag, np.ones((3, 3), np.uint8))
    vals = mag[ok]
    if vals.size < 50:
        return {"edges": 0}
    thr = np.quantile(vals, 1 - top_frac)
    ys, xs = np.nonzero(ok & (mag >= thr))
    if len(ys) > max_edges:
        sel = np.random.default_rng(0).choice(len(ys), max_edges, replace=False)
        ys, xs = ys[sel], xs[sel]
    n = np.stack([gx[ys, xs], gy[ys, xs]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    t = np.arange(-half, half + 0.001, 0.25)
    px = xs[:, None] + n[:, :1] * t[None]
    py = ys[:, None] + n[:, 1:] * t[None]
    prof = map_coordinates(lum, [py, px], order=1, mode="nearest")
    lo = prof[:, :8].mean(1)
    hi = prof[:, -8:].mean(1)
    contrast = hi - lo
    good = contrast >= min_contrast
    prof, lo, contrast = prof[good], lo[good], contrast[good]
    if len(prof) < 20:
        return {"edges": int(len(prof))}
    f = (prof - lo[:, None]) / contrast[:, None]
    f = np.maximum.accumulate(f, axis=1)  # monotone envelope
    i10 = np.argmax(f >= 0.1, axis=1)
    i90 = np.argmax(f >= 0.9, axis=1)
    width = (i90 - i10) * 0.25
    return {"edges": int(len(width)), "median_px": float(np.median(width)),
            "p25_px": float(np.quantile(width, 0.25)), "p75_px": float(np.quantile(width, 0.75))}
