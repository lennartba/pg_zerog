"""View-angle brightness normalisation.

OpenScan lights move with the camera: a surface seen head-on is lit head-on, one seen at an
angle is lit at that angle and looks darker. The blend then mixes head-on and oblique samples;
where oblique views must win (edges: the blur window), the texture gets their darker look.

Model, fitted per scan from the photos themselves (no assumed reflectance law):

    sample = albedo * g[view] * f(cos)

f: brightness vs the cosine between surface normal and view ray (f(1) = 1), g: per-photo
exposure. Both are fitted on random surface points away from edges/outlines (log domain,
alternating medians), then every sample is divided by g * f before blending: each view
contributes the head-on brightness. Nothing else changes (shadows, ambient occlusion, paint).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("pgz")

BINS = np.linspace(0.0, 1.0, 21)


def fit(trace: list, cos_min: float, iterations: int = 8) -> dict:
    """trace: blend_points records of sample points. Returns {'cos': .., 'f': .., 'gain': {view: g}}."""
    idx = np.concatenate([t["idx"] for t in trace])
    vid = np.concatenate([np.full(len(t["idx"]), i) for i, t in enumerate(trace)])
    rgb = np.concatenate([t["rgb"] for t in trace]).astype(float) / 255
    cos = np.concatenate([t["cos"] for t in trace])
    cw = np.concatenate([t["cos_win"] for t in trace])
    cpx = np.concatenate([t["contour_px"] for t in trace])
    lin = rgb ** 2.2
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    # clean samples only: not mixed with another face / background, not too dark, not clipped
    ok = (cw >= 0.9 * cos) & (cpx > 16) & (lum > 0.003) & (rgb.max(1) < 0.98) & (cos > cos_min)
    idx, vid, cos, y = idx[ok], vid[ok], cos[ok], np.log(lum[ok])
    b = np.clip(np.digitize(cos, BINS) - 1, 0, len(BINS) - 2)
    nv, npnt = len(trace), idx.max() + 1
    lg = np.zeros(nv)
    lf = np.zeros(len(BINS) - 1)
    for _ in range(iterations):
        r = y - lg[vid] - lf[b]
        la = _group_median(idx, r, npnt)
        r = y - la[idx] - lf[b]
        lg = _group_median(vid, r, nv)
        lg -= np.median(lg)
        r = y - la[idx] - lg[vid]
        lf = _group_median(b, r, len(BINS) - 1, fill=np.nan)
        lf = _fill_monotone(lf)
        lf -= lf[-1]  # f(1) = 1
    res = y - la[idx] - lg[vid] - lf[b]
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    counts = np.bincount(b, minlength=len(BINS) - 1)
    return {"cos": centers.tolist(), "f": np.exp(lf).tolist(), "bin_samples": counts.tolist(),
            "gain": {t["view"]: float(np.exp(g)) for t, g in zip(trace, lg)},
            "samples": int(len(y)), "residual_mad": float(np.median(np.abs(res))),
            "spread_before_mad": float(np.median(np.abs(y - _group_median(idx, y, npnt)[idx])))}


def _group_median(g, v, n, fill=0.0):
    out = np.full(n, fill, float)
    o = np.argsort(g, kind="stable")
    gs, vs = g[o], v[o]
    starts = np.r_[0, np.nonzero(np.diff(gs))[0] + 1]
    ends = np.r_[starts[1:], len(gs)]
    for s, e in zip(starts, ends):
        out[gs[s]] = np.median(vs[s:e])
    return out


def _fill_monotone(lf):
    """Bins without data take the nearest bin's value; brightness must not drop with rising cos."""
    x = np.arange(len(lf))
    good = np.isfinite(lf)
    lf = np.interp(x, x[good], lf[good])
    return np.maximum.accumulate(lf)


def f_of(model: dict, cos: np.ndarray) -> np.ndarray:
    """f(cos), 0 for cos <= 0 (background, back faces)."""
    return np.where(cos > 0, np.interp(cos, model["cos"], model["f"]), 0.0)


def factor(model: dict, view_name: str, cos: np.ndarray) -> np.ndarray:
    """g * f(cos) for samples of one view."""
    f = np.interp(cos, model["cos"], model["f"])
    return model["gain"].get(view_name, 1.0) * np.maximum(f, 0.05)


def save(model: dict, path: Path) -> None:
    path.write_text(json.dumps(model, indent=1))
