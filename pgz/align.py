"""Align the reconstruction to the prop's STL and cut away holder/clutter.

COLMAP's scale is arbitrary, so the transform is a similarity (rotation, translation,
scale). The STL then gives true millimetres.

1. Segment the prop in the dense cloud by base-coat colour; the holder (putty,
   stick, label) has a different colour. Keep the largest connected cluster.
2. Coarse: PCA axes and extents of cloud vs STL, all 24 axis assignments,
   trimmed ICP with scale against dense STL samples. Lowest error wins.
3. Fine: trimmed ICP with exact closest points on the STL surface.
"""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

log = logging.getLogger("pgz")
RNG = np.random.default_rng(0)


def _signed_permutations() -> list[np.ndarray]:
    """All 48 signed permutation matrices; the caller keeps those giving a proper rotation
    (PCA eigenvector bases can be left- or right-handed)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            m[range(3), perm] = signs
            out.append(m)
    return out


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> np.ndarray:
    """Similarity (or rigid) transform (4x4) minimising |s R src + t - dst|^2."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum() if with_scale else 1.0
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = md - s * R @ ms
    return T


def apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def scale_of(T: np.ndarray) -> float:
    return float(np.cbrt(np.linalg.det(T[:3, :3])))


def _icp(src, T, tgt, tgt_tree, iters, keep, with_scale, rev=None, keep_rev=0.8):
    """Trimmed ICP. Rigid: scale stays as in T. With scale: correspondences in both
    directions (scan->STL and STL->scan), otherwise the scan shrinks into one region
    of the STL, which lowers one-sided distances."""
    s0 = scale_of(T)
    err = np.inf
    for _ in range(iters):
        p = apply(T, src)
        d, i = tgt_tree.query(p)
        sel = d <= np.quantile(d, keep)
        a, b = src[sel], tgt[i[sel]]
        if with_scale:
            rv = tgt if rev is None else rev
            dr, j = cKDTree(p).query(rv)
            selr = dr <= np.quantile(dr, keep_rev)
            a, b = np.r_[a, src[j[selr]]], np.r_[b, rv[selr]]
            T = umeyama(a, b, True)
        else:
            T = umeyama(a * s0, b, False) @ np.diag([s0, s0, s0, 1.0])
        err = float(np.sqrt(np.mean(d[sel] ** 2)))
    return T, err


def segment_prop(points: np.ndarray, colors: np.ndarray, cfg: dict) -> np.ndarray:
    """Indices of the prop: base-coat coloured points, largest connected cluster."""
    sc = cfg["segment"]
    hsv = cv2.cvtColor(colors[:, :3].astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h = hsv[:, 0]
    hue_ok = np.zeros(len(h), bool)
    for lo, hi in sc["hue_ranges"]:
        hue_ok |= (h >= lo) & (h <= hi)
    idx = np.where(hue_ok & (hsv[:, 1] >= sc["min_saturation"]) & (hsv[:, 2] >= sc["min_value"]))[0]
    if len(idx) < 1000:
        raise RuntimeError(f"only {len(idx)} points match the base-coat colour; check [segment] in the config")
    if len(idx) > 300_000:
        idx = RNG.choice(idx, 300_000, replace=False)
    p = points[idx]
    tree = cKDTree(p)
    nn = tree.query(p[RNG.choice(len(p), min(5000, len(p)), replace=False)], k=2)[0][:, 1]
    r = 4 * np.median(nn)
    pairs = tree.query_pairs(r, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(p), len(p)))
    _, lab = connected_components(g, directed=False)
    big = np.bincount(lab).argmax()
    keep = idx[lab == big]
    log.info("  prop segment: %d coloured points, largest cluster %d", len(idx), len(keep))
    return keep


def align_to_stl(dense_ply: Path, mesh_ply: Path, stl: Path, work: Path, cfg: dict) -> np.ndarray:
    out = work / "align.json"
    if out.exists():
        return np.array(json.loads(out.read_text())["scan_to_stl"])

    cloud = trimesh.load(dense_ply, process=False)
    pts = np.asarray(cloud.vertices, float)
    cols = np.asarray(cloud.colors) if getattr(cloud, "colors", None) is not None and len(cloud.colors) else None
    if cols is None:
        raise RuntimeError(f"{dense_ply} has no colours")
    prop = pts[segment_prop(pts, cols, cfg)]
    src = prop[RNG.choice(len(prop), min(8000, len(prop)), replace=False)]

    target = trimesh.load(stl, force="mesh")
    samples, sf = trimesh.sample.sample_surface_even(target, 200_000, seed=0)
    tree = cKDTree(samples)
    # STL side the holder covers is never seen; keep it out of the STL->scan term
    hn = np.asarray(cfg["report"]["holder_normal"], float)
    visible = samples[target.face_normals[sf] @ hn <= 0.9][::4]

    # PCA frames
    cs, ct = src.mean(0), samples.mean(0)
    ws, Es = np.linalg.eigh(np.cov((src - cs).T))
    wt, Et = np.linalg.eigh(np.cov((samples - ct).T))
    Es, Et = Es[:, ::-1], Et[:, ::-1]
    ss, st = np.sqrt(ws[::-1]), np.sqrt(wt[::-1])
    s0 = float(np.mean(st[:2] / ss[:2]))

    # coarse: rigid at the PCA scale, all axis assignments
    coarse = []
    for D in _signed_permutations():
        R = Et @ D @ Es.T
        if np.linalg.det(R) < 0:
            continue
        T = np.eye(4)
        T[:3, :3] = s0 * R
        T[:3, 3] = ct - s0 * R @ cs
        T, err = _icp(src, T, samples, tree, 20, 0.85, with_scale=False)
        coarse.append((err, T))
    coarse.sort(key=lambda c: c[0])
    # fine: scale free, bidirectional - for the best few, because near-symmetric props
    # can rank a flipped pose first at the (slightly wrong) PCA scale
    best = (np.inf, None)
    for _, T in coarse[: cfg["align"]["fine_candidates"]]:
        T, _ = _icp(src, T, samples, tree, 40, 0.90, with_scale=True, rev=visible)
        T, err = _icp(src, T, samples, tree, 10, 0.90, with_scale=False)
        if err < best[0]:
            best = (err, T)
    T = best[1]
    _, d, _ = trimesh.proximity.closest_point(target, apply(T, src))
    info = {
        "scan_to_stl": T.tolist(),
        "scale_mm_per_unit": scale_of(T),
        "fine_err_mm": best[0],
        "prop_points_rms_mm": float(np.sqrt(np.mean(d**2))),
        "prop_points_p95_mm": float(np.quantile(d, 0.95)),
    }
    out.write_text(json.dumps(info, indent=2))
    log.info("  aligned: scale %.4f mm/unit, prop-point RMS %.3f mm, p95 %.3f mm",
             info["scale_mm_per_unit"], info["prop_points_rms_mm"], info["prop_points_p95_mm"])
    return T


def crop_mesh(mesh_ply: Path, stl: Path, T: np.ndarray, out: Path, cfg: dict) -> Path:
    """Keep faces within crop_tolerance_mm of the STL, largest connected piece.
    Output stays in scan coordinates, because OpenMVS textures in that frame."""
    if out.exists():
        return out
    m = trimesh.load(mesh_ply, process=False)
    target = trimesh.load(stl, force="mesh")
    samples, _ = trimesh.sample.sample_surface_even(target, 1_000_000, seed=1)
    d, _ = cKDTree(samples).query(apply(T, m.triangles_center))
    keep = d <= cfg["mesh"]["crop_tolerance_mm"]
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda p: p.area)
    log.info("  crop: kept %d of %d faces (%.1f%%)", len(m.faces), len(keep), 100 * len(m.faces) / len(keep))
    m.export(out)
    return out
