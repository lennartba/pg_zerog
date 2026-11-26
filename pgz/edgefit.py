"""Measure how the print's convex edges really look: radius and position, from photo outlines.

Wherever a sharp STL edge is the outline of the prop in a photo (one face towards the camera,
the other away, background behind), the photo shows where the real edge is. A rounded edge
of radius r has its outline inside the sharp corner by

    delta = r * g(view) - D . n_s,   g = (nA . n_s) - 1 - tan(a/2) * (uA . n_s)

n_s: the fillet normal perpendicular to the view ray (outline direction), a: turn angle of
the faces, uA: direction into face A away from the edge, D: displacement of the real corner
in the edge's cross-section. g runs from 0 (view along a face) to 0.41 (90 deg edge seen along
the bisector), so views from different angles separate r from D.

The outline is measured at the 50 % crossing between the prop and the (black) background,
perpendicular to the projected edge, averaged over a few px along it: sub-pixel, no shading
model involved. Fit per edge line (robust least squares), with a scan-wide fit as fallback.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

from .align import apply, scale_of
from .edges import sharp_convex_edges

log = logging.getLogger("pgz")


def edge_lines(mesh: trimesh.Trimesh, min_angle_deg: float):
    """Group sharp convex STL edge segments into lines (same face normals, collinear)."""
    A, B, nA, nB = sharp_convex_edges(mesh, min_angle_deg)
    key = []
    for a, b, na, nb in zip(A, B, nA, nB):
        if tuple(np.round(na, 3)) > tuple(np.round(nb, 3)):
            na, nb = nb, na
        d = (b - a) / np.linalg.norm(b - a)
        # line id: normals + position of the line in the plane perpendicular to it
        off = a - (a @ d) * d
        key.append(tuple(np.round(np.r_[na, nb, off], 2)))
    ids = {k: i for i, k in enumerate(dict.fromkeys(key))}
    line = np.array([ids[k] for k in key])
    return A, B, nA, nB, line


def _g(na, nb, ns):
    a = np.arccos(np.clip(na @ nb, -1, 1))
    d = np.cross(na, nb)
    d /= np.linalg.norm(d)
    ua = np.cross(d, na)
    if ua @ nb > 0:
        ua = -ua
    return (na @ ns) - 1 - np.tan(a / 2) * (ua @ ns)


def measure(stl_mesh, lattice, T, views, scene, cfg) -> list[dict]:
    """Outline offsets of sharp convex edges in all views (STL frame, mm)."""
    e = cfg["edges"]
    S = np.linalg.inv(T)
    k = scale_of(T)
    Rs = S[:3, :3] / scale_of(S)
    A, B, nA, nB, line = edge_lines(stl_mesh, cfg["bake"]["edge_min_angle_deg"])
    # sample points along each segment, away from its ends (corners of the prop)
    pts, seg = [], []
    for i, (a, b) in enumerate(zip(A, B)):
        L = np.linalg.norm(b - a)
        if L < 2 * e["end_margin_mm"] + 0.1:
            continue
        t = np.arange(e["end_margin_mm"], L - e["end_margin_mm"] + 1e-9, e["sample_step_mm"]) / L
        pts.append(a + t[:, None] * (b - a))
        seg.append(np.full(len(t), i))
    if not pts:
        return []
    P = np.concatenate(pts)
    sg = np.concatenate(seg)
    Pd = lattice(P) if lattice is not None else P
    Xs = apply(S, Pd)
    D = (B - A) / np.linalg.norm(B - A, axis=1, keepdims=True)
    half, step, along = e["profile_px"], 0.25, e["along_px"]
    tq = np.arange(-half, half + 1e-9, step)
    out = []
    for v in views:
        C = apply(T, v.center[None])[0]  # camera centre, STL frame
        vd = Pd - C
        vd /= np.linalg.norm(vd, axis=1, keepdims=True)
        fa = np.einsum("ij,ij->i", nA[sg], -vd)
        fb = np.einsum("ij,ij->i", nB[sg], -vd)
        cand = np.nonzero(((fa > 0.05) & (fb < -0.05)) | ((fa < -0.05) & (fb > 0.05)))[0]
        if len(cand) == 0:
            continue
        uv, z = v.project(Xs[cand])
        inimg = (uv[:, 0] > half + 10) & (uv[:, 1] > half + 10) & (uv[:, 0] < v.w - half - 10) & (uv[:, 1] < v.h - half - 10)
        cand, uv, z = cand[inimg], uv[inimg], z[inimg]
        if len(cand) == 0:
            continue
        # outline normal: fillet normal perpendicular to the view ray, in the cross-section
        ed = D[sg[cand]]
        ns = np.cross(ed, vd[cand])
        ns /= np.linalg.norm(ns, axis=1, keepdims=True)
        ns = np.where((np.einsum("ij,ij->i", ns, nA[sg[cand]] + nB[sg[cand]]) < 0)[:, None], -ns, ns)
        # corner itself visible (ray from the camera reaches it) and background behind the outline
        Xc = Xs[cand]
        q_in = apply(S, Pd[cand] - 0.03 * (nA[sg[cand]] + nB[sg[cand]]))
        q_out = apply(S, Pd[cand] + e["background_probe_mm"] * ns)
        rays = []
        for q in (q_in, q_out):
            dvec = q - v.center
            dn = np.linalg.norm(dvec, axis=1, keepdims=True)
            rays.append(np.c_[np.repeat(v.center[None], len(q), 0), dvec / dn])
        hit = scene.cast_rays(o3d.core.Tensor(np.concatenate(rays).astype(np.float32)))["t_hit"].numpy()
        n = len(cand)
        d_in = np.linalg.norm(q_in - v.center, axis=1)
        vis = hit[:n] >= d_in - 0.1 / k
        bg = ~np.isfinite(hit[n:]) | (hit[n:] > np.linalg.norm(q_out - v.center, axis=1) + 2.0 / k)
        keep = vis & bg
        if not keep.any():
            continue
        miss = ~np.isfinite(hit[n:])[keep]  # nothing behind the outline (not a far wall)
        cand, uv, z, ns = cand[keep], uv[keep], z[keep], ns[keep]
        # image directions: along the projected edge and outward (towards q_out)
        uv2, _ = v.project(apply(S, Pd[cand] + 0.1 * D[sg[cand]]))
        uo, _ = v.project(apply(S, Pd[cand] + 0.1 * ns))
        de = uv2 - uv
        de /= np.linalg.norm(de, axis=1, keepdims=True)
        do = np.c_[-de[:, 1], de[:, 0]]
        do = np.where((np.einsum("ij,ij->i", do, uo - uv) < 0)[:, None], -do, do)
        img = cv2.imread(str(v.image))[:, :, 2].astype(np.float32)  # red: base coat vs black velvet
        offs = np.arange(-along, along + 1e-9, 1.0)
        for j in range(len(cand)):
            c0 = uv[j] - 0.5
            grid = c0[None, None] + tq[None, :, None] * do[j][None, None] + offs[:, None, None] * de[j][None, None]
            prof = cv2.remap(img, grid[..., 0].astype(np.float32), grid[..., 1].astype(np.float32),
                             cv2.INTER_LINEAR).mean(0)
            lvl_in = np.median(prof[tq <= -half + 4])
            lvl_out = np.median(prof[tq >= half - 4])
            if lvl_in - lvl_out < e["min_contrast"]:
                continue
            half_lvl = 0.5 * (lvl_in + lvl_out)
            below = np.nonzero((prof[:-1] >= half_lvl) & (prof[1:] < half_lvl))[0]
            if len(below) != 1:  # ambiguous (texture, other edges): skip
                continue
            i0 = below[0]
            f = (prof[i0] - half_lvl) / (prof[i0] - prof[i0 + 1])
            cross_px = tq[i0] + f * step
            mm_per_px = z[j] * k / v.K[0, 0]
            s = sg[cand[j]]
            out.append({"view": v.name, "seg": int(s), "line": int(line[s]), "p": P[cand[j]].tolist(),
                        "ns": ns[j].tolist(), "delta_mm": float(-cross_px * mm_per_px),
                        "g": float(_g(nA[s], nB[s], ns[j])), "contrast": float(lvl_in - lvl_out), "bg_miss": bool(miss[j]),
                        "mm_per_px": float(mm_per_px)})
    return out


def _fit(rows, nA, nB, sigma, prior_r, sigma_r, sigma_d, r_min, huber=0.03):
    """delta = r g - D . ns with D = dA nA + dB nB; robust weighted least squares with a prior
    (r -> prior_r, D -> 0) so poorly observed lines stay near the scan-wide values.
    Samples of one view are correlated: the data term counts views, not samples."""
    M = np.array([[r["g"], -(nA @ r["ns"]), -(nB @ r["ns"])] for r in rows])
    y = np.array([r["delta_mm"] for r in rows])
    nviews = len({r["view"] for r in rows})
    scale = nviews / len(rows) / sigma ** 2
    Pw = np.diag([1 / sigma_r ** 2, 1 / sigma_d ** 2, 1 / sigma_d ** 2])
    p0 = np.array([prior_r, 0.0, 0.0])
    x = p0.copy()
    fix_r = None
    for _ in range(10):
        res = M @ x - y
        w = np.where(np.abs(res) <= huber, 1.0, huber / np.maximum(np.abs(res), 1e-9)) * scale
        if fix_r is None:
            x = np.linalg.solve(M.T @ (w[:, None] * M) + Pw, M.T @ (w * y) + Pw @ p0)
            if x[0] < r_min:
                fix_r = r_min
        if fix_r is not None:
            Mr = M[:, 1:]
            x = np.r_[fix_r, np.linalg.solve(Mr.T @ (w[:, None] * Mr) + Pw[1:, 1:], Mr.T @ (w * (y - fix_r * M[:, 0])))]
    res = M @ x - y
    cov = np.linalg.inv(M.T @ (w[:, None] * M) + Pw)
    return x, float(np.sqrt(np.median(res ** 2))), np.sqrt(np.diag(cov)), nviews


def fit(stl: Path, lattice, T, views, scene, work: Path, cfg) -> dict:
    """Per edge line: radius and corner displacement. Writes work/edges.json (incl. per-segment
    arrays for edges.fillet_remap)."""
    outp = work / "edges.json"
    if outp.exists():
        return json.loads(outp.read_text())
    e = cfg["edges"]
    m = trimesh.load(stl, force="mesh")
    rows = measure(m, lattice, T, views, scene, cfg)
    A, B, nA, nB, line = edge_lines(m, cfg["bake"]["edge_min_angle_deg"])
    nseg = len(A)
    res = {"measurements": len(rows), "lines": {}}
    rg = e["fallback_radius_mm"]
    by_line = {}
    for r in rows:
        by_line.setdefault(r["line"], []).append(r)
    usable = {li: rl for li, rl in by_line.items() if len({r["view"] for r in rl}) >= e["min_views"]}
    if usable:
        # scan-wide radius: weighted median of loosely regularised per-line fits
        rs, ws = [], []
        for li, rl in usable.items():
            s0 = rl[0]["seg"]
            x, _, sd, nv = _fit(rl, nA[s0], nB[s0], e["sigma_mm"], 0.0, 1.0, 1.0, e["min_radius_mm"])
            rs.append(x[0])
            ws.append(1 / max(sd[0], 1e-3) ** 2)
        o = np.argsort(rs)
        cw = np.cumsum(np.asarray(ws)[o])
        rg = float(np.asarray(rs)[o][np.searchsorted(cw, cw[-1] / 2)])
    rg = max(rg, e["min_radius_mm"])
    seg_r = np.full(nseg, rg)
    seg_dA = np.zeros(nseg)
    seg_dB = np.zeros(nseg)
    for li, rl in usable.items():
        s0 = rl[0]["seg"]
        x, rms, sd, nv = _fit(rl, nA[s0], nB[s0], e["sigma_mm"], rg, e["prior_radius_sd_mm"],
                              e["prior_offset_sd_mm"], e["min_radius_mm"])
        Dv = x[1] * nA[s0] + x[2] * nB[s0]
        for s in np.nonzero(line == li)[0]:  # express in each segment's own normal pair
            sol = np.linalg.lstsq(np.c_[nA[s], nB[s]], Dv, rcond=None)[0]
            seg_r[s], seg_dA[s], seg_dB[s] = x[0], sol[0], sol[1]
        seg_ids = np.nonzero(line == li)[0]
        pts = np.r_[A[seg_ids], B[seg_ids]]
        res["lines"][str(li)] = {"n": len(rl), "views": nv, "from": pts.min(0).round(2).tolist(),
                                 "to": pts.max(0).round(2).tolist(), "nA": nA[s0].round(3).tolist(),
                                 "nB": nB[s0].round(3).tolist(), "radius_mm": float(x[0]),
                                 "dA_mm": float(x[1]), "dB_mm": float(x[2]), "sd_mm": sd.tolist(), "rms_mm": rms}
    res["global_radius_mm"] = rg
    res["lines_fitted"] = len(usable)
    res["segments"] = {"radius_mm": seg_r.tolist(), "dA_mm": seg_dA.tolist(), "dB_mm": seg_dB.tolist()}
    outp.write_text(json.dumps(res, indent=1))
    (work / "edges_measurements.json").write_text(json.dumps(rows))
    log.info("  edges: %d outline measurements, %d edge lines fitted, scan-wide radius %.0f µm",
             len(rows), len(usable), 1000 * rg)
    for li, v_ in res["lines"].items():
        log.info("    edge %s..%s: r %.0f µm, offsets %+.0f / %+.0f µm (%d views)", v_["from"], v_["to"],
                 1000 * v_["radius_mm"], 1000 * v_["dA_mm"], 1000 * v_["dB_mm"], v_["views"])
    return res
