"""Silhouette check and refinement of the scan->STL alignment.

The STL is ray-cast into every camera with the current alignment; its outline is
compared with the object outline in the photo (distance transform of the image
boundary). Outline stretches that border the holder instead of background are
skipped. At ~15 µm per pixel this measures the alignment independently of the noisy
dense cloud, and the same distances drive a 7-DOF least-squares refinement.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .align import apply, scale_of
from .views import View, load_views

log = logging.getLogger("pgz")


def _image_outline(view: View, scale: float, thr: int):
    """Foreground mask and distance-to-outline map of the photo at `scale`."""
    bgr = cv2.imread(str(view.image), cv2.IMREAD_COLOR)
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    fg = (bgr.max(axis=2) > thr).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # fill dark regions enclosed by the object (glossy black underside, shadows)
    ff = fg.copy()
    flood = np.zeros((fg.shape[0] + 2, fg.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 1)
    fg = fg | (1 - ff)
    edge = fg - cv2.erode(fg, np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform((1 - edge).astype(np.uint8), cv2.DIST_L2, 5)
    return fg.astype(bool), dt


class _Caster:
    def __init__(self, verts: np.ndarray, faces: np.ndarray):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(verts.astype(np.float32)),
                                 o3d.core.Tensor(faces.astype(np.uint32)))

    def cast(self, view: View, scale: float):
        w, h = int(round(view.w * scale)), int(round(view.h * scale))
        K = view.K.copy()
        K[:2] *= scale
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(K), o3d.core.Tensor(view.extrinsic()), w, h)
        ans = self.scene.cast_rays(rays)
        t = ans["t_hit"].numpy()
        r = rays.numpy()
        hit = np.isfinite(t)
        pts = r[..., :3] + r[..., 3:] * np.where(hit, t, 0)[..., None]
        return hit, pts


def _contour_samples(view, caster, scale, img_fg, img_dt, ring_px, max_px):
    """3D points (scan frame) on the STL outline in this view, valid ones only."""
    hit, pts = caster.cast(view, scale)
    m = hit.astype(np.uint8)
    contour = (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)
    r = max(2, int(round(ring_px)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    ring = cv2.dilate(m, k).astype(bool) & ~hit
    inner = hit & ~cv2.erode(m, k).astype(bool)

    def frac(sel, of):
        num = cv2.boxFilter(sel.astype(np.float32), -1, (2 * r + 1,) * 2, normalize=False)
        den = cv2.boxFilter(of.astype(np.float32), -1, (2 * r + 1,) * 2, normalize=False)
        return num / np.maximum(den, 1)

    # usable where the photo shows background just outside (not holder/putty) and the
    # object just inside (not a dark, invisible part of the prop)
    ok = (contour & (frac(ring & img_fg, ring) <= 0.1) & (frac(inner & img_fg, inner) >= 0.7)
          & (img_dt < max_px))
    ys, xs = np.nonzero(ok)
    return pts[ys, xs], ok.sum(), contour.sum()


def _residuals(X, views_dt, scale):
    res = []
    for (view, dt), P in zip(views_dt, X):
        if len(P) == 0:
            continue
        uv, _ = view.project(P, scale)
        res.append(map_coordinates(dt, [uv[:, 1], uv[:, 0]], order=1, mode="nearest"))
    return np.concatenate(res) if res else np.zeros(0)


def _similarity(p, c):
    """x -> s R (x - c) + c + t with p = (rotvec, t, log s)."""
    R = Rotation.from_rotvec(p[:3]).as_matrix()
    s = np.exp(p[6])
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = c + p[3:6] - s * R @ c
    return M


def measure_and_refine(dense: Path, stl: Path, T: np.ndarray, work: Path, cfg: dict) -> np.ndarray:
    """Returns the refined scan->STL transform; writes silhouette.json."""
    sc = cfg["silhouette"]
    out = work / "silhouette.json"
    if out.exists():
        return np.array(json.loads(out.read_text())["scan_to_stl"])

    views = load_views(dense)
    stl_mesh = trimesh.load(stl, force="mesh")
    Vstl, F = np.asarray(stl_mesh.vertices), np.asarray(stl_mesh.faces)

    def stats(Tcur, vlist, scale, imgs):
        S = np.linalg.inv(Tcur)
        caster = _Caster(apply(S, Vstl), F)
        per_view, X, used = [], [], 0
        for v, (fg, dt) in zip(vlist, imgs):
            P, n_ok, n_all = _contour_samples(v, caster, scale, fg, dt, sc["ring_px"] * scale,
                                              sc["max_px"] * scale)
            X.append(P)
            if n_ok < 50:
                continue
            uv, z = v.project(P, scale)
            d = map_coordinates(dt, [uv[:, 1], uv[:, 0]], order=1, mode="nearest") / scale
            # sign: + if the STL outline lies outside the photo's object (STL too big)
            inside = map_coordinates(fg.astype(np.float32), [uv[:, 1], uv[:, 0]], order=0, mode="nearest") > 0.5
            sd = np.where(inside, -d, d)
            um_per_px = 1000 * np.median(z) / v.K[0, 0] * scale_of(Tcur)
            per_view.append((np.median(d), np.median(d) * um_per_px, n_ok / max(n_all, 1),
                             np.median(sd) * um_per_px))
            used += n_ok
        return np.array(per_view), X, used

    # refinement on a subset of views at reduced resolution
    sub = views[:: sc["view_stride"]]
    rs = sc["refine_scale"]
    with ThreadPoolExecutor() as ex:
        imgs_sub = list(ex.map(lambda v: _image_outline(v, rs, sc["threshold"]), sub))
    Tcur = T.copy()
    for it in range(sc["iterations"]):
        S0 = np.linalg.inv(Tcur)
        _, Xscan, used = stats(Tcur, sub, rs, imgs_sub)
        c = np.concatenate([x for x in Xscan if len(x)]).mean(0)

        def fun(p):
            M = _similarity(p, c)
            return _residuals([apply(M, x) for x in Xscan], list(zip(sub, [d for _, d in imgs_sub])), rs)

        sol = least_squares(fun, np.zeros(7), loss="soft_l1", f_scale=1.0, x_scale=[1e-3] * 3 + [1e-3] * 3 + [1e-3])
        S1 = _similarity(sol.x, c) @ S0
        Tcur = np.linalg.inv(S1)
        log.info("  silhouette refine %d: %d outline points, rot %.4f°, shift %.4f, scale %+.3f%%",
                 it + 1, used, np.degrees(np.linalg.norm(sol.x[:3])), np.linalg.norm(sol.x[3:6]) * scale_of(Tcur),
                 100 * (np.exp(sol.x[6]) - 1))

    # final measurement: all views, higher resolution
    ms = sc["measure_scale"]
    with ThreadPoolExecutor() as ex:
        imgs_all = list(ex.map(lambda v: _image_outline(v, ms, sc["threshold"]), views))
    pv0, _, _ = stats(T, views, ms, imgs_all)
    pv1, _, _ = stats(Tcur, views, ms, imgs_all)

    def summary(pv):
        if len(pv) == 0:
            return {"views": 0, "median_px": float("nan"), "median_um": float("nan"),
                    "p90_um": float("nan"), "usable_outline": 0.0, "signed_median_um": float("nan")}
        return {"views": int(len(pv)), "median_px": float(np.median(pv[:, 0])),
                "median_um": float(np.median(pv[:, 1])), "p90_um": float(np.quantile(pv[:, 1], 0.9)),
                "usable_outline": float(np.median(pv[:, 2])),
                "signed_median_um": float(np.median(pv[:, 3]))}

    info = {"scan_to_stl": Tcur.tolist(), "before": summary(pv0), "after": summary(pv1),
            "scale_mm_per_unit": scale_of(Tcur)}
    out.write_text(json.dumps(info, indent=2))
    if info["after"]["views"] < 10:
        log.warning("  silhouette check: only %d usable views - alignment NOT verified", info["after"]["views"])
    log.info("  silhouette distance (median over %d views): %.1f µm before, %.1f µm after refinement",
             info["after"]["views"], info["before"]["median_um"], info["after"]["median_um"])
    return Tcur
