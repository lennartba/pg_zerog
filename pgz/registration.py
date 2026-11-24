"""Texture-on-mesh registration, measured directly.

The baked texture is rendered into photos that were NOT used for baking (held-out views)
and compared patch by patch with the photo: phase correlation gives the local shift
between where the texture puts a surface detail and where the camera saw it. Converted
to µm on the surface with the pixel footprint at that patch.

This catches what the outline check cannot: a print that deviates from its STL inside the
outline (ef01: cube tops 0.5 mm low -> texture shifted by parallax in every oblique view).
"""
from __future__ import annotations

import cv2
import numpy as np


def _highpass(lum: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    return lum - cv2.GaussianBlur(lum, (0, 0), sigma)


def texture_registration(obj, tex_rgb: np.ndarray, S: np.ndarray, views, mm_per_unit: float,
                         patch: int = 128, step: int = 96, min_response: float = 0.2,
                         holder_normal=None) -> dict:
    """obj in STL mm (already deformed to the real surface if a deformation exists),
    S: STL -> scan frame, views: held-out views. Returns shift statistics in µm.
    Patches on the holder side (normal . holder_normal > 0.5) are skipped, as for coverage."""
    import open3d as o3d
    from .align import apply
    from .sharpness import _lum, render_texture
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(apply(S, obj.v).astype(np.float32)),
                        o3d.core.Tensor(obj.fv.astype(np.uint32)))
    win = cv2.createHanningWindow((patch, patch), cv2.CV_64F)
    shifts, per_view = [], []
    for v in views:
        photo = cv2.imread(str(v.image))[:, :, ::-1]
        rend, hit, z = render_texture(scene, obj, tex_rgb, v)
        if holder_normal is not None:
            import open3d as o3d
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                o3d.core.Tensor(v.K), o3d.core.Tensor(v.extrinsic()), v.w, v.h)
            nrm = scene.cast_rays(rays)["primitive_normals"].numpy()
            Rs = S[:3, :3] / np.cbrt(np.linalg.det(S[:3, :3]))
            hit &= ~(nrm @ (Rs @ np.asarray(holder_normal, float)) > 0.5)
        mask = cv2.imread(str(v.image.with_suffix("")) + ".mask.png", cv2.IMREAD_GRAYSCALE)
        ok = hit & ((mask > 0) if mask is not None else True)
        # patch must lie fully on the object (outline shifts are measured elsewhere)
        inner = cv2.erode(ok.astype(np.uint8), np.ones((patch // 2 * 2 + 1,) * 2, np.uint8)).astype(bool)
        a_img = _highpass(_lum(photo)).astype(np.float64)
        b_img = _highpass(_lum(rend)).astype(np.float64)
        vs = []
        h = patch // 2
        for cy in range(h, v.h - h, step):
            for cx in range(h, v.w - h, step):
                if not inner[cy, cx]:
                    continue
                a = a_img[cy - h:cy + h, cx - h:cx + h]
                b = b_img[cy - h:cy + h, cx - h:cx + h]
                if a.std() < 1.5 or b.std() < 0.75:  # no detail to correlate
                    continue
                (dx, dy), resp = cv2.phaseCorrelate(b, a, win)
                if resp < min_response:
                    continue
                um = np.hypot(dx, dy) * z[cy, cx] * mm_per_unit * 1000 / v.K[0, 0]
                vs.append(um)
        if vs:
            shifts += vs
            per_view.append({"view": v.name, "patches": len(vs), "median_um": float(np.median(vs)),
                             "p90_um": float(np.quantile(vs, 0.9))})
    if len(shifts) < 20:
        return {"patches": len(shifts)}
    s = np.asarray(shifts)
    return {"patches": int(len(s)), "views": len(per_view), "median_um": float(np.median(s)),
            "p90_um": float(np.quantile(s, 0.9)), "frac_over_50um": float((s > 50).mean()),
            "per_view": per_view}
