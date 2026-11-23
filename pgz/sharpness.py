"""Edge spread (10-90 % rise width).

Two measurements:

* ``edge_spread`` - strongest edges of one image (texture atlas or photo). Cheap, but texture
  and photos then contain different edges (outline, holder, label), so the numbers are not
  directly comparable.
* ``matched_edge_spread`` - the baked texture is rendered back into real cameras (STL + UVs,
  ray-cast at photo resolution) and the SAME edges are measured in the render and in the
  photo. The difference is what the baking loses; the photo value is the ceiling. Edges on
  the object outline are excluded (only interior texture edges count).

On a plain base coat the strongest edges are shading/label edges, not painted strokes; the
number is a proxy until a highlighted scan is available.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates


def _lum(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _edges(lum, ok, top_frac, half, max_edges):
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ok = cv2.erode(ok.astype(np.uint8), np.ones((2 * half + 3,) * 2, np.uint8)).astype(bool)
    ok &= mag >= cv2.dilate(mag, np.ones((3, 3), np.uint8))  # one sample per edge crossing
    vals = mag[ok]
    if vals.size < 50:
        return None
    thr = np.quantile(vals, 1 - top_frac)
    ys, xs = np.nonzero(ok & (mag >= thr))
    if len(ys) > max_edges:
        sel = np.random.default_rng(0).choice(len(ys), max_edges, replace=False)
        ys, xs = ys[sel], xs[sel]
    n = np.stack([gx[ys, xs], gy[ys, xs]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return ys, xs, n


def _widths(lum, ys, xs, n, half, min_contrast):
    """10-90 % widths in px along n; NaN where contrast is too low."""
    t = np.arange(-half, half + 0.001, 0.25)
    px = xs[:, None] + n[:, :1] * t[None]
    py = ys[:, None] + n[:, 1:] * t[None]
    prof = map_coordinates(lum, [py, px], order=1, mode="nearest")
    lo, hi = prof[:, :8].mean(1), prof[:, -8:].mean(1)
    contrast = hi - lo
    f = (prof - lo[:, None]) / np.where(np.abs(contrast) > 1e-6, contrast, 1)[:, None]
    f = np.maximum.accumulate(f, axis=1)  # monotone envelope
    w = (np.argmax(f >= 0.9, axis=1) - np.argmax(f >= 0.1, axis=1)) * 0.25
    return np.where(contrast >= min_contrast, w, np.nan), contrast


def edge_spread(rgb: np.ndarray, valid: np.ndarray | None = None, top_frac: float = 0.003,
                half: int = 8, min_contrast: float = 40.0, max_edges: int = 4000) -> dict:
    lum = _lum(rgb)
    ok = np.ones(lum.shape, bool) if valid is None else valid
    e = _edges(lum, ok, top_frac, half, max_edges)
    if e is None:
        return {"edges": 0}
    w, _ = _widths(lum, *e, half, min_contrast)
    w = w[np.isfinite(w)]
    if len(w) < 20:
        return {"edges": int(len(w))}
    return {"edges": int(len(w)), "median_px": float(np.median(w)),
            "p25_px": float(np.quantile(w, 0.25)), "p75_px": float(np.quantile(w, 0.75))}


def render_texture(scene, obj, tex: np.ndarray, view, ds: float = 1.0):
    """Ray-cast the textured mesh (already in the view's frame) into a camera.
    Returns (rgb uint8, valid mask, depth z)."""
    import open3d as o3d
    w, h = int(view.w * ds), int(view.h * ds)
    K = view.K.copy()
    K[:2] *= ds
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3d.core.Tensor(K), o3d.core.Tensor(view.extrinsic()), w, h)
    ans = scene.cast_rays(rays)
    pid = ans["primitive_ids"].numpy().astype(np.int64)
    hit = pid != scene.INVALID_ID
    bc = ans["primitive_uvs"].numpy()[hit]
    f = pid[hit]
    uvc = obj.vt[obj.ft[f]]
    wts = np.c_[1 - bc.sum(1), bc]
    uv = (uvc * wts[:, :, None]).sum(1)
    H, W = tex.shape[:2]
    from .bake import _sample
    col = _sample(tex, (uv[:, 0] * W - 0.5).astype(np.float32), ((1 - uv[:, 1]) * H - 0.5).astype(np.float32),
                  cv2.INTER_LINEAR)
    img = np.zeros((h, w, 3), np.uint8)
    img[hit] = col
    z = ans["t_hit"].numpy() * (rays.numpy()[..., 3:] @ view.R[2])
    return img, hit, np.where(hit, z, np.nan)


def matched_edge_spread(obj, tex_rgb: np.ndarray, S: np.ndarray, views, mm_per_unit: float,
                        n_views: int = 8, top_frac: float = 0.01, half: int = 8,
                        min_contrast: float = 25.0, max_edges: int = 3000) -> dict:
    """Same edges in photo and in the re-rendered texture. obj in STL mm, S: STL -> scan frame."""
    import open3d as o3d
    from .align import apply
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(apply(S, obj.v).astype(np.float32)),
                        o3d.core.Tensor(obj.fv.astype(np.uint32)))
    sel = views[:: max(1, len(views) // n_views)][:n_views]
    wp, wr, um_px = [], [], []
    for v in sel:
        photo = cv2.imread(str(v.image))[:, :, ::-1]
        rend, valid, z = render_texture(scene, obj, tex_rgb, v)
        mask = cv2.imread(str(v.image.with_suffix("")) + ".mask.png", cv2.IMREAD_GRAYSCALE)
        ok = valid & ((mask > 0) if mask is not None else True)
        lp, lr = _lum(photo), _lum(rend)
        e = _edges(lp, ok, top_frac, half, max_edges)  # edges chosen in the photo, interior only
        if e is None:
            continue
        a, ca = _widths(lp, *e, half, min_contrast)
        b, cb = _widths(lr, *e, half, min_contrast * 0.5)
        good = np.isfinite(a) & np.isfinite(b)
        wp.append(a[good])
        wr.append(b[good])
        ys, xs = e[0][good], e[1][good]
        um_px.append(z[ys, xs] * mm_per_unit * 1000 / v.K[0, 0])
    if not wp or sum(len(x) for x in wp) < 20:
        return {"edges": 0}
    wp, wr, um = map(np.concatenate, (wp, wr, um_px))
    return {"edges": int(len(wp)), "views": len(sel), "photo_px": float(np.median(wp)),
            "texture_px": float(np.median(wr)), "um_per_px": float(np.median(um)),
            "photo_um": float(np.median(wp * um)), "texture_um": float(np.median(wr * um)),
            "texture_um_p75": float(np.quantile(wr * um, 0.75))}
