"""Texture baking onto the STL: own UV atlas + blended best views.

OpenMVS's TextureMesh picks one view per face and packs every resulting patch on its
own (ef01: ~19,000 islands, seams everywhere). Here, as in Metashape:

1. UV atlas of the STL with Microsoft UVAtlas (Open3D): a few large charts.
2. Every texel gets its 3D point and normal (STL frame, mm).
3. Every view: project the texels, keep those visible (ray-cast depth of the STL plus
   the holder taken from the scan mesh), inside the object mask and not too oblique.
4. Colour = weighted mean in linear light; weight = (cos angle)^a * (pixel footprint
   ratio)^b, raised to `sharpness_power`, so the best view dominates (mild blending,
   allowed by the requirements). Weights change smoothly, so switching views makes no
   seams.
5. Charts are padded by dilation so bilinear sampling never reads background.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from .align import apply, scale_of
from .views import load_views

log = logging.getLogger("pgz")


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    s = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)
    return np.round(s * 255).astype(np.uint8)


def _sample(img: np.ndarray, x: np.ndarray, y: np.ndarray, interp: int) -> np.ndarray:
    """cv2.remap for long point lists (remap maps must be < 32767 wide)."""
    n = len(x)
    W = 4096
    rows = (n + W - 1) // W
    pad = rows * W - n
    mx = np.pad(x.astype(np.float32), (0, pad)).reshape(rows, W)
    my = np.pad(y.astype(np.float32), (0, pad)).reshape(rows, W)
    out = cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(rows * W, *img.shape[2:])[:n]


def uv_atlas(stl_mesh: trimesh.Trimesh, cfg: dict) -> tuple[np.ndarray, int]:
    """Per-corner UVs (F,3,2) in [0,1] and the atlas size, chosen for the texel target."""
    b = cfg["bake"]
    fill = 0.65  # typical UVAtlas packing on CAD-like props
    target_mm = b["texel_um"] / 1000.0
    need = np.sqrt(stl_mesh.area / fill) / target_mm
    size = int(min(b["max_size"], max(1024, 512 * np.ceil(need / 512))))
    t = o3d.t.geometry.TriangleMesh()
    t.vertex.positions = o3d.core.Tensor(np.asarray(stl_mesh.vertices, np.float32))
    t.triangle.indices = o3d.core.Tensor(np.asarray(stl_mesh.faces, np.int32))
    stretch, charts, _ = t.compute_uvatlas(size=size, gutter=float(2 * b["padding_px"]),
                                           max_stretch=b["max_stretch"])
    log.info("  UV atlas: %d charts, %d x %d, max stretch %.3f", charts, size, size, stretch)
    return t.triangle.texture_uvs.numpy().astype(np.float64), size


def rasterize(stl_mesh: trimesh.Trimesh, uv: np.ndarray, size: int):
    """Texel centres covered by each UV triangle -> pixel index, 3D point, normal."""
    V, F = np.asarray(stl_mesh.vertices), np.asarray(stl_mesh.faces)
    N = np.asarray(stl_mesh.face_normals)
    pix, pts, nrm = [], [], []
    owner = np.full(size * size, -1, np.int64)
    for f in range(len(F)):
        p = np.c_[uv[f, :, 0] * size, (1 - uv[f, :, 1]) * size]  # x right, y down
        x0, y0 = np.floor(p.min(0)).astype(int)
        x1, y1 = np.ceil(p.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        q = np.c_[xs.ravel(), ys.ravel()]
        T = np.array([p[1] - p[0], p[2] - p[0]]).T
        det = np.linalg.det(T)
        if abs(det) < 1e-12:
            continue
        l12 = np.linalg.solve(T, (q - p[0]).T).T
        bary = np.c_[1 - l12.sum(1), l12]
        inside = (bary >= -1e-6).all(1)
        if not inside.any():
            continue
        q, bary = q[inside], bary[inside]
        idx = (q[:, 1] - 0.5).astype(np.int64) * size + (q[:, 0] - 0.5).astype(np.int64)
        free = owner[idx] < 0
        owner[idx[free]] = f
        pix.append(idx[free])
        pts.append(bary[free] @ V[F[f]])
        nrm.append(np.repeat(N[f][None], free.sum(), 0))
    return np.concatenate(pix), np.concatenate(pts), np.concatenate(nrm)


def holder_mesh(raw_mesh: Path, stl_mesh: trimesh.Trimesh, T: np.ndarray, tol_mm: float):
    """Parts of the scan mesh that are not the prop (stick, putty): occluders, scan frame."""
    m = trimesh.load(raw_mesh, process=False)
    samples, _ = trimesh.sample.sample_surface_even(stl_mesh, 500_000, seed=2)
    d, _ = cKDTree(samples).query(apply(T, m.triangles_center))
    keep = d > tol_mm
    return np.asarray(m.vertices), np.asarray(m.faces)[keep]


def build_scene(stl_mesh, P, T, raw_mesh, cfg, lattice=None, edges=None):
    """Ray-casting scene (scan frame) of the surface the photos show + holder occluders.
    edges: (radius, dA, dB, fade) per sharp convex edge segment -> the STL's edges are replaced
    by the measured ones first (edges.corrected_mesh).
    Returns (scene, P moved onto the real surface, mm per scan unit)."""
    # surface the photos actually show: the STL deformed onto the print (fine remesh so the
    # deformation is not flattened by big CAD triangles)
    if edges is not None:
        from .edges import corrected_mesh
        radius, dA, dB, fade = edges
        real = corrected_mesh(stl_mesh, cfg["mesh"]["stl_max_edge_mm"], radius, dA, dB, fade,
                              cfg["bake"]["edge_min_angle_deg"])
        rv, rf = real.vertices, real.faces
    else:
        rv, rf = trimesh.remesh.subdivide_to_size(stl_mesh.vertices, stl_mesh.faces,
                                                  max_edge=cfg["mesh"]["stl_max_edge_mm"])
    if lattice is not None:
        real = trimesh.Trimesh(lattice(rv), rf, process=False)
        P = lattice(P)
    else:
        real = trimesh.Trimesh(rv, rf, process=False)

    # ray-casting scene in the scan frame: real surface + holder occluders
    S = np.linalg.inv(T)
    k = scale_of(T)  # mm per scan unit
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(apply(S, np.asarray(real.vertices)).astype(np.float32)),
                        o3d.core.Tensor(np.asarray(real.faces).astype(np.uint32)))
    if raw_mesh is not None and raw_mesh.exists():
        hv, hf = holder_mesh(raw_mesh, real, T, cfg["mesh"]["crop_tolerance_mm"])
        if len(hf):
            scene.add_triangles(o3d.core.Tensor(hv.astype(np.float32)), o3d.core.Tensor(hf.astype(np.uint32)))
            log.info("  holder occluder: %d faces from the scan mesh", len(hf))

    return scene, P, k


def blend_points(X, Nrm, views, scene, k, b, trace=None, Xc=None):
    """Colour of surface points X (scan frame, normals Nrm) blended over the views, linear light.
    Xc: where to sample the colour if not at X (rounded print edges, edges.py; Nrm is then the
    normal there); visibility is always tested at X, which lies on the ray-cast surface.
    Returns (acc, wsum, nviews); colour = acc / wsum. trace: list -> per-sample records."""
    if Xc is None:
        Xc = X
    acc = np.zeros((len(X), 3), np.float64)
    wsum = np.zeros(len(X), np.float64)
    wmax = np.zeros(len(X), np.float64)
    nviews = np.zeros(len(X), np.int32)
    ds = b["depth_scale"]
    tol = b["depth_tol_mm"] / k

    for i, v in enumerate(views):
        C = v.center
        to_cam = C[None] - Xc
        dist = np.linalg.norm(to_cam, axis=1)
        cosang = np.einsum("ij,ij->i", Nrm, to_cam / dist[:, None])
        uvp, z = v.project(X)
        cand = (cosang > b["min_cos"]) & (z > 0) & (uvp[:, 0] >= 1) & (uvp[:, 1] >= 1) \
            & (uvp[:, 0] < v.w - 2) & (uvp[:, 1] < v.h - 2)
        if not cand.any():
            continue
        # depth map of STL + holder at reduced resolution
        w2, h2 = int(v.w * ds), int(v.h * ds)
        K2 = v.K.copy()
        K2[:2] *= ds
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(K2), o3d.core.Tensor(v.extrinsic()), w2, h2)
        ans = scene.cast_rays(rays)
        hit = ans["t_hit"].numpy()
        # ray length -> z depth
        dirs = rays.numpy()[..., 3:]
        # blur footprint: a photo pixel mixes everything within a few px. Near a crease the other
        # face is often seen at a grazing angle (dark), near an outline there is background: the
        # sample is only as good as the worst surface around it -> weight by the smallest cosine
        # within the window (background = 0). Views seeing both faces alike win at edges.
        cwin = int(round(b["blur_window_px"] * ds))
        if cwin > 0:
            cmap = -np.einsum("ijk,ijk->ij", ans["primitive_normals"].numpy(), dirs / np.linalg.norm(dirs, axis=2, keepdims=True))
            cmap = np.where(np.isfinite(hit), np.clip(cmap, 0, 1), 0).astype(np.float32)
            cmin = cv2.erode(cmap, np.ones((2 * cwin + 1, 2 * cwin + 1), np.uint8))
        dmap = hit * (dirs @ v.R[2])  # z = t * (d . optical axis) (R[2] = camera z axis in world)
        dmap = np.where(np.isfinite(dmap), dmap, np.inf).astype(np.float32)
        # occluding contours (outline vs background, one part in front of another): the photo
        # blends both sides there over a few px (lens blur, focus stacking), ef01: black
        # background darkened cube edges seen from behind. Samples close to a depth jump
        # get a low weight, so edges come from views where the edge is a crease, not an outline.
        far = np.where(np.isfinite(dmap), dmap, 1e6)
        k3 = np.ones((3, 3), np.uint8)
        jump = (cv2.dilate(far, k3) - cv2.erode(far, k3)) > b["contour_jump_mm"] / k
        cdist = cv2.distanceTransform((~jump).astype(np.uint8), cv2.DIST_L2, 3) / ds  # full-res px
        # conservative: nearest surface in a 3x3 neighbourhood (occlusion edges)
        dmap = -cv2.dilate(-dmap, k3)
        ci = np.nonzero(cand)[0]
        u2 = np.clip((uvp[ci, 0] * ds).astype(int), 0, w2 - 1)
        v2 = np.clip((uvp[ci, 1] * ds).astype(int), 0, h2 - 1)
        vis = z[ci] <= dmap[v2, u2] + tol
        ci, u2, v2 = ci[vis], u2[vis], v2[vis]
        cpx = cdist[v2, u2]
        cw = np.clip(cpx / b["contour_margin_px"], 1e-3, 1.0)
        if cwin > 0:  # replace the own cosine by the worst one in the blur window
            cosw = np.maximum(cmin[v2, u2], 1e-3)
            cw = cw * np.minimum(cosw / np.maximum(cosang[ci], 1e-3), 1.0) ** b["angle_power"]
        else:
            cosw = cosang[ci]
        if len(ci) == 0:
            continue
        img = cv2.imread(str(v.image), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(v.image.with_suffix("")) + ".mask.png", cv2.IMREAD_GRAYSCALE)
        # COLMAP: pixel centre of the top-left pixel at (0.5, 0.5); OpenCV: at (0, 0)
        uvc = uvp[ci] if Xc is X else v.project(Xc[ci])[0]
        mx = (uvc[:, 0] - 0.5).astype(np.float32)
        my = (uvc[:, 1] - 0.5).astype(np.float32)
        if mask is not None:
            mk = _sample(mask, mx, my, cv2.INTER_NEAREST)
            okm = mk > 0
            ci, mx, my, cw, cpx, cosw = ci[okm], mx[okm], my[okm], cw[okm], cpx[okm], cosw[okm]
        col = _sample(img, mx, my, cv2.INTER_LINEAR)
        lin = _srgb_to_linear(col[:, ::-1])
        footprint = (v.K[0, 0] / z[ci])  # px per scan unit: higher = finer
        w = (cosang[ci] ** b["angle_power"]) * (footprint ** b["resolution_power"]) * cw
        w = w.astype(np.float64)
        if trace is not None:
            trace.append({"view": v.name, "idx": ci.copy(), "w": w.copy(), "rgb": col[:, ::-1].copy(),
                          "contour_px": cpx.copy(), "cos": cosang[ci].copy(), "cos_win": cosw.copy(),
                          "footprint": footprint.copy(), "uv": np.c_[mx, my]})
        # keep numbers in range: normalise against a running per-texel max
        newmax = np.maximum(wmax[ci], w)
        rescale = np.where(wmax[ci] > 0, (wmax[ci] / newmax) ** b["sharpness_power"], 0)
        acc[ci] *= rescale[:, None]
        wsum[ci] *= rescale
        wmax[ci] = newmax
        acc_w = (w / newmax) ** b["sharpness_power"]
        acc[ci] += acc_w[:, None] * lin
        wsum[ci] += acc_w
        nviews[ci] += 1
        if (i + 1) % 25 == 0:
            log.info("  baked %d/%d views", i + 1, len(views))

    return acc, wsum, nviews


def bake(dense: Path, stl: Path, T: np.ndarray, raw_mesh: Path | None, out_dir: Path, cfg: dict,
         lattice=None) -> Path:
    """Write out_dir/model.obj (+ .mtl, texture PNG) in STL millimetres.

    lattice: deformation STL -> photographed surface (deform.py). Texels are projected from
    their deformed position, the output keeps the original STL and its UVs."""
    b = cfg["bake"]
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "model.obj"
    if obj_path.exists():
        return obj_path

    stl_mesh = trimesh.load(stl, force="mesh")
    uv, size = uv_atlas(stl_mesh, cfg)
    pix, P, Nrm = rasterize(stl_mesh, uv, size)
    log.info("  %d texels on the prop (%.1f%% of the atlas)", len(pix), 100 * len(pix) / size**2)

    P0 = P.copy()
    scene, P, k = build_scene(stl_mesh, P, T, raw_mesh, cfg, lattice)
    views = load_views(dense)
    # held-out views: not used for the texture, so the report can check registration honestly
    hs = int(b.get("holdout_stride", 0))
    holdout = [v.name for v in views[hs // 2::hs]] if hs > 0 else []
    views = [v for v in views if v.name not in holdout]
    # printed edges are not where/how the STL has them (rounded, shifted by up to ~0.1 mm):
    # measured from the photo outlines (edgefit.py); texels near sharp convex edges sample the
    # real edge (edges.py). Geometry and UVs of the output stay the STL's.
    e = cfg["edges"]
    if e["enabled"]:
        from . import edgefit
        ef = edgefit.fit(stl, lattice, T, views, scene, out_dir, cfg)
        sg = ef["segments"]
        radius, dA, dB = np.asarray(sg["radius_mm"]), np.asarray(sg["dA_mm"]), np.asarray(sg["dB_mm"])
    else:
        radius, dA, dB = b["edge_radius_mm"], None, None
    from .edges import fillet_remap
    Pc, Nrm, on_edge = fillet_remap(P0, Nrm, stl_mesh, radius, b["edge_min_angle_deg"], dA=dA, dB=dB,
                                    fade=e["fade_mm"])
    log.info("  %.1f%% of texels sample the measured edge geometry", 100 * on_edge.mean())
    if e["enabled"]:
        # ray-cast the measured edges too: visibility and outline distance where the print has them
        scene, Pc, k = build_scene(stl_mesh, Pc, T, raw_mesh, cfg, lattice, (radius, dA, dB, e["fade_mm"]))
        P = Pc
    elif lattice is not None:
        Pc = lattice(Pc)
    S = np.linalg.inv(T)
    X = apply(S, P)  # texel points for the visibility test, scan frame
    Xc = apply(S, Pc)  # where their colour is sampled
    # normals into the scan frame: rotation part of S (a similarity rotates, it does not keep normals)
    Rs = S[:3, :3] / scale_of(S)
    Nrm = Nrm @ Rs.T
    acc, wsum, nviews = blend_points(X, Nrm, views, scene, k, b, Xc=Xc)

    tex_lin = np.zeros((size * size, 3), np.float32)
    have = wsum > 0
    tex_lin[pix[have]] = (acc[have] / wsum[have, None]).astype(np.float32)
    covered = np.zeros(size * size, np.uint8)
    covered[pix[have]] = 1
    tex = _linear_to_srgb(tex_lin).reshape(size, size, 3)
    covered = covered.reshape(size, size)

    # padding: grow charts outward so bilinear filtering never reads background
    filled = covered.copy()
    img = tex.copy()
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(b["padding_px"]):
        grow = cv2.dilate(filled, k3) & (1 - filled)
        if not grow.any():
            break
        s = cv2.blur(img.astype(np.float32) * filled[..., None], (3, 3))
        n = cv2.blur(filled.astype(np.float32), (3, 3))
        gy, gx = np.nonzero(grow)
        img[gy, gx] = (s[gy, gx] / n[gy, gx, None]).astype(np.uint8)
        filled |= grow
    tex_name = "model_albedo.png"
    cv2.imwrite(str(out_dir / tex_name), img[..., ::-1])

    # OBJ in STL millimetres, per-corner UVs
    V, F = np.asarray(stl_mesh.vertices), np.asarray(stl_mesh.faces)
    with open(out_dir / "model.mtl", "w") as fh:
        fh.write(f"newmtl albedo\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {tex_name}\n")
    with open(obj_path, "w") as fh:
        fh.write("# pgz bake; units: mm (STL frame)\nmtllib model.mtl\n")
        fh.write("".join(f"v {a:.6f} {b_:.6f} {c:.6f}\n" for a, b_, c in V))
        fh.write("".join(f"vt {a:.7f} {b_:.7f}\n" for a, b_ in uv.reshape(-1, 2)))
        fh.write("usemtl albedo\n")
        fh.write("".join(f"f {a+1}/{3*i+1} {b_+1}/{3*i+2} {c+1}/{3*i+3}\n" for i, (a, b_, c) in enumerate(F)))

    stats = {
        "atlas": size, "texels": int(len(pix)), "covered_frac": float(have.mean()),
        "views_per_texel_median": float(np.median(nviews[have])) if have.any() else 0.0,
        "holdout_views": holdout,
    }
    (out_dir / "bake.json").write_text(json.dumps(stats, indent=2))
    log.info("  baked: %.1f%% of texels covered, median %d views per texel",
             100 * stats["covered_frac"], stats["views_per_texel_median"])
    return obj_path
