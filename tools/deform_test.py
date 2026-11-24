"""Robustness test of the STL deformation + texture transfer on a real scan.

The real photos stay the same; the STL is warped with a known synthetic defect, so the
print now deviates from the "nominal" STL by (real print defect) + (inverse warp).
Each case runs alignment -> deformation fit -> bake with held-out views, then reports:

* registration: texture rendered into held-out photos vs the photo (patch phase correlation)
* edge spread on the same held-out photos (texture vs photo)
* shape: fitted surface vs the fitted surface of the unwarped STL, in the scan frame
  (both should be the real print)

    python tools/deform_test.py <work dir of a finished run> <stl> <out dir> [case ...]
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pgz import align, bake, deform  # noqa: E402
from pgz.align import apply, scale_of  # noqa: E402
from pgz.config import load_config  # noqa: E402
from pgz.export import read_obj  # noqa: E402
from pgz.registration import texture_registration  # noqa: E402
from pgz.sharpness import matched_edge_spread  # noqa: E402
from pgz.views import load_views  # noqa: E402

log = logging.getLogger("pgz")

# warps in STL mm (z up, bottom at z=0)
CASES = {
    "nodeform": None,                    # control: real STL, no deformation fit
    "base": lambda v: v,                 # real STL (real print defect ~0.5 mm in z)
    "shear": lambda v: v + np.c_[0.08 * v[:, 2], 0 * v[:, 2], 0 * v[:, 2]],   # leaning, 0.64 mm at top
    "yscale": lambda v: v * [1.0, 1.03, 1.0],                              # +3 % in y (±0.27 mm)
    "zstretch": lambda v: v * [1.0, 1.0, 1.06],                            # print now ~1 mm short at top
    "bump": lambda v: v + np.c_[0 * v[:, 0], 0 * v[:, 0],                  # local 0.4 mm bump on the plate
                                0.4 * np.exp(-((v[:, 0] - 5) ** 2 + (v[:, 1] + 5) ** 2) / 8.0)],
}


def run_case(name, work, stl, out, cfg, base_surface=None):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    warp = CASES[name]
    m = trimesh.load(stl, force="mesh")
    if warp is not None:
        m = trimesh.Trimesh(warp(np.asarray(m.vertices, float)), m.faces, process=False)
    case_stl = d / "nominal.stl"
    m.export(case_stl)
    c = json.loads(json.dumps(cfg))
    c["deform"]["enabled"] = warp is not None
    t0 = time.time()
    T = align.align_to_stl(work / "mvs/scene_dense.ply", work / "mvs/mesh_raw.ply", case_stl, d, c)
    lat = deform.fit(work / "mvs/scene_dense.ply", case_stl, T, d, c) if c["deform"]["enabled"] else None
    obj_path = bake.bake(work / "dense", case_stl, T, work / "mvs/mesh_raw.ply", d / "baked", c, lat)
    held = json.loads((d / "baked/bake.json").read_text())["holdout_views"]
    views = [v for v in load_views(work / "dense") if v.name in held]
    obj = read_obj(obj_path)
    tex = cv2.imread(str(next(iter(obj.textures.values()))))[:, :, ::-1]
    if lat is not None:
        obj.v = lat(obj.v)
    S, k = np.linalg.inv(T), scale_of(T)
    reg = texture_registration(obj, tex, S, views, k, holder_normal=c["report"]["holder_normal"])
    es = matched_edge_spread(obj, tex, S, views, k, n_views=len(views))
    res = {"case": name, "minutes": round((time.time() - t0) / 60, 1),
           "registration_median_um": reg.get("median_um"), "registration_p90_um": reg.get("p90_um"),
           "registration_patches": reg.get("patches"),
           "edge_texture_um": es.get("texture_um"), "edge_photo_um": es.get("photo_um")}
    if lat is not None:
        res.update({k_: v for k_, v in json.loads((d / "deform.json").read_text()).items()
                    if k_ in ("deviation_rms_mm", "deviation_max_mm", "residual_after_median_um")})
    # fitted real surface in the scan frame (for comparison between cases)
    smp, _ = trimesh.sample.sample_surface_even(trimesh.load(stl, force="mesh"), 100000, seed=5)
    smp = np.asarray(smp)
    w = smp if warp is None else warp(smp)
    surf = apply(S, lat(w) if lat is not None else w)
    np.save(d / "surface_scan.npy", surf)
    if base_surface is not None:
        dd, _ = cKDTree(base_surface[0]).query(surf)
        dd *= base_surface[1] * 1000
        res["shape_vs_base_median_um"] = float(np.median(dd))
        res["shape_vs_base_p95_um"] = float(np.quantile(dd, 0.95))
    (d / "result.json").write_text(json.dumps(res, indent=2))
    log.info("RESULT %s", json.dumps(res))
    return res, surf, k


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    work, stl, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    names = sys.argv[4:] or list(CASES)
    cfg = load_config(Path(__file__).resolve().parents[1] / "config/container.toml")
    cfg["bake"]["texel_um"] = 20  # faster; same behaviour
    base = None
    if "base" in names:
        names = ["base"] + [n for n in names if n != "base"]
    rows = []
    for n in names:
        r, surf, k = run_case(n, work, stl, out, cfg, base)
        if n == "base":
            base = (surf, k)
        rows.append(r)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
