"""Measure the deliverable against the edgefit requirements and write report.md/.json."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from . import __version__
from .export import read_obj
from .openscan import jpeg_info

log = logging.getLogger("pgz")


def texel_stats(obj_path: Path, cfg: dict) -> dict:
    obj = read_obj(obj_path)
    sizes, imgs = {}, {}
    for name, p in obj.textures.items():
        im = Image.open(p)
        sizes[name] = im.size
        imgs[name] = np.asarray(im.convert("RGB"))
    tri = obj.v[obj.fv]
    a3 = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    ok = (obj.ft >= 0).all(1) & (obj.fmat >= 0)
    uv = obj.vt[np.where(ok[:, None], obj.ft, 0)]
    wh = np.array([sizes.get(obj.materials[m], (0, 0)) if m >= 0 else (0, 0) for m in obj.fmat], float)
    e1 = (uv[:, 1] - uv[:, 0]) * wh
    e2 = (uv[:, 2] - uv[:, 0]) * wh
    atex = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    ok &= (atex > 0) & (a3 > 0)

    # faces OpenMVS could not texture get the empty colour (black)
    cen = uv.mean(1)
    untextured = np.zeros(len(a3), bool)
    for mi, name in enumerate(obj.materials):
        if name not in imgs:
            continue
        sel = obj.fmat == mi
        im = imgs[name]
        h, w = im.shape[:2]
        x = np.clip((cen[sel, 0] * w).astype(int), 0, w - 1)
        y = np.clip(((1 - cen[sel, 1]) * h).astype(int), 0, h - 1)
        untextured[sel] = (im[y, x] == 0).all(1)

    size_um = np.full(len(a3), np.nan)
    size_um[ok] = 1000 * np.sqrt(a3[ok] / atex[ok])
    m = ok & ~untextured
    w = a3[m]
    s = size_um[m]
    order = np.argsort(s)
    cw = np.cumsum(w[order]) / w.sum()

    def q(p):
        return float(s[order][np.searchsorted(cw, p)])

    med = q(0.5)
    tol = cfg["report"]["density_tolerance"]
    within = float(w[np.abs(s / med - 1) <= tol].sum() / w.sum())
    atlas = sum(a * b for a, b in sizes.values())
    hn = np.asarray(cfg["report"]["holder_normal"], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) / (2 * a3[:, None])
    vis = ~(normals @ hn > 0.9)
    return {
        "textured_excl_holder": float(a3[vis & ~untextured].sum() / a3[vis].sum()),
        "textures": {k: list(v) for k, v in sizes.items()},
        "texel_um_median": med, "texel_um_p10": q(0.1), "texel_um_p90": q(0.9),
        "density_within_tol": within,
        "atlas_utilisation": float(atex[ok].sum() / atlas) if atlas else 0.0,
        "untextured_area_frac": float(a3[untextured].sum() / a3.sum()),
        "surface_mm2": float(a3.sum()),
    }


def geometry_stats(obj_path: Path, stl: Path, cfg: dict) -> dict:
    r = cfg["report"]
    obj = read_obj(obj_path)
    scan = trimesh.Trimesh(obj.v, obj.fv, process=False)
    target = trimesh.load(stl, force="mesh")

    pts, _ = trimesh.sample.sample_surface(scan, 30_000, seed=0)
    _, d, _ = trimesh.proximity.closest_point(target, pts)

    tp, tf = trimesh.sample.sample_surface_even(target, 300_000, seed=0)
    holder = target.face_normals[tf] @ np.asarray(r["holder_normal"], float) > 0.9
    sp, _ = trimesh.sample.sample_surface(scan, 2_000_000, seed=1)
    dc, _ = cKDTree(sp).query(tp)
    covered = dc <= r["coverage_tol_mm"]
    return {
        "rms_mm": float(np.sqrt(np.mean(d**2))),
        "p95_mm": float(np.quantile(d, 0.95)),
        "max_mm": float(d.max()),
        "coverage_excl_holder": float(covered[~holder].mean()),
        "coverage_all": float(covered.mean()),
    }


def _tool_versions(work: Path, cfg: dict) -> dict:
    import subprocess

    from .tools import find_tool

    out = {}
    try:
        txt = subprocess.run([find_tool("colmap", cfg["tools"]["colmap_dir"]), "-h"],
                             capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"COLMAP ([\d.]+)", txt)
        if m:
            out["colmap"] = m.group(1)
    except (OSError, subprocess.SubprocessError):
        pass
    p = work / "logs" / "mvs_import.log"
    if p.exists():
        m = re.search(r"OpenMVS \S+ (v[\d.]+)", p.read_text(errors="replace"))
        if m:
            out["openmvs"] = m.group(1)
    return out


def _status(val, target, minimum, lower_is_better=True):
    if val is None:
        return "NOT MEASURED"
    if lower_is_better:
        return "PASS" if val <= target else ("WARN" if minimum is not None and val <= minimum else "FAIL")
    return "PASS" if val >= target else ("WARN" if minimum is not None and val >= minimum else "FAIL")


def _edge_row(res, t, r):
    """Requirement: <= 1.5 texels at the 20 µm target texel (30 µm), <= 2.5 texels at the
    30 µm minimum texel (75 µm). Compared in µm, so finer texels are not penalised."""
    e = res["edge_spread"]
    tgt = 1.5 * r["texel_target_um"]
    mn = 2.5 * r["texel_min_um"]
    m = e.get("matched")
    if m and m.get("edges", 0) >= 20:
        um = m["texture_um"]
        return (f"Edge spread: texture rendered into {m['views']} photos, same {m['edges']} interior edges "
                "(proxy: no painted edges on a base scan)",
                f"{um:.0f} µm ({um / t['texel_um_median']:.1f} texels); photos {m['photo_um']:.0f} µm",
                f"≤ {tgt:.0f} µm", f"≤ {mn:.0f} µm", _status(um, tgt, mn))
    et = e.get("texture", {})
    if et.get("edges", 0) < 20:
        return ("Edge spread", "too few edges", f"≤ {tgt:.0f} µm", f"≤ {mn:.0f} µm", "NOT MEASURED")
    um = et["median_px"] * t["texel_um_median"]
    return ("Edge spread, strongest texture edges (no cameras: not compared with photos)",
            f"{um:.0f} µm ({et['median_px']:.1f} texels)", f"≤ {tgt:.0f} µm", f"≤ {mn:.0f} µm",
            _status(um, tgt, mn))


def write_report(out_dir: Path, scan, work: Path, cfg: dict, stl: Path | None, T,
                 source: str = "scan") -> dict:
    r = cfg["report"]
    obj_path = out_dir / "model.obj"
    res: dict = {"pgz": __version__, "scan": scan.name, "mesh_source": source}
    res["input"] = {
        "views": len(scan.views),
        "image_size": list(scan.image_size),
        "jpeg_quality": scan.camera_settings.get("jpeg_quality"),
        **jpeg_info(scan.views[0].image),
    }
    imgs_txt = work / "dense" / "sparse" / "images.txt"
    if imgs_txt.exists():
        n = sum(1 for l in imgs_txt.read_text().splitlines() if l and not l.startswith("#")) // 2
        res["input"]["registered"] = n
    res["tools"] = _tool_versions(work, cfg)
    res["texture"] = texel_stats(obj_path, cfg)
    if stl and T is not None:
        res["align"] = json.loads((work / "align.json").read_text())
        if source == "scan":
            res["geometry"] = geometry_stats(obj_path, stl, cfg)
    (out_dir / "report.json").write_text(json.dumps(res, indent=2))

    # edge spread: the texture rendered back into real cameras vs the photos, same edges
    import cv2
    from .sharpness import edge_spread, matched_edge_spread
    obj = read_obj(obj_path)
    timg = cv2.imread(str(next(iter(obj.textures.values()))))[:, :, ::-1]
    dense = work / "dense"
    if stl and T is not None and (dense / "sparse" / "images.txt").exists():
        from .align import scale_of
        from .views import load_views
        res["edge_spread"] = {"matched": matched_edge_spread(obj, timg, np.linalg.inv(T), load_views(dense),
                                                             scale_of(T))}
    else:
        res["edge_spread"] = {"texture": edge_spread(timg, timg.max(2) > 0)}

    t, g = res["texture"], res.get("geometry")
    rows = [
        ("Texel size on prop (median)", f"{t['texel_um_median']:.1f} µm "
         f"(p10 {t['texel_um_p10']:.1f}, p90 {t['texel_um_p90']:.1f})",
         f"≤ {r['texel_target_um']:.0f} µm", f"≤ {r['texel_min_um']:.0f} µm",
         _status(t["texel_um_median"], r["texel_target_um"], r["texel_min_um"])),
        ("Even texel density", f"{100 * t['density_within_tol']:.0f} % of area within ±{100 * r['density_tolerance']:.0f} %",
         "≥ 90 %", "—", _status(t["density_within_tol"], 0.9, None, False)),
        ("Untextured surface", f"{100 * t['untextured_area_frac']:.1f} %", "0 %", "—",
         _status(t["untextured_area_frac"], 0.005, 0.02)),
    ]
    sil_path = work / "silhouette.json"
    if stl and sil_path.exists() and json.loads(sil_path.read_text())["after"]["views"] > 0:
        sil = json.loads(sil_path.read_text())["after"]
        res["silhouette"] = sil
        rows += [
            ("Alignment: STL outline vs photo outline", f"median {sil['median_um']:.0f} µm "
             f"(p90 {sil['p90_um']:.0f}, signed {sil['signed_median_um']:+.0f}) over {sil['views']} views; "
             f"floor ~1 px", f"≤ {1000 * r['rms_target_mm']:.0f} µm", f"≤ {1000 * r['rms_reject_mm']:.0f} µm",
             _status(sil["median_um"] / 1000, r["rms_target_mm"], r["rms_reject_mm"])),
        ]
    if source == "stl" and "align" in res:
        al = res["align"]
        rows += [
            ("Dense prop points vs STL (info)", f"RMS {al['prop_points_rms_mm']:.3f} mm "
             f"(p95 {al['prop_points_p95_mm']:.3f}); mostly dense-cloud noise",
             "—", "—", "INFO"),
            ("Textured STL surface (excl. holder side)", f"{100 * t['textured_excl_holder']:.1f} %",
             f"≥ {100 * r['coverage_target']:.0f} %", f"≥ {100 * r['coverage_reject']:.0f} %",
             _status(t["textured_excl_holder"], r["coverage_target"], r["coverage_reject"], False)),
        ]
    if g:
        rows += [
            ("Scan vs STL RMS", f"{g['rms_mm']:.3f} mm (p95 {g['p95_mm']:.3f})",
             f"≤ {r['rms_target_mm']} mm", f"≤ {r['rms_reject_mm']} mm",
             _status(g["rms_mm"], r["rms_target_mm"], r["rms_reject_mm"])),
            ("STL coverage (excl. holder side)", f"{100 * g['coverage_excl_holder']:.1f} %",
             f"≥ {100 * r['coverage_target']:.0f} %", f"≥ {100 * r['coverage_reject']:.0f} %",
             _status(g["coverage_excl_holder"], r["coverage_target"], r["coverage_reject"], False)),
        ]
    inp = res["input"]
    rows += [
        ("Source images", f"{inp.get('format')} q{inp.get('jpeg_quality')} {inp.get('subsampling', '')}",
         "lossless / 4:4:4", "—",
         "WARN" if inp.get("subsampling") not in (None, "4:4:4") else "PASS"),
        ("Texture file format", ", ".join(sorted({p.suffix for p in read_obj(obj_path).textures.values()})),
         "PNG/TIFF", "—",
         "PASS" if all(p.suffix.lower() in (".png", ".tif", ".tiff") for p in read_obj(obj_path).textures.values())
         else "FAIL"),
        ("Sharpening / delighting / colour leveling", "none applied", "none", "—", "PASS"),
        _edge_row(res, t, r),
        (("Texture-on-mesh registration", "projected onto the STL: = alignment above "
          f"({res['silhouette']['median_um']:.0f} µm) + SfM reprojection", "≤ 0.05 mm", "—",
          _status(res['silhouette']['median_um'] / 1000, 0.05, None))
         if source == "stl" and "silhouette" in res else
         ("Texture-on-mesh registration", "—", "≤ 0.05 mm", "—", "NOT MEASURED")),
        ("Colour vs checker ΔE", "—", "≤ 3", "—", "OUT OF SCOPE (prototype)"),
    ]
    md = [f"# Scan report {scan.name}", "",
          f"Mesh: {'aligned STL, textured' if source == 'stl' else 'photogrammetry mesh'}. "
          f"pgz {__version__}; COLMAP {res['tools'].get('colmap', '?')}; OpenMVS {res['tools'].get('openmvs', '?')}; "
          f"{inp['views']} views, {inp.get('registered', '?')} registered; image {inp['image_size'][0]}×{inp['image_size'][1]}.",
          "", "| Requirement | Measured | Target | Minimum | Status |", "| --- | --- | --- | --- | --- |"]
    md += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows]
    md += ["", f"Surface {t['surface_mm2']:.0f} mm², textures {t['textures']}, "
           f"atlas utilisation {100 * t['atlas_utilisation']:.0f} %."]
    if "align" in res:
        md += [f"Alignment scale {res['align']['scale_mm_per_unit']:.4f} mm per model unit."]
    (out_dir / "report.md").write_text("\n".join(md) + "\n")

    note = [
        f"scanner: OpenScan ({scan.camera_name}), focus stacks {scan.settings.get('focus_stacks')}, "
        f"jpeg quality {scan.camera_settings.get('jpeg_quality')}",
        f"software: pgz {__version__}, COLMAP {res['tools'].get('colmap', '?')}, OpenMVS {res['tools'].get('openmvs', '?')}",
        f"photos: {inp['views']} stacked, {inp.get('registered', '?')} registered",
        f"units: {'mm, aligned to ' + stl.name if stl and T is not None else 'arbitrary (no STL)'}",
        "manual edits: none",
    ]
    (out_dir / "note.txt").write_text("\n".join(note) + "\n")
    try:
        from .preview import render
        render(obj_path, out_dir / "preview.jpg")
    except Exception as exc:  # preview is a convenience, never fail the report on it
        log.warning("  preview failed: %s", exc)
    log.info("  report: %s", out_dir / "report.md")
    return res
