"""pgz - OpenScan photogrammetry pipeline prototype.

    python -m pgz run <openscan_folder> --out <work_dir> [--stl prop.stl] [--config cfg.toml]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .config import load_config

log = logging.getLogger("pgz")

STEPS = ["prepare", "sfm", "undistort", "dense", "mesh", "crop", "texture", "export", "report"]


def _setup_logging(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(work / "pgz.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def cmd_run(args) -> int:
    from . import align, export, mvs, report, sfm
    from .openscan import load_scan

    cfg = load_config(args.config)
    work = Path(args.out).resolve()
    _setup_logging(work)
    stl = Path(args.stl).resolve() if args.stl else None
    stop = STEPS.index(args.until) if args.until else len(STEPS) - 1

    scan = load_scan(Path(args.scan))
    log.info("pgz %s | scan %s | %d views | %s", __version__, scan.name, len(scan.views), scan.camera_name)
    (work / "config_used.json").write_text(json.dumps(cfg, indent=2))
    t0 = time.time()

    def step(name):
        i = STEPS.index(name)
        if i > stop:
            return False
        log.info("== %s", name)
        return True

    if step("prepare"):
        sfm.prepare_images(scan, work, cfg)
    if step("sfm"):
        model = sfm.run_colmap(scan, work, cfg)
    if step("undistort"):
        dense_dir = sfm.undistort_for_mvs(model, work, cfg)
    mvs_dir = work / "mvs"
    if step("dense"):
        scene = mvs.import_scene(dense_dir, mvs_dir, cfg)
        dense_scene = mvs.densify(scene, cfg)
    if step("mesh"):
        raw_mesh = mvs.mesh(dense_scene, cfg)
    if step("crop"):
        tf = None
        source = cfg["mesh"]["source"] if stl else "scan"
        if stl:
            tf = align.align_to_stl(mvs_dir / "scene_dense.ply", raw_mesh, stl, work, cfg)
            if cfg["silhouette"]["enabled"]:
                from . import silhouette
                tf = silhouette.measure_and_refine(dense_dir, stl, tf, work, cfg)
        lattice = None
        if stl and cfg["deform"]["enabled"]:
            from . import deform
            lattice = deform.fit(mvs_dir / "scene_dense.ply", stl, tf, work, cfg)
        if source == "stl":
            mesh_in = align.stl_to_scan_frame(stl, tf, mvs_dir / "stl_in_scan.ply",
                                              cfg["mesh"]["stl_max_edge_mm"])
        elif stl:
            mesh_in = align.crop_mesh(raw_mesh, stl, tf, mvs_dir / "mesh_crop.ply", cfg)
        else:
            log.warning("  no STL: holder is NOT removed and no scale is known")
            mesh_in = raw_mesh
        if cfg["mesh"]["refine"] and source == "scan":
            mesh_in = mvs.refine(dense_scene, mesh_in, cfg)
    baked = source == "stl" and cfg["bake"]["enabled"]
    if step("texture"):
        if baked:
            from . import bake
            obj = bake.bake(dense_dir, stl, tf, raw_mesh, mvs_dir / "baked", cfg, lattice)
        else:
            obj = mvs.texture(dense_scene, mesh_in, cfg, name=f"textured_{source}")
    if step("export"):
        # baked OBJ is already in STL millimetres
        out_dir = export.export(obj, np.eye(4) if baked else tf, scan, work, cfg, stl,
                                suffix="baked" if baked else source)
    if step("report"):
        report.write_report(out_dir, scan, work, cfg, stl, tf, source)
    log.info("finished in %.1f min", (time.time() - t0) / 60)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pgz", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline (finished steps are skipped)")
    r.add_argument("scan", help="OpenScan project or scan folder (unzipped)")
    r.add_argument("--out", required=True, help="work/output folder")
    r.add_argument("--stl", help="the prop's STL (mm); enables alignment, cropping, metrics")
    r.add_argument("--config", type=Path, help="TOML file overriding defaults")
    r.add_argument("--until", choices=STEPS, help="stop after this step")
    r.set_defaults(func=cmd_run)
    args = ap.parse_args(argv)
    return args.func(args)
