"""Write the deliverable folder: model.obj/.mtl in STL millimetres + lossless textures."""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__

log = logging.getLogger("pgz")


@dataclass
class Obj:
    v: np.ndarray
    vt: np.ndarray
    fv: np.ndarray  # (n,3) vertex indices
    ft: np.ndarray  # (n,3) uv indices
    fmat: np.ndarray  # (n,) material index
    materials: list[str]
    mtllib: str | None
    textures: dict[str, Path] = field(default_factory=dict)  # material -> image


def read_obj(path: Path) -> Obj:
    v, vt, fv, ft, fm, mats = [], [], [], [], [], []
    mtllib, cur = None, -1
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append(line.split()[1:4])
            elif line.startswith("vt "):
                vt.append(line.split()[1:3])
            elif line.startswith("f "):
                parts = [p.split("/") for p in line.split()[1:]]
                if len(parts) != 3:
                    raise ValueError("only triangulated OBJ supported")
                fv.append([int(p[0]) - 1 for p in parts])
                ft.append([int(p[1]) - 1 if len(p) > 1 and p[1] else -1 for p in parts])
                fm.append(cur)
            elif line.startswith("usemtl "):
                name = line.split(maxsplit=1)[1].strip()
                if name not in mats:
                    mats.append(name)
                cur = mats.index(name)
            elif line.startswith("mtllib "):
                mtllib = line.split(maxsplit=1)[1].strip()
    obj = Obj(np.array(v, float), np.array(vt, float).reshape(-1, 2), np.array(fv, int),
              np.array(ft, int), np.array(fm, int), mats, mtllib)
    if mtllib and (path.parent / mtllib).exists():
        cur = None
        for line in (path.parent / mtllib).read_text().splitlines():
            if line.startswith("newmtl "):
                cur = line.split(maxsplit=1)[1].strip()
            elif line.strip().startswith("map_Kd ") and cur:
                obj.textures[cur] = path.parent / line.split(maxsplit=1)[1].strip()
    return obj


def _transform_obj(src: Path, dst: Path, T: np.ndarray | None) -> None:
    R = None
    if T is not None:
        R = T[:3, :3] / np.cbrt(np.linalg.det(T[:3, :3]))
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        fo.write(f"# pgz {__version__}; units: {'mm (aligned to STL)' if T is not None else 'arbitrary (no STL)'}\n")
        for line in fi:
            if T is not None and line.startswith("v "):
                p = np.array(line.split()[1:4], float)
                q = T[:3, :3] @ p + T[:3, 3]
                fo.write(f"v {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}\n")
            elif R is not None and line.startswith("vn "):
                n = R @ np.array(line.split()[1:4], float)
                fo.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            else:
                fo.write(line)


def export(obj_path: Path, T, scan, work: Path, cfg: dict, stl: Path | None) -> Path:
    out = work / "deliver" / scan.name
    out.mkdir(parents=True, exist_ok=True)
    src = obj_path.parent
    _transform_obj(obj_path, out / "model.obj", T)
    obj = read_obj(obj_path)
    if obj.mtllib:
        shutil.copy2(src / obj.mtllib, out / obj.mtllib)
    for tex in obj.textures.values():
        with Image.open(tex) as im:
            fmt = im.format
        if fmt not in ("PNG", "TIFF"):
            raise RuntimeError(f"texture {tex.name} is {fmt}, not lossless")
        shutil.copy2(tex, out / tex.name)
    if obj.mtllib and (out / obj.mtllib).exists() and obj.mtllib != "model.mtl":
        # keep the requirement's name model.mtl
        (out / obj.mtllib).rename(out / "model.mtl")
        txt = (out / "model.obj").read_text().replace(f"mtllib {obj.mtllib}", "mtllib model.mtl")
        (out / "model.obj").write_text(txt)
    if stl:
        shutil.copy2(stl, out / stl.name)
    log.info("  deliverable: %s", out)
    return out
