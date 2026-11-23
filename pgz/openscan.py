"""Reading an OpenScan (openscan3 firmware) scan folder."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class View:
    step: int  # original_step = number in the stacked file name
    image: Path
    direction: np.ndarray  # unit vector from object to camera, scanner frame
    theta: float
    phi: float


@dataclass
class Scan:
    root: Path
    name: str
    settings: dict
    camera_settings: dict
    camera_name: str
    views: list[View] = field(default_factory=list)
    stack_transforms: list | None = None

    @property
    def image_size(self) -> tuple[int, int]:
        with Image.open(self.views[0].image) as im:
            return im.size


def _find(root: Path, name: str) -> Path | None:
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def load_scan(root: Path) -> Scan:
    """Accepts the unzipped project folder or the scanNN folder inside it."""
    root = Path(root)
    scan_json = _find(root, "scan.json")
    path_json = _find(root, "path.json")
    if not scan_json or not path_json:
        raise FileNotFoundError(f"scan.json / path.json not found below {root}")
    scan_dir = scan_json.parent
    stacked = scan_dir / "stacked"
    if not stacked.is_dir():
        raise FileNotFoundError(f"{stacked} missing - only focus-stacked scans are supported")

    meta = json.loads(scan_json.read_text())
    path = json.loads(path_json.read_text())
    prior = {p["original_step"]: p for p in path["points"]}

    views = []
    for img in sorted(stacked.glob("*.jp*g")):
        m = re.search(r"_(\d+)\.jpe?g$", img.name, re.I)
        if not m:
            continue
        step = int(m.group(1))
        p = prior.get(step)
        if p is None:
            raise ValueError(f"{img.name}: step {step} not in path.json")
        c = p["cartesian"]
        d = np.array([c["x"], c["y"], c["z"]], float)
        views.append(View(step, img, d / np.linalg.norm(d), p["polar"]["theta"], p["polar"]["fi"]))
    if not views:
        raise FileNotFoundError(f"no stacked images in {stacked}")

    calib = _find(stacked, "calibration_*.json")
    transforms = json.loads(calib.read_text()).get("transforms") if calib else None

    return Scan(
        root=scan_dir,
        name=f"{meta.get('project_name', 'scan')}_{scan_dir.name}",
        settings=meta.get("settings", {}),
        camera_settings=meta.get("camera_settings", {}),
        camera_name=meta.get("camera_name", "unknown"),
        views=views,
        stack_transforms=transforms,
    )


def match_pairs(scan: Scan, neighbors: int, max_angle_deg: float) -> list[tuple[str, str]]:
    """Image pairs from the scanner's pose prior: each view with its angular neighbours.

    Saves most of the O(n^2) exhaustive matching and avoids false matches between
    views from opposite sides, which look alike on symmetric props.
    """
    dirs = np.stack([v.direction for v in scan.views])
    ang = np.degrees(np.arccos(np.clip(dirs @ dirs.T, -1, 1)))
    pairs = set()
    for i in range(len(scan.views)):
        order = np.argsort(ang[i])
        for j in order[1 : neighbors + 1]:
            if ang[i, j] <= max_angle_deg:
                pairs.add((min(i, j), max(i, j)))
    names = [v.image.name for v in scan.views]
    return [(names[i], names[j]) for i, j in sorted(pairs)]


def jpeg_info(path: Path) -> dict:
    """Chroma subsampling of a JPEG: 4:4:4 keeps full colour resolution, 4:2:0 halves it."""
    from PIL import JpegImagePlugin

    with Image.open(path) as im:
        if im.format != "JPEG":
            return {"format": im.format}
        s = JpegImagePlugin.get_sampling(im)
    return {"format": "JPEG", "subsampling": {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(s, "unknown")}
