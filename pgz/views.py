"""Undistorted (PINHOLE) cameras of the dense folder, for projection and ray casting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sfm import read_cameras_txt, read_images_txt


@dataclass
class View:
    name: str
    image: Path
    K: np.ndarray  # 3x3, full resolution
    R: np.ndarray  # world -> camera
    t: np.ndarray
    w: int
    h: int

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, X: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coords (at image scale `scale`) and depth of world points."""
        c = X @ self.R.T + self.t
        z = c[:, 2]
        uv = (c[:, :2] / z[:, None]) * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]]
        return uv * scale, z

    def extrinsic(self) -> np.ndarray:
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = self.R, self.t
        return E


def qvec_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_views(dense: Path) -> list[View]:
    cams = read_cameras_txt(dense / "sparse" / "cameras.txt")
    imgs = read_images_txt(dense / "sparse" / "images.txt")
    out = []
    for im in imgs.values():
        c = cams[im["cam"]]
        if c["model"] != "PINHOLE":
            raise ValueError("expected undistorted PINHOLE cameras")
        fx, fy, cx, cy = c["params"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        out.append(View(im["name"], dense / "images" / im["name"], K, qvec_to_R(im["q"]),
                        np.array(im["t"], float), c["w"], c["h"]))
    return sorted(out, key=lambda v: v.name)
