"""Preview renders of a textured OBJ (ray-cast, no GPU/display needed)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from .export import read_obj

DIRECTIONS = {
    "top": (0, 0, 1), "front": (0, -1, 0.6), "back": (0, 1, 0.6),
    "left": (-1, 0, 0.5), "right": (1, 0, 0.5), "bottom (holder side)": (0, 0, -1),
}


def render(obj_path: Path, out: Path, px_per_mm: float = 25.0, shade: bool = True) -> Path:
    obj = read_obj(obj_path)
    tex = cv2.imread(str(next(iter(obj.textures.values()))))
    H, W = tex.shape[:2]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(obj.v.astype(np.float32)), o3d.core.Tensor(obj.fv.astype(np.uint32)))
    c = (obj.v.min(0) + obj.v.max(0)) / 2
    radius = np.linalg.norm(obj.v.max(0) - obj.v.min(0)) / 2
    size = int(2 * radius * px_per_mm)
    tri = obj.v[obj.fv]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    tiles = []
    for name, d in DIRECTIONS.items():
        d = np.asarray(d, float)
        d /= np.linalg.norm(d)
        up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([0, 1.0, 0])
        x = np.cross(up, d)
        x /= np.linalg.norm(x)
        y = np.cross(d, x)
        s = (np.arange(size) + 0.5) / px_per_mm - radius
        gx, gy = np.meshgrid(s, -s)
        origins = c + d * 3 * radius + gx[..., None] * x + gy[..., None] * y
        rays = np.concatenate([origins, np.broadcast_to(-d, origins.shape)], -1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays))
        pid = ans["primitive_ids"].numpy().astype(np.int64)
        bc = ans["primitive_uvs"].numpy()
        hit = pid != scene.INVALID_ID
        img = np.full((size, size, 3), 255, np.uint8)
        f = pid[hit]
        uvc = obj.vt[obj.ft[f]]  # (n,3,2)
        w = np.c_[1 - bc[hit].sum(1), bc[hit]]
        uv = (uvc * w[:, :, None]).sum(1)
        px = np.clip(uv[:, 0] * W - 0.5, 0, W - 1).astype(np.float32)
        py = np.clip((1 - uv[:, 1]) * H - 0.5, 0, H - 1).astype(np.float32)
        col = cv2.remap(tex, px[None], py[None], cv2.INTER_LINEAR)[0] if len(px) < 32000 else \
            np.concatenate([cv2.remap(tex, px[i:i + 30000][None], py[i:i + 30000][None], cv2.INTER_LINEAR)[0]
                            for i in range(0, len(px), 30000)])
        if shade:
            lam = np.clip(np.abs(fn[f] @ d), 0, 1)
            col = (col * (0.75 + 0.25 * lam[:, None])).astype(np.uint8)
        img[hit] = col
        cv2.putText(img, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        tiles.append(img)
    row1 = np.concatenate(tiles[:3], 1)
    row2 = np.concatenate(tiles[3:], 1)
    cv2.imwrite(str(out), np.concatenate([row1, row2], 0))
    return out
