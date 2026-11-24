"""Non-rigid fit of the STL to the photographed surface.

A print never matches its STL exactly (ef01: cube tops 0.5 mm lower, elephant foot on
the first layers). Projecting photos onto the undeformed STL then puts every surface
point at the wrong depth: each view sees it shifted by parallax, the blend ghosts, and
texels near edges pick up the gap or side face (dark corners).

So the texture is transferred like a classic high-to-low bake: the STL is deformed onto
the real surface, the photos are projected from the deformed surface, and the colours
land on the original STL's UVs.

Deformation: a smooth displacement field on a regular lattice (trilinear), fitted to the
dense cloud with point-to-plane residuals (the STL normal at the closest point),
Laplacian smoothness, robust (Huber) weights, a few re-correspondence iterations.
Holder-side points are ignored (putty).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .align import apply, read_point_ply

log = logging.getLogger("pgz")


class Lattice:
    def __init__(self, lo: np.ndarray, spacing: float, shape: tuple, U: np.ndarray | None = None):
        self.lo, self.h, self.shape = np.asarray(lo, float), float(spacing), tuple(int(s) for s in shape)
        self.U = np.zeros((int(np.prod(self.shape)), 3)) if U is None else U

    def weights(self, p: np.ndarray) -> sp.csr_matrix:
        g = (p - self.lo) / self.h
        i0 = np.clip(np.floor(g).astype(int), 0, np.array(self.shape) - 2)
        f = np.clip(g - i0, 0, 1)
        rows, cols, vals = [], [], []
        n = len(p)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * \
                        (f[:, 2] if dz else 1 - f[:, 2])
                    idx = np.ravel_multi_index((i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz), self.shape)
                    rows.append(np.arange(n)), cols.append(idx), vals.append(w)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, len(self.U)))

    def displacement(self, p: np.ndarray) -> np.ndarray:
        return self.weights(p) @ self.U

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return p + self.displacement(p)

    def laplacian(self) -> sp.csr_matrix:
        idx = np.arange(len(self.U)).reshape(self.shape)
        a, b = [], []
        for ax in range(3):
            s0 = [slice(None)] * 3
            s1 = [slice(None)] * 3
            s0[ax], s1[ax] = slice(0, -1), slice(1, None)
            a.append(idx[tuple(s0)].ravel())
            b.append(idx[tuple(s1)].ravel())
        a, b = np.concatenate(a), np.concatenate(b)
        m = len(a)
        return sp.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[a, b])),
                             shape=(m, len(self.U)))

    def save(self, path: Path) -> None:
        np.savez(path, lo=self.lo, h=self.h, shape=np.array(self.shape), U=self.U)

    @classmethod
    def load(cls, path: Path) -> "Lattice":
        z = np.load(path)
        return cls(z["lo"], float(z["h"]), tuple(z["shape"]), z["U"])


def fit(dense_ply: Path, stl: Path, T: np.ndarray, work: Path, cfg: dict) -> Lattice:
    """Fit the lattice (STL frame, mm); writes work/deform.npz and work/deform.json."""
    out = work / "deform.npz"
    if out.exists():
        return Lattice.load(out)
    d = cfg["deform"]
    m = trimesh.load(stl, force="mesh")
    hn = np.asarray(cfg["report"]["holder_normal"], float)

    xyz, _ = read_point_ply(dense_ply)
    X = apply(T, xyz)
    smp, fi = trimesh.sample.sample_surface_even(m, d["samples"], seed=3)
    smp = np.asarray(smp)
    nrm = m.face_normals[fi]
    tree = cKDTree(smp)
    dist, j = tree.query(X, distance_upper_bound=d["max_dist_mm"])
    keep = np.isfinite(dist)
    keep[keep] &= nrm[j[keep]] @ hn < 0.5  # no holder-side (putty) points
    X = X[keep]
    if len(X) > d["max_points"]:
        X = X[np.random.default_rng(0).choice(len(X), d["max_points"], replace=False)]
    log.info("  deform: %d dense points within %.1f mm of the STL", len(X), d["max_dist_mm"])

    lo = m.bounds[0] - 2 * d["lattice_mm"]
    shape = tuple(np.ceil((m.bounds[1] + 2 * d["lattice_mm"] - lo) / d["lattice_mm"]).astype(int) + 1)
    lat = Lattice(lo, d["lattice_mm"], shape)
    Ws = lat.weights(smp)  # sample -> nodes
    L = lat.laplacian()
    nn = lat.U.shape[0]
    # per-node data normalisation: ~1 per active node, so `smooth` is resolution independent
    active = max(1, int((np.asarray(Ws.sum(0)).ravel() > 0).sum()))
    wdata = active / len(X)

    def residuals(U):
        Q = smp + Ws @ U
        dq, jq = cKDTree(Q).query(X)
        r = np.einsum("ij,ij->i", X - Q[jq], nrm[jq])
        return jq, r

    r0 = None
    for it in range(d["iterations"]):
        jq, r = residuals(lat.U)
        if r0 is None:
            r0 = r
        hub = d["huber_mm"]
        w = np.where(np.abs(r) <= hub, 1.0, hub / np.maximum(np.abs(r), 1e-9)) * wdata
        # rows: n_i . (W_s[j_i] U) = n_i . (X_i - s_j)   (total displacement, not increment)
        Wj = Ws[jq]
        target = np.einsum("ij,ij->i", X - smp[jq], nrm[jq])
        A = sp.hstack([Wj.multiply(nrm[jq, c][:, None]) for c in range(3)]).tocsr()
        Dw = sp.diags(w)
        Lb = sp.block_diag([L, L, L]).tocsr()
        M = A.T @ Dw @ A + d["smooth"] * (Lb.T @ Lb) + d["damping"] * sp.eye(3 * nn)
        rhs = A.T @ (w * target)
        u = spsolve(M.tocsc(), rhs)
        lat.U = u.reshape(3, nn).T
        log.info("  deform iteration %d: robust residual median |r| %.0f µm", it + 1,
                 1000 * np.median(np.abs(residuals(lat.U)[1])))
    _, r1 = residuals(lat.U)

    disp = Ws @ lat.U
    vis = nrm @ hn < 0.5
    dn = np.einsum("ij,ij->i", disp, nrm)[vis]
    mag = np.linalg.norm(disp[vis], axis=1)
    stats = {
        "points": int(len(X)),
        "lattice_mm": d["lattice_mm"],
        "residual_before_median_um": float(1000 * np.median(np.abs(r0))),
        "residual_after_median_um": float(1000 * np.median(np.abs(r1))),
        "deviation_rms_mm": float(np.sqrt(np.mean(dn ** 2))),
        "deviation_p95_mm": float(np.quantile(np.abs(dn), 0.95)),
        "deviation_max_mm": float(np.abs(dn).max()),
        "displacement_max_mm": float(mag.max()),
    }
    (work / "deform.json").write_text(json.dumps(stats, indent=2))
    lat.save(out)
    log.info("  print vs STL (fitted deformation, excl. holder side): RMS %.3f mm, p95 %.3f, max %.3f; "
             "residual %.0f -> %.0f µm", stats["deviation_rms_mm"], stats["deviation_p95_mm"],
             stats["deviation_max_mm"], stats["residual_before_median_um"], stats["residual_after_median_um"])
    return lat
