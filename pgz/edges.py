"""Rounded print edges vs sharp CAD edges.

A printed convex edge is never sharp: resin and FDM round it off (ef01: ~0.1-0.2 mm). The STL
keeps a sharp corner, so texels within ~r of it lie where the print has no material. Projected
into the photos they land on the rounded edge seen at a grazing angle, or past it on the other
face, depending on the view: the texture gets a dark line along every edge.

Here each such texel gets the point and normal on a fillet of radius r instead (the strip of
width r*tan(a/2) on each face maps onto its half of the arc). Colours are sampled there; the
texel keeps its place on the STL, so the output geometry and UVs do not change.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def sharp_convex_edges(mesh: trimesh.Trimesh, min_angle_deg: float = 30.0):
    """Segments (a, b) of convex edges whose faces turn by more than min_angle_deg, with the
    two face normals."""
    ang = mesh.face_adjacency_angles
    sel = (ang > np.radians(min_angle_deg)) & mesh.face_adjacency_convex
    e = mesh.face_adjacency_edges[sel]
    f = mesh.face_adjacency[sel]
    return mesh.vertices[e[:, 0]], mesh.vertices[e[:, 1]], mesh.face_normals[f[:, 0]], mesh.face_normals[f[:, 1]]


def fillet_remap(P: np.ndarray, N: np.ndarray, mesh: trimesh.Trimesh, radius: float,
                 min_angle_deg: float = 30.0, spacing: float = 0.05, k: int = 32):
    """Points P (on the STL, normals N) near sharp convex edges -> points/normals on a fillet of
    the given radius. Returns (P2, N2, moved mask)."""
    P2, N2 = P.copy(), N.copy()
    moved = np.zeros(len(P), bool)
    if radius <= 0:
        return P2, N2, moved
    A, B, nA, nB = sharp_convex_edges(mesh, min_angle_deg)
    if len(A) == 0:
        return P2, N2, moved
    L = np.linalg.norm(B - A, axis=1)
    D = (B - A) / L[:, None]
    alpha = np.arccos(np.clip(np.einsum("ij,ij->i", nA, nB), -1, 1))  # turn angle between faces
    reach = radius * np.tan(alpha / 2)  # strip width on each face
    # dense samples along the edges; the k nearest give candidate segments
    cnt = np.maximum(2, np.ceil(L / spacing).astype(int) + 1)
    seg = np.repeat(np.arange(len(A)), cnt)
    t = np.concatenate([np.linspace(0, 1, c) for c in cnt])
    samples = A[seg] + t[:, None] * (B - A)[seg]
    dist, j = cKDTree(samples).query(P, k=min(k, len(samples)),
                                     distance_upper_bound=float(reach.max()) + spacing)
    dist, j = np.atleast_2d(dist), np.atleast_2d(j)
    rows = np.nonzero(np.isfinite(dist[:, 0]))[0]
    best_d = np.full(len(rows), np.inf)
    best_s = np.full(len(rows), -1)
    best_flip = np.zeros(len(rows), bool)
    Pr, Nr = P[rows], N[rows]
    for c in range(j.shape[1]):
        ok = np.isfinite(dist[rows, c])
        s = np.where(ok, seg[np.minimum(j[rows, c], len(seg) - 1)], 0)
        # the texel must lie on one of the edge's two faces
        onA = np.einsum("ij,ij->i", Nr, nA[s]) > 0.99
        onB = np.einsum("ij,ij->i", Nr, nB[s]) > 0.99
        na = np.where(onB[:, None] & ~onA[:, None], nB[s], nA[s])
        nb = np.where(onB[:, None] & ~onA[:, None], nA[s], nB[s])
        e = A[s] + np.clip(np.einsum("ij,ij->i", Pr - A[s], D[s]), 0, L[s])[:, None] * D[s]
        u = np.cross(D[s], na)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        u = np.where((np.einsum("ij,ij->i", u, nb) > 0)[:, None], -u, u)
        d = np.einsum("ij,ij->i", Pr - e, u)
        good = ok & (onA | onB) & (d > -1e-6) & (d < reach[s]) & (d < best_d)
        best_d[good], best_s[good], best_flip[good] = d[good], s[good], (onB & ~onA)[good]
    m = best_s >= 0
    r_, s, d, flip = rows[m], best_s[m], best_d[m], best_flip[m]
    na = np.where(flip[:, None], nB[s], nA[s])
    nb = np.where(flip[:, None], nA[s], nB[s])
    e = A[s] + np.clip(np.einsum("ij,ij->i", P[r_] - A[s], D[s]), 0, L[s])[:, None] * D[s]
    u = np.cross(D[s], na)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u = np.where((np.einsum("ij,ij->i", u, nb) > 0)[:, None], -u, u)
    phi = 0.5 * alpha[s] * (1 - d / reach[s])  # 0 at the tangent line, a/2 at the edge
    w = nb - np.einsum("ij,ij->i", nb, na)[:, None] * na  # na rotated towards nb
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    n = np.cos(phi)[:, None] * na + np.sin(phi)[:, None] * w
    along = P[r_] - e - d[:, None] * u  # keep the texel's position along the edge
    c = e + reach[s][:, None] * u - radius * na  # fillet centre
    P2[r_] = c + radius * n + along
    N2[r_] = n
    moved[r_] = True
    return P2, N2, moved
