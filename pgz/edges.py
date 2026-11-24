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


def fillet_remap(P: np.ndarray, N: np.ndarray, mesh: trimesh.Trimesh, radius,
                 min_angle_deg: float = 30.0, spacing: float = 0.05, k: int = 32,
                 dA=None, dB=None, fade: float = 0.5):
    """Points P (on the STL, normals N) near sharp convex edges -> points/normals on the real
    edge: corner moved by D = dA nA + dB nB (per segment, measured: edgefit.py), rounded with
    radius r (scalar or per segment). The strip of face a up to the real tangent line maps onto
    face a's half of the arc; beyond it the face offset fades out over `fade` mm.
    Returns (P2, N2, moved mask)."""
    P2, N2 = P.copy(), N.copy()
    moved = np.zeros(len(P), bool)
    A, B, nA, nB = sharp_convex_edges(mesh, min_angle_deg)
    if len(A) == 0:
        return P2, N2, moved
    ns_ = len(A)
    r = np.broadcast_to(np.asarray(radius, float), (ns_,)).copy()
    dA = np.zeros(ns_) if dA is None else np.asarray(dA, float)
    dB = np.zeros(ns_) if dB is None else np.asarray(dB, float)
    Dv = dA[:, None] * nA + dB[:, None] * nB
    if not (r > 0).any() and not np.abs(Dv).any():
        return P2, N2, moved
    L = np.linalg.norm(B - A, axis=1)
    Dd = (B - A) / L[:, None]
    alpha = np.arccos(np.clip(np.einsum("ij,ij->i", nA, nB), -1, 1))  # turn angle between faces
    reach = r * np.tan(alpha / 2)
    has_off = np.linalg.norm(Dv, axis=1) > 0
    zone = reach + np.linalg.norm(Dv, axis=1) + np.where(has_off, fade, 0.0)  # max influence
    cnt = np.maximum(2, np.ceil(L / spacing).astype(int) + 1)
    seg = np.repeat(np.arange(ns_), cnt)
    t = np.concatenate([np.linspace(0, 1, c) for c in cnt])
    samples = A[seg] + t[:, None] * (B - A)[seg]
    dist, j = cKDTree(samples).query(P, k=min(k, len(samples)), distance_upper_bound=float(zone.max()) + spacing)
    dist, j = np.atleast_2d(dist), np.atleast_2d(j)
    rows = np.nonzero(np.isfinite(dist[:, 0]))[0]
    Pr, Nr = P[rows], N[rows]

    def frame(s, Pp, flip):
        na = np.where(flip[:, None], nB[s], nA[s])
        nb = np.where(flip[:, None], nA[s], nB[s])
        e = A[s] + np.clip(np.einsum("ij,ij->i", Pp - A[s], Dd[s]), 0, L[s])[:, None] * Dd[s]
        u = np.cross(Dd[s], na)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        u = np.where((np.einsum("ij,ij->i", u, nb) > 0)[:, None], -u, u)
        return na, nb, e, u

    best_d = np.full(len(rows), np.inf)
    best_s = np.full(len(rows), -1)
    best_flip = np.zeros(len(rows), bool)
    for c in range(j.shape[1]):
        ok = np.isfinite(dist[rows, c])
        s = np.where(ok, seg[np.minimum(j[rows, c], len(seg) - 1)], 0)
        onA = np.einsum("ij,ij->i", Nr, nA[s]) > 0.99
        onB = np.einsum("ij,ij->i", Nr, nB[s]) > 0.99
        flip = onB & ~onA
        _, _, e, u = frame(s, Pr, flip)
        d = np.einsum("ij,ij->i", Pr - e, u)
        good = ok & (onA | onB) & (d > -1e-6) & (d < zone[s]) & (d < best_d)
        best_d[good], best_s[good], best_flip[good] = d[good], s[good], flip[good]
    m = best_s >= 0
    r_, s, d, flip = rows[m], best_s[m], best_d[m], best_flip[m]
    na, nb, e, u = frame(s, P[r_], flip)
    D = Dv[s]
    Ws = np.maximum(np.einsum("ij,ij->i", D, u) + reach[s], 1e-4)  # real tangent line, STL coords
    along = P[r_] - e - d[:, None] * u
    arc = d < Ws
    # on the arc
    phi = 0.5 * alpha[s] * (1 - np.clip(d / Ws, 0, 1))
    w = nb - np.einsum("ij,ij->i", nb, na)[:, None] * na
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    n = np.cos(phi)[:, None] * na + np.sin(phi)[:, None] * w
    c = e + D + reach[s][:, None] * u - r[s][:, None] * na
    p_arc = c + r[s][:, None] * n + along
    # on the face: the face's own offset, fading out
    tau = np.clip(1 - (d - Ws) / fade, 0, 1)
    p_face = P[r_] + (np.einsum("ij,ij->i", D, na) * tau)[:, None] * na
    P2[r_] = np.where(arc[:, None], p_arc, p_face)
    N2[r_] = np.where(arc[:, None], n, na)
    moved[r_] = arc | (tau > 0) & has_off[s]
    return P2, N2, moved


def corrected_mesh(mesh: trimesh.Trimesh, coarse_edge: float, radius, dA=None, dB=None, fade: float = 0.5,
                   min_angle_deg: float = 30.0, fine_edge: float = 0.05) -> trimesh.Trimesh:
    """The STL with its sharp convex edges replaced by the measured ones (same mapping as
    fillet_remap), for ray casting: visibility and outline detection must see the edge where
    the print has it. Faces near the edges are subdivided finely, the rest coarsely; per-face
    (unwelded) vertices carry their face normal, so both faces map the corner to the same arc."""
    A, B, nA, nB = sharp_convex_edges(mesh, min_angle_deg)
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=coarse_edge)
    coarse = trimesh.Trimesh(v, f, process=False)
    if len(A) == 0:
        return coarse
    r = np.broadcast_to(np.asarray(radius, float), (len(A),))
    Dn = np.zeros(len(A)) if dA is None else np.hypot(np.asarray(dA), np.asarray(dB))
    zone = float((r * 1.0 + Dn).max() + fade + coarse_edge)
    L = np.linalg.norm(B - A, axis=1)
    cnt = np.maximum(2, np.ceil(L / 0.1).astype(int) + 1)
    seg = np.repeat(np.arange(len(A)), cnt)
    t = np.concatenate([np.linspace(0, 1, c) for c in cnt])
    samples = A[seg] + t[:, None] * (B - A)[seg]
    tri = coarse.triangles
    near = np.isfinite(cKDTree(samples).query(tri.reshape(-1, 3), distance_upper_bound=zone)[0]).reshape(-1, 3).any(1)
    near |= np.isfinite(cKDTree(samples).query(tri.mean(1), distance_upper_bound=zone)[0])
    fv, ff = trimesh.remesh.subdivide_to_size(coarse.vertices, coarse.faces[near], max_edge=fine_edge)
    # unweld the fine part: every corner gets its face's normal
    fn = trimesh.Trimesh(fv, ff, process=False).face_normals
    Pv = fv[ff].reshape(-1, 3)
    Nv = np.repeat(fn, 3, 0)
    P2, _, _ = fillet_remap(Pv, Nv, mesh, radius, min_angle_deg, dA=dA, dB=dB, fade=fade)
    keep = coarse.faces[~near]
    verts = np.r_[coarse.vertices, P2]
    faces = np.r_[keep, len(coarse.vertices) + np.arange(len(P2)).reshape(-1, 3)]
    return trimesh.Trimesh(verts, faces, process=False)
