"""Configuration: built-in defaults, overridden by a TOML file."""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path

DEFAULTS: dict = {
    "tools": {
        # Directories holding the executables. Empty = search PATH.
        "colmap_dir": "",
        "openmvs_dir": "",
        "use_gpu": False,
        "threads": 0,  # 0 = all cores
    },
    "mask": {
        # Background is black on the OpenScan; foreground = max(R,G,B) above this.
        "threshold": 15,
        "close_px": 15,  # closes small gaps in dark paint
        "dilate_px": 20,  # safety margin so object edges are never masked away
        "min_area_frac": 0.002,  # drop specks smaller than this share of the image
    },
    "sfm": {
        "camera_model": "OPENCV",
        "max_image_size": 3200,  # feature extraction only; dense/texture use full res
        "max_num_features": 16000,
        "clahe_clip": 4.0,  # contrast boost of the feature-only images
        # Plain base coat gives few verified matches per pair (~40); COLMAP's default
        # 30-inlier registration threshold then rejects almost every image.
        "mapper": {
            "abs_pose_min_num_inliers": 12,
            "abs_pose_min_inlier_ratio": 0.1,
            "init_min_num_inliers": 40,
            "init_min_tri_angle": 4,
            "min_model_size": 5,
            "ba_refine_principal_point": 0,
        },
        "mapper_type": "global",  # "global" (robust here) or "incremental"
        "random_seed": 1,  # fixed: reproducible mapper runs
        "neighbors": 30,  # matching pairs from the scanner's pose prior
        "max_angle_deg": 40.0,
    },
    "dense": {
        "resolution_level": 1,  # 0 = full resolution
        "max_resolution": 4200,
        "number_views": 8,
        "number_views_fuse": 3,
    },
    "mesh": {
        # "stl": texture the aligned STL (edgefit takes geometry from the STL anyway;
        #        the scan mesh has holes on plain base coat).
        # "scan": texture the photogrammetry mesh (cropped to the STL if given).
        "source": "stl",
        "stl_max_edge_mm": 0.3,  # STL is remeshed to this edge length before texturing
        "smooth": 0,  # OpenMVS default is 2: rounds off bevels -> off
        "refine": False,  # RefineMesh: better edges, slow on CPU
        "refine_resolution_level": 1,
        "crop_tolerance_mm": 0.6,  # faces farther than this from the STL are holder/clutter
    },
    "segment": {
        # Colour of the prop's base coat in OpenCV HSV (H 0..180). Used to find the
        # prop among holder/putty/label for the first STL alignment.
        "hue_ranges": [[0, 12], [165, 180]],
        "min_saturation": 90,
        "min_value": 25,
    },
    "silhouette": {
        "enabled": True,
        "threshold": 35,  # outline = max(R,G,B) above this (background <= 8)
        "ring_px": 16,  # full-res px; neighbourhood checked for holder / dark prop parts
        "max_px": 40,  # full-res px; farther outline matches are outliers
        "view_stride": 3,  # every n-th view for the refinement
        "refine_scale": 0.25,
        "measure_scale": 0.5,
        # 7-DOF refinement on outline distances. On ef01 the ICP alignment was already at
        # the measurement floor (~1 px = 15 µm) and refining did not improve it -> 0.
        "iterations": 0,
    },
    # non-rigid fit of the STL to the photographed surface before projecting (see deform.py).
    # Prints deviate from their STL; projecting onto the undeformed STL ghosts and darkens edges.
    "deform": {
        "enabled": True,
        # coarse to fine: (lattice spacing mm, capture radius mm) per stage
        "stages": [[4.0, 3.0], [2.0, 1.5], [1.0, 0.8]],
        "smooth": 0.1,          # Laplacian weight (per active node, data ~1); ef01: 1.0 under-fits, 0.02 adds noise
        "damping": 1e-4,
        "huber_mm": 0.3,        # robust weight beyond this residual
        "point_to_point": 0.2,
        "holder_capture_mm": 0.3,  # holder-side points closer than this are used (label, bare bottom)  # weight of the closest-point term relative to point-to-plane
        "iterations": 3,        # per stage
        "samples": 300000,
        "max_points": 400000,
    },
    "align": {
        "fine_candidates": 6,
    },
    "bake": {
        "holdout_stride": 24,   # every n-th view is left out of the texture and used to check it (0 = off)
        # own texturing of the STL (mesh.source = "stl"); OpenMVS TextureMesh otherwise
        "enabled": True,
        "texel_um": 15.0,  # ~ image sampling (15 µm/px on ef01); requirement <= 20
        "max_size": 8192,
        "max_stretch": 0.05,
        "padding_px": 8,
        "min_cos": 0.25,  # ignore views flatter than ~75°
        "angle_power": 2.0,
        "resolution_power": 1.0,
        # weights are raised to this power: high = best view dominates (sharp),
        # low = more averaging (smoother, blurs if registration is off)
        "sharpness_power": 8.0,
        "depth_scale": 0.5,  # occlusion depth maps at half resolution
        "depth_tol_mm": 0.05,
        # samples within this many px of an occluding contour get a lower weight (linear ramp);
        # contour = depth jump larger than contour_jump_mm between neighbouring depth-map pixels
        "contour_margin_px": 8,
        "contour_jump_mm": 0.3,
        # printed convex edges are rounded, CAD edges are sharp: texels within the fillet strip
        # sample the photos on a fillet of this radius (edges.py). ef01: dark line on every edge
        # with sharp sampling; profiles flat from ~0.2 mm, 0.25 chosen. 0 = off
        "edge_radius_mm": 0.25,
        "edge_min_angle_deg": 30.0,
    },
    "texture": {
        "resolution_level": 0,  # always full-res images for texture
        "sharpness_weight": 0.0,  # OpenMVS default 0.5 = sharpening -> forbidden
        "global_seam_leveling": False,  # shifts colours per patch -> off
        # Local seam leveling blends colours at patch borders. On the near single-channel
        # red base coat it produced black holes and green/blue blobs (31 % of the
        # surface untextured with it, 0 % without) -> off.
        "local_seam_leveling": False,
        "cost_smoothness_ratio": 0.01,  # lower = more compact patches
        "virtual_face_images": 3,  # groups coplanar faces into one view choice
        "outlier_threshold": 0.0,  # 0.06 rejected views and left faces untextured
        "max_texture_size": 8192,
        "empty_color": 0,
        # faces kept for texturing (quadric decimation keeps edges); 0 = no limit.
        # 600k ~ 0.08 mm edge length on a 30 mm prop and fits in 15 GB RAM.
        "max_faces": 600_000,
    },
    "report": {
        "texel_target_um": 20.0,
        "texel_min_um": 30.0,
        "rms_target_mm": 0.05,
        "rms_reject_mm": 0.3,
        "coverage_target": 0.90,
        "coverage_reject": 0.80,
        "coverage_tol_mm": 0.15,
        # STL faces whose normal points along this axis are the holder side and are
        # excluded from coverage (OpenScan holder sits on the bottom).
        "holder_normal": [0.0, 0.0, -1.0],
        "density_tolerance": 0.20,
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "rb") as fh:
            cfg = _merge(cfg, tomllib.load(fh))
    return cfg
