"""Dense reconstruction, meshing and texturing with OpenMVS."""
from __future__ import annotations

from pathlib import Path

from .tools import find_tool, run


def _tool(cfg, name):
    return find_tool(name, cfg["tools"]["openmvs_dir"])


def _common(cfg, mvs: Path):
    return ["-w", mvs, "--max-threads", str(cfg["tools"]["threads"])]


def import_scene(dense: Path, mvs: Path, cfg: dict) -> Path:
    scene = mvs / "scene.mvs"
    if not scene.exists():
        mvs.mkdir(parents=True, exist_ok=True)
        run([_tool(cfg, "InterfaceCOLMAP"), "-i", dense.resolve(), "-o", scene.resolve(),
             "--image-folder", (dense / "images").resolve(), *_common(cfg, mvs)],
            mvs.parent / "logs" / "mvs_import.log")
    return scene


def densify(scene: Path, cfg: dict) -> Path:
    d = cfg["dense"]
    out = scene.with_name("scene_dense.mvs")
    if not out.exists():
        run([_tool(cfg, "DensifyPointCloud"), scene.resolve(), "-o", out.resolve(),
             "--resolution-level", str(d["resolution_level"]),
             "--max-resolution", str(d["max_resolution"]),
             "--number-views", str(d["number_views"]),
             "--number-views-fuse", str(d["number_views_fuse"]),
             "--ignore-mask-label", "0",
             "--remove-dmaps", "1",
             *_common(cfg, scene.parent)],
            scene.parent.parent / "logs" / "mvs_densify.log")
    return out


def mesh(dense_scene: Path, cfg: dict) -> Path:
    out = dense_scene.with_name("mesh_raw.ply")
    if not out.exists():
        run([_tool(cfg, "ReconstructMesh"), dense_scene.resolve(), "-o", out.resolve(),
             "--smooth", str(cfg["mesh"]["smooth"]),
             *_common(cfg, dense_scene.parent)],
            dense_scene.parent.parent / "logs" / "mvs_mesh.log")
    return out


def refine(dense_scene: Path, mesh_in: Path, cfg: dict) -> Path:
    out = mesh_in.with_name(mesh_in.stem + "_refined.ply")
    if not out.exists():
        run([_tool(cfg, "RefineMesh"), dense_scene.resolve(), "-m", mesh_in.resolve(),
             "-o", out.resolve(),
             "--resolution-level", str(cfg["mesh"]["refine_resolution_level"]),
             "--close-holes", "0",
             *_common(cfg, dense_scene.parent)],
            dense_scene.parent.parent / "logs" / "mvs_refine.log")
    return out


def ply_face_count(path: Path) -> int:
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("element face"):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise ValueError(f"no face element in {path}")


def texture(dense_scene: Path, mesh_in: Path, cfg: dict) -> Path:
    t = cfg["texture"]
    # TextureMesh memory grows with faces x views (2.35 M faces, 179 views > 12 GB).
    # Texture detail comes from the images, not the mesh density.
    faces = ply_face_count(mesh_in)
    decimate = min(1.0, t["max_faces"] / faces) if t["max_faces"] else 1.0
    out_dir = dense_scene.parent / "textured"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "model.ply"
    if not out.exists():
        run([_tool(cfg, "TextureMesh"), dense_scene.resolve(), "-m", mesh_in.resolve(),
             "-o", out.resolve(), "--export-type", "ply",
             "--resolution-level", str(t["resolution_level"]),
             "--min-resolution", "8192",
             "--sharpness-weight", str(t["sharpness_weight"]),
             "--global-seam-leveling", str(int(t["global_seam_leveling"])),
             "--local-seam-leveling", str(int(t["local_seam_leveling"])),
             "--cost-smoothness-ratio", str(t["cost_smoothness_ratio"]),
             "--outlier-threshold", str(t["outlier_threshold"]),
             "--max-texture-size", str(t["max_texture_size"]),
             "--empty-color", str(t["empty_color"]),
             "--close-holes", "0",
             "--decimate", f"{decimate:.4f}",
             "--ignore-mask-label", "0",
             *_common(cfg, dense_scene.parent)],
            dense_scene.parent.parent / "logs" / "mvs_texture.log")
    return out
