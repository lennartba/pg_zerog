"""Structure from motion with COLMAP, then lossless undistortion for OpenMVS."""
from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import masks
from .openscan import Scan, match_pairs
from .tools import find_tool, run

log = logging.getLogger("pgz")


def feature_image(rgb: np.ndarray, cfg: dict) -> np.ndarray:
    """Contrast-enhanced grey image used ONLY for SIFT.

    The prop is nearly single-channel (red base coat: G, B ~ 3/255). Standard grey
    conversion makes it a dark, flat area with ~150 features per image. max(R,G,B)
    + CLAHE gives ~10x more. Dense matching and texture use the untouched originals.
    """
    g = rgb.max(axis=2)
    clahe = cv2.createCLAHE(clipLimit=cfg["sfm"]["clahe_clip"], tileGridSize=(16, 16))
    return clahe.apply(g)


def prepare_images(scan: Scan, work: Path, cfg: dict) -> None:
    """Link source images into work/images, write feature images and COLMAP masks."""
    img_dir, mask_dir, feat_dir = work / "images", work / "masks", work / "images_feat"
    for d in (img_dir, mask_dir, feat_dir):
        d.mkdir(parents=True, exist_ok=True)

    def one(view):
        dst = img_dir / view.image.name
        if not dst.exists():
            try:
                dst.symlink_to(view.image.resolve())
            except OSError:  # Windows without symlink rights
                shutil.copy2(view.image, dst)
        mpath = mask_dir / (view.image.name + ".png")
        fpath = feat_dir / view.image.name
        if not mpath.exists() or not fpath.exists():
            rgb = masks.read_rgb(view.image)
            cv2.imwrite(str(mpath), masks.foreground_mask(rgb, cfg["mask"]))
            cv2.imwrite(str(fpath), feature_image(rgb, cfg), [cv2.IMWRITE_JPEG_QUALITY, 97])

    with ThreadPoolExecutor() as ex:
        list(ex.map(one, scan.views))


def run_colmap(scan: Scan, work: Path, cfg: dict) -> Path:
    """Features, pose-prior pair matching, incremental mapping. Returns the model dir."""
    t, s = cfg["tools"], cfg["sfm"]
    colmap = find_tool("colmap", t["colmap_dir"])
    logs = work / "logs"
    db = work / "colmap.db"
    sparse = work / "sparse"
    gpu = "1" if t["use_gpu"] else "0"
    threads = str(t["threads"] or -1)

    if not db.exists():
        run([colmap, "feature_extractor",
             "--database_path", db, "--image_path", work / "images_feat",
             "--ImageReader.mask_path", work / "masks",
             "--ImageReader.camera_model", s["camera_model"],
             "--ImageReader.single_camera", "1",
             "--FeatureExtraction.use_gpu", gpu,
             "--FeatureExtraction.num_threads", threads,
             "--FeatureExtraction.max_image_size", str(s["max_image_size"]),
             "--SiftExtraction.max_num_features", str(s["max_num_features"]),
             ], logs / "colmap_features.log")

        pairs = match_pairs(scan, s["neighbors"], s["max_angle_deg"])
        pairs_file = work / "pairs.txt"
        pairs_file.write_text("".join(f"{a} {b}\n" for a, b in pairs))
        log.info("  %d image pairs from pose prior (exhaustive would be %d)",
                 len(pairs), len(scan.views) * (len(scan.views) - 1) // 2)
        run([colmap, "matches_importer",
             "--database_path", db, "--match_list_path", pairs_file,
             "--match_type", "pairs",
             "--FeatureMatching.use_gpu", gpu,
             "--FeatureMatching.num_threads", threads,
             ], logs / "colmap_matching.log")

    if not sparse.exists() or not any(sparse.iterdir()):
        sparse.mkdir(exist_ok=True)
        run([colmap, "mapper",
             "--database_path", db, "--image_path", work / "images_feat",
             "--output_path", sparse,
             "--Mapper.num_threads", threads,
             "--Mapper.ba_refine_principal_point", "1",
             ], logs / "colmap_mapper.log")

    # choose the model with the most registered images, export as text
    best, best_n = None, -1
    for d in sorted(p for p in sparse.iterdir() if p.is_dir()):
        txt = work / "sparse_txt" / d.name
        if not (txt / "images.txt").exists():
            txt.mkdir(parents=True, exist_ok=True)
            run([colmap, "model_converter", "--input_path", d, "--output_path", txt,
                 "--output_type", "TXT"], logs / "colmap_convert.log")
        n = len(read_images_txt(txt / "images.txt"))
        if n > best_n:
            best, best_n = txt, n
    if best is None:
        raise RuntimeError("COLMAP produced no model")
    log.info("  model %s: %d of %d images registered", best.name, best_n, len(scan.views))
    if best_n < len(scan.views):
        log.warning("  %d images NOT registered", len(scan.views) - best_n)
    return best


# --- COLMAP text model I/O -------------------------------------------------

def read_cameras_txt(path: Path) -> dict:
    cams = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        cams[int(p[0])] = {"model": p[1], "w": int(p[2]), "h": int(p[3]),
                           "params": [float(x) for x in p[4:]]}
    return cams


def read_images_txt(path: Path) -> dict:
    """image_id -> dict(q, t, cam, name, xys, pids)."""
    lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    imgs = {}
    for i in range(0, len(lines) - 1, 2):
        p = lines[i].split()
        if not p:
            continue
        pts = lines[i + 1].split()
        xy = np.array(pts, float).reshape(-1, 3) if pts else np.zeros((0, 3))
        imgs[int(p[0])] = {
            "q": [float(x) for x in p[1:5]], "t": [float(x) for x in p[5:8]],
            "cam": int(p[8]), "name": " ".join(p[9:]),
            "xys": xy[:, :2], "pids": xy[:, 2].astype(np.int64),
        }
    return imgs


def _cv_intrinsics(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    m, p = cam["model"], cam["params"]
    if m == "PINHOLE":
        fx, fy, cx, cy = p; dist = [0, 0, 0, 0]
    elif m == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1:3]; dist = [0, 0, 0, 0]
    elif m == "SIMPLE_RADIAL":
        fx = fy = p[0]; cx, cy = p[1:3]; dist = [p[3], 0, 0, 0]
    elif m == "RADIAL":
        fx = fy = p[0]; cx, cy = p[1:3]; dist = [p[3], p[4], 0, 0]
    elif m == "OPENCV":
        fx, fy, cx, cy = p[:4]; dist = p[4:8]
    else:
        raise ValueError(f"camera model {m} not supported")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    return K, np.array(dist, float)


def undistort_for_mvs(model_txt: Path, work: Path, cfg: dict) -> Path:
    """Undistort images to lossless PNG (COLMAP's undistorter would re-encode JPEG),
    write a PINHOLE model and OpenMVS masks. Returns the dense folder."""
    dense = work / "dense"
    img_out = dense / "images"
    sp_out = dense / "sparse"
    if (sp_out / "images.txt").exists():
        return dense
    img_out.mkdir(parents=True, exist_ok=True)
    sp_out.mkdir(parents=True, exist_ok=True)

    cams = read_cameras_txt(model_txt / "cameras.txt")
    imgs = read_images_txt(model_txt / "images.txt")
    maps = {}
    with open(sp_out / "cameras.txt", "w") as fh:
        for cid, cam in cams.items():
            K, dist = _cv_intrinsics(cam)
            size = (cam["w"], cam["h"])
            maps[cid] = (K, dist, cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_32FC1))
            fh.write(f"{cid} PINHOLE {cam['w']} {cam['h']} {K[0,0]:.10g} {K[1,1]:.10g} {K[0,2]:.10g} {K[1,2]:.10g}\n")

    src_dir = work / "images"

    def one(item):
        iid, im = item
        K, dist, (mx, my) = maps[im["cam"]]
        stem = Path(im["name"]).stem
        rgb = masks.read_rgb(src_dir / im["name"])
        und = cv2.remap(rgb, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(img_out / f"{stem}.png"), cv2.cvtColor(und, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        # OpenMVS reads <image>.mask.png; 0 = ignore (passed as --ignore-mask-label 0)
        mk = masks.foreground_mask(und, cfg["mask"])
        cv2.imwrite(str(img_out / f"{stem}.mask.png"), mk)
        xy = im["xys"]
        if len(xy):
            xy = cv2.undistortPoints(xy.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        return iid, stem, xy

    with ThreadPoolExecutor() as ex:
        results = list(ex.map(one, imgs.items()))

    with open(sp_out / "images.txt", "w") as fh:
        for iid, stem, xy in results:
            im = imgs[iid]
            fh.write(f"{iid} {' '.join(f'{v:.12g}' for v in im['q'])} {' '.join(f'{v:.12g}' for v in im['t'])} "
                     f"{im['cam']} {stem}.png\n")
            fh.write(" ".join(f"{x:.4f} {y:.4f} {pid}" for (x, y), pid in zip(xy, im["pids"])) + "\n")
    shutil.copy2(model_txt / "points3D.txt", sp_out / "points3D.txt")
    return dense
