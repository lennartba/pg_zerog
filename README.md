# pg_zerog

Photogrammetry pipeline prototype for focus-stacked scans from an **OpenScan Mini + macro add-on**,
producing textured scans for edgefit (requirements doc: *Photogrammetry output requirements for edgefit*).

Open source only: [COLMAP](https://colmap.github.io/) (structure from motion),
[OpenMVS](https://github.com/cdcseacave/openMVS) (dense cloud, mesh),
[Open3D](https://www.open3d.org/) (UVAtlas, ray casting), orchestrated in Python.

## Approach

edgefit takes geometry from the prop's STL and only colour from the scan. So the pipeline uses
photogrammetry to find the cameras and the prop's exact position, then **projects the photos onto the
STL** and bakes the texture itself (like Metashape: a UV atlas with few large charts, blended best views).
The photogrammetry mesh is only used to find the holder (stick, putty) as an occluder.

Why not texture the scan mesh with OpenMVS: on the plain base coat the dense reconstruction has holes
(ef01: 63 % coverage), and OpenMVS TextureMesh packs every view patch separately
(ef01: ~19,000 islands, seams everywhere; its OBJ export also writes JPEG only).
The old route is still available with `mesh.source = "scan"`.

## Pipeline

| Step | Tool | What happens |
| --- | --- | --- |
| prepare | Python | Link images, foreground masks (black background), contrast-enhanced copies for SIFT only (red base coat: ~150 → ~2,000 features/image) |
| sfm | COLMAP | SIFT, matching only angular neighbours from the scanner's `path.json`, mapping with one shared camera and relaxed registration thresholds, global SfM (view-graph calibration + global positioning), then leftover images registered and bundle-adjusted (ef01: 194/200, 0.9 px) |
| undistort | Python/OpenCV | Undistort to **PNG** (COLMAP's undistorter would re-encode JPEG), OpenMVS masks |
| dense, mesh | OpenMVS | Dense cloud and mesh (only for alignment and the holder occluder) |
| crop / align | Python | Similarity alignment to the STL: prop found by base-coat colour → PCA → trimmed ICP (rigid, then scaled with bidirectional correspondences). **Independent check**: STL outline ray-cast into every photo vs the photo's object outline (ef01: median 15 µm) |
| texture | Python/Open3D | UV atlas (UVAtlas, ef01: 28 charts), every texel projected into every view, visibility by ray casting STL + holder, blend in linear light with the best view dominating, chart padding. No sharpening, no colour leveling |
| export | Python | `model.obj/.mtl` in STL mm, PNG texture, STL copy |
| report | Python | Texel size and density, coverage, alignment, edge spread (texture rendered back into photos, same edges measured in both), `report.md`, `preview.jpg` |

Finished steps are skipped on re-run (delete a step's output to redo it).

## Install

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), then:

```bash
micromamba env create -f environment.yml
micromamba env create -f environment-colmap.yml
```

COLMAP lives in its own env because it cannot be solved together with OpenMVS. On Windows with an
NVIDIA driver conda installs the CUDA builds. Put the two env `bin` folders (Windows: `<env>/Library/bin`)
into a config file, see `config/`.

## Run

```bash
micromamba run -n pgz python -m pgz run <unzipped OpenScan project> --out <work dir> --stl <prop.stl> --config config/windows-rtx2070.toml
```

Result: `<work dir>/deliver/<project>_<scan>_baked/` with `model.obj`, `model.mtl`, `model_albedo.png`,
the STL, `note.txt`, `report.md`, `report.json`, `preview.jpg`.

`--until <step>` stops early.

## Configs

- `config/container.toml` – Linux CPU test container (reduced feature/depth resolution)
- `config/windows-rtx2070.toml` – full resolution, GPU

All keys, defaults and the reasons for non-default values: `pgz/config.py`.

## ef01 results (container, CPU)

Measured on the sample scan with the current defaults; see the report in the work folder for the latest run.

| | OpenMVS on scan mesh | OpenMVS on STL | own baking on STL |
| --- | --- | --- | --- |
| UV islands / patches | tens of thousands | 19,314 | 28 charts |
| Textured surface (excl. holder side) | 63 % | 100 % | 96.9 % (rest: putty contact) |
| Alignment check (outline vs photo) | – | – | median 15 µm |
| Edge spread, same edges in photo and re-rendered texture | – | – | texture 72 µm vs photos 60 µm (WARN: target 30, minimum 75) |

## Known limits

- Source images are JPEG q90 with 4:2:0 chroma subsampling (OpenScan firmware): colour edges have half
  resolution. On ef01 the interior edges in the photos are ~60 µm wide; that is the ceiling for the texture.
  The bake adds ~20 %; finer texels (10 µm) or a harder best-view selection did not reduce it, so the
  rest is most likely view-to-view registration (SfM reprojection ~0.9 px ≈ 12 µm).
- Prop segmentation for the first alignment uses the base-coat colour (`[segment]`, default: red).
- Edge spread is measured on the strongest texture edges (label, shading) – a proxy until a scan with
  painted highlights exists.
- No colour calibration (checker out of scope for the prototype).
- UV chart borders follow UVAtlas (often sharp edges). Colours are continuous across them because every
  texel is computed by the same blend, but the requirement "seams away from edges" is not enforced.
