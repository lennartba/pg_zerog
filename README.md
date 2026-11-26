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
| deform | Python/SciPy | Non-rigid fit of the STL to the photographed surface (dense cloud, smooth lattice, point-to-plane). Prints deviate from their STL (ef01: cube tops 0.5 mm lower, elephant foot); projecting onto the undeformed STL ghosts details and darkens corners. Output keeps the original STL |
| texture | Python/Open3D | UV atlas (UVAtlas, ef01: 28 charts), every texel projected into every view **from its position on the deformed (real) surface**, visibility by ray casting STL + holder, blend in linear light with the best view dominating, chart padding. No sharpening, no colour leveling |
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
| UV islands / patches | tens of thousands | 19,314 | 22 charts |
| Textured surface (excl. holder side) | 63 % | 100 % | 94.8 % (rest: putty contact, narrow gaps between cubes) |
| Alignment check (outline vs photo) | – | – | median 15 µm |
| Texture registration, 8 held-out photos (excl. holder side) | – | – | median 15 µm, p90 30 µm (PASS, target ≤ 50 µm; measurement floor ~13–17 µm) |
| Print vs STL (fitted deformation) | – | – | RMS 0.18 mm, max 0.56 mm (cube tops −0.45 mm); side-view check of cube tops after fit: −21 µm (before: −505 µm) |
| Edge spread, same edges in photo and re-rendered texture | – | – | NOT MEASURED: excl. holder side and occlusion edges, a plain base coat leaves too few texture edges (earlier 66 vs 59 µm came mostly from the label) |

## Batch: calibration props ef02–ef_h02

Six props (red base coat, putty on a stick), container defaults, 200 views each, ~100–120 min per scan
(4 CPUs). Registration recomputed with the subdivided render (see above).

| Scan | Texel | Textured (excl. holder) | Outline alignment | Registration median / p90 | Print vs STL RMS (fit residual) | Edge radius (lines fitted) |
| --- | --- | --- | --- | --- | --- | --- |
| ef02 angles | 13.6 µm | 100 % | 15 µm | 20 / 41 µm | 0.136 mm (35 µm) | 30 µm (19) |
| ef03 round | 15.0 µm | 99.3 % | 15 µm | 14 / 33 µm | 0.163 mm (30 µm) | – (0, default 100 µm) |
| ef04 fins | 14.3 µm | 100 % | 14 µm | 13 / 30 µm | 0.095 mm (26 µm) | 55 µm (23) |
| ef05 details | 14.1 µm | 99.7 % | 15 µm | 15 / 37 µm | 0.154 mm (32 µm) | 179 µm (11, hole rim) |
| ef_h01 console | 15.2 µm | 100 % | 15 µm | 14 / 29 µm | 0.086 mm (26 µm) | 30 µm (6) |
| ef_h02 tank | 14.4 µm | 99.9 % | 14 µm | 14 / 32 µm | 0.163 mm (18 µm) | – (0, default 100 µm) |

Findings:
- ef04: the printed plate is 0.42 mm thinner than the STL; the holder-side capture limit dropped the
  whole bottom (fixed: the limit now applies outwards only, where the putty is).
- Edge spread on base-coat props measured the paper label (holder side) and occlusion edges; both are
  now excluded, which leaves too few texture edges -> NOT MEASURED. Needs a scan with painted detail.
- ef_h02: the 2.3 mm max deviation is the putty around the leg ends, not the print.
- Dark rings around raised features (ef03–ef05) are in the paint (spray shadow), visible in the photos.
- A single edge line's radius is determined to ±25–70 µm; outlines against a far wall instead of the
  background are not biased (ef04: +16 µm on average).
- Peak disk use is ~25 GB per scan during DensifyPointCloud (depth maps), on top of ~4 GB kept per scan.

## Robustness against print deformation

`tools/deform_test.py` applies synthetic defects to the STL (the real ef01 print already deviates by
~0.5 mm at the cube tops), runs fit + bake with 8 held-out views, and measures registration by rendering
the texture into the held-out photos (patch-wise phase correlation, median / p90 in µm on the surface):

| Case | Plate top | Sides | Cube tops | Fitted shape vs real-STL fit |
| --- | --- | --- | --- | --- |
| no fit (texture on the plain STL) | 23 / 42 | 20 / 36 | 127 / 175 (3 of ~100 patches correlate) | – |
| fit, real STL | 28 / 63 | 22 / 47 | 26 / 61 | – |
| STL sheared x += 0.08·z | 34 / 84 | 25 / 58 | 40 / 70 | 49 / 106 |
| STL y × 1.03 | 29 / 62 | 26 / 47 | 29 / 57 | 21 / 49 |
| STL z × 1.06 (total z error ~1 mm) | 29 / 65 | 33 / 66 | 32 / 111 | 19 / 94 |
| 0.4 mm local bump | 32 / 68 | 20 / 50 | 26 / 58 | 8 / 51 |

Note: these numbers were measured with the texture rendered on the STL deformed only at its CAD
vertices; inside big faces the deformation was lost, which inflated the shifts (ef01: 29 -> 15 µm once
the mesh is subdivided first, as the bake does). Relative comparisons stand; absolute values are high.

Reading: the fit fixes the cube tops (127 -> 26 µm) at a small cost on the flat plate top (p90 42 -> 63 µm,
mostly next to the cube bases where a smooth field cannot follow a 0.5 mm step exactly). Scale errors,
a 1 mm height error and local bumps are recovered to the level of the plain fit. Shear is the weakest
case: tangential shifts of flat tops are only constrained by the side faces. Run time ~13 min per case.

## Edges

Printed edges are neither where nor how the STL has them. ef01, left cube: the edges sit 20-104 µm
inside the deformed STL (the dense cloud cannot resolve this right at an edge) and are nearly sharp
(radius at the 30 µm floor). Sampling the photos at the STL's corner gave dark lines along the edges.
The bake now handles this from the data of every scan, no tuned constant:

1. `edgefit.py`: wherever a sharp convex STL edge is the outline of the prop in a photo, the outline
   position is measured sub-pixel (50 % crossing prop/background). Per edge line a robust fit gives
   radius and corner displacement (views from different angles separate the two; priors keep poorly
   observed lines at the scan-wide values). ef01: 4,715 measurements, 8 edge lines, residual 6-17 µm.
2. `edges.py`: texels next to sharp edges sample the measured edge (shifted, rounded); face offsets fade
   out over 0.5 mm. The ray-casting surface carries the same correction (visibility, outlines).
3. Blur window (`bake.blur_window_px` 6): a photo pixel mixes its surroundings, so the weight uses the
   smallest view cosine within the window (background = 0). At an edge, views seeing both faces alike
   win over views that see the other face at a grazing angle (dark) or the background.

Red across the top/back edge in view 141 (top face 0.4 mm | edge | back face 0.4 mm):
photo 147 | 149-151 | 129; sharp sampling 159 | 120 | 144; now 160 | 134-138 | 144.
Edges that are never an outline against the background (concave edges, edges only seen against the
prop itself) use the scan-wide radius without displacement.

## View-angle normalisation (`bake.shading_normalise`, on by default; prototype)

OpenScan's light moves with the camera, so a surface seen at an angle is also lit at that angle. Fitted
per scan from ~100k surface points (`shading.py`): brightness = albedo x per-photo factor x f(cos), with
f(cos) close to cos (Lambert, light at the camera; ef01: f(0.5) = 0.54, f(0.75) = 0.81). Every sample is
divided by the brightness the model expects for that pixel (f over the photo blur, background 0), so
each view contributes its head-on brightness. The spread of one surface point across views drops from
22 % to 12 %. Shadows, occlusion and paint stay; only the view-angle term is removed. Effects on ef01:
edge bands within -12..+5 % of the neighbouring faces; faces of different orientation closer together;
side faces in narrow gaps (only seen obliquely) up to 30 % brighter; shading of geometric detail
(engraved text) weaker. This touches the "no delighting" requirement: it removes view-dependent
lighting, not baked-in shading. Off: `shading_normalise = false`.

## Known limits

- The outline alignment check (median 15 µm) is dominated by the plate outline and did not reveal the
  0.5 mm cube-top deviation; the print-vs-STL row (deformation fit) is the relevant number for that.
- Surfaces no photo sees (putty contact, the bottom of the narrow gaps between cubes) stay black in
  the texture.

- Source images are JPEG q90 with 4:2:0 chroma subsampling (OpenScan firmware): colour edges have half
  resolution. On ef01 the interior edges in the photos are ~60 µm wide; that is the ceiling for the texture.
  The bake adds ~12 %. Most of the earlier loss came from projecting onto the undeformed STL.
- Prop segmentation for the first alignment uses the base-coat colour (`[segment]`, default: red).
- Edge spread is measured on the strongest texture edges (label, shading) – a proxy until a scan with
  painted highlights exists.
- No colour calibration (checker out of scope for the prototype).
- UV chart borders follow UVAtlas (often sharp edges). Colours are continuous across them because every
  texel is computed by the same blend, but the requirement "seams away from edges" is not enforced.
