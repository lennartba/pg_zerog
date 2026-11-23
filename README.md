# pg_zerog

Photogrammetry pipeline prototype for focus-stacked scans from an **OpenScan Mini + macro add-on**,
producing textured scans for edgefit (see the requirements doc *Photogrammetry output requirements for edgefit*).

Open source only: [COLMAP](https://colmap.github.io/) (structure from motion) and
[OpenMVS](https://github.com/cdcseacave/openMVS) (dense cloud, mesh, texture), orchestrated in Python.

## Pipeline

| Step | Tool | What happens | Why |
| --- | --- | --- | --- |
| prepare | Python | Link images, foreground masks (threshold on black background) | Holder stays in: it rotates with the prop and gives the matcher texture |
| sfm | COLMAP | SIFT, matching only angular neighbours from the scanner's `path.json`, mapping with one shared camera | Pose prior saves ~85 % of matching and avoids wrong matches between similar sides |
| undistort | Python/OpenCV | Undistort to **PNG**, PINHOLE model, OpenMVS masks | COLMAP's own undistorter would re-encode JPEG |
| dense | OpenMVS | Depth maps, fused point cloud, masked | |
| mesh | OpenMVS | Delaunay mesh, **smoothing off** | Smoothing rounds off bevels |
| crop | Python | Similarity alignment to the STL (colour segmentation → PCA → ICP with scale), keep faces within 0.6 mm of the STL, optional RefineMesh | Removes holder/putty; gives true mm |
| texture | OpenMVS | Per-face best view, **sharpening off**, **no global seam leveling**, full-res images | Requirements forbid sharpening and colour shifts |
| export | Python | `model.obj/.mtl` in STL mm, PNG textures, STL copy | Deliverable layout from the requirements |
| report | Python | Texel size on the prop, density, untextured area, RMS vs STL, coverage | Pass/warn/fail against the requirements |

Finished steps are skipped on re-run (delete the step's output to redo it).

## Install

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), then:

```bash
micromamba env create -f environment.yml
micromamba env create -f environment-colmap.yml
```

On Windows with an NVIDIA driver conda installs the CUDA builds of both. Put the two env `bin`
folders (Windows: `<env>/Library/bin`) into a config file, see `config/`.

## Run

```bash
micromamba run -n pgz python -m pgz run <unzipped OpenScan project> --out <work dir> --stl <prop.stl> --config config/windows-rtx2070.toml
```

Result: `<work dir>/deliver/<project>_<scan>/` with `model.obj`, `model.mtl`, texture PNGs, the STL,
`note.txt` and `report.md`.

`--until <step>` stops early; useful for checking the alignment before texturing.

## Configs

- `config/container.toml` – Linux CPU test container (reduced feature/depth resolution)
- `config/windows-rtx2070.toml` – full resolution, GPU, RefineMesh on

All keys and defaults: `pgz/config.py`.

## Known limits (prototype)

- Source images are JPEG q90 with 4:2:0 chroma subsampling (OpenScan firmware); colour edges have half resolution.
- Prop segmentation for the first alignment uses the base-coat colour (`[segment]`, default: red).
- Not measured automatically yet: edge spread on painted edges, texture-to-mesh registration, colour (ΔE, no checker).
