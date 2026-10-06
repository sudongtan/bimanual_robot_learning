# YCB 010_potted_meat_can — provenance

- Source: `https://ycb-benchmarks.s3.amazonaws.com/data/google/010_potted_meat_can_google_16k.tgz`, imported by `scripts/import_ycb.py`.
  B. Calli et al., "The YCB Object and Model Set", 2015 — https://arxiv.org/abs/1502.03143
- License: CC BY 4.0 (`../LICENSE`).
- `textured.obj` / `textured.mtl` unmodified; `texture_map.png` downscaled to
  1024×1024 (Lanczos).
- Mass 370 g from the YCB paper's Table II. Scan bounding box
  102.1 × 60.1 × 83.5 mm (x × y × z).
- Collision: the scan's convex hull (MuJoCo convexifies mesh geoms for contact).
