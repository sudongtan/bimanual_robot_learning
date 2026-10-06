"""Write simplified copies of the SO-101 visual meshes (scene-decisions D19).

Reads sim/assets/so101/assets/*.stl (originals, unchanged) and writes
quadric-decimated copies to sim/assets/so101/assets_visual/. The arm model
points its meshdir there. Safe for physics: none of the arm's mesh geoms can
collide (collision is boxes/spheres/capsule) and every arm body has an explicit
<inertial>, so meshes affect only how the arm looks.

Usage:
    uv run python scripts/simplify_meshes.py                 # keep 10% of faces
    uv run python scripts/simplify_meshes.py --keep 0.05
"""

import argparse
from pathlib import Path

import trimesh

SRC = Path("sim/assets/so101/assets")
DST = Path("sim/assets/so101/assets_visual")
MIN_FACES = 400  # small meshes are left with at least this many faces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=float, default=0.10, help="fraction of faces to keep")
    args = ap.parse_args()

    DST.mkdir(parents=True, exist_ok=True)
    total_in = total_out = 0
    for src in sorted(SRC.glob("*.stl")):
        mesh = trimesh.load(src, force="mesh")
        target = max(MIN_FACES, int(len(mesh.faces) * args.keep))
        out = mesh if target >= len(mesh.faces) else mesh.simplify_quadric_decimation(face_count=target)
        out.export(DST / src.name)
        total_in += len(mesh.faces)
        total_out += len(out.faces)
        print(f"{src.name:40s} {len(mesh.faces):7d} -> {len(out.faces):6d} faces")
    print(f"total (unique meshes): {total_in} -> {total_out} faces ({100 * total_out / total_in:.1f}%)")


if __name__ == "__main__":
    main()
