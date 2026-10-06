"""Download YCB objects and turn each into a MuJoCo model under sim/assets/objects/ycb/<id>/.

For each object:
  1. download <id>_google_16k.tgz from the YCB S3 bucket (cached in outputs/ycb_cache/)
  2. copy textured.obj/.mtl unmodified; downscale texture_map.png to --texture px
  3. compile the mesh, measure its bounds, and offset the visual geom so the
     object's bounding-box centre sits on the body origin
  4. write model.xml: a freejoint body with
       - visual geom: the scan (group 2, no contact, no mass)
       - collision geom: an exact cylinder ("cylinder") or the mesh's convex
         hull ("hull", group 3), carrying the YCB mass
  5. write SOURCE.md with provenance

Masses come from Calli et al. 2015, "The YCB Object and Model Set", Table II
(https://arxiv.org/abs/1502.03143). The 065 cups are a set of ten; their
masses are listed in size order, and the mass for a cup is matched by its
measured diameter.

Usage:
    uv run python scripts/import_ycb.py                    # every object in ITEMS
    uv run python scripts/import_ycb.py 025_mug 029_plate  # just these
"""

import argparse
import io
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

BASE_URL = "https://ycb-benchmarks.s3.amazonaws.com/data/google/"
OUT_ROOT = Path("sim/assets/objects/ycb")
CACHE = Path("outputs/ycb_cache")

# id: (mass in kg from YCB Table II, collision type, waste stream in our task)
ITEMS = {
    "002_master_chef_can": (0.414, "cylinder", "metal"),
    "005_tomato_soup_can": (0.349, "cylinder", "metal"),
    "007_tuna_fish_can": (0.171, "cylinder", "metal"),
    "010_potted_meat_can": (0.370, "hull", "metal"),  # rectangular tin, not a cylinder
    "006_mustard_bottle": (0.431, "hull", "plastic"),
    "011_banana": (0.066, "hull", "food waste"),
    "012_strawberry": (0.018, "hull", "food waste"),
    "013_apple": (0.068, "hull", "food waste"),
    "014_lemon": (0.029, "hull", "food waste"),
    "015_peach": (0.033, "hull", "food waste"),
    "016_pear": (0.049, "hull", "food waste"),
    "017_orange": (0.047, "hull", "food waste"),
    "018_plum": (0.025, "hull", "food waste"),
    "023_wine_glass": (0.133, "hull", "to wash"),  # no google_16k scan on the server (404): skipped
    "024_bowl": (0.147, "hull", "to wash"),
    "025_mug": (0.118, "hull", "to wash"),
    "029_plate": (0.279, "hull", "to wash"),
    "030_fork": (0.034, "hull", "to wash"),
    "031_spoon": (0.030, "hull", "to wash"),
    "032_knife": (0.031, "hull", "to wash"),
}
# 065 cups, smallest to largest: (diameter mm, height mm, mass g) from Table II.
CUPS = [(55, 60, 13), (60, 62, 14), (65, 64, 17), (70, 66, 19), (75, 68, 21),
        (80, 70, 26), (85, 72, 28), (90, 74, 31), (95, 76, 35), (100, 78, 38)]


def fetch(obj_id: str) -> Path | None:
    """Download and unpack one object; return its google_16k directory, or None if YCB has no such scan."""
    out = CACHE / obj_id / "google_16k"
    if not (out / "textured.obj").exists():
        print(f"  downloading {obj_id} …")
        try:
            data = urllib.request.urlopen(f"{BASE_URL}{obj_id}_google_16k.tgz", timeout=300).read()
        except urllib.error.HTTPError as e:
            print(f"  {obj_id}: no google_16k scan on the YCB server (HTTP {e.code}); skipped")
            return None
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(CACHE, filter="data")
    return out


def measure(src: Path) -> np.ndarray:
    """Compiled mesh vertices in the body frame of a geom with no user offset."""
    m = mujoco.MjModel.from_xml_string(
        f'<mujoco><compiler meshdir="{src.resolve()}"/><asset><mesh name="m" file="textured.obj"/></asset>'
        '<worldbody><body><geom type="mesh" mesh="m"/></body></worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    v = m.mesh_vert[: m.mesh_vertnum[0]]
    return v @ d.geom_xmat[0].reshape(3, 3).T + d.geom_xpos[0]


def write_model(obj_id: str, mass: float, collision: str, stream: str, texture_px: int) -> dict | None:
    src = fetch(obj_id)
    if src is None:
        return None
    dst = OUT_ROOT / obj_id
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("textured.obj", "textured.mtl"):
        (dst / f).write_bytes((src / f).read_bytes())
    Image.open(src / "texture_map.png").resize((texture_px, texture_px), Image.LANCZOS).save(
        dst / "texture_map.png", optimize=True)

    v = measure(src)
    lo, hi = v.min(0), v.max(0)
    centre, size = (lo + hi) / 2, hi - lo
    off = " ".join(f"{-c:.4f}" for c in centre)
    name = obj_id.split("_", 1)[1]

    if collision == "cylinder":
        r, hh = (size[0] + size[1]) / 4, size[2] / 2
        coll = (f'<geom name="{name}_collision" type="cylinder" size="{r:.4f} {hh:.4f}" mass="{mass}"\n'
                f'            group="3" condim="4" friction="0.9 0.005 0.0001"/>')
        coll_note = f"exact cylinder r {r*1000:.1f} mm, height {2*hh*1000:.1f} mm (from the scan's bounds)"
    else:
        coll = (f'<geom name="{name}_collision" type="mesh" mesh="{name}" pos="{off}" mass="{mass}"\n'
                f'            group="3" condim="4" friction="0.9 0.005 0.0001"/>')
        coll_note = "the scan's convex hull (MuJoCo convexifies mesh geoms for contact)"

    (dst / "model.xml").write_text(f"""<!--
  YCB {obj_id} (Calli et al. 2015). CC BY 4.0, see ../LICENSE and SOURCE.md.
  Generated by scripts/import_ycb.py. Waste stream in our task: {stream}.

  Body origin = centre of the scan's bounding box. Visual: the textured scan
  (group 2, no contact, no mass). Collision: {coll_note}; mass {mass*1000:.0f} g
  (YCB Table II).
-->
<mujoco model="ycb_{obj_id}">
  <compiler angle="radian" meshdir="." texturedir="."/>
  <asset>
    <texture name="{name}" type="2d" file="texture_map.png"/>
    <material name="{name}" texture="{name}"/>
    <mesh name="{name}" file="textured.obj"/>
  </asset>
  <worldbody>
    <body name="{name}">
      <freejoint name="{name}_free"/>
      <geom name="{name}_visual" type="mesh" mesh="{name}" material="{name}" pos="{off}"
            contype="0" conaffinity="0" group="2" mass="0"/>
      {coll}
    </body>
  </worldbody>
</mujoco>
""")
    (dst / "SOURCE.md").write_text(f"""# YCB {obj_id} — provenance

- Source: `{BASE_URL}{obj_id}_google_16k.tgz`, imported by `scripts/import_ycb.py`.
  B. Calli et al., "The YCB Object and Model Set", 2015 — https://arxiv.org/abs/1502.03143
- License: CC BY 4.0 (`../LICENSE`).
- `textured.obj` / `textured.mtl` unmodified; `texture_map.png` downscaled to
  {texture_px}×{texture_px} (Lanczos).
- Mass {mass*1000:.0f} g from the YCB paper's Table II. Scan bounding box
  {size[0]*1000:.1f} × {size[1]*1000:.1f} × {size[2]*1000:.1f} mm (x × y × z).
- Collision: {coll_note}.
""")
    return {"id": obj_id, "size_mm": (size * 1000).round(1), "mass_g": round(mass * 1000), "collision": collision}


def pick_cup(target_diameter_mm: float) -> tuple[str, float]:
    """Return the 065 cup whose scanned diameter is closest to the target, with its Table II mass."""
    best = None
    for letter in "abcdefghij":
        cid = f"065-{letter}_cups"
        size = np.ptp(measure(fetch(cid)), axis=0) * 1000
        dia = (size[0] + size[1]) / 2
        mass = min(CUPS, key=lambda c: abs(c[0] - dia))[2] / 1000
        print(f"  {cid}: diameter {dia:.0f} mm, height {size[2]:.0f} mm → {mass*1000:.0f} g")
        if best is None or abs(dia - target_diameter_mm) < abs(best[2] - target_diameter_mm):
            best = (cid, mass, dia)
    return best[0], best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="YCB ids (default: all of ITEMS plus one cup)")
    ap.add_argument("--texture", type=int, default=1024)
    ap.add_argument("--cup-diameter", type=float, default=75.0, help="mm; picks the closest 065 cup")
    args = ap.parse_args()

    items = dict(ITEMS)
    if not args.ids:
        cid, mass = pick_cup(args.cup_diameter)
        items[cid] = (mass, "hull", "plastic")
    ids = args.ids or list(items)
    for obj_id in ids:
        mass, coll, stream = items[obj_id]
        r = write_model(obj_id, mass, coll, stream, args.texture)
        if r is None:
            continue
        print(f"{r['id']:22s} {stream:10s} {r['mass_g']:4d} g  {r['collision']:8s} bbox {r['size_mm']} mm")


if __name__ == "__main__":
    main()
