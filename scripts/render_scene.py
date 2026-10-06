"""Render every camera in a MuJoCo scene to one PNG, after letting physics settle.

Usage:
    uv run python scripts/render_scene.py sim/scenes/dinner_table.xml outputs/renders/step1.png
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--settle", type=float, default=1.0, help="seconds of physics before rendering")
    ap.add_argument("--keyframe", default=None, help="keyframe name to reset into")
    ap.add_argument("--size", type=int, nargs=2, default=(480, 640), metavar=("H", "W"))
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(args.scene))
    d = mujoco.MjData(m)
    if args.keyframe:
        mujoco.mj_resetDataKeyframe(m, d, m.key(args.keyframe).id)
    else:
        mujoco.mj_resetData(m, d)
    for _ in range(int(args.settle / m.opt.timestep)):
        mujoco.mj_step(m, d)

    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} ncam={m.ncam}")
    print(f"after {args.settle}s: qpos finite={np.isfinite(d.qpos).all()} ncon={d.ncon}")

    h, w = args.size
    renderer = mujoco.Renderer(m, h, w)
    frames = []
    for i in range(m.ncam):
        renderer.update_scene(d, camera=i)
        frames.append(renderer.render())
        print(f"camera {i}: {m.camera(i).name}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, np.concatenate(frames, axis=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
