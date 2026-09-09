"""Phase 0 checkpoint: verify the local environment can import the stack and
render an offscreen MuJoCo frame.

    uv run python scripts/check_env.py
"""

import platform
import sys


def main() -> int:
    print(f"python   {sys.version.split()[0]}  ({platform.machine()}, {sys.platform})")

    import mujoco
    import lerobot
    import openvino
    import torch

    print(f"mujoco   {mujoco.__version__}")
    print(f"lerobot  {lerobot.__version__}")
    print(f"openvino {openvino.__version__}")
    print(f"torch    {torch.__version__}  (mps={torch.backends.mps.is_available()}, "
          f"cuda={torch.cuda.is_available()})")

    devices = openvino.Core().available_devices
    print(f"openvino devices: {devices}")

    # Offscreen render of a trivial scene. On macOS MuJoCo renders headless via
    # CGL with no display; on Linux set MUJOCO_GL=egl (GPU) or osmesa (software).
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <light pos="0 0 2"/>
            <geom type="plane" size="1 1 0.1" rgba=".8 .8 .8 1"/>
            <body pos="0 0 .3"><freejoint/><geom type="box" size=".1 .1 .1" rgba=".8 .2 .2 1"/></body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()
    print(f"render   ok: frame {frame.shape} {frame.dtype}, mean={frame.mean():.1f}")

    print("\nPhase 0 checkpoint: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
