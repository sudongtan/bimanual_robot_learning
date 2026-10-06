"""Search for a collision-free home pose for both SO-101 arms and print it as a <key>.

The search runs on the left arm (arm A): random joint samples, then local
refinement, minimising distance from the gripper tip to --target plus a penalty
for not pointing down. The right arm (arm B) gets the same joint vector with
shoulder_pan and wrist_roll negated: its base is rotated 180 deg about z, so
this lands its tip at the target mirrored in x. The result is then checked
under physics: contacts at the pose, and how well the servos hold it.

Usage:
    uv run python scripts/find_home_pose.py
    uv run python scripts/find_home_pose.py --target -0.12 -0.14 0.435 --seed 0

Prints a <keyframe> block with two keys, `home` and `bins_open` (every slide
joint at its upper limit); free objects keep their spawn pose (qpos0). Paste it
over the scene's <keyframe> block.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
POINT_DOWN_WEIGHT = 0.03  # metres of position error traded for pointing straight up instead of down
RANGE_MARGIN = 0.8  # only sample the middle 80% of each joint range


def tip_and_direction(m, d, side):
    """Gripper tip position and the unit vector from the wrist-roll body to the tip."""
    tip = d.site(f"{side}_gripperframe").xpos.copy()
    v = tip - d.xpos[m.body(f"{side}_gripper").id]
    return tip, v / np.linalg.norm(v)


def search_left(m, d, target, samples, seed):
    jids = [m.joint(f"left_{j}").id for j in ARM_JOINTS]
    adr = [m.jnt_qposadr[j] for j in jids]
    lo, hi = m.jnt_range[jids, 0] * RANGE_MARGIN, m.jnt_range[jids, 1] * RANGE_MARGIN

    def cost(q):
        d.qpos[:] = 0
        d.qpos[adr] = q
        mujoco.mj_kinematics(m, d)
        tip, v = tip_and_direction(m, d, "left")
        return np.linalg.norm(tip - target) + POINT_DOWN_WEIGHT * (1 + v[2])

    rng = np.random.default_rng(seed)
    best_q, best_c = None, np.inf
    for q in rng.uniform(lo, hi, size=(samples, len(adr))):
        c = cost(q)
        if c < best_c:
            best_q, best_c = q, c
    q = best_q
    for i in range(4000):
        cand = np.clip(q + np.random.default_rng(seed + 1 + i).normal(0, 0.02, len(adr)), lo, hi)
        c = cost(cand)
        if c < best_c:
            best_q, best_c, q = cand, c, cand
    return best_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("sim/scenes/dinner_table.xml"))
    ap.add_argument("--target", type=float, nargs=3, default=(-0.12, -0.14, 0.435),
                    metavar=("X", "Y", "Z"), help="left gripper tip target, world frame (m)")
    ap.add_argument("--samples", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(args.scene))
    d = mujoco.MjData(m)
    left = search_left(m, d, np.array(args.target), args.samples, args.seed)
    right = left.copy()
    right[0], right[4] = -right[0], -right[4]

    # Full qpos in the model's joint order: every non-arm joint (bins, objects)
    # stays at its default (qpos0); arm joints get the solution; grippers at 0.
    q = m.qpos0.copy()
    for side, vals in (("left", left), ("right", right)):
        for name, v in zip(ARM_JOINTS, vals):
            q[m.jnt_qposadr[m.joint(f"{side}_{name}").id]] = v
        q[m.jnt_qposadr[m.joint(f"{side}_gripper").id]] = 0.0
    q = q.round(3)
    # ctrl in actuator order: each position servo targets its joint's home value.
    ctrl = np.array([q[m.jnt_qposadr[m.actuator_trnid[i, 0]]] for i in range(m.nu)])
    arm_adr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]

    d.qpos[:] = q
    d.ctrl[:] = ctrl
    mujoco.mj_forward(m, d)
    ncon0 = d.ncon
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    for side in ("left", "right"):
        tip, v = tip_and_direction(m, d, side)
        print(f"{side:5s} tip {tip.round(3)}  pointing {v.round(2)}")
    err = np.abs(d.qpos[arm_adr] - ctrl).max()
    print(f"contacts at pose: {ncon0}; after 3 s: {d.ncon}; max arm |q - target| = {err:.4f} rad")

    # bins_open: the same, with every slide joint (the pull-out bins) at its upper limit.
    q_open = q.copy()
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
            q_open[m.jnt_qposadr[j]] = m.jnt_range[j, 1]

    fmt = lambda v: " ".join(f"{x:g}" for x in v)
    print("\n<keyframe>")
    for name, qq in (("home", q), ("bins_open", q_open)):
        print(f'  <key name="{name}"\n       qpos="{fmt(qq)}"\n       ctrl="{fmt(ctrl)}"/>')
    print("</keyframe>")


if __name__ == "__main__":
    main()
