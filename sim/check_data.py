"""First-batch checks on a recorded dataset (2data_collection.md, "Checks on the first batch").

    python -m sim.check_data data/a0                # all checks; report and contact sheets in data/a0/check/
    python -m sim.check_data data/a0 --no-sheets    # skip the contact sheets (decoding video is the slow part)
    python -m sim.check_data data/a0 --stride 5     # cube visibility on every 5th frame (default 10)

Check 1 (exact replay) already ran in the recorder, on every episode. This script runs the other
three, on what training will read (the LeRobot dataset), plus the episode log:

    summary   per variant: episodes recorded and failed (with the failed checks), S2 start kinds, cube
              colours, episode lengths
    actions   largest arm-command change between frames (posture jumps), longest still stretch before
              the final hold, gripper closes per arm
    labels    the command names the variant's arm(s) and target; only the named arm moves (composites:
              the first arm before the switch, the second after); subtask labels switch once; still hold
    images    contact sheets (three cameras at a few moments per episode), and from re-rendered
              segmentation: how many overview pixels the cube covers (at 480 x 640 and resized to
              224 x 224, as π0) and its colour contrast with each surround (arm parts, table, drawer, handle, cabinet)

Every line that breaks a threshold is listed under FLAGS at the top of the report.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path

import mujoco
import numpy as np

from sim import episode as E
from sim import randomize as RZ
from sim import record as R
from sim.scene import ARM_NAME, GRIPPER_CLOSED, STATE_SPEC
from sim.skills import HOLD_S
from sim.tasks import HANDLE_OPEN

HOLD_FRAMES = int(HOLD_S * R.FPS)
JUMP = 0.15  # rad; an arm command changing more than this between frames is a posture jump (tests' MAX_JUMP)
STILL = 1e-3  # rad per frame; every joint (and gripper) below this = nothing moving
LONG_STILL = 1.0  # s; a still stretch longer than this before the final hold is flagged
MOVED = 0.3  # rad; an arm that acts moves at least this far from where its step started
STAYED = 0.05  # rad; an arm that doesn't act stays within this (as the hold check)
GRIP_MID = (GRIPPER_CLOSED + HANDLE_OPEN) / 2  # a gripper command crossing this downwards is one close
MIN_CONTRAST = 0.4  # RGB distance (0-1 per channel) between the cube and its least different surround in the
                    # overview; every palette colour measured >= 0.45, the rejected ones <= 0.28
MIN_PX_224 = 20  # cube pixels in the overview after resizing to 224 x 224
SHEET_SCALE = 3  # contact sheet images are 1/3 size (160 x 213)


def arm_joints(task_arm: str) -> slice:
    """Columns of one arm's 5 joints in observation.state / action (actuator order: left 6, then right 6)."""
    i = 0 if ARM_NAME[task_arm] == "left" else 6
    return slice(i, i + 5)


def gripper(task_arm: str) -> int:
    return 5 if ARM_NAME[task_arm] == "left" else 11


def acting_arms(variant) -> tuple[str, str]:
    """(first, second): the arm that acts first and the other one."""
    return variant[1], E.other(variant[1])


def check_command(variant, command: str) -> list[str]:
    """The command names the variant's arms (in order, for composites) and its target, and nothing else."""
    task, arm, target = variant
    problems = []
    named = re.findall(r"arm ([AB])", command)
    first, second = acting_arms(variant)
    want = [first, second] if task in ("T1", "T2") else [first]
    if named != want:
        problems.append(f"names arms {named}, expected {want}")
    targets = [t for t in RZ.TARGETS if re.search(rf"\b{t}\b", command)]
    if targets != ([target] if target else []):
        problems.append(f"names targets {targets}, expected {target}")
    return problems


def longest_still(state: np.ndarray) -> float:
    """Longest run (s) of frames with nothing moving, before the final hold."""
    moving = np.abs(np.diff(state[: len(state) - HOLD_FRAMES], axis=0)).max(axis=1) > STILL
    run = best = 0
    for m in moving:
        run = 0 if m else run + 1
        best = max(best, run)
    return best / R.FPS


def closes(g: np.ndarray) -> np.ndarray:
    """Frames where a gripper command crosses GRIP_MID downwards."""
    return np.flatnonzero((g[:-1] >= GRIP_MID) & (g[1:] < GRIP_MID)) + 1


def check_episode(e: dict, state: np.ndarray, action: np.ndarray, subtask: np.ndarray) -> tuple[dict, list[str]]:
    """Actions and labels for one episode. Returns its metrics and its flags."""
    variant = tuple(e["variant"])
    task = variant[0]
    first, second = acting_arms(variant)
    flags = [f"command: {p}" for p in check_command(variant, e["command"])]
    m = {}

    arms = np.r_[0:5, 6:11]
    jump = np.abs(np.diff(action[:, arms], axis=0)).max()
    m["max_jump"] = round(float(jump), 3)
    if jump > JUMP:
        k = int(np.abs(np.diff(action[:, arms], axis=0)).max(axis=1).argmax())
        flags.append(f"command jump {jump:.3f} rad at frame {k}")
    m["longest_still_s"] = longest_still(state)
    if m["longest_still_s"] > LONG_STILL:
        flags.append(f"still for {m['longest_still_s']:.2f} s before the hold")

    composite = task in ("T1", "T2")
    switches = np.flatnonzero(np.diff(subtask))
    if composite:
        if len(switches) != 1 or subtask[0] != 0 or subtask[-1] != 1:
            flags.append(f"subtask labels switch at {switches.tolist()}, expected once from 0 to 1")
        cut = int(switches[0]) + 1 if len(switches) else len(subtask)
        m["switch_frame"] = cut
        steps = [(first, second, slice(0, cut)), (second, first, slice(cut, None))]
    else:
        if subtask.any():
            flags.append("subtask label set in a primitive")
        steps = [(first, second, slice(None))]
    for k, (actor, idle, part) in enumerate(steps):
        s = state[part]
        if not len(s):
            continue
        moved = np.abs(s[:, arm_joints(actor)] - s[0, arm_joints(actor)]).max()
        stayed = np.abs(s[:, arm_joints(idle)] - s[0, arm_joints(idle)]).max()
        m[f"step{k}_arm_{actor}_moved"], m[f"step{k}_arm_{idle}_moved"] = round(float(moved), 3), round(float(stayed), 3)
        if moved < MOVED:
            flags.append(f"step {k}: arm {actor} should act but moved only {moved:.3f} rad")
        if stayed > STAYED:
            flags.append(f"step {k}: arm {idle} should stay but moved {stayed:.3f} rad")
        g = action[part, gripper(actor)]
        n_closes, idle_closes = len(closes(g)), len(closes(action[part, gripper(idle)]))
        m[f"step{k}_arm_{actor}_closes"] = n_closes
        if n_closes != 1 or idle_closes:
            flags.append(f"step {k}: gripper closes arm {actor} {n_closes} (expected 1), arm {idle} {idle_closes} (expected 0)")
    tail = state[-HOLD_FRAMES:]
    hold = np.abs(tail - tail[0]).max()
    if hold > STAYED:
        flags.append(f"hold not still: {hold:.3f} rad")
    return m, flags


def randomization(info: dict) -> RZ.Randomization:
    names = {f.name for f in fields(RZ.Randomization)}
    return RZ.Randomization(**{k: v for k, v in info["randomization"].items() if k in names})


def cube_visibility(e: dict, frames: dict, rand, stride: int) -> dict:
    """Re-render the overview (colour and segmentation) on every `stride`-th frame from the recorded
    states: cube pixels when visible, how often it is visible, and its contrast with each surround
    (median over frames; the lowest is reported)."""
    sim, _ = E.new_sim(e["seed"], rand)
    model = sim.model
    data = mujoco.MjData(model)
    rgb, seg = mujoco.Renderer(model, R.HEIGHT, R.WIDTH), mujoco.Renderer(model, R.HEIGHT, R.WIDTH)
    seg.enable_segmentation_rendering()
    cube = model.geom("cube").id
    surrounds = surround_geoms(sim)
    px, contrast = [], defaultdict(list)
    for state in frames["state"][::stride]:
        mujoco.mj_setState(model, data, state, STATE_SPEC)
        mujoco.mj_forward(model, data)
        seg.update_scene(data, camera="overview")
        ids = seg.render()
        geom = np.where(ids[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), ids[..., 0], -1)
        on_cube = geom == cube
        px.append(int(on_cube.sum()))
        if on_cube.sum() < 20:
            continue
        rgb.update_scene(data, camera="overview")
        img = rgb.render() / 255.0
        for name, geoms in surrounds.items():
            on = np.isin(geom, geoms)
            if on.sum() >= 20:
                contrast[name].append(float(np.linalg.norm(img[on_cube].mean(0) - img[on].mean(0))))
    rgb.close()
    seg.close()
    px = np.array(px)
    visible = px[px > 0]
    scale = (224 / R.HEIGHT) * (224 / R.WIDTH)
    med = {k: float(np.median(v)) for k, v in contrast.items()}
    worst = min(med, key=med.get) if med else None
    return {"cube_visible_frac": round(float((px > 0).mean()), 2),
            "cube_px_median": int(np.median(visible)) if len(visible) else 0,
            "cube_px_min_visible": int(visible.min()) if len(visible) else 0,
            "cube_px_224_median": round(float(np.median(visible) * scale), 1) if len(visible) else 0.0,
            "cube_contrast": round(med[worst], 3) if worst else None, "cube_contrast_vs": worst}


def surround_geoms(sim) -> dict:
    """Geoms of what the cube is seen against: the arms' light (yellow) and dark (motor) parts apart,
    since a mean over the whole arm hides a cube that matches one of them (as black did), then the
    table, drawer, handle and cabinet. Every drawn geom, not just sim.arms[...].geoms (collision only)."""
    m = sim.model
    body = lambda g: m.body(m.geom_bodyid[g]).name
    out = defaultdict(list)
    for g in range(m.ngeom):
        b = body(g)
        if b.split("/")[0] in sim.arms and m.geom_matid[g] >= 0:
            out["arm_dark" if m.material(m.geom_matid[g]).rgba[:3].mean() < 0.3 else "arm_light"].append(g)
        elif b == "drawer":
            out["handle" if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else "drawer"].append(g)
        elif b == "cabinet":
            out["cabinet"].append(g)
    out["table"] = [m.geom("tabletop").id]
    return {k: np.array(v) for k, v in out.items()}


def contact_sheet(ds, ep: dict, index: int, e: dict, switch: int | None, path: Path):
    """One row per moment (start, quarters, the switch, end), the three cameras side by side."""
    from PIL import Image, ImageDraw

    n = ep["length"]
    moments = {"start": 0, "25%": n // 4, "50%": n // 2, "75%": 3 * n // 4, "end": n - 1}
    if switch is not None:
        moments["switch"] = switch
    rows = []
    for name, k in sorted(moments.items(), key=lambda kv: kv[1]):
        x = ds[ep["dataset_from_index"] + k]
        imgs = [(x[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[::SHEET_SCALE, ::SHEET_SCALE]
                for c in R.CAMERAS]
        row = Image.fromarray(np.concatenate(imgs, axis=1))
        ImageDraw.Draw(row).text((4, 4), f"{name}  frame {k}  subtask {int(x['subtask_index'])}", fill=(255, 255, 255))
        rows.append(np.asarray(row))
    sheet = Image.fromarray(np.concatenate(rows, axis=0))
    ImageDraw.Draw(sheet).text((4, sheet.height - 14), f"#{index} seed {e['seed']}: {e['command']}", fill=(255, 255, 0))
    sheet.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path)
    p.add_argument("--no-sheets", action="store_true")
    p.add_argument("--stride", type=int, default=10, help="frames between cube-visibility renders")
    a = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root, out = a.root, a.root / "check"
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    info = json.loads((root / "sim" / "info.json").read_text())
    rand = randomization(info)
    log = [json.loads(line) for line in open(root / "sim" / "episodes.jsonl")]
    ds = LeRobotDataset(R.REPO_ID, root=root)
    cols = ds.hf_dataset.with_format("numpy")[:]
    ep_col = cols["episode_index"].reshape(-1)

    lines, flags, per_episode = [], [], []
    by_variant = defaultdict(list)
    for e in log:
        by_variant[tuple(e["variant"])].append(e)

    lines.append("SUMMARY")
    for variant, es in by_variant.items():
        good = [e for e in es if e["success"]]
        lengths = [e["frames"] for e in good]
        colours = Counter(e["scene"].get("cube_colour") for e in good)
        line = f"  {':'.join(x for x in variant if x)}: {len(good)} recorded, {len(es) - len(good)} failed"
        if lengths:
            line += f", {min(lengths)}-{max(lengths)} frames"
        if variant[0] == "S2":
            line += f", starts {dict(Counter('cube in drawer' if e['cube_in_drawer'] else 'cube on table' for e in good))}"
        lines.append(line + f", colours {dict(colours)}")
        for e in es:
            if not e["success"]:
                failed = [k for k, v in e.get("checks", {}).items() if not v]
                lines.append(f"    failed seed {e['seed']}: checks {failed}, pushed {e.get('pushed')}, "
                             f"hold {e.get('hold_motion')}/{e.get('hold_moved')}")
    all_colours = Counter(e["scene"].get("cube_colour") for e in log if e["success"])
    lines.append(f"  cube colours overall: {dict(all_colours)}")

    for e in (e for e in log if e["success"]):
        i = e["episode_index"]
        rows = ep_col == i
        state, action = cols["observation.state"][rows], cols["action"][rows]
        subtask = cols["subtask_index"][rows].reshape(-1)
        variant = tuple(e["variant"])
        if not (cols["variant_index"][rows] == RZ.VARIANTS.index(variant)).all():
            flags.append(f"#{i}: variant_index does not match the log")
        m, fl = check_episode(e, state, action, subtask)
        with np.load(root / "sim" / e["states"]) as f:
            frames = dict(f)
        m.update(cube_visibility(e, frames, rand, a.stride))
        if m["cube_px_224_median"] < MIN_PX_224:
            fl.append(f"cube small in the overview at 224: median {m['cube_px_224_median']} px")
        if m["cube_contrast"] is not None and m["cube_contrast"] < MIN_CONTRAST:
            fl.append(f"cube colour ({e['scene'].get('cube_colour')}) close to the {m['cube_contrast_vs']}: "
                      f"contrast {m['cube_contrast']}")
        flags += [f"#{i} {':'.join(x for x in variant if x)} seed {e['seed']}: {f}" for f in fl]
        per_episode.append({"episode_index": i, "variant": e["variant"], "seed": e["seed"],
                            "cube_colour": e["scene"].get("cube_colour"), **m})
        if not a.no_sheets:
            name = f"{i:03d}_{'_'.join(x for x in variant if x)}_{e['seed']}.png"
            contact_sheet(ds, ds.meta.episodes[i], i, e, m.get("switch_frame"), out / "sheets" / name)
        print(f"#{i} checked", flush=True)

    lines.append("ACTIONS AND IMAGES, per variant (max over episodes; cube pixels: min of the medians)")
    for variant in by_variant:
        ms = [m for m in per_episode if tuple(m["variant"]) == variant]
        if not ms:
            continue
        contrast = [m["cube_contrast"] for m in ms if m["cube_contrast"] is not None]
        lines.append(f"  {':'.join(x for x in variant if x)}: jump {max(m['max_jump'] for m in ms):.3f} rad, "
                     f"still {max(m['longest_still_s'] for m in ms):.2f} s, "
                     f"cube visible {min(m['cube_visible_frac'] for m in ms):.2f}-{max(m['cube_visible_frac'] for m in ms):.2f} of frames, "
                     f"{min(m['cube_px_median'] for m in ms)} px ({min(m['cube_px_224_median'] for m in ms)} at 224), "
                     f"contrast {min(contrast) if contrast else '-'}")
    by_colour = defaultdict(list)
    for m in per_episode:
        if m["cube_contrast"] is not None:
            by_colour[m["cube_colour"]].append(m["cube_contrast"])
    lines.append("  cube contrast with its least different surround, by colour (median): "
                 + ", ".join(f"{c} {np.median(v):.2f}" for c, v in sorted(by_colour.items(), key=lambda kv: np.median(kv[1]))))

    report = [f"FLAGS ({len(flags)})"] + [f"  {f}" for f in flags] + [""] + lines
    (out / "report.txt").write_text("\n".join(report) + "\n")
    (out / "episodes.json").write_text(json.dumps(per_episode, indent=1))
    print("\n".join(report))
    print(f"\nwritten: {out / 'report.txt'}, {out / 'episodes.json'}" + ("" if a.no_sheets else f", {out / 'sheets'}/"))


if __name__ == "__main__":
    main()
