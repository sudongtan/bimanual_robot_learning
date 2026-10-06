# How to run

Every command, grouped by what you want to do. Run them from the repo root.
What the scene contains and why: [building-a-mujoco-scene.md](building-a-mujoco-scene.md);
tasks: [README](../README.md) and [tasks.md](tasks.md).

| I want to… | Section |
|---|---|
| set up the environment | [1](#1-set-up) |
| check that everything works | [2](#2-check-that-everything-works) |
| look at the scene | [3](#3-look-at-the-scene) |
| watch an episode of a task | [4](#4-watch-an-episode) |
| smoke-test a task's scripted expert | [5](#5-smoke-test-a-scripted-expert) |
| record expert demonstrations (phase 2) | [6](#6-record-demonstrations) |
| add objects, change the arms, regenerate keyframes | [7](#7-change-the-scene) |
| find an output file | [8](#8-where-outputs-go) |
| understand a warning | [9](#9-messages-that-look-like-errors-but-are-not) |

All scripts that take `--task` accept any task in `sim/waste_env.py` `TASKS`
and pick the matching scripted expert from `sim/experts.py` `EXPERTS`.
Implemented so far: `can_to_metal_bin` (T1).

---

## 1. Set up

```bash
uv sync --extra local
```

Python 3.12, MuJoCo, LeRobot (with its dataset tools), OpenVINO. Test and
asset-tool dependencies (`pytest`, `trimesh`, `fast-simplification`) are in the
`dev` group, which `uv sync` installs by default.

## 2. Check that everything works

```bash
uv run pytest                    # everything, ~25 s
uv run pytest -m "not slow"      # skip the IK reach searches and expert runs
```

`tests/test_scene.py` checks the scene (structure, physics settings, keyframes,
bins, every item standing on the table, cameras, reach for task T1);
`tests/test_env.py` checks the environment, the expert and replaying saved
states. To test a modified copy of the scene:
`SCENE_XML=sim/scenes/<copy>.xml uv run pytest tests/test_scene.py`.

## 3. Look at the scene

The scene file holds the table, bins, arms and cameras — **no items** (items
are added per episode, see section 4).

**Render all cameras into one image:**

```bash
uv run python scripts/render_scene.py sim/scenes/dinner_table.xml outputs/renders/scene.png --keyframe home
uv run python scripts/render_scene.py sim/scenes/dinner_table.xml outputs/renders/open.png --keyframe bins_open
open outputs/renders/scene.png
```

Prints model sizes, whether the state stays finite, and the contact count.

**Interactive viewer:**

```bash
uv run python -m mujoco.viewer --mjcf=sim/scenes/dinner_table.xml
```

It opens with every joint at zero — **the arms start upright and collide**.
First: left panel → **Simulation** → **Key** slider to `home` (0) →
**Load key**. Then **Space** to run.

| To… | Do |
|---|---|
| Rotate / pan / zoom | left-drag / right-drag / scroll |
| Run, pause, single step | **Space**, **→** |
| Reset (back to all-zeros — load `home` again after) | **Backspace** |
| Reload the XML after editing it | **Ctrl+L** |
| Look through `overview` / `left_wrist` / `right_wrist` | **]** / **[**; **Esc** = free camera |
| Show camera / light markers | **Q** / **Z** |
| Show collision geometry / hide visual meshes | **3** / **2** |
| Show contact points / forces | **C** / **F** |
| Move an arm joint | right panel → **Control** sliders |
| See bin positions | right panel → **Joint**: `plastic_bin_slide`, `metal_bin_slide` (0 closed, 0.18 open) |
| Push or drag an object | double-click it, then **Ctrl** + right-drag |
| Help overlay | **F1** |

Keys from MuJoCo's [simulate shortcut reference](https://mujoco.readthedocs.io/en/stable/programming/samples.html#saSimulateShortcuts).

## 4. Watch an episode

An episode's scene includes its items, placed by the task for the given seed.

```bash
uv run python scripts/view_episode.py --task can_to_metal_bin --seed 7            # you drive the viewer
uv run mjpython scripts/view_episode.py --task can_to_metal_bin --seed 7 --expert # watch the scripted expert
```

`--expert` needs `mjpython` on macOS (installed with MuJoCo). In the plain
viewer, **Load key** resets the arms and bins but puts items at the world
origin — restart the script to get the episode back.

## 5. Smoke-test a scripted expert

```bash
uv run python scripts/smoke_test.py --task can_to_metal_bin                  # seeds 0-4, per-phase log
uv run python scripts/smoke_test.py --task can_to_metal_bin --seeds $(seq 0 49) --quiet
open outputs/smoke/can_to_metal_bin/seed_0.mp4
```

Prints, per seed: success or failure, each item's start and end position,
how far each bin is open, and a warning if the grasped item moved before the
grasp. Saves an overview-camera video of the first seed (`--video-all` for
all). Expected for T1: every seed succeeds at ≈ 15.7 s.

## 6. Record demonstrations

Two steps (scene-decisions D20): run the expert without rendering and save
the states, then render camera images from the saved states into a LeRobot
dataset.

**Step 1 — run the expert, save states (≈ 1 s per episode):**

```bash
uv run python scripts/record_states.py --task can_to_metal_bin --seeds $(seq 0 49)
```

Writes `data/raw/can_to_metal_bin/seed_<n>.npz` (full simulator state, arm
joint positions and actions per step, success flag).

**Step 2 — render into a LeRobot dataset:**

```bash
uv run python scripts/render_dataset.py --task can_to_metal_bin                 # 3 cameras, 224x224
uv run python scripts/render_dataset.py --task can_to_metal_bin --cameras overview left_wrist --size 96 96
```

Writes `data/lerobot/can_to_metal_bin/`. Options: `--workers N` (parallel
rendering, default = CPU cores − 2), `--limit N` (first N episodes only),
`--no-reflections`, `--include-failures`, `--raw` / `--out` (input and output
folders).

⚠ **`render_dataset.py` deletes its output folder first.** For a quick test,
write somewhere else:

```bash
uv run python scripts/record_states.py --seeds 100 101 --out outputs/smoke_raw
uv run python scripts/render_dataset.py --raw outputs/smoke_raw --out outputs/smoke_ds --workers 2
```

**Check a dataset:**

```bash
uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/can_to_metal_bin', root='data/lerobot/can_to_metal_bin')
print(ds.num_episodes, 'episodes,', ds.num_frames, 'frames at', ds.fps, 'fps')
f = ds[0]; print({k: tuple(v.shape) for k, v in f.items() if hasattr(v, 'shape')}); print(f['task'])"
```

Videos per camera are in `data/lerobot/can_to_metal_bin/videos/`.

Expected output of the check for the T1 dataset: `50 episodes, 40598 frames
at 50 fps`, each camera image `(3, 224, 224)`, state and action `(12,)`, and
the T1 instruction as the task.

Measured on this Mac (M-series, 10 cores), 50 episodes of T1:

| Step | Time | Size |
|---|---|---|
| 1 `record_states.py` | 40 s | 6.9 MB |
| 2 `render_dataset.py` (8 workers, 3 cameras at 224², shadows + reflections on) | 29 min (23 frames/s) | 346 MB, almost all video |

## 7. Change the scene

| To… | Do |
|---|---|
| Add a YCB object | add its id to `ITEMS` in `scripts/import_ycb.py` (mass from YCB Table II, collision `cylinder` or `hull`), then `uv run python scripts/import_ycb.py <id>`. Tasks refer to it by name; nothing to add to the scene file |
| Put an item on the table for a task | list it in the task's `items` and place it in the task's `place` function (`sim/waste_env.py`) |
| Add a scripted expert for a task | write the expert class in `sim/experts.py` (with `grasped_item` and `grasp_phase`) and add it to `EXPERTS` |
| Change anything that adds, removes or moves a joint in the scene file | regenerate the keyframes: `uv run python scripts/find_home_pose.py`, paste the printed `<keyframe>` block over the scene's |
| Move the arms or change the home pose | change the mounts, then `uv run python scripts/find_home_pose.py --target X Y Z` (left gripper target; the right arm is mirrored) |
| Simplify the arms' display meshes (scene-decisions D19, not adopted) | `uv run python scripts/simplify_meshes.py --keep 0.1` writes `sim/assets/so101/assets_visual/` |

After any change: `uv run pytest`.

`import_ycb.py` downloads into `outputs/ycb_cache/` (~260 MB for everything).
`find_home_pose.py` checks its result for contacts and a 3 s hold before
printing it.

## 8. Where outputs go

| Path | What | In git? |
|---|---|---|
| `outputs/renders/` | scene images from `render_scene.py` and experiments | no |
| `outputs/smoke/<task>/` | smoke-test videos | no |
| `outputs/ycb_cache/` | YCB downloads | no |
| `data/raw/<task>/` | recorded states (`record_states.py`) | no |
| `data/lerobot/<task>/` | LeRobot datasets (`render_dataset.py`) | no |
| `sim/assets/objects/ycb/` | imported YCB models | yes |

## 9. Messages that look like errors but are not

| Message | Meaning |
|---|---|
| `WARNING: Attach conflict when attaching 'so101' … keeping parent value` | the arm file's solver settings lose to the scene's — intended (scene-decisions D3) |
| Long `torchcodec` traceback ending in `Falling back to 'pyav'` | no system FFmpeg; LeRobot uses pyav instead, recording works (debug-log #23) |
| `Svt[info]: …` lines | the AV1 video encoder's log while writing a dataset |
| A new `MUJOCO_LOG.TXT` in the repo root | MuJoCo writes its warnings there; safe to delete |
| Viewer: arms fold up and collide when you press Space | no keyframe loaded — load `home` first (section 3) |
