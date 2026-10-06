# Design decisions

Every design decision in the project — the MuJoCo scene (D1–D23), recording
(D24–D27), evaluation, policies and training (D28–D29, D31–D33) and task scope (D30) — with what
was chosen, the alternatives, why, who decided, and the evidence. "User" =
decided by the project owner; "proposed" = chosen by the assistant while building, not
explicitly approved. Bugs and investigations are in
[debug-log.md](debug-log.md); the current state of the scene is in
[scene.md](scene.md).

Status: **active**, **superseded** (by a later decision), or **open**.

| ID | Decision | Status |
|---|---|---|
| D1 | World frame and units | active |
| D2 | Table size and height | active |
| D3 | Physics options | active |
| D4 | SO-101 model source | active |
| D5 | Two arms by attaching one model twice | active |
| D6 | Arm bases on the tabletop | active |
| D7 | Arm placement and orientation | active |
| D8 | Home pose | active |
| D9 | Recycling bins as pull-out drawers under the front edge | active |
| D10 | Bin handle 5 cm off the panel | active |
| D11 | Two bins at x = ±0.10; middle bin removed | active |
| D12 | Plastic bin left, metal bin right | active |
| D13 | Items from YCB scans | active |
| D14 | Item collision shapes | active |
| D15 | Item textures at 1024² | active |
| D16 | Cameras | active |
| D17 | Build the scene per episode with only the items in use | active (replaces D17a) |
| D17a | All items always in the scene, unused ones parked | superseded by D17 |
| D18 | One shadow-casting light | active |
| D19 | Simplified visual meshes for the arms | **open** — tested, not adopted |
| D20 | Record states first, render images afterwards | active |
| D21 | Floor reflections | **open** |
| D22 | Lights and floor appearance | active |
| D23 | Keep shadows | active |
| D24 | Dataset format: LeRobot, one frame per control step (50 fps) | active |
| D25 | What a recorded episode contains | active |
| D26 | Default cameras and image size for rendered datasets | **open** |
| D27 | Seeds 0–199 recorded, 0–49 in T0's training set; evaluation seeds 1000–1049 | active |
| D28 | Evaluate with LeRobot's evaluator; the environment registered as a LeRobot env | active |
| D29 | BC baseline: own single-step network (`bc_mlp`) as a LeRobot policy plugin | active |
| D30 | T0 = T1 with only the can on the table, as the pipeline smoke test; T1 keeps its other items | active |
| D31 | VLA: fine-tune LeRobot's SmolVLA (`lerobot/smolvla_base`) | **open** (proposed) |
| D32 | OpenVINO: convert the LeRobot policy network with `ov.convert_model`; pre/post-processing stays in LeRobot | **pending** (deployment left out for now) |
| D33 | T0 training: BC and ACT, 20k steps, batch 8, on this Mac's GPU | active (proposed) |

---

### D1 World frame and units
- **Choice:** origin on the floor under the table centre, z up, metres,
  radians. Arms on the −y edge (the front); the overview camera looks from there.
- **Why:** MuJoCo's defaults; a fixed convention keeps every position in the
  docs and code comparable.
- **Decided by:** proposed (step 1).

### D2 Table size and height
- **Choice:** 0.70 × 0.60 m top, surface at z = 0.315, four legs.
- **Why:** same dimensions as physicalai's tested dual-SO-101 garment scene.
- **Evidence later:** the arms reach only up to y ≈ 0.18 at grasp height, so
  the far 12 cm of the table is out of reach; a bigger table would not help
  (reach, not table size, limits the usable area).
- **Decided by:** proposed.

### D3 Physics options
- **Choice:** `timestep 0.002`, `implicitfast`, elliptic cones,
  `impratio 10`, CG solver — physicalai's tested values, declared in the scene.
- **Why:** an attached model's `<option>` is not inherited (MuJoCo prints
  "Attach conflict … keeping parent value"); elliptic cones and impratio > 1
  reduce slip in grasps (MuJoCo docs).
- **Decided by:** proposed.

### D4 SO-101 model source
- **Choice:** physicalai's single-arm `so101.xml`, unmodified, with meshes from
  TheRobotStudio SO-ARM100 (byte-identical to physicalai's Git LFS copies).
- **Alternatives:** TheRobotStudio's `so101_new_calib.xml`.
- **Why:** same joints, ranges and servo parameters, but physicalai replaced
  the gripper's mesh collision (convex hulls, which fill the jaw faces) with
  boxes and fingertip spheres, and adds a wrist camera.
- **Decided by:** this model — proposed; the SO-101 is the robot this project
  is built around (plan of record).

### D5 Two arms by attaching one model twice
- **Choice:** `<attach model="so101" body="base" prefix="left_|right_"/>`.
- **Alternatives:** physicalai's hand-written two-arm files; `<include>`.
- **Why:** one arm definition instead of two hand-synced copies; `include`
  allows a file only once; MuJoCo docs: "Prefer attach to include".
- **Decided by:** proposed.

### D6 Arm bases on the tabletop
- **Choice:** bases rest on the table (mount z = 0.3174, because the base mesh
  extends 2.4 mm below its origin).
- **Alternatives:** physicalai's z = 0.40 (bases floating 8.5 cm above the
  table) or a visible stand.
- **Why:** looks right in the demo video; reach is about the same (0.47 vs
  0.45 m at grasp height); costs a re-tuned home pose.
- **Decided by:** user (2026-10-05).

### D7 Arm placement and orientation
- **Choice:** arm A (`left_`) base at (−0.23, −0.25), arm B (`right_`) at
  (+0.23, −0.25), arm B rotated 180° about z.
- **Why:** physicalai's dual layout. Consequence: the arms face each other
  along x; arm B is rotated, not mirrored, so its joint signs and some reach
  limits differ from arm A's.
- **Decided by:** proposed (layout copied); "arm A = left" from the user's
  command examples.

### D8 Home pose
- **Choice:** grippers ~12 cm above the table, between each base and the table
  centre, angled down; arm B = arm A with `shoulder_pan` and `wrist_roll`
  negated. Generated by `scripts/find_home_pose.py`.
- **Why:** collision-free and holds (0.0004 rad); all-zeros is unusable (the
  upright arms collide).
- **Decided by:** proposed.

### D9 Recycling bins as pull-out drawers under the front edge
- **Choice:** drawers built from boxes inside the `table` body, each
  0.16 × 0.20 × 0.12 m, on a `slide` joint (0–0.18 m toward −y, damping 3,
  frictionloss 0.2), with handle, hook and interior sites.
- **Alternatives:** aloha_sim's drawer cabinet (a 24 cm countertop unit, not a
  bin).
- **Why:** user's story ("like in real life, the arm pulls it out like a
  drawer"); inside the table body they never collide with the table
  (parent–child filter).
- **Decided by:** under-table drawers — user; construction — proposed.

### D10 Bin handle 5 cm off the panel
- **Choice:** handle bar 5 cm off the front panel (was 2.5 cm).
- **Why:** at 2.5 cm no arm could reach behind the handle of a closed bin
  (gripper hits the table edge); sweep 1–6 cm: ≥ 4 cm works for every arm
  (debug-log #9).
- **Decided by:** proposed.

### D11 Two bins at x = ±0.10; middle bin removed
- **Choice:** plastic at x = −0.10, metal at +0.10.
- **History:** three bins at −0.20 / 0 / +0.20 (user added the middle one);
  then the user asked to remove the middle bin and move the two to the
  middle. ±0.10 is the closest pair where both bins can be opened by their
  arm and both arms can drop into both (sweep ±0.09 … ±0.12).
- **Caveat:** the move was partly motivated by a wrong IK result (debug-log
  #12): at ±0.20 arm A could drop into the metal bin at 44° tilt; at ±0.10 it
  needs 15°.
- **Decided by:** user (2026-10-05).

### D12 Plastic bin left, metal bin right
- **Decided by:** user. Consequence: only arm A can open the plastic bin and
  only arm B the metal bin.

### D13 Items from YCB scans
- **Choice:** 20 YCB items imported with `scripts/import_ycb.py`: metal (4
  cans), plastic (mustard bottle, cup `065-e`), food waste (8 fruits), to wash
  (plate, bowl, mug, fork, spoon, knife). Masses from the YCB paper, Table II.
- **Left out:** chips can (cardboard), bleach/Windex bottles (not dinner
  waste), pitcher, sponge; wine glass (no scan on the server).
- **Alternatives:** RoboCasa (pack downloads 0.8–5.8 GB), simple shapes.
- **Decided by:** YCB, "all items that fit" — user; which items fit — proposed.

### D14 Item collision shapes
- **Choice:** cans = exact cylinder from the scan's bounds; everything else =
  the scan's convex hull. Visual scan in group 2, collision in group 3.
- **Why:** a convex object loses nothing as a primitive; hulls are the quick
  default.
- **Known limit:** hollow items (mug, bowl, cup) collide as solids — they
  cannot hold liquid until rebuilt.
- **Decided by:** proposed.

### D15 Item textures at 1024²
- **Choice:** downscale YCB textures from 4096² (8.9 MB) to 1024² (~1 MB).
- **Why:** repo size; 1024² is ample for 224–640 px renders.
- **Decided by:** proposed.

### D16 Cameras
- **Choice:** `overview` at (0, −0.75, 0.95), aimed at the table
  (`mode="targetbody"`), fovy 58°; `left_wrist` / `right_wrist` from the arm
  model. `<map znear="0.002"/>`.
- **Why:** overview position copied from physicalai's rig, approximately; the
  near plane must stay closer than the jaws (debug-log #8).
- **Open point:** the arms look small in the overview image; reframing was
  deferred until the scene is complete.
- **Decided by:** proposed.

### D17 Build the scene per episode with only the items in use
- **Choice:** the scene file holds floor, table, bins, arms, cameras — no
  items. On `reset`, the environment attaches the task's items with MjSpec and
  compiles; the model is cached per item set.
- **Replaces D17a:** all 20 items always in the scene, unused ones parked on
  the floor out of view.
- **Why:** parking caused the near-plane bug, 311k extra triangles per
  render pass, 140 unused state values, and the keyframe-padding trap. A
  per-episode build costs 48–164 ms (vs 207–516 ms for all 20 items) against a
  ~16 s episode. robosuite offers the same pattern (`hard_reset`: "re-loads
  model, sim, and render object upon a reset call").
- **Consequences:** the full state size depends on the items in play (the
  policy's state input — 12 joint positions — does not); the scene file alone
  shows no items (use `scripts/view_episode.py`).
- **Decided by:** user approved (2026-10-06).

### D17a All items always in the scene, unused ones parked — superseded
- **Choice (2026-10-05):** one fixed model with all 20 items; `reset` moves
  unused ones to a row on the floor behind the overview camera.
- **Decided by:** proposed, not discussed with the user.

### D18 One shadow-casting light
- **Choice:** only the `top` light casts shadows; the `key` light still
  lights the scene.
- **Why:** each shadow-casting light adds a full drawing pass per camera.
  Measured (3 cameras, 224², reflections off, parked items excluded): two
  shadow lights 83 ms/step, one 56 ms/step.
- **Trade-off:** one shadow per object instead of two.
- **Decided by:** user approved (2026-10-06).

### D19 Simplified visual meshes for the arms — open (tested, not adopted)
- **Proposal:** decimate the arms' visual meshes; collision and inertias are
  unaffected (no arm mesh can collide; all 7 arm bodies have an explicit
  `<inertial>`).
- **Why considered:** the arms' visual meshes are ~400k triangles each, the
  largest share of what is drawn (debug-log #19).
- **Tool:** `scripts/simplify_meshes.py` (trimesh + fast-simplification,
  quadric decimation; dev dependencies).
- **Result (asked for 10 % of faces, got 29 %: some meshes stop early):**
  rendering 3 cameras at 224² went from 50 to 25 ms/step. Pixels differing by
  more than 30/255: wrist cameras 0.1–0.2 %, overview 1.8 %. But a close-up
  ([image](img_mesh_simplify_closeup.png): original, simplified, differences
  in red) shows faceted surfaces, lost detail (the mounting plate's holes) and
  jagged small parts — visible in a demo video.
- **Not adopted:** with D20 (render from saved states, in parallel) the
  original meshes are fast enough for now. Options if rendering becomes the
  bottleneck: a milder reduction; or keep both mesh sets in different geom
  groups and render the simplified one only for policy inputs.
- **Decided by:** user approved trying it (2026-10-06); adoption pending the
  user, given the visual result.

### D20 Record states first, render images afterwards
- **Choice:** run the expert without rendering, save the full state at every
  step, then render camera images from the saved states in parallel.
- **Why:** physics is 1.8 ms per step; rendering is the cost. The same demos
  can be re-rendered with other cameras or image sizes without re-running the
  expert. robomimic does the same (`dataset_states_to_obs.py`).
- **Decided by:** user approved (2026-10-06).

### D21 Floor reflections — open
- **Current:** floor material `reflectance="0.2"` (from physicalai's scene).
- **Measured:** reflections roughly double render time (shadows on: 141 →
  113 ms/step when turned off, other settings unchanged).
- **Options:** keep; turn off for recorded images only; make the floor matte
  in the scene. Background randomisation can vary the floor's colour/texture
  either way.
- **Decided by:** not decided.

### D22 Lights and floor appearance
- **Choice:** two lights (`key` spot at (1, 1, 3.5), `top` directional at
  (0, 0, 3.5)), headlight off, checkered blue floor, gradient skybox — copied
  from physicalai's scene.
- **Decided by:** proposed. To be randomised (lighting, background) later.

### D23 Keep shadows
- **Choice:** shadows stay on in rendered images.
- **Alternatives:** shadows off (25.7 vs 80.6 ms/step).
- **Why:** a policy has to cope with lighting changes — lighting is one of the
  factors that most reduces imitation-policy success when it changes (THE
  COLOSSEUM, arXiv 2402.08191; Xie et al., arXiv 2307.03659) — and with
  shadows off, moving a light only changes brightness.
- **Decided by:** user (2026-10-06).

---

## Recording decisions (phase 2)

### D24 Dataset format: LeRobot, one frame per control step (50 fps)
- **Choice:** LeRobot v3 dataset (`LeRobotDataset`, video-encoded images),
  fps = the environment's control rate, 50 Hz; every control step is a frame.
- **Why:** the project trains BC/ACT/VLA with LeRobot; the dataset's actions
  must be at the rate the policy acts. ACT's real-robot data was also 50 Hz.
- **Alternative:** a lower rate (LeRobot's real-robot guide records at 30 fps):
  smaller dataset, but the policy would then act at that rate too.
- **Dependency:** `lerobot[dataset,training]` (`dataset` adds `datasets`,
  pandas, pyarrow, torchcodec; `training` adds `accelerate` and `wandb` for
  `lerobot-train`, debug-log #29). On this Mac torchcodec cannot load (no
  system FFmpeg) and LeRobot falls back to pyav — works, but prints a long
  warning (debug-log #23).
- **Decided by:** LeRobot format — plan of record; 50 fps — proposed.

### D25 What a recorded episode contains
- **Choice:** `observation.state` = the 12 arm joint positions; `action` =
  the 12 joint targets sent at that step; one video per camera; `task` = the
  instruction. Recording continues 0.5 s after success, then stops. Only
  successful episodes go into the dataset (failures are kept in `data/raw/`).
- **Why:** state/action are what the policy will see and output; the short
  tail after success shows the end state; ACT's simulated datasets were
  "50 successful demonstrations".
- **Raw data:** `data/raw/<task>/seed_<n>.npz` holds the full simulator state
  per step, so images can be re-rendered (D20). `data/` is gitignored.
- **Decided by:** proposed.

### D26 Default cameras and image size — open
- **Current default:** all three cameras (`overview`, `left_wrist`,
  `right_wrist`) at 224×224, floor reflections on (D21 open).
- **Why open:** asked earlier, not answered; with D20 it is a rendering
  setting (`render_dataset.py --cameras … --size …`), changeable without
  re-recording.

### D27 Seeds 0–199 recorded, 0–49 in T0's training set; evaluation seeds 1000–1049 — active
- **Choice:** demonstrations are recorded from seeds 0–199 (200 episodes,
  first 0–49; expanded 2026-10-06, by the user's decision, to cover the can
  patch corner near arm A's base, debug-log #28). T0's training dataset uses
  seeds 0–49 only: "since this is a smoke test, it does not need to be
  perfect" (user, 2026-10-06); 50–199 stay recorded for later. Every policy (and
  the expert) is evaluated on seeds 1000–1049, 50 episodes, the same for
  every phase.
- **Why:** must not overlap the training seeds; must be a consecutive range
  (LeRobot's evaluator uses `--seed` + 0, 1, 2, …, D28); 50 episodes because
  LeRobot's policy guide says "Use `n_episodes ≥ 50` per suite for stable
  success-rate estimates" (huggingface.co/docs/lerobot/bring_your_own_policies).
  Starting at 1000 leaves room to record more training seeds (200–999) later.
- **Alternative:** 10 episodes (1000–1009) — faster, but one failure moves the
  rate by 10 points.
- **Decided by:** proposed (a setting with a conventional default; change it
  here if needed — earlier results must then be re-run).

### D28 Evaluate with LeRobot's evaluator — active
- **Choice:** register the environment with LeRobot (`sim/lerobot_plugin`:
  an `EnvConfig` named `waste_sort`, gym ids `gym_waste_sort/<task>`) and
  evaluate trained policies with `lerobot-eval`. The scripted expert, which
  is not a LeRobot policy, is evaluated by `scripts/evaluate_expert.py`
  through LeRobot's `rollout()` (same env creation, seeding, stepping and
  success reading) and reported in the same `eval_info.json` layout.
- **Alternative:** our own evaluation script (full control, e.g. extra
  metrics), rejected: it duplicates LeRobot, needs its own policy loading,
  and `lerobot-train` can only evaluate during training through a registered
  env.
- **Consequences:**
  - observation keys renamed to LeRobot's simulation names: `state` →
    `agent_pos`, `images` → `pixels`; info `success` → `is_success`; the
    env exposes `task_description` (the instruction).
  - seeds are consecutive from a start seed (`--seed` in `lerobot-eval`,
    `eval_policy(start_seed=…)`), so the evaluation seeds (D27) must be a
    range.
  - episodes still end at the first success step (debug-log #25 open point).
  - environments are wrapped in LeRobot's `FreezeAfterEpisodeEnd`
    (debug-log #26).
- **Evidence:** `lerobot/scripts/lerobot_eval.py` (`rollout`, `eval_policy`
  requires a `PreTrainedPolicy`), `lerobot/envs/configs.py` (`EnvConfig`,
  `register_subclass`), `lerobot/configs/parser.py` (`load_plugin`,
  `--env.discover_packages_path`), `lerobot/envs/utils.py`
  (`preprocess_observation` key names) — LeRobot 0.6.1 as installed.
- **Decided by:** user (option A, 2026-10-06).

### D29 BC baseline: our own single-step network as a LeRobot policy plugin — active
- **Choice:** `policies/lerobot_policy_bc_mlp/` — policy type `bc_mlp`:
  each camera image through one shared ResNet-18 (ImageNet weights, frozen
  batch-norm, global-average-pooled, 512 values per camera), concatenated
  with the 12 joint positions, MLP 512–512 (ReLU, dropout 0.1) → the next
  12-value action. L1 loss, mean/std normalisation, AdamW (lr 1e-4,
  backbone 1e-5) — the same as LeRobot's ACT, so BC vs ACT isolates action
  chunking and the CVAE. Installed as an editable path dependency; LeRobot
  imports `lerobot_policy_*` distributions at startup.
- **Alternative:** LeRobot's ACT with `chunk_size=1`, `use_vae=false` (no new
  code).
- **Evidence:** the ACT paper's BC-ConvMLP baseline (arXiv 2304.13705);
  LeRobot "Adding a Policy" guide; `lerobot/policies/factory.py` (naming
  conventions), `policies/act/` (encoder, loss, processors).
- **Decided by:** user ("write a simple neural network to do BC",
  2026-10-06); architecture details proposed.

### D30 T0: T1 with only the can on the table, as the pipeline smoke test — active
- **Choice:** T0 (`t0_can_to_metal_bin`) is the simplest case of T1: the same
  instruction and success check, with only the tomato soup can on the table.
  It is used to take the whole pipeline through once. T1
  (`can_to_metal_bin`, README) is the real task: the same instruction with
  other items on the table besides the can (user, 2026-10-06); built later.
- **History:** at scene step 5 the user decided on several items per episode
  ("a different subset per episode", scene.md A3), and the
  basic scene had three items on the table (tomato can, tuna can, plastic
  cup). When the first task was built (2026-10-05), the assistant left out all but
  the can to get the first grasp and expert working — stated as a fact, not
  raised as a decision, not recorded here. That task was then called T1. On
  2026-10-06 the user decided: what was built is T0, a simplified T1 for the
  pipeline smoke test; the original T1 is untouched. Code name, data folders
  and tests were renamed from `can_to_metal_bin` to `t0_can_to_metal_bin`
  (debug-log #32).
- **Consequence:** a policy trained only on T0 never has to tell the target
  from other items, and its instruction carries no information;
  distractor-free training is the setting THE COLOSSEUM and LIBERO-Plus show
  to be brittle. T0 results measure the pipeline, not those abilities.
- **Open (for T1):** which other items, how many, how they are placed (with a
  clearance around the can for the expert's grasp).
- **Decided by:** user (2026-10-06).

### D31 VLA: fine-tune LeRobot's SmolVLA — open (proposed)
- **Proposal:** phase 5 fine-tunes `lerobot/smolvla_base` (450 M parameters;
  LeRobot's own VLA, installed with the `smolvla` extra) on the task dataset
  with `lerobot-train --policy.path=lerobot/smolvla_base`. Cameras are mapped
  with `--rename_map`: overview → camera1, left_wrist → camera2,
  right_wrist → camera3 (debug-log #34).
- **Why:** it runs through the same LeRobot training and evaluation as BC and
  ACT; LeRobot's guide recommends ≈ 50 episodes, the size of T0's dataset;
  the joint state (12) and actions (12) fit its 32-value inputs.
- **Alternatives:** LeRobot's pi0 / pi0.5 (`pi` extra; larger), GR00T, X-VLA.
- **Cost:** the guide's example recipe, 20 k steps at batch 64, takes ≈ 4 h on
  one A100. Shorter fine-tunes run on a Mac: Hugging Face's blog says SmolVLA
  can "train on a single consumer GPU, or even a MacBook", and an M5 MacBook
  example ran 3,000-step fine-tunes at batch 4–16 in ≈ 20–85 min each
  (huggingface.co/Twu31/smolvla-cross-embodiment-mps). Steps and batch for
  T0 to be chosen from a timed run on this Mac (debug-log #34).
- **Evidence:** huggingface.co/docs/lerobot/smolvla;
  `lerobot/policies/smolvla/configuration_smolvla.py`.
- **Decided by:** proposed — which VLA is the user's choice.

### D32 OpenVINO export: convert the LeRobot network with `ov.convert_model` — pending
- **Status:** deployment (phase 8) is left out for now (user, 2026-10-06);
  the code below exists and was tested on test checkpoints only. A Jetson
  route (ONNX → TensorRT) was discussed and also left out.
- **Choice:** `scripts/export_openvino.py` converts the policy network
  (normalised state + images → normalised action chunk) with OpenVINO's
  PyTorch route (`ov.convert_model`, `ov.save_model`); LeRobot's
  pre/post-processors from the same checkpoint do the normalisation at run
  time. `scripts/evaluate_openvino.py` runs the compiled model in the
  simulator through the expert's evaluation loop (`scripts/evaluate_expert.py`).
- **Alternative:** Intel's Physical AI Studio export (`physicalai export
  --policy physicalai.policies.ACT --backend openvino`). It exports policies
  trained with its own trainer (`.ckpt`); nothing in its documentation covers
  LeRobot checkpoints, and the `physicalai` LeRobot plugin covers robot
  hardware, not policies.
- **Scope:** single-pass policies (bc_mlp, ACT) — tested on 2-episode test
  checkpoints: exact at f32 (debug-log #33), 2.4–2.8× faster than PyTorch
  on this Mac's CPU. SmolVLA (tokeniser, iterative sampling) is not handled.
- **Evidence:** docs.openvino.ai "Converting PyTorch Models";
  github.com/openvinotoolkit/physicalai (packages/);
  skills.sh/open-edge-platform/physical-ai-studio (export workflow).
- **Decided by:** proposed.

### D33 T0 training: BC and ACT, 20k steps, batch 8, on this Mac's GPU — active (proposed)
- **Choice:** `bc_mlp` and ACT each trained for 20,000 steps at batch 8
  (≈ 4 passes over T0's 40,598 frames), `--policy.device=mps`, a checkpoint
  every 5,000 steps, no evaluation during training; one after the other
  (`scripts/train_policy.sh`, defaults = these settings). Other settings are
  each policy's defaults (ACT: chunk 100, lr 1e-5; bc_mlp: D29).
- **Why:** LeRobot's defaults (batch 8, 100k steps; its guide: "training
  should take several hours") would take ≈ 4.7 h (bc_mlp) + 8.3 h (ACT) here —
  measured 0.17 and 0.30 s/step at 224×224. T0 is a smoke test (D30), so a
  fifth of that; the same budget for both keeps BC vs ACT fair. A run can be
  continued with `--resume=true` if 20k is too short.
- **Evidence:** timed 60-step runs (2026-10-06); LeRobot "Imitation Learning
  on Real-World Robots" guide; `lerobot/configs/train.py` defaults.
- **Decided by:** proposed (the user asked to train BC and ACT).
