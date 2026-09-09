# CLAUDE.md — Intel Physical AI Online Challenge: Bimanual VLA Table-Setting

When answering questions, reference latest documentation or reputable repo. don't answer from memory, dont reinvent things, dont make up things. 


## Project Context

This repo implements a submission for Intel's Physical AI Online Challenge
("Setting Up a Dinner Table" track): a bimanual VLA pipeline controlling two
simulated SO-101 arms in MuJoCo, driven by natural-language instructions +
camera observations, optimized for inference on Intel Core Ultra Series 2/3
via OpenVINO.

**Engineer profile:** experienced ML engineer, moderate RL/VLA background —
skip basic explanations of training loops, gradient descent, etc. Do flag
robotics-specific gotchas (coordinate frames, sim-to-real-ish randomization,
action space conventions) since those are less familiar territory.

**Ground truth spec:** see `docs/challenge_spec.pdf` (copy the original PDF
into the repo under this path before starting — Claude should treat that file,
not this one, as the source of truth for scoring/deliverables if they ever
diverge).

---

## Hardware reality check (current setup)

**Available now:**
- MacBook (Apple Silicon or Intel, native MuJoCo support either way)
- Windows/Linux laptop with an **Intel Core i7-8565U** — an 8th-gen mobile
  CPU (2019, Whiskey Lake). This does **not** qualify as the "Intel Core
  Ultra Series 2/3" hardware the challenge requires for the final benchmark
  and demo — no NPU, not the target generation. Fine for development, not
  valid for the submission deliverable.

**Not yet available, needs sourcing:**
- A real **Intel Core Ultra Series 2/3** system (NUC or AI PC) for Phase 6
  (OpenVINO benchmark) and the final demo. This is a hard, non-negotiable
  requirement for the submission — not something the project needs
  technically, but something the *hackathon* requires, since Intel is
  scoring 20/100 points on OpenVINO + Core Ultra optimization specifically.
  **Action item, resolve early:** check whether Intel provides DevCloud
  access / loaner hardware to challenge entrants; otherwise arrange
  borrowed/purchased access. Do not let this slip to the final week.
- A CUDA GPU for policy training at reasonable speed. Neither the Mac nor
  the i7-8565U has one. **Plan: rent cloud GPU compute** (RunPod, Lambda,
  Vast.ai, etc.) for Phase 2 training runs — the spec explicitly allows any
  training hardware, cloud or local.

**What this means for sequencing (confirmed against the phase plan below):**
- **Phases 0–5 are fully doable on the Mac (+ cloud GPU for training),
  no Intel Core Ultra hardware needed yet.** MuJoCo has native, fully
  supported Mac builds (including Apple Silicon) — unlike Gazebo (Linux/ROS-
  first, Mac support is unreliable) or Isaac Sim (NVIDIA/Omniverse-only, no
  Mac path at all, hard blocker). This is one reason MuJoCo is a good fit
  for this challenge regardless of the Intel angle.
- Rollouts/data collection can run on Mac CPU (or i7-8565U CPU) — slower
  rendering than a GPU box, acceptable for one-time dataset generation.
  Actual policy training (backprop) happens on the rented cloud GPU against
  the already-rendered dataset — these two steps don't need to be on the
  same machine.
- **De-risk Phase 6 early anyway:** install OpenVINO on the Mac or the
  i7-8565U now and test the ONNX → IR conversion path on your trained model
  as soon as you have one. This shakes out export bugs (dynamic shapes,
  unsupported ops) well before you have hands on qualifying hardware. Note
  OpenVINO will run generic CPU inference on non-Intel-Ultra hardware for
  this testing purpose, but the *actual* benchmark numbers and final demo
  must be captured on the real Core Ultra Series 2/3 system — CPU-only
  numbers from the i7-8565U or Mac are not a valid substitute for
  submission.
- Bottom line: **start the project now on what you have.** Time-box sourcing
  Core Ultra access to land before Phase 6 begins, not before Phase 0.

---

## Non-negotiable deliverables (map directly to the 100-pt rubric)

| # | Deliverable | Rubric weight |
|---|---|---|
| 1 | Reproducible GitHub repo (setup, deps, MuJoCo assets, train/eval/infer code) | → Technical Quality (10) |
| 2 | Reproducible MuJoCo sim w/ domain randomization + eval config | → contributes to Task Completion (30) + Robustness (15) |
| 3 | Intel inference benchmark script (latency, throughput, device, precision) | → OpenVINO/Intel (20) |
| 4 | Demo video, 10 randomized seeds, command+scene+outcome legible | → Robustness (15) + Task Completion (30) |
| 5 | Technical README / architecture summary | → Technical Quality (10) + Innovation (5) |

Total addressable: Task Completion 30, VLA/Multi-modal Reasoning 20,
Robustness 15, OpenVINO/Intel 20, Reproducibility 10, Innovation 5.

**Priority order if time-constrained:** Task Completion > OpenVINO conversion
(don't skip — it's 20 pts and often forgotten until the end) > Robustness >
Reasoning polish > Innovation. A working single-seed pipeline with a real
OpenVINO benchmark beats a fancier policy with no Intel deployment story.

---

## Phase 0 — Environment & Repo Scaffolding

**Hardware:** Mac or i7-8565U laptop — either is fine, this phase is just
repo/dependency setup. No training or Intel-specific hardware needed yet.

1. Create repo structure:
   ```
   repo/
     docs/                  # challenge_spec.pdf, architecture.md, decisions log
     sim/                   # MuJoCo scene XML, asset meshes, task/env wrappers
     data/                  # collected demos (gitignored, use DVC or just note path)
     policy/                # training code, model defs, configs
     inference/             # runtime pipeline: perception -> reasoning -> action
     openvino/              # conversion scripts, IR models, benchmark script
     eval/                  # seed configs, success-rate harness, video capture
     scripts/               # setup.sh, run_train.sh, run_eval.sh, run_bench.sh
     README.md
     CLAUDE.md
   ```
2. **Dependency management: use `uv`**, not conda/plain pip. MuJoCo ships
   proper precompiled wheels for Mac (Apple Silicon + Intel) and Linux, so
   conda's binary-package handling isn't needed here — `uv` resolves fast
   from PyPI and gives a cross-platform lockfile, which matters since you're
   splitting work between the Mac/i7-8565U (local) and a cloud GPU instance
   (training).

   Single `pyproject.toml`, with optional-dependency groups instead of
   separate requirements files:
   ```toml
   [project]
   name = "bimanual-vla-challenge"
   requires-python = ">=3.10,<3.12"   # confirm against LeRobot's current
                                       # supported range before locking
   dependencies = [
       "lerobot",
       "mujoco",
       "gymnasium",
       "ikpy",
       "numpy",
   ]

   [project.optional-dependencies]
   local = ["openvino", "openvino-dev", "nncf"]   # sim dev + early OpenVINO
                                                    # export testing (Phase 6
                                                    # de-risk pass)
   train = ["wandb", "tensorboard"]                # cloud GPU training extras

   [tool.uv.sources]
   # pin CPU vs CUDA torch per environment — point at the correct PyTorch
   # index for CPU (Mac/laptop) vs the cloud instance's CUDA version
   ```
   - **Mac / i7-8565U:** `uv sync --extra local` → CPU-only torch, sim +
     rollout generation + early OpenVINO export testing.
   - **Cloud GPU instance:** `uv sync --extra train` with the CUDA torch
     index set for that instance's CUDA version → actual policy fine-tuning.
   - Commit `uv.lock` — uv supports resolving a single lockfile across
     platforms, so judges (or you, later) get correct reproducibility with
     `uv sync` regardless of their OS, no separate Mac/Linux files to juggle.
   - Confirm rendering works headless where needed (`MUJOCO_GL=egl` or
     `osmesa`), since the Intel Core Ultra box may not share the same GPU
     stack as your training machine.
3. Repo README setup instructions should point to `uv sync --extra local`
   as the reproduction path for sim/eval/demo — that's what judges will run.
   The `train` extra is documented for completeness but isn't something
   judges need to execute, since they're evaluating your trained checkpoint
   and inference pipeline, not re-training from scratch.
4. [pending]Confirm access to target Intel hardware NOW, not later — see "Hardware
   reality check" above. Current setup (Mac + i7-8565U) does not qualify;
   resolve Core Ultra Series 2/3 access in week 1 via Intel DevCloud, a
   borrowed/purchased Core Ultra NUC, or equivalent. This blocks 20 rubric
   points and the deployment deliverable — do not leave it until the end.
   Nothing else in Phase 0-5 is blocked by this, so proceed in parallel.
5. `git init`, set up `.gitignore` for large model checkpoints and raw video,
   and decide early whether checkpoints are pulled via HF Hub / Git LFS /
   external link in the README (judges will clone this repo — don't make it
   50GB).

**Checkpoint:** `uv run python -c "import mujoco, lerobot, openvino"` runs
clean on the local (`--extra local`) environment; `mjpython` (or your
headless render check) produces a frame from a trivial scene.

---

## Phase 1 — MuJoCo Scene: Dual SO-101 + Dinner Table

**Hardware:** Mac (native MuJoCo support, including Apple Silicon) or the
i7-8565U laptop. Pure CPU scene-building/debugging work — no GPU needed.

1. Get SO-101 MJCF/URDF. LeRobot's repo or the SO-101 hardware repo
   (Hugging Face / TheRobotStudio) should have a MuJoCo-compatible model or
   a URDF you convert with `mujoco`'s URDF importer. Verify joint limits and
   actuator gains against the real hardware spec even though you're
   sim-only — the challenge explicitly cares about physically plausible
   manipulation, not a toy sim.
2. Instantiate two SO-101 arms in one scene, spaced for a shared workspace
   over a table. Get the frame/base offsets right first — bimanual bugs are
   disproportionately caused by wrong relative transforms between arm bases.
3. Add task assets: table, drawer (with a hinge/slide joint, not fixed
   geometry — you need it openable), plate, cup, 2 spoons, 2 forks, a
   pourable "water" proxy (MuJoCo doesn't do real fluids well — common
   workaround: a small number of free-body "droplet" spheres, or fake it
   with a state flag if simulation-realism isn't scored, only task
   completion signal is). Decide and document this pragmatically.
4. Define the MDP wrapper (Gymnasium-style env):
   - Observation: RGB from 1-2 camera viewpoints (wrist + scene, or overhead
     + scene — pick based on what your chosen policy expects), proprioception
     (joint pos/vel) for both arms, optionally object poses for debugging
     (not necessarily fed to the policy).
   - Action space: per-arm joint or end-effector delta commands (decide
     based on the policy family in Phase 2 — ACT typically likes joint-space
     or EE-space depending on config).
   - Reward/success signal: written as a scripted checker (drawer opened
     beyond threshold, correct items on correct table zones, cup upright and
     "filled" flag set) — you need this for auto-eval across 10 seeds, not
     just visual judgment.
5. **Domain randomization hooks** (build this in now, not bolted on later
   for Phase 5): object mass/friction ranges, initial pose jitter, table
   texture/lighting randomization, camera pose jitter. Expose these as env
   config parameters with a fixed-seed sampler so eval runs are
   reproducible.

**Checkpoint:** a random/scripted policy can open the drawer and touch each
object at least once across several randomized resets without physics
blowing up (check for interpenetration, NaN states).

---

## Phase 2 — Data Collection & Policy Training

**Hardware:** split across two steps. Data collection / rollout rendering
(steps 1-3 below) runs fine on the Mac or i7-8565U CPU — slower rendering
than a GPU box but acceptable for one-time dataset generation. Actual policy
fine-tuning (step 5, the backprop step) needs a **rented cloud GPU**
(RunPod/Lambda/Vast.ai — an RTX 3060-4090-class card is enough for ACT,
lean toward 12-24GB VRAM if going SmolVLA). These two steps don't need to be
on the same machine — render locally, upload the dataset, train in the
cloud.

**Environment:** local side runs on `uv sync --extra local` (Phase 0). On
the cloud GPU instance, run `uv sync --extra train` with the CUDA torch
index set for that instance's CUDA version — same `pyproject.toml`/`uv.lock`,
different extra. **Dataset handoff between the two:** push the collected
LeRobot dataset to the Hugging Face Hub via LeRobot's dataset tooling
(`push_to_hub`) — cleanest, versioned option, cloud side just pulls it down
after `uv sync`. If you'd rather not host it publicly, `rsync`/`scp` to the
cloud box or stage through S3/GCS if the rental provider supports mounting.

You have two realistic paths — pick one and commit, don't try both:

**Path A — Scripted/motion-planning demonstrations (recommended given your
timeline and RL-not-expert status):** write a scripted policy (inverse
kinematics + waypoints) that solves each subtask (open drawer, pick X, place
X, hand-off, pour) reliably in sim. Roll it out under domain randomization to
generate a demonstration dataset in LeRobot's dataset format. This sidesteps
training a from-scratch RL policy for contact-rich bimanual tasks, which is
genuinely hard, and gets you to imitation learning faster.

**Path B — Teleoperation:** only pursue if you have a working teleop rig
(e.g. leader-follower SO-101 setup or a spacemouse/VR controller mapped into
sim). Higher-quality data but a real time sink to set up — likely not worth
it for a sim-only online challenge unless you already have this infra.

Steps (Path A):
1. Write per-subtask waypoint scripts using MuJoCo's IK utilities or a
   lightweight IK solver (`ikpy`, or MuJoCo's native `mj_jac`-based solver).
2. Chain subtasks into full episode rollouts that match the language
   commands you'll support (start with a fixed command template set, expand
   later — e.g. "open the top drawer", "pick up the {object} with arm
   {A|B}", "place it on the table", "pour {liquid} into the {container}").
3. Log episodes in LeRobot dataset format (images, proprioception, actions,
   language annotation per episode). Target order-of-magnitude: hundreds of
   episodes per subtask minimum for imitation learning to generalize across
   your randomization ranges — scale up if success rate is poor.
4. Pick your policy: **ACT** is the safer choice for a non-VLA-expert on a
   deadline (well-documented in LeRobot, robust for bimanual/dual-arm
   setups, doesn't require a full VLM backbone). **SmolVLA** if you want the
   multi-modal-reasoning rubric points to look stronger (it's explicitly
   language-conditioned and smaller than Pi0.5, easier to fine-tune and
   later export to OpenVINO). Recommendation: **start with ACT to get an
   end-to-end pipeline working, then swap in SmolVLA once the scaffolding
   (sim, data, eval, OpenVINO export) is proven** — don't let policy choice
   block everything else.
5. Fine-tune using LeRobot's training scripts, pointing at your dataset.
   Track: task success rate per subtask, not just training loss.

**Checkpoint:** trained policy solves the full command sequence on the
training distribution (non-randomized or lightly randomized) at a
non-trivial success rate (define your own bar, e.g. >50%, before moving on —
don't proceed to robustness eval on a policy that can't do the task once).

---

## Phase 3 — Multi-Modal Reasoning Layer

**Hardware:** Mac or i7-8565U. This is Python logic wired around your
trained policy checkpoint — no training compute needed, and inference at
this stage (dev/debug only, not the benchmark) is light enough for CPU.

This is the layer that turns a raw language instruction + scene into a
sequence of subtask calls to your Phase 2 policy (or, if using a native VLA
like SmolVLA, this may partially collapse into the policy itself).

1. Decide your architecture honestly:
   - **Simple/robust option:** a lightweight instruction parser (rule-based
     or a small LLM call) that decomposes a compound command into an ordered
     list of subtask primitives + arguments (object, arm, target). Your
     trained policy executes each primitive. State tracking = "which
     primitive are we on" + scripted preconditions (don't try to place the
     plate before the drawer is open).
   - **End-to-end option:** feed the raw instruction + image stream directly
     into SmolVLA/Pi0.5 and let the model handle sequencing internally. Higher
     ceiling on the reasoning rubric line, higher risk given your stated VLA
     experience level and timeline.
   - Given "experienced ML but not VLA expert," the hybrid (parser +
     learned low-level policy) is the pragmatic default — document this
     trade-off explicitly in your README, since "Innovation" is scored
     separately and reviewers will read intent, not just outcome.
2. Implement scene-state tracking: object detection/pose estimation from
   camera frames (can be privileged sim state for speed, or a real
   perception model if you want the reasoning score higher — again, be
   explicit in the README about which you chose and why).
3. Implement replanning/adaptation: if a grasp fails or an object isn't
   where expected, the system should retry or replan rather than blindly
   executing a fixed action sequence. Even a simple "check post-condition,
   retry subtask up to N times" loop scores meaningfully better than open-loop
   execution.

**Checkpoint:** given a compound natural-language command, the system
produces a correct ordered subtask plan and executes it end-to-end in sim.

---

## Phase 4 — Bimanual Coordination

**Hardware:** Mac or i7-8565U. Same as Phase 3 — this is coordination logic
on top of the sim + policy, CPU-only dev work.

1. Implement explicit **hand-off**: arm A moves object to a rendezvous pose,
   arm B closes gripper on it, arm A releases — sequence this with
   force/contact checks if available, or a fixed dwell time + gripper-state
   check if not.
2. Implement at least one **complementary dual-arm action**: e.g. arm A
   holds the cup steady while arm B pours — model this as two concurrent
   subtask executions with a shared synchronization point, not fully
   independent single-arm control loops.
3. Add **collision-aware sequencing**: simplest robust approach is workspace
   partitioning / turn-taking with a shared "who's moving through the
   central zone right now" lock, rather than full joint trajectory
   collision-checking, unless you have time budget for the latter.

**Checkpoint:** the demo sequence in the spec ("open drawer... plate with arm
A... mug with arm B... pour with arm A") runs without arms colliding or
deadlocking, across several randomized seeds.

---

## Phase 5 — Robustness Evaluation

**Hardware:** Mac or i7-8565U for running the eval harness (repeated
MuJoCo rollouts + policy inference, CPU is adequate since this isn't a
speed-sensitive step — just runs slower than on a GPU box, budget time
accordingly). If CPU inference across 10 seeds feels too slow, optionally
point the harness at your cloud GPU instance for the inference call, but
it's not required. **Not** the Core Ultra system yet — that's Phase 6 only.

1. Finalize the 10 randomized seed configs (object placement, weight,
   friction, shape variant if you have multiple object meshes, lighting,
   background). Store these as versioned config files in `eval/seeds/`.
2. Run full evaluation harness: for each seed, execute the full command
   sequence, log success/failure per subtask and overall, log video.
3. Compute and report: overall success rate across 10 seeds, per-subtask
   breakdown, and qualitative failure analysis (what breaks, and why) — this
   failure analysis is worth including in the README even though it's not a
   separate rubric line; it signals technical maturity for the
   "Innovation & Technical Demonstration" score.
4. If success rate is poor on held-out randomization, the fix is almost
   always more diverse training data (Phase 2) rather than tweaking the
   reasoning layer — diagnose before "fixing."

**Checkpoint:** documented success rate across all 10 seeds with per-seed
video captured, ready to cut into the final demo video.

---

## Phase 6 — OpenVINO Conversion & Intel Benchmarking

**Hardware:** two sub-stages, different requirements.
- Steps 1-2 (ONNX export, IR conversion, NNCF quantization) can be **de-risked
  early on the Mac or i7-8565U**, using the `local` uv extra (already has
  `openvino`/`nncf` from Phase 0) — OpenVINO installs and runs generic CPU
  inference on non-Intel-Ultra hardware, which is enough to shake out export
  bugs (dynamic shapes, unsupported ops) before you have the real chip.
- Steps 3-5 (the actual benchmark script, the "preserve system behavior"
  re-eval, and the final demo run) **must happen on the real Intel Core
  Ultra Series 2/3 system.** CPU-only numbers from the i7-8565U or Mac are
  not valid substitutes for the submitted benchmark — this is the hard
  hardware requirement flagged in "Hardware reality check" above. Get this
  hardware lined up before you reach this phase, not during it.

Do not leave this to the last few days — model export quirks (unsupported
ops, dynamic shapes from vision transformers) are common and take longer
than expected.

1. Export trained policy to ONNX first (`torch.onnx.export`), then convert
   to OpenVINO IR (`ovc` / `openvino.convert_model`). Vision-transformer
   backbones (common in SmolVLA/ACT) sometimes need static input shapes —
   fix batch size / image resolution for export if dynamic shapes fail.
2. Quantize with NNCF (post-training quantization to INT8 as a first pass;
   only pursue quantization-aware training if PTQ tanks accuracy).
3. Benchmark script (`openvino/benchmark.py`) must report, per the rubric:
   latency (mean/p95), throughput, device used (CPU/iGPU/NPU — test at least
   two), and precision (FP32 vs INT8). Use `openvino.runtime.Core` with
   explicit `AUTO`/`CPU`/`GPU`/`NPU` device strings — don't just rely on
   default device selection, the rubric wants device utilization reported.
4. **Re-run the Phase 5 success-rate eval with the OpenVINO-optimized model**
   to confirm the "preserve system behavior" requirement — quantization
   silently degrading task success is a common failure mode judges will
   likely check for.
5. Run the actual end-to-end MuJoCo sim + inference on the target Intel Core
   Ultra hardware for the final demonstration, not just the benchmark
   script in isolation.

**Checkpoint:** benchmark script runs standalone on Intel hardware and
produces a clean latency/throughput/device/precision report; optimized model
success rate is within an acceptable margin of the unoptimized model (state
your margin explicitly in the README).

---

## Phase 7 — Packaging & Submission

**Hardware:** Mac or i7-8565U for writing/editing the README and cutting the
demo video. The video content itself (footage of the sim + inference running)
needs to have been captured on the Core Ultra system during Phase 6 — this
phase is assembly, not new hardware-bound work.

1. **README / architecture summary** — cover: architecture diagram
   (perception → reasoning/planning → bimanual policy → action), policy
   choice and why, training approach, robustness methodology, OpenVINO
   optimization results (numbers, not just "we optimized it"), Intel
   hardware mapping, known limitations.
2. **Repo reproducibility pass**: fresh clone, follow your own README
   top-to-bottom on a clean environment (or at minimum a clean venv) —
   fix anything that doesn't just work. Pin all dependency versions.
3. **Demo video**: script it to hit every line in the "Recommended
   Demonstration Sequence" from the spec explicitly — command + scene shown,
   perception/inference visibly running, hand-off/complementary action
   clearly shown, task completion shown, 10-seed summary shown, benchmark
   results shown on-screen. Judges are scoring against a checklist; make
   each item easy to find in the video (timestamps in the description help).
4. Final check against the deliverables table at the top of this file —
   confirm all 5 are present and linked from the README.

---

## Working notes / decisions log

Keep a running log here (or in `docs/decisions.md`) of choices made and why,
especially: policy family (ACT vs SmolVLA vs Pi0.5), pouring simulation
approach, perception approach (privileged state vs learned), and any rubric
trade-offs made under time pressure. This doubles as README material later.

## Timeline sanity check

Rough estimate for one experienced engineer, sim-only, building on existing
LeRobot examples: **4-6 weeks** of focused part-time work, front-loaded on
Phase 0-2 (sim + data + baseline policy) and Phase 6 (OpenVINO) since both
have hidden yak-shaving. Don't let policy architecture experimentation eat
the time budget needed for robustness eval and Intel deployment — both are
worth more rubric points (35 combined) than a marginally better policy.