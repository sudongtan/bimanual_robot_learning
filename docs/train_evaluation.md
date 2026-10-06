# Training and evaluation

How policies are trained on a task's demonstrations and how every policy —
scripted or learned — is measured. The same for every task; the walkthrough
for T0, with its results, is [pipeline.md](pipeline.md). Where the data comes
from: [tasks_data.md](tasks_data.md). Commands run from the repo root after
`uv sync --extra local` ([pipeline.md](pipeline.md), before you start).
References: "Dnn" = a decision in [design-decisions.md](design-decisions.md),
"#nn" = an entry in [debug-log.md](debug-log.md).

Everything here uses LeRobot: `lerobot-train` trains, `lerobot-eval` (or
LeRobot's episode runner) evaluates. Our code connects the task environment
and our own BC network to it.

---

## 1. The policies

Three learners, each adding one idea to the one before: `bc_mlp` imitates
step by step; ACT keeps its inputs but predicts chunks of actions; SmolVLA
adds the instruction text and pretraining on other robots' data. All three
imitate the scripted expert, which is listed first as the reference.

| Policy | What it is | Learns from | Code |
|---|---|---|---|
| Scripted expert | plans each episode from the true simulator state (tasks_data.md A3) | — | `sim/experts.py` |
| `bc_mlp` — behaviour cloning | each camera image through a shared ResNet-18 (ImageNet weights), joined with the 12 joint positions; an MLP predicts the next action. One action per step, no memory — the ACT paper's "BC-ConvMLP" baseline. 12 M parameters | the task's dataset | our LeRobot policy plugin, [`policies/lerobot_policy_bc_mlp/`](../policies/lerobot_policy_bc_mlp/) (D29) |
| `act` — ACT | LeRobot's Action Chunking Transformer: predicts the next 100 actions at once, with a CVAE during training. Same image encoder, loss (L1) and normalisation as `bc_mlp`, so BC vs ACT shows what chunking adds. 52 M parameters | the task's dataset | LeRobot, built in |
| `smolvla` — VLA | LeRobot's SmolVLA, fine-tuned from the pretrained `lerobot/smolvla_base`: also reads the instruction text, starts from a model pretrained on other robot data. 450 M parameters, 100 M trained (vision encoder frozen) | the task's dataset + pretraining | LeRobot, built in (D31, proposed) |

The learned policies see only what a real robot would: the three camera
images, the 12 joint positions, and (SmolVLA) the instruction. They output the
12 joint targets.

## 2. Training

[`scripts/train_policy.sh`](../scripts/train_policy.sh) `<task> <policy>
[steps] [batch] [device]` runs LeRobot's `lerobot-train` on
`data/lerobot/<task>/` and writes `outputs/train/<task>/<policy>/`
(checkpoints every 5,000 steps and `last/`) plus the log
`outputs/train/<task>/<policy>.log`. It refuses to start if that folder
exists, so a finished run is never overwritten; a run is continued with
LeRobot's `--resume=true`.

```bash
scripts/train_policy.sh t0_can_to_metal_bin bc_mlp                # <task> <policy> [steps] [batch] [device]
scripts/train_policy.sh t0_can_to_metal_bin act
scripts/train_policy.sh t0_can_to_metal_bin smolvla STEPS BATCH  # fine-tunes lerobot/smolvla_base
# continue a run:
uv run python -m lerobot.scripts.lerobot_train \
    --config_path=outputs/train/<task>/<policy>/checkpoints/last/pretrained_model/train_config.json --resume=true
```

- **`bc_mlp`, `act`:** trained from scratch (`--policy.type`); the image
  encoder starts from ImageNet weights.
- **`smolvla`:** fine-tuned from `lerobot/smolvla_base` (`--policy.path`,
  downloaded on first use). Its pretrained config names the cameras
  `camera1`–`camera3`, so ours are renamed (overview → camera1, left_wrist →
  camera2, right_wrist → camera3; debug-log #34).

**Settings** (D33): 20,000 steps, batch 8, on this Mac's GPU (`mps`) —
a fifth of LeRobot's default length, chosen for T0's smoke test; the same
budget for every policy keeps the comparison fair. Measured at 224×224:
`bc_mlp` 0.17 s/step (≈ 1 h), ACT 0.30 s/step (≈ 1 h 40 min). SmolVLA on this
Mac: to be timed (an M5 MacBook example fine-tuned it for 3,000 steps at
batch 4–16 in ≈ 20–85 min; debug-log #34).

**No evaluation during training** (`--env_eval_freq=0`): one evaluation
episode is slow on this Mac, and choosing a checkpoint by its score on the
evaluation seeds would make the reported score optimistic — those seeds would
have been used for selection. The last checkpoint is the one evaluated.

## 3. Evaluation — the yardstick

### 3a. Preparing evaluation (phase 2)

The measuring is set up before anything is trained, in phase 2 together with
the demonstrations (README), so every policy — the expert, BC, ACT, the VLA —
is scored on exactly the same test. Six steps:

**1. Choose the evaluation seeds: 1000–1049** (D27). One seed is one starting
scene (tasks_data.md A2), so the test set is a seed range: 50 starting scenes
no learned policy has trained on (training uses seeds 0–199). They are the
default of the evaluation script (3b).

**2. Check that the success rule is fit for scoring.** The rule is the task's
own success check, the same one used when recording (tasks_data.md A1); an
episode ends at the first success or at the time limit. For recording it only
had to work for the expert; for scoring it must not credit an item that falls
in and rolls out again — success fires while the item is still falling.
Checked: the expert run 3 s past success on 50 seeds, the can always ended at
rest inside the bin (#25; guarded by `test_success_still_holds_after_settling`).
Open: a learned policy might drop an item on the bin's edge.

**3. Connect our environment to LeRobot's evaluator.** Learned policies are
evaluated by LeRobot's `lerobot-eval`, which loads a checkpoint with its
normalisation, runs the episodes and counts successes.
[`sim/lerobot_plugin/`](../sim/lerobot_plugin/__init__.py) makes our
environment available to it: it registers each task as a Gymnasium
environment `gym_waste_sort/<task>` with its time limit and defines a LeRobot
environment type `waste_sort` (cameras, image size); the flag
`--env.discover_packages_path=sim.lerobot_plugin` loads it (D28). For this the
environment reports LeRobot's names (`agent_pos`, `pixels`,
`info["is_success"]`, `task_description`), and a wrapper keeps an episode's
last frame instead of resetting at once (#26).

**4. Build an evaluator for the expert.** `lerobot-eval` only loads LeRobot
policy checkpoints, and the expert is not one.
[`scripts/evaluate_expert.py`](../scripts/evaluate_expert.py) wraps the expert
behind LeRobot's policy interface and runs it in the same `waste_sort`
environment (without cameras — the expert does not look) through LeRobot's
own episode runner (`rollout()`, the one `lerobot-eval` uses), counting
successes the same way.

**5. Score the expert.** It is the first entry on the scoreboard and the
ceiling for the learners, and it checks that every evaluation seed can be
solved — a failure there is a bug in the expert, not a hard seed for the
learners. For T0 the first run found one (48/50, a grasp bug, #28); fixed, the
expert scores 50/50 (pipeline.md).

**6. Test the chain with a trained checkpoint.** Before real training, a tiny
`bc_mlp` (20 steps on 2 episodes) went through training, a saved checkpoint
and `lerobot-eval` in our environment: it runs end to end (#30).

### 3b. Running an evaluation

[`scripts/evaluate.sh`](../scripts/evaluate.sh) `<task> <policy>` is one command for
every policy: it picks the evaluator (step 3 or 4), uses the evaluation seeds by default, and
writes `outputs/eval/<task>/<policy>/`:

| File | What |
|---|---|
| `eval_info.json` | the evaluator's full result — per episode and overall; the layout differs between `lerobot-eval` and the expert script |
| `summary.json` | the same for every policy: task, policy, seeds, number of episodes, number and percent successful |
| videos | the first episodes |

It refuses to evaluate a policy whose training has not finished (`last/`
already exists mid-training, debug-log #35) and never overwrites a result.

```bash
scripts/evaluate.sh t0_can_to_metal_bin expert           # <task> <policy> [first_seed] [n_episodes] [device]; ~80 s
scripts/evaluate.sh t0_can_to_metal_bin bc_mlp
scripts/evaluate.sh t0_can_to_metal_bin act
scripts/evaluate.sh t0_can_to_metal_bin smolvla
scripts/evaluate.sh t0_can_to_metal_bin act 1000 1       # one episode, e.g. to time it (own folder)
```

Under the hood: `expert` → `scripts/evaluate_expert.py`; the others →
`uv run python -m lerobot.scripts.lerobot_eval --policy.path=<checkpoint>
--env.type=waste_sort --env.discover_packages_path=sim.lerobot_plugin
--env.task=<task> --eval.n_episodes=50 --seed=1000 …` (`python -m`, not the
`lerobot-eval` shortcut, so the repo root is on the import path and `sim` is
found). The log goes to `outputs/eval/<task>/<policy>.log`. A message
`objc … AVFFrameReceiver is implemented in both … cv2 … av` in it is two
bundled copies of FFmpeg (OpenCV and PyAV) in one process — no effect
observed.

## 4. Deployment (pending)

Deployment is left out for now (user, 2026-10-06). What exists for later:
[`scripts/export_openvino.py`](../scripts/export_openvino.py) converts a
trained `bc_mlp` or ACT network to OpenVINO (`ov.convert_model`; D32), checks
it gives the same output as PyTorch (exact at f32; this Mac's CPU defaults to
f16, debug-log #33) and times both;
[`scripts/evaluate_openvino.py`](../scripts/evaluate_openvino.py) runs the
converted model on the same yardstick. Tested only on 2-episode test
checkpoints. A Jetson route (ONNX → TensorRT) was discussed and also left out.

```bash
uv run python scripts/export_openvino.py \
    --checkpoint outputs/train/t0_can_to_metal_bin/act/checkpoints/last/pretrained_model \
    --out outputs/openvino/t0_act                     # --device CPU|GPU|NPU|AUTO, --precision f32|f16|bf16
uv run python scripts/evaluate_openvino.py --export outputs/openvino/t0_act \
    --task t0_can_to_metal_bin --seed 1000 --n-episodes 50
```

`export_openvino.py` writes `model.xml`/`.bin` and `export_info.json` (parity
at f32 and at the run precision, time per call against PyTorch);
`evaluate_openvino.py` writes `outputs/eval/<task>/openvino_<export>/`.

## 5. Checks and outputs

**Check:** `uv run pytest tests/test_bc_mlp.py` (LeRobot finds the plugin; a
training step and an action step; the OpenVINO export matches PyTorch) and
`tests/test_env.py::test_lerobot_creates_the_env` (LeRobot builds and reads
our environment).

**Outputs** (none in git): `outputs/train/<task>/<policy>/` (checkpoints) and
`<policy>.log`; `outputs/eval/<task>/<policy>/` (results, videos) and
`<policy>.log`; `outputs/openvino/<name>/` (pending phase).

## 6. Open points

- Evaluation speed: one 25 s episode took up to 13 min on this Mac's CPU while
  other jobs ran (debug-log #30); to be measured on `mps` with nothing else
  running.
- Score success at the end of the episode, not at the first success
  (debug-log #25) — not needed for the expert; to revisit with learned
  policies.
- Which VLA (D31).
