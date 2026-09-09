# Decisions log

Running record of choices and their rationale. Doubles as README material.

## 2026-09-09 — Phase 0

**Python 3.12, not 3.10/3.11.** `CLAUDE.md` proposed `>=3.10,<3.12`; LeRobot
0.6.x declares `requires-python = ">=3.12"` upstream, so that window resolves
to nothing. Upper bound `<3.14`: lerobot caps torch at `<2.12`, and cp314
wheels are not available across all target platforms in that range.

**`openvino`, not `openvino-dev`.** `openvino-dev` was deprecated and
discontinued after the 2024.6 release (latest `openvino-dev` is 2024.6.0 vs
`openvino` 2026.3.1). Model conversion is now `ovc` / `openvino.convert_model`,
shipped inside the runtime package. `CLAUDE.md`'s dependency list is stale here.

**Benchmark dir is `openvino_bench/`, not `openvino/`.** Verified: a bare
directory named `openvino/` on `sys.path` does *not* shadow the installed
package — a namespace portion loses to a regular package found later on the
path. But the moment it gains an `__init__.py` (which it would, since these
directories are declared as wheel packages) it becomes a regular package and
wins, and `import openvino` from the repo root silently resolves to our own
empty module. Renamed rather than relying on never adding that file.

**Conflicting `local` / `train` extras.** Declared via `tool.uv.conflicts` so
uv refuses to install both into one environment — they resolve different torch
builds (PyPI CPU/MPS for Mac, cu128 for the Linux GPU box). Single `uv.lock`
covers both.

## 2026-09-09 — spec read, checked against CLAUDE.md

Read `docs/challenge_spec.pdf` (5pp). Rubric matches CLAUDE.md exactly
(30/20/15/20/10/5) and so do the five deliverables. Two divergences worth
recording:

**Core Ultra is "preferred", not the only permitted target.** CLAUDE.md calls
Core Ultra Series 2/3 a hard, non-negotiable requirement. The spec's Platform
Requirements say: *target system* is "Intel Core Ultra AI PC / NUC, **or a local
execution environment with Intel CPU and iGPU acceleration**"; Core Ultra
Series 2/3 is listed separately as *preferred deployment hardware*. The hard
requirement is "run the final simulation on Intel hardware". The i7-8565U
(Intel CPU + UHD 620 iGPU) therefore satisfies the platform floor and gives a
real fallback — but the OpenVINO rubric line names Core Ultra 2/3 explicitly,
so the fallback likely costs points rather than zeroing them. Keep sourcing
Core Ultra; stop treating it as an existential blocker. Spec wins over
CLAUDE.md here, per CLAUDE.md's own ground-truth rule.

**Asset list is short a bottle, and the cup is a mug.** The spec's dual-arm
example is "one arm holding a mug while the other pours **from a bottle**", and
the sample command is "pick up the mug with arm B, pour water into the mug with
arm A". CLAUDE.md's Phase 1 asset list has plate, cup, 2 spoons, 2 forks — no
bottle. Scene needs: table, drawer, plate, mug, bottle, 2 spoons, 2 forks,
water proxy.

**SO-101 model sourced and verified.** TheRobotStudio/SO-ARM100
`Simulation/SO101`, Apache 2.0, so vendoring the meshes into `sim/assets/` with
the license notice is fine. Loads and steps stably under mujoco 3.12.0. Using
the `new_calib` variant. Details in the untracked notes.

## Pending decisions

- **Policy family** — ACT vs SmolVLA vs Pi0.5. Plan of record: ACT first to
  prove the end-to-end pipeline, then swap in SmolVLA. _Not yet committed._
- **Pouring simulation** — free-body droplet spheres vs a state flag.
- **Perception** — privileged sim state vs a learned pose estimator.
- **Core Ultra Series 2/3 access** — unresolved. Blocks 20 rubric points and
  the Phase 6 deliverable. Options: Intel DevCloud / Tiber, borrowed or
  purchased Core Ultra NUC. Must land before Phase 6.
