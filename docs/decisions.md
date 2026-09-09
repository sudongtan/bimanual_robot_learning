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

## Pending decisions

- **Policy family** — ACT vs SmolVLA vs Pi0.5. Plan of record: ACT first to
  prove the end-to-end pipeline, then swap in SmolVLA. _Not yet committed._
- **Pouring simulation** — free-body droplet spheres vs a state flag.
- **Perception** — privileged sim state vs a learned pose estimator.
- **Core Ultra Series 2/3 access** — unresolved. Blocks 20 rubric points and
  the Phase 6 deliverable. Options: Intel DevCloud / Tiber, borrowed or
  purchased Core Ultra NUC. Must land before Phase 6.
