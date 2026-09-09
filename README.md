# Bimanual VLA — Table Setting (Intel Physical AI Online Challenge)

Two simulated **SO-101** arms in **MuJoCo**, driven by natural-language
instructions plus camera observations, with inference optimized via
**OpenVINO** for Intel Core Ultra Series 2/3.

> **Status: Phase 0 (scaffolding).** The environment, task, policy and
> benchmark are not implemented yet. This README documents the reproduction
> path as it lands; sections marked _(pending)_ are placeholders.

---

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone <this-repo> && cd bimanual_vla
uv sync --extra local
uv run python scripts/check_env.py
```

`--extra local` is the path judges should use: sim, evaluation, demo, and
OpenVINO export/benchmark. It installs a CPU/MPS torch and does not need a GPU.

### Training environment (not needed to reproduce results)

```bash
uv sync --extra train    # on a CUDA Linux box; pulls torch from the cu128 index
```

`local` and `train` are declared as **conflicting extras** in
[pyproject.toml](pyproject.toml) — they resolve different torch builds and must
not be installed into the same environment. The CUDA index is pinned to
`cu128`; change it to match the rented instance's CUDA version.

---

## Repository layout

| Path | Contents |
|---|---|
| [docs/](docs/) | `challenge_spec.pdf` (source of truth), architecture notes, [decisions log](docs/decisions.md) |
| [sim/](sim/) | MuJoCo scene XML, meshes, Gymnasium env wrapper, domain-randomization config |
| [data/](data/) | Collected LeRobot demonstration datasets (gitignored — see [data/README.md](data/README.md)) |
| [policy/](policy/) | Model definitions, training configs, checkpoints |
| [inference/](inference/) | Runtime pipeline: perception → reasoning/planning → bimanual policy → action |
| [openvino_bench/](openvino_bench/) | ONNX/IR conversion, NNCF quantization, latency+throughput benchmark |
| [eval/](eval/) | 10 fixed seed configs, success-rate harness, video capture |
| [scripts/](scripts/) | `check_env.py`, `run_train.sh`, `run_eval.sh`, `run_bench.sh` |

---

## Deliverables → rubric

| # | Deliverable | Rubric | Status |
|---|---|---|---|
| 1 | Reproducible repo (setup, deps, assets, train/eval/infer) | Reproducibility 10 | scaffolded |
| 2 | MuJoCo sim + domain randomization + eval config | Task Completion 30 / Robustness 15 | _pending_ |
| 3 | Intel inference benchmark (latency, throughput, device, precision) | OpenVINO/Intel 20 | _pending_ |
| 4 | Demo video, 10 randomized seeds | Robustness 15 / Task Completion 30 | _pending_ |
| 5 | Technical README / architecture summary | Technical Quality 10 / Innovation 5 | in progress |

---

## Architecture

_(pending — Phase 3.)_ Planned: instruction parser decomposes a compound
command into ordered subtask primitives; a learned low-level policy (ACT
first, SmolVLA as the upgrade path) executes each primitive; post-condition
checks drive retry/replan. Rationale and trade-offs in
[docs/decisions.md](docs/decisions.md).

## OpenVINO results

_(pending — Phase 6.)_ Will report mean/p95 latency, throughput, device
(CPU / iGPU / NPU) and precision (FP32 / INT8), plus the success-rate delta
between the FP32 PyTorch policy and the INT8 IR model.

## Known limitations

_(pending.)_
