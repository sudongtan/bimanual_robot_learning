ignore deprecated/
when answer question or suggest design, always use reference from similar projects

## Start here

- Read README.md, then STATUS.md (what is running now and what's next), then the numbered docs as needed: 0background.md (why), 1tasks_scene.md (tasks, scene, plans), 2data_collection.md (recording), 3train_evaluation.md (results and findings), cloud/README.md (rented GPU).
- The project is simulation only, a portfolio project: the aim is meaningful and honest results, not novelty.

## Conventions

- Python: the project's `.venv` (`.venv/bin/python -m ...`, `.venv/bin/lerobot-train`). `uv run` fails (see STATUS.md, open questions). LeRobot is pinned at 0.6.1.
- `data/` (recordings) and `outputs/` (checkpoints, evaluations, logs) are git-ignored and exist only on this Mac. Recordings are deterministic from their seeds: the commands are in 2data_collection.md.
- Fast tests before committing code: `.venv/bin/python -m pytest tests -m "not slow"` (~40 s).
- Evaluate ACT with `--temporal-ensemble 0.01` (3train_evaluation.md, finding 2). Report results per variant with the failed checks, not just the success rate.
- The normalisation floor (`STD_FLOOR`, `floor_std` in sim/dataset_view.py) must be in every dataset's stats; the recorder and dataset views apply it.
- Long jobs (training, evaluation, recording): run them detached (`nohup caffeinate -i -s ... &` here, tmux on the rented GPU) so they survive a Claude Code restart, keep the Mac on mains power, and record them in STATUS.md. One GPU job at a time on the Mac.
- Record every result and finding in 3train_evaluation.md, and keep the status lines in README.md, 1tasks_scene.md and STATUS.md in step.
