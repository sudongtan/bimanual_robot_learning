# Datasets

Contents of this directory are **gitignored**. Demonstration data lives on the
Hugging Face Hub and is pulled down by the training/eval scripts.

- Dataset repo: _(pending — set once the Phase 2 collection run is pushed)_
- Format: LeRobot dataset (images, proprioception, actions, per-episode
  language annotation).

Local layout after a pull:

```
data/<dataset-name>/    # LeRobot dataset root
```

The cloud GPU box pulls the same dataset by id — do not rsync episodes by hand
unless the Hub route is unavailable.
