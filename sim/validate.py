"""Validate every scripted task over random starts, both arms (S4, T2: all three targets per start).

    python -m sim.validate          # 40 starts per task and arm, all randomisation on (~3 min on 8 cores)
    python -m sim.validate 10       # quicker
    python -m sim.validate 40 none  # no randomisation at all (no start or demonstration noise either)
    python -m sim.validate 40 cube_yaw,distractors   # only these factors (add them one at a time)
    python -m sim.validate 40 all eval               # held-out colours and phrasings

Prints successes per task and arm, the largest joint jump between IK waypoints (a big jump
means the arm switched posture mid-move), and every failure with its seed.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from sim.randomize import NONE, Randomization

FACTORS = {f: getattr(Randomization(), f) for f in
           ("start_noise", "demo_noise", "colours", "distractors", "cube_yaw", "lighting", "wording")}
EXTRA = {"cube_yaw": {"cube_yaw_drawer": Randomization().cube_yaw_drawer}}  # settings that go with a factor


def config(factors: str = "all", split: str = "train") -> Randomization:
    if factors == "all":
        return Randomization(split=split)
    if factors == "none":
        return replace(NONE, split=split)
    kw = {}
    for f in factors.split(","):
        kw[f] = FACTORS[f]
        kw.update(EXTRA.get(f, {}))
    return replace(NONE, split=split, **kw)


def _run(job):
    from sim import skills as K

    name, args, seed, rand = job
    return getattr(K, f"run_{name}")(*args, seed=seed, rand=rand)


def main(n: int = 40, factors: str = "all", split: str = "train"):
    rand = config(factors, split)
    print(rand)
    jobs = []
    for arm in ("A", "B"):
        for s in range(n):
            jobs += [("S1", (arm,), s, rand), ("S2", (arm,), s, rand), ("S3", (arm,), s, rand)]
            jobs += [(task, (arm, target), s, rand) for task in ("S4", "T2") for target in ("left", "middle", "right")]
            jobs += [("T1", (arm,), s, rand)]
    with ProcessPoolExecutor() as pool:
        res = list(pool.map(_run, jobs, chunksize=2))
    for task in ("S1", "S2", "S3", "S4", "T1", "T2"):
        for arm in ("A", "B"):
            rs = [r for r in res if r["task"] == task and r.get("arm", r.get("opener", r.get("giver"))) == arm]
            fails = [r for r in rs if not r["success"]]
            print(f"{task} arm {arm}: {len(rs) - len(fails)}/{len(rs)}  max jump {max(r['max_jump'] for r in rs):.2f}"
                  f"  max distractor push {max(r['distractor_moved'] for r in rs) * 1000:.1f} mm")
            for r in fails:
                print("    fail", {k: v for k, v in r.items() if k != "task"})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40, *sys.argv[2:4])
