"""Train/val/test splits + norm fitting. Matches paper protocol exactly."""
import random

import numpy as np

from lib.categories import EXCLUDE_TASKS

# Unseen split: 7 unseen / 15 seen, randomized per-seed (matches paper)
N_UNSEEN = 7


def split_allseen(task_eps, seed):
    """All-Seen Protocol (Paper §2): 70/15/15 stratified per task per label.

    Args:
      task_eps: dict[task_id, list[ep]]   eps must have "label"
      seed: int

    Returns:
      {"train": [...], "val": [...], "test": [...]}
    """
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for tid in sorted(task_eps.keys()):
        if tid in EXCLUDE_TASKS:
            continue
        for lv in [0, 1]:
            grp = [e for e in task_eps[tid] if e["label"] == lv]
            rng.shuffle(grp)
            n = len(grp)
            ntr = int(n * 0.70)
            nva = int(n * 0.15)
            splits["train"].extend(grp[:ntr])
            splits["val"].extend(grp[ntr:ntr + nva])
            splits["test"].extend(grp[ntr + nva:])
    return splits


def split_unseen(task_eps, seed):
    """Unseen Protocol (Paper §2): N_UNSEEN tasks held out as test.

    For seen tasks: 70% train, 30% val (no test).
    For unseen tasks: all eps → test.
    """
    rng = random.Random(seed + 1000)   # offset matches existing convention
    all_tasks = sorted([t for t in task_eps if t not in EXCLUDE_TASKS])
    rng.shuffle(all_tasks)
    unseen = all_tasks[:N_UNSEEN]
    seen   = all_tasks[N_UNSEEN:]

    splits = {"train": [], "val": [], "test": []}
    rng2 = random.Random(seed)
    for tid in seen:
        for lv in [0, 1]:
            grp = [e for e in task_eps[tid] if e["label"] == lv]
            rng2.shuffle(grp)
            n = len(grp)
            ntr = int(n * 0.70)
            splits["train"].extend(grp[:ntr])
            splits["val"].extend(grp[ntr:])
    for tid in unseen:
        splits["test"].extend(task_eps[tid])
    return splits


def fit_norm(eps, key="raw"):
    """Z-score normalization fitted on train split.

    Args:
      eps: list of dicts with key field
      key: name of array field to normalize over

    Returns:
      (mean, std) — both np.float32
    """
    arr = np.concatenate([e[key] for e in eps], axis=0)
    mean = arr.mean(0).astype(np.float32)
    std  = (arr.std(0) + 1e-8).astype(np.float32)
    return mean, std


def apply_norm(eps, mean, std, src_key="raw", dst_key="normed"):
    for e in eps:
        e[dst_key] = ((e[src_key] - mean) / std).astype(np.float32)


def compute_task_min(eps):
    """Per-task minimum episode length. Used by `metrics.q_auc` for cutoff."""
    from collections import defaultdict
    by_task = defaultdict(list)
    for e in eps:
        by_task[e["task_id"]].append(e["length"])
    return {tid: min(L) for tid, L in by_task.items()}
