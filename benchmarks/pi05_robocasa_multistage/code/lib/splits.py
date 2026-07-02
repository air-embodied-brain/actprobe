"""Train/val/test splits for multi-stage (allseen-only, no unseen split)."""
import random
import numpy as np

from lib.categories import EXCLUDE_TASKS


def split_allseen(task_eps, seed):
    """70/15/15 stratified per task per label."""
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
    """Multi-stage has only 5 tasks → no unseen-task split. Raises."""
    raise NotImplementedError(
        "Multi-stage has only 5 tasks; unseen protocol not supported per paper §4.")


def fit_norm(eps, key="raw"):
    arr = np.concatenate([e[key] for e in eps], axis=0)
    return arr.mean(0).astype(np.float32), (arr.std(0) + 1e-8).astype(np.float32)


def apply_norm(eps, mean, std, src_key="raw", dst_key="normed"):
    for e in eps:
        e[dst_key] = ((e[src_key] - mean) / std).astype(np.float32)


def compute_task_min(eps):
    from collections import defaultdict
    by_task = defaultdict(list)
    for e in eps:
        by_task[e["task_id"]].append(e["length"])
    return {tid: min(L) for tid, L in by_task.items()}
