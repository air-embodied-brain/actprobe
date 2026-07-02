"""Data loading. All paths resolved relative to OPENVLA_LIBERO_ROOT (env.sh).

OpenVLA+LIBERO 与 Pi0+LIBERO 的差异：
- jsonl 中 `steps[*]` 只有 6 个 metric keys（pi0 是 10）；其他 4 个非 paper-required，省略。
- hidden states 是 (T, 4096) f32（pi0 是 (T, 1024)），来自 OpenVLA 7-token mean-pool（在
  scripts/build_hidden_states.py 一次性完成；本模块只读已 pool 后的数据）。
- 200 raw eps/task → subsample 50 (paper protocol；pi0 没有这一步)。
"""
import json
import os
import pickle
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("OPENVLA_LIBERO_ROOT", Path(__file__).resolve().parent.parent.parent))
DATA = ROOT / "data"

# Canonical 6-feat schema (jsonl natural order). Used by:
# - lib/methods/actprobe.py (PAPER_FEAT_IDX subset = action_norm + chunk_mse)
# - any feat-based ablation
METRIC_KEYS = [
    "action_norm",          # 0
    "chunk_mse",            # 1
    "action_jerk",          # 2
    "gripper_oscillation",  # 3
    "denoising_curvature",  # 4
    "eef_z",                # 5
]
N_FEAT_FULL = len(METRIC_KEYS)

# Key-driven: do not hard-code feature indices.
PAPER_FEAT_IDX = [
    METRIC_KEYS.index("action_norm"),
    METRIC_KEYS.index("chunk_mse"),
]   # = [0, 1] in canonical schema

EPS_PER_TASK_PAPER = 50  # paper-faithful subsample (200 raw → 50)


def load_metrics_logs():
    """Load 6-feat episodes from data/metrics_logs/task_*.jsonl.

    Returns:
      dict[task_id, list[ep]]   each ep:
        {raw: (T, 6) f32, label, task_id, length, episode_id}
      task_id is `task_<i>` (str) for symmetry with hidden_states keys.
    """
    out = {}
    src = DATA / "metrics_logs"
    for p in sorted(src.glob("task_*.jsonl")):
        m = re.match(r"task_(\d+)\.jsonl", p.name)
        if not m:
            continue
        tid = f"task_{int(m.group(1))}"

        eps = []
        with open(p) as f:
            for line in f:
                ep = json.loads(line)
                steps = ep["steps"]
                T = len(steps)
                if T < 5:
                    continue
                arr = np.zeros((T, N_FEAT_FULL), dtype=np.float32)
                for t, s in enumerate(steps):
                    for i, k in enumerate(METRIC_KEYS):
                        arr[t, i] = float(s.get(k, 0.0) or 0.0)
                eps.append({
                    "raw":        arr,
                    "label":      0 if ep["success"] else 1,
                    "task_id":    tid,
                    "length":     T,
                    "episode_id": ep.get("episode_id"),
                })
        out[tid] = eps
    return out


def load_hs():
    """Load pre-pooled OpenVLA hidden states from data/hidden_states/task_*.pkl.

    Each pkl is a list of dicts; HS already mean-pooled over the 7-token axis
    in scripts/build_hidden_states.py, so shape here is (T, 4096) f32.

    Returns:
      dict[task_id, list[ep]]   each ep:
        {raw_hs: (T, 4096) f32, label, task_id, length, episode_id}
    """
    out = {}
    src = DATA / "hidden_states"
    for p in sorted(src.glob("task_*.pkl")):
        m = re.match(r"task_(\d+)\.pkl", p.name)
        if not m:
            continue
        tid = f"task_{int(m.group(1))}"
        with open(p, "rb") as f:
            raw_eps = pickle.load(f)
        eps = []
        for ep in raw_eps:
            hs = np.asarray(ep["hidden_states"], dtype=np.float32)
            if hs.ndim != 2 or hs.shape[1] != 4096:
                raise ValueError(f"unexpected hs shape {hs.shape} in {p.name} "
                                 f"(expected (T, 4096) — did build_hidden_states.py mean-pool?)")
            if hs.shape[0] < 5:
                continue
            success = bool(ep.get("success", False))
            eps.append({
                "raw_hs":     hs,
                "label":      0 if success else 1,
                "task_id":    tid,
                "length":     hs.shape[0],
                "episode_id": ep.get("episode_id"),
            })
        if eps:
            out[tid] = eps
    return out


def load_task_embeddings():
    """Load Qwen3 task embeddings.

    The .npy stores dict[str task_id → (1024,) f32] (re-keyed in Step 3).

    Returns:
      dict[str task_id, (1024,) f32]
    """
    raw = np.load(DATA / "task_embeddings.npy", allow_pickle=True).item()
    return {str(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def load_task_instructions():
    """Load human-readable task instructions. Returns dict[task_id, str]."""
    with open(DATA / "task_instructions.json") as f:
        return json.load(f)


def subsample_50eps(task_eps, eps_per_task=EPS_PER_TASK_PAPER, seed=42):
    """Subsample to <eps_per_task> eps per task, stratified by label (success vs failure).

    Paper protocol uses 50 eps/task (out of 200 raw). Always called BEFORE split.

    Args:
      task_eps: dict[task_id, list[ep]]   eps must have "label"
      eps_per_task: target eps per task (default 50)
      seed: RNG seed (paper default 42)

    Returns:
      dict[task_id, list[ep]] — proportional class-stratified subsample
    """
    rng = random.Random(seed)
    out = {}
    for tid, eps in task_eps.items():
        if len(eps) <= eps_per_task:
            out[tid] = list(eps)
            continue
        succ = [e for e in eps if e["label"] == 0]
        fail = [e for e in eps if e["label"] == 1]
        total = len(eps)
        n_s = min(max(1, round(eps_per_task * len(succ) / total)), len(succ))
        n_f = min(max(1, round(eps_per_task * len(fail) / total)), len(fail))
        rng.shuffle(succ)
        rng.shuffle(fail)
        out[tid] = succ[:n_s] + fail[:n_f]
    return out


def flatten(task_eps):
    """{task_id: [eps]} → [eps] flat list, sorted by task_id."""
    out = []
    for tid in sorted(task_eps.keys()):
        out.extend(task_eps[tid])
    return out
