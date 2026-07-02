"""Data loading. All paths resolved relative to PI0_ROOT (env.sh).

π0+LIBERO 与 GR00T 的差异：
- jsonl 中所有 10 features 都是顶层 step keys（无嵌套 physics_info）。
- task_id 是整数 0..9，没有像 GR00T 那种语义化任务名。
- hidden states 是单层 (T, 1024)，没有 4-iter。
"""
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PI0_ROOT", Path(__file__).resolve().parent.parent.parent))
DATA = ROOT / "data"

# Canonical 10-feat schema (jsonl natural order). Used by:
# - lib/methods/actprobe.py (PAPER_FEAT_IDX subset)
# - lib/methods/stac.py (chunk_mse only)
# - any feat-based ablation
METRIC_KEYS = [
    "action_norm",          # 0
    "chunk_mse",            # 1
    "action_jerk",          # 2
    "gripper_oscillation",  # 3
    "denoising_curvature",  # 4
    "eef_z",                # 5
    "eef_z_vel",            # 6
    "eef_speed_3d",         # 7
    "gripper_qpos_mean",    # 8
    "gripper_qpos_vel",     # 9
]
N_FEAT_FULL = len(METRIC_KEYS)

# Key-driven: do not hard-code feature indices.
PAPER_FEAT_IDX = [
    METRIC_KEYS.index("action_norm"),
    METRIC_KEYS.index("chunk_mse"),
]   # = [0, 1] in canonical schema


def load_metrics_logs():
    """Load 10-feat episodes from data/metrics_logs/task_*.jsonl.

    Returns:
      dict[task_id, list[ep]]   each ep:
        {raw: (T, 10) f32, label, task_id, length, episode_id}
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
    """Load single-layer hidden states from data/hidden_states/task_*.pkl.

    Pi0 hidden states are (T, 1024) per ep — single layer, NOT 4-iter.

    Returns:
      dict[task_id, list[ep]]   each ep: {raw_hs: (T, 1024) f32, label, task_id, length}
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
            hs = ep["hidden_states"]
            if hasattr(hs, "numpy"):
                hs = hs.numpy()
            hs = np.asarray(hs, dtype=np.float32)
            if hs.ndim != 2 or hs.shape[1] != 1024:
                raise ValueError(f"unexpected hs shape {hs.shape} in {p}")
            if hs.shape[0] < 5:
                continue
            eps.append({
                "raw_hs":     hs,
                "label":      0 if ep["success"] else 1,
                "task_id":    tid,
                "length":     hs.shape[0],
                "episode_id": ep.get("episode_id"),
            })
        if eps:
            out[tid] = eps
    return out


def load_task_embeddings():
    """Load Qwen3 task embeddings.

    The .npy stores dict[int task_id → (1024,) f32]. We re-key to `task_<i>`
    for consistency with metrics_logs / hidden_states.

    Returns:
      dict[str task_id, (1024,) f32]
    """
    raw = np.load(DATA / "task_embeddings.npy", allow_pickle=True).item()
    return {f"task_{int(k)}": np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def load_task_instructions():
    """Load human-readable task instructions. Returns dict[task_id, str]."""
    with open(DATA / "task_instructions.json") as f:
        return json.load(f)


def flatten(task_eps):
    """{task_id: [eps]} → [eps] flat list, sorted by task_id."""
    out = []
    for tid in sorted(task_eps.keys()):
        out.extend(task_eps[tid])
    return out
