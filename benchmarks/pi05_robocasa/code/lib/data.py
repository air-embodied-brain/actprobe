"""Pi0.5 + RoboCasa data loading. Reads from data/probe_data/<task>.pkl."""
import os
import pickle
from pathlib import Path

import numpy as np

from lib.categories import EXCLUDE_TASKS, TASK2CAT

ROOT = Path(os.environ.get("PI05_ROOT",
                            Path(__file__).resolve().parent.parent.parent))
DATA = ROOT / "data"

# 10-feat layout (matches the GR00T metrics_logs METRIC_KEYS)
METRIC_KEYS = [
    "action_norm",          # 0
    "gripper_oscillation",  # 1
    "denoising_curvature",  # 2
    "eef_z",                # 3
    "chunk_mse",            # 4
    "action_jerk",          # 5
    "gripper_qpos_mean",    # 6
    "eef_z_vel",            # 7
    "eef_speed_3d",         # 8
    "gripper_qpos_vel",     # 9
]
N_FEAT_FULL = 10
PAPER_FEAT_IDX = [0, 4]   # action_norm + chunk_mse


def load_probe_data():
    """Load all 24 task pkls. Returns dict[task_id, list[ep]].

    Each ep dict: {raw: (T,10), hidden_states: (T,1024), label, task_id, length, instruction, episode_id}.
    label: 0 = success, 1 = fail (matches groot convention).
    """
    out = {}
    for p in sorted((DATA / "probe_data").glob("*.pkl")):
        tid = p.stem
        if tid in EXCLUDE_TASKS:
            continue
        with open(p, "rb") as f:
            raw_eps = pickle.load(f)
        eps = []
        for r in raw_eps:
            eps.append({
                "raw":           r["probe_features_10"].astype(np.float32),
                "hidden_states": r["hidden_states"].astype(np.float32),
                "label":         0 if r["success"] else 1,
                "task_id":       tid,
                "length":        int(r["length"]),
                "instruction":   r["instruction"],
                "episode_id":    r["episode_id"],
                "task_emb_512":  r["task_emb"].astype(np.float32),  # legacy 512-dim
            })
        out[tid] = eps
    return out


def load_qwen3_embeddings():
    """Per-instruction Qwen3 1024-dim embedding for ActProbe lang conditioning.

    Returns dict[instruction, (1024,) float32]. Computed by
    `scripts/build_qwen3_emb.py` and cached at `data/qwen3_emb.pkl`.
    """
    p = DATA / "qwen3_emb.pkl"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Run: python scripts/build_qwen3_emb.py")
    with open(p, "rb") as f:
        return pickle.load(f)


# ── Backward-compat shims for groot-style train scripts ──────────────────────
def load_metrics_logs():
    """Alias: returns episodes with 10-feat 'raw' key.

    Mirrors actProbe groot's API. Pi0.5 uses pre-computed
    probe_features_10 (already in the same shape).
    """
    return load_probe_data()


def load_hs_meanpool():
    """Returns episodes with 'raw_hs' key (1024-dim hidden states).

    Pi0.5 hidden_states are already mean-pooled at extraction time, so
    this just adds the 'raw_hs' alias on top of load_probe_data().
    """
    out = {}
    for tid, eps in load_probe_data().items():
        new_eps = []
        for e in eps:
            new = dict(e)
            new["raw_hs"] = e["hidden_states"]
            new_eps.append(new)
        out[tid] = new_eps
    return out
