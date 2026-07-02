"""Pi0.5 + RoboCasa multi-stage data loading.

Reads per-task pkls from data/probe_data/<task>.pkl (5 multi-stage tasks).
Same format as single-stage pi05_robocasa.
"""
import os
import pickle
from pathlib import Path

import numpy as np

from lib.categories import LONG_TASKS, EXCLUDE_TASKS

ROOT = Path(os.environ.get("PI05_ROOT",
                            Path(__file__).resolve().parent.parent.parent))
DATA = ROOT / "data"

METRIC_KEYS = [
    "action_norm", "gripper_oscillation", "denoising_curvature", "eef_z",
    "chunk_mse", "action_jerk", "gripper_qpos_mean",
    "eef_z_vel", "eef_speed_3d", "gripper_qpos_vel",
]
N_FEAT_FULL = 10
PAPER_FEAT_IDX = [0, 4]


def load_probe_data():
    out = {}
    for tid in LONG_TASKS:
        if tid in EXCLUDE_TASKS:
            continue
        p = DATA / "probe_data" / f"{tid}.pkl"
        if not p.exists():
            print(f"[skip] {tid}: missing {p}")
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
            })
        out[tid] = eps
    return out


def load_qwen3_embeddings():
    p = DATA / "qwen3_emb.pkl"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing.")
    with open(p, "rb") as f:
        return pickle.load(f)


# Backward-compat shims
def load_metrics_logs():
    return load_probe_data()


def load_hs_meanpool():
    out = {}
    for tid, eps in load_probe_data().items():
        new_eps = []
        for e in eps:
            new = dict(e); new["raw_hs"] = e["hidden_states"]
            new_eps.append(new)
        out[tid] = new_eps
    return out
