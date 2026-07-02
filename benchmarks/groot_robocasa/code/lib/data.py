"""Data loading. All paths resolved relative to GROOT_ROOT (env.sh)."""
import json
import os
import pickle
from pathlib import Path

import numpy as np

from lib.categories import EXCLUDE_TASKS, TASK2CAT

ROOT = Path(os.environ.get("GROOT_ROOT", Path(__file__).resolve().parent.parent.parent))
DATA = ROOT / "data"

# 10 features extracted from JSONL steps (paper main 2-feat = indices [0, 4])
METRIC_KEYS = [
    "action_norm",          # 0
    "gripper_oscillation",  # 1
    "denoising_curvature",  # 2
    "eef_z",                # 3 (extracted from physics_info)
    "chunk_mse",            # 4
    "action_jerk",          # 5
    "gripper_qpos_mean",    # 6 (extracted from physics_info)
    "eef_z_vel",            # 7 (derived: |diff(eef_z)|)
    "eef_speed_3d",         # 8 (derived: sqrt(dx²+dy²+dz²))
    "gripper_qpos_vel",     # 9 (derived: |diff(gripper_qpos_mean)|)
]
N_FEAT_FULL = 10
PAPER_FEAT_IDX = [0, 4]   # action_norm + chunk_mse


def load_metrics_logs():
    """Load 10-feat episodes from data/metrics_logs/*.jsonl.

    Returns:
      dict[task_id, list[dict]]   each ep: {raw: (T,10) f32, label, task_id, length}
    """
    out = {}
    for p in sorted((DATA / "metrics_logs").glob("*.jsonl")):
        tname = p.stem.replace("_PandaOmron_Env", "")
        if tname in EXCLUDE_TASKS:
            continue
        eps = []
        xyz_list, grip_list = [], []

        with open(p) as f:
            for line in f:
                ep = json.loads(line)
                steps = ep["steps"]
                T = len(steps)
                if T < 5:
                    continue

                arr  = np.zeros((T, 7), dtype=np.float32)
                xyz  = np.zeros((T, 3), dtype=np.float32)
                grip = np.zeros(T,       dtype=np.float32)

                for t, s in enumerate(steps):
                    arr[t, 0] = s.get("action_norm", 0) or 0
                    arr[t, 1] = s.get("gripper_oscillation", 0) or 0
                    arr[t, 2] = s.get("denoising_curvature", 0) or 0
                    pi = s.get("physics_info", {})
                    eef = pi.get("state.end_effector_position_absolute", [0, 0, 0])
                    arr[t, 3] = float(eef[2])
                    arr[t, 4] = s.get("chunk_mse", 0) or 0
                    arr[t, 5] = s.get("action_jerk", 0) or 0
                    gqpos = pi.get("state.gripper_qpos", [0, 0])
                    arr[t, 6] = float(np.mean(gqpos)) if hasattr(gqpos, "__len__") else float(gqpos)

                    xyz[t]  = [float(eef[0]), float(eef[1]), float(eef[2])]
                    grip[t] = arr[t, 6]

                eps.append({"raw7": arr, "xyz": xyz, "grip": grip,
                            "label": 0 if ep["success"] else 1,
                            "task_id": tname, "length": T})

        # Derived features 7/8/9
        for ep in eps:
            T = ep["length"]
            zvel = np.zeros((T, 1), dtype=np.float32)
            spd  = np.zeros((T, 1), dtype=np.float32)
            gvel = np.zeros((T, 1), dtype=np.float32)
            if T > 1:
                zvel[1:, 0] = np.abs(np.diff(ep["raw7"][:, 3]))
                v = np.diff(ep["xyz"], axis=0)
                spd[1:, 0] = np.sqrt((v ** 2).sum(axis=1))
                gvel[1:, 0] = np.abs(np.diff(ep["grip"]))
            ep["raw"] = np.hstack([ep["raw7"], zvel, spd, gvel]).astype(np.float32)
            del ep["raw7"], ep["xyz"], ep["grip"]

        out[tname] = eps
    return out


def load_hs_meanpool():
    """Load mean-pooled hidden states. Shape per ep: (T, 1024) fp16."""
    out = {}
    for p in sorted((DATA / "hidden_states_meanpool").glob("*.pkl")):
        tid = p.stem
        if tid in EXCLUDE_TASKS:
            continue
        with open(p, "rb") as f:
            raw_eps = pickle.load(f)
        eps = []
        for ep in raw_eps:
            hs = np.asarray(ep["hidden_states"], dtype=np.float32)
            if hs.shape[0] == 0:
                continue
            eps.append({
                "raw_hs": hs,                                # (T, 1024)
                "label": 0 if ep["success"] else 1,
                "task_id": tid,
                "length": hs.shape[0],
                "episode_id": ep.get("episode_id"),
            })
        if eps:
            out[tid] = eps
    return out


def load_hs_raw():
    """Load raw 4-iter hidden states. Shape per ep: (T, 4, 1024) fp16."""
    out = {}
    for p in sorted((DATA / "hidden_states").glob("*.pkl")):
        tid = p.stem
        if tid in EXCLUDE_TASKS:
            continue
        with open(p, "rb") as f:
            raw_eps = pickle.load(f)
        eps = []
        for ep in raw_eps:
            hs = ep["hidden_states"]
            if hs.shape[0] == 0:
                continue
            eps.append({
                "raw_hs_iter": hs.astype(np.float32),
                "label": 0 if ep["success"] else 1,
                "task_id": tid,
                "length": hs.shape[0],
                "episode_id": ep.get("episode_id"),
            })
        if eps:
            out[tid] = eps
    return out


def load_task_embeddings():
    """Load Qwen3 task embeddings. Returns dict[task_id, (1024,) fp32]."""
    embs = np.load(DATA / "task_embeddings.npy", allow_pickle=True).item()
    return {k: v.astype(np.float32) for k, v in embs.items()}


def load_task_instructions():
    """Load human-readable task instructions."""
    with open(DATA / "task_instructions.json") as f:
        return json.load(f)


def flatten(task_eps):
    """{task_id: [eps]} → [eps] flat list."""
    return [ep for eps in task_eps.values() for ep in eps]
