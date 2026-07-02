"""
Extract action-space features from saved OpenVLA LIBERO-10 rollouts.

OpenVLA is a token-based (autoregressive) VLA with execution horizon 1, so the
"chunk" is a single 7-D action and TCE reduces to the MSE between consecutive
actions. This is the *data-collection / feature-extraction* stage and runs
**post-hoc** on rollout records — no VLA is needed to re-run it. The output JSONL
is the per-step feature stream that `data/probe_data/` ships and that
`code/train/train_actprobe.py` consumes.

ActProbe itself uses only two of these features:
    TCE (Temporal Consistency Error) := chunk_mse   (consecutive-action MSE here)
    ACM (Action Chunk Magnitude)     := action_norm
The remaining features are computed for completeness / ablations.

Expected rollout layout (produced by the SAFE / vla-safe OpenVLA stack):
    <rollout_dir>/task<id>--ep<idx>--succ<0|1>.csv   # per-step metrics: the 7
        action components action/d{x,y,z,roll,pitch,yaw,gripper}, a token-entropy
        column (action/mean_token_entropy), and optionally obs/eef_z.
    <rollout_dir>/task<id>--ep<idx>--succ<0|1>.pkl   # metadata (task_description).

Usage:
    python extract_features.py \
        --rollout_dir /path/to/openvla/libero_10 \
        --out_feat   /path/to/probe_data \
        [--max_episodes N]
"""

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Filenames: task{id}--ep{idx}--succ{0|1}.csv
FNAME_RE = re.compile(r"task(\d+)--ep(\d+)--succ(\d+)")

ACTION_COLS = ["action/dx", "action/dy", "action/dz",
               "action/droll", "action/dpitch", "action/dyaw",
               "action/dgripper"]


def compute_action_norm(action):
    """ACM: RMS magnitude of the 7-D action."""
    return float(np.sqrt(np.mean(action ** 2)))


def compute_action_jerk(action_history):
    if len(action_history) < 3:
        return 0.0
    a = np.array(action_history[-3:])
    jerk = a[2] - 2 * a[1] + a[0]
    return float(np.sqrt(np.mean(jerk ** 2)))


def compute_gripper_oscillation(gripper_history, window=10):
    if len(gripper_history) < 2:
        return 0
    recent = gripper_history[-window:]
    signs = np.sign(recent)
    return int(np.sum(signs[1:] != signs[:-1]))


def extract_probes(rollout_dir, out_feat, max_episodes=None):
    rollout_dir = Path(rollout_dir)
    if not rollout_dir.exists():
        print(f"ERROR: {rollout_dir} does not exist. Run OpenVLA rollout collection first.")
        return

    out_dir = Path(out_feat)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(rollout_dir.glob("task*--ep*--succ*.csv"))
    print(f"Found {len(csv_files)} CSV rollout files in {rollout_dir}")

    # Dedup: group by (task_id, episode_idx), keep the latest file
    file_by_key = {}
    for csv_path in csv_files:
        m = FNAME_RE.search(csv_path.stem)
        if not m:
            continue
        file_by_key[(int(m.group(1)), int(m.group(2)))] = csv_path

    # Group by task and apply per-task episode cap
    task_files = defaultdict(list)
    for (task_id, episode_idx), csv_path in sorted(file_by_key.items()):
        task_files[task_id].append((episode_idx, csv_path))

    deduped = []
    for task_id in sorted(task_files.keys()):
        files = task_files[task_id]
        if max_episodes and len(files) > max_episodes:
            print(f"  task {task_id}: truncated {len(files)} -> {max_episodes} episodes")
            files = files[:max_episodes]
        deduped.extend([f for _, f in files])

    print(f"After dedup: {len(deduped)} CSV files (from {len(csv_files)} originals)")

    task_episodes = defaultdict(list)

    for csv_path in tqdm(deduped, desc="Extracting features"):
        m = FNAME_RE.search(csv_path.stem)
        if not m:
            continue
        task_id = int(m.group(1))
        episode_idx = int(m.group(2))
        success = bool(int(m.group(3)))

        df = pd.read_csv(csv_path)

        # Task description from the sidecar pkl, if present
        task_description = f"libero_10_task_{task_id}"
        pkl_path = csv_path.with_suffix(".pkl")
        if pkl_path.exists():
            try:
                meta = pickle.load(open(pkl_path, "rb"))
                task_description = meta.get("task_description", task_description)
            except Exception:
                pass

        has_actions = all(c in df.columns for c in ACTION_COLS)
        entropy_col = next((c for c in ["action/mean_token_entropy",
                                        "action/avg_token_entropy"]
                            if c in df.columns), None)
        eef_z_col = next((c for c in ["obs/eef_z", "obs/robot0_eef_pos_z"]
                          if c in df.columns), None)

        steps = []
        action_history = []
        gripper_history = []
        prev_action = None

        for t in range(len(df)):
            row = df.iloc[t]
            if has_actions:
                action = np.array([row[c] for c in ACTION_COLS], dtype=np.float64)
            else:
                action = np.zeros(7)

            # TCE: consecutive-action MSE (exec_horizon = 1)
            chunk_mse = float(np.mean((action - prev_action) ** 2)) if prev_action is not None else 0.0
            action_norm = compute_action_norm(action)  # ACM

            action_history.append(action)
            action_jerk = compute_action_jerk(action_history)

            gripper_history.append(float(action[-1]))
            gripper_osc = compute_gripper_oscillation(gripper_history)

            # denoising_curvature is approximated by token entropy for this VLA
            token_ent = float(row[entropy_col]) if entropy_col is not None else 0.0
            eef_z = float(row[eef_z_col]) if eef_z_col is not None else 0.0

            steps.append({
                "t": t,
                "chunk_mse": chunk_mse,
                "action_norm": action_norm,
                "action_jerk": action_jerk,
                "gripper_oscillation": gripper_osc,
                "denoising_curvature": token_ent,
                "eef_z": eef_z,
            })
            prev_action = action

        task_episodes[task_id].append({
            "episode_id": episode_idx,
            "task_id": task_id,
            "task_description": task_description,
            "success": success,
            "length": len(steps),
            "steps": steps,
        })

    for task_id in sorted(task_episodes.keys()):
        episodes = task_episodes[task_id]
        out_path = out_dir / f"task_{task_id}.jsonl"
        n_succ = sum(1 for e in episodes if e["success"])
        with open(out_path, "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")
        print(f"  task_{task_id}: {len(episodes)} eps ({n_succ} succ, "
              f"{len(episodes) - n_succ} fail)")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout_dir", required=True,
                   help="Directory with task*--ep*--succ*.csv rollout files")
    p.add_argument("--out_feat", required=True,
                   help="Output directory for per-task feature JSONL")
    p.add_argument("--max_episodes", type=int, default=None,
                   help="Optional cap on episodes per task")
    args = p.parse_args()
    extract_probes(args.rollout_dir, args.out_feat, args.max_episodes)
