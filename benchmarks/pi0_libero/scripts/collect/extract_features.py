"""
Extract action-space features from saved Pi0 (diffusion) LIBERO-10 rollouts.

This is the *data-collection / feature-extraction* stage. It runs **post-hoc** on
rollout records that were dumped by the Pi0 inference stack — no VLA is needed to
re-run it. The output JSONL is exactly the per-step feature stream that
`data/probe_data/` ships and that `code/train/train_actprobe.py` consumes.

ActProbe itself uses only two of these features:
    TCE (Temporal Consistency Error) := chunk_mse
    ACM (Action Chunk Magnitude)     := action_norm
The remaining features are computed here for completeness / ablations.

Expected rollout layout (produced by the Pi0 inference stack):
    <rollout_dir>/
        env_records/*.pkl       # one per (task, episode): task_id, task_description,
                                 #   episode_idx, episode_success, replan_steps
        policy_records/*.pkl     # one per inference step, named
                                 #   task_<id>--ep_<idx>--t_<step>...; each holds
                                 #   actions (1, H, A) | (H, A), observation/state (8,)
                                 #   = eef_pos(3)+eef_axisangle(3)+gripper_qpos(2),
                                 #   and optional pre_velocity (n_diff, H, D) for the
                                 #   denoising-curvature / hidden-state features.

Usage:
    python extract_features.py \
        --rollout_dir /path/to/pi0-libero_10 \
        --out_feat   /path/to/probe_data \
        [--out_hs    /path/to/hidden_states]   # optional 1024-d SAFE features
"""

import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPLAN_STEPS = 5


# ── Feature computation helpers ─────────────────────────────────────────────

def compute_chunk_mse(prev_actions, curr_actions, replan_steps):
    """TCE: MSE between the overlapping horizons of consecutive action chunks."""
    if prev_actions is None:
        return 0.0
    overlap_prev = prev_actions[replan_steps:]
    overlap_curr = curr_actions[:-replan_steps]
    n = min(len(overlap_prev), len(overlap_curr))
    if n == 0:
        return 0.0
    return float(np.mean((overlap_prev[:n] - overlap_curr[:n]) ** 2))


def compute_action_norm(actions, replan_steps):
    """ACM: RMS magnitude of the executed portion of the action chunk."""
    executed = actions[:replan_steps]
    return float(np.sqrt(np.mean(executed ** 2)))


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
    flips = np.sum(signs[1:] != signs[:-1])
    return int(flips)


def compute_denoising_curvature(pre_velocity):
    if pre_velocity is None:
        return 0.0
    dv = np.diff(pre_velocity, axis=0)
    return float(np.sqrt(np.mean(dv ** 2)))


def compute_eef_speed_3d(prev_eef_pos, curr_eef_pos):
    """3D end-effector speed: sqrt(dx^2 + dy^2 + dz^2)."""
    if prev_eef_pos is None:
        return 0.0
    diff = curr_eef_pos - prev_eef_pos
    return float(np.sqrt(np.sum(diff ** 2)))


# ── Main extraction ─────────────────────────────────────────────────────────

def extract_all(rollout_dir, out_feat, out_hs=None):
    rollout_dir = Path(rollout_dir)
    out_feat = Path(out_feat)
    out_feat.mkdir(parents=True, exist_ok=True)
    if out_hs is not None:
        out_hs = Path(out_hs)
        out_hs.mkdir(parents=True, exist_ok=True)

    env_dir = rollout_dir / "env_records"
    pol_dir = rollout_dir / "policy_records"

    env_pkls = sorted([f for f in os.listdir(env_dir) if f.endswith(".pkl")])

    # Index policy records by (task_id, episode_idx) -> sorted by timestep
    print("Indexing policy records...")
    pol_index = defaultdict(list)
    pat = re.compile(r'task_(\d+)--ep_(\d+)--t_(\d+)')
    for fname in os.listdir(pol_dir):
        m = pat.search(fname)
        if m:
            tid, eid, t = int(m.group(1)), int(m.group(2)), int(m.group(3))
            pol_index[(tid, eid)].append((t, str(pol_dir / fname)))
    for key in pol_index:
        pol_index[key] = [p for _, p in sorted(pol_index[key])]

    print(f"Found {len(env_pkls)} env records, "
          f"{sum(len(v) for v in pol_index.values())} policy records "
          f"across {len(pol_index)} episodes")

    task_episodes_feat = defaultdict(list)
    task_episodes_hs = defaultdict(list)
    seen = set()
    n_skip = 0

    for env_file in tqdm(env_pkls, desc="Extracting features"):
        env = pickle.load(open(env_dir / env_file, "rb"))
        task_id = env["task_id"]
        task_desc = env["task_description"]
        episode_idx = env["episode_idx"]
        success = env["episode_success"]
        replan_steps = env.get("replan_steps", REPLAN_STEPS)

        key = (task_id, episode_idx)
        if key in seen:
            n_skip += 1
            continue
        pol_list = pol_index.get(key, [])
        if not pol_list:
            n_skip += 1
            continue
        seen.add(key)

        # Load all policy records for this episode
        policy_records = []
        ok = True
        for p in pol_list:
            try:
                policy_records.append(pickle.load(open(p, "rb")))
            except Exception:
                ok = False
                break
        if not ok or not policy_records:
            n_skip += 1
            continue

        # Extract features and hidden states
        steps = []
        hs_list = []
        prev_actions = None
        prev_eef_pos = None
        prev_eef_z = None
        prev_gripper_mean = None
        action_history = []
        gripper_history = []

        for t, pr in enumerate(policy_records):
            # Actions: (1, pred_horizon, action_dim) -> (pred_horizon, action_dim)
            actions = pr["actions"]
            if actions.ndim == 3:
                actions = actions[0]

            # pre_velocity: (n_diff_steps, pred_horizon, dim_feat)
            pre_vel = pr.get("pre_velocity", None)

            # Optional hidden state: mean over diff_steps and horizon -> (1024,)
            if out_hs is not None:
                if pre_vel is not None:
                    hs = pre_vel.mean(axis=0).mean(axis=0).astype(np.float32)
                else:
                    hs = np.zeros(1024, dtype=np.float32)
                hs_list.append(hs)

            # observation/state: (8,) = eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
            obs_state = pr["observation/state"]
            eef_pos = obs_state[:3].astype(np.float32)
            eef_z = float(obs_state[2])
            gripper_qpos = obs_state[6:8].astype(np.float32)
            gripper_mean = float(gripper_qpos.mean())

            # Executed actions for action_history
            executed = actions[:replan_steps]
            for step_a in executed:
                action_history.append(step_a)
                gripper_history.append(float(step_a[-1]))

            action_norm = compute_action_norm(actions, replan_steps)      # ACM
            chunk_mse = compute_chunk_mse(prev_actions, actions, replan_steps)  # TCE
            action_jerk = compute_action_jerk(action_history)
            gripper_osc = compute_gripper_oscillation(gripper_history)
            denoising_curv = compute_denoising_curvature(pre_vel)
            eef_z_vel = abs(eef_z - prev_eef_z) if prev_eef_z is not None else 0.0
            eef_speed = compute_eef_speed_3d(prev_eef_pos, eef_pos)
            gripper_vel = (abs(gripper_mean - prev_gripper_mean)
                           if prev_gripper_mean is not None else 0.0)

            step_log = {
                "t": t,
                "action_norm": action_norm,
                "chunk_mse": chunk_mse,
                "action_jerk": action_jerk,
                "gripper_oscillation": gripper_osc,
                "denoising_curvature": denoising_curv,
                "eef_z": eef_z,
                "eef_z_vel": eef_z_vel,
                "eef_speed_3d": eef_speed,
                "gripper_qpos_mean": gripper_mean,
                "gripper_qpos_vel": gripper_vel,
            }
            steps.append(step_log)

            prev_actions = actions
            prev_eef_pos = eef_pos
            prev_eef_z = eef_z
            prev_gripper_mean = gripper_mean

        task_episodes_feat[task_id].append({
            "episode_id": episode_idx,
            "task_id": task_id,
            "task_description": task_desc,
            "success": bool(success),
            "length": len(steps),
            "steps": steps,
        })

        if out_hs is not None:
            task_episodes_hs[task_id].append({
                "episode_id": episode_idx,
                "task_id": task_id,
                "success": bool(success),
                "length": len(hs_list),
                "hidden_states": np.stack(hs_list, axis=0),  # (T, 1024)
            })

    # ── Write feature JSONL ──
    total_eps = sum(len(v) for v in task_episodes_feat.values())
    print(f"\nExtracted {total_eps} episodes (skipped {n_skip})")
    print(f"\n── Writing feature JSONL to {out_feat} ──")
    for task_id in sorted(task_episodes_feat.keys()):
        episodes = task_episodes_feat[task_id]
        out_path = out_feat / f"task_{task_id}.jsonl"
        n_succ = sum(1 for e in episodes if e["success"])
        n_fail = len(episodes) - n_succ
        with open(out_path, "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")
        print(f"  task_{task_id}: {len(episodes)} eps ({n_succ} succ, {n_fail} fail)")

    # ── Optionally write 1024-d hidden-state features (SAFE baseline) ──
    if out_hs is not None:
        print(f"\n── Writing hidden states to {out_hs} ──")
        for task_id in sorted(task_episodes_hs.keys()):
            episodes = task_episodes_hs[task_id]
            out_path = out_hs / f"task_{task_id}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(episodes, f)
            print(f"  task_{task_id}: {len(episodes)} eps, "
                  f"size={out_path.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"\nDone. Features: {out_feat}"
          + (f", Hidden states: {out_hs}" if out_hs is not None else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout_dir", required=True,
                   help="Directory with env_records/ and policy_records/")
    p.add_argument("--out_feat", required=True,
                   help="Output directory for per-task feature JSONL")
    p.add_argument("--out_hs", default=None,
                   help="Optional output dir for 1024-d hidden-state features (SAFE)")
    args = p.parse_args()
    extract_all(args.rollout_dir, args.out_feat, args.out_hs)
