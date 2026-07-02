"""
GR00T RoboCasa rollout with per-step action-space metric logging  [REFERENCE].

For GR00T the data-collection stage is *model-coupled*: the action-space features
are logged online, inside the rollout loop, because some of them (notably the
denoising / ODE-trajectory curvature) come from the policy server's per-step
output. There is no separable post-hoc extraction step the way there is for the
LIBERO benchmarks.

This file is therefore provided as **reference** — it documents exactly how the
shipped `data/metrics_logs/*.jsonl` were produced. It is NOT runnable from this
repository alone: it depends on the GR00T evaluation stack (`gr00t.*`), RoboCasa,
robosuite and a GR00T policy (local checkpoint or a policy server). To reproduce
the rollouts, drop this into the GR00T eval package and run it against your GR00T
checkpoint / server.

Per-step features written to `metrics_log.jsonl` (one JSON object per episode,
each with a `steps` list):
    chunk_mse           -> TCE (Temporal Consistency Error), used by ActProbe
    action_norm         -> ACM (Action Chunk Magnitude),     used by ActProbe
    action_jerk         -> 2nd-order action difference RMS
    gripper_oscillation -> gripper sign-change count in a sliding window
    physics_info        -> eef_pos + gripper_qpos (auxiliary, not an ActProbe input)
    denoising_curvature -> ODE-trajectory curvature reported by the policy server
"""

import argparse
import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# These imports come from the GR00T evaluation package (not part of this repo).
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy import BasePolicy
import gymnasium as gym


@dataclass
class VideoConfig:
    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = 720
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    overlay_text: bool = True
    n_action_steps: int = 8


@dataclass
class MultiStepConfig:
    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 720
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


def get_robocasa_env_fn(env_name: str):
    def env_fn():
        import os

        import robocasa  # noqa: F401
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
        import robosuite  # noqa: F401

        os.environ["MUJOCO_GL"] = "egl"
        return gym.make(env_name, enable_render=True)

    return env_fn


def get_gym_env(env_name: str):
    """RoboCasa env factory (the embodiment used in the paper's GR00T benchmark)."""
    env_embodiment = get_embodiment_tag_from_env_name(env_name)
    if env_embodiment in (EmbodimentTag.GR1, EmbodimentTag.ROBOCASA_PANDA_OMRON):
        return get_robocasa_env_fn(env_name)()
    raise ValueError(f"Unsupported embodiment for this reference script: {env_embodiment}")


def create_eval_env(env_name: str, wrapper_configs: WrapperConfigs) -> gym.Env:
    env = get_gym_env(env_name)
    if wrapper_configs.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            overlay_text=wrapper_configs.video.overlay_text,
        )

    env = MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    n_episodes: int = 10,
    n_envs: int = 1,
    metrics_dir: str = "",
) -> Any:
    """Run policy rollouts in parallel envs and log per-step action-space features."""
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    env_fns = [
        partial(create_eval_env, env_name=env_name, wrapper_configs=wrapper_configs)
        for _ in range(n_envs)
    ]
    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = gym.vector.AsyncVectorEnv(env_fns, shared_memory=False, context="spawn")

    episode_lengths = []
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)

    # --- Metrics logging state (per env) ---
    n_action_steps = wrapper_configs.multistep.n_action_steps
    prev_chunks = [None] * n_envs                                   # for chunk_mse / TCE
    action_history = [deque(maxlen=3) for _ in range(n_envs)]       # for action_jerk
    gripper_history = [deque(maxlen=10) for _ in range(n_envs)]     # for gripper_oscillation
    episode_step_logs = [[] for _ in range(n_envs)]
    all_episode_logs = []

    observations, _ = env.reset()
    policy.reset()

    pbar = tqdm(total=n_episodes, desc="Episodes")
    while completed_episodes < n_episodes:
        actions, policy_info = policy.get_action(observations)
        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)

        # --- Compute per-step metrics per env ---
        for env_idx in range(n_envs):
            step_log = {"t": current_lengths[env_idx], "is_replan": True}

            # Build a flat action chunk by concatenating all action keys.
            action_arrays = []
            gripper_val = None
            for k, v in sorted(actions.items()):
                chunk = v[env_idx]                       # (action_horizon, dim) or (action_horizon,)
                if chunk.ndim == 1:
                    chunk = chunk[:, np.newaxis]
                if "gripper" in k:
                    gripper_val = v[env_idx]
                action_arrays.append(chunk)
            full_chunk = np.concatenate(action_arrays, axis=-1)  # (action_horizon, total_dim)

            # 1. chunk_mse (TCE): overlap MSE between consecutive chunks
            if prev_chunks[env_idx] is not None:
                overlap = min(
                    full_chunk.shape[0] - n_action_steps,
                    prev_chunks[env_idx].shape[0] - n_action_steps,
                )
                if overlap > 0:
                    prev_overlap = prev_chunks[env_idx][n_action_steps:n_action_steps + overlap]
                    curr_overlap = full_chunk[:overlap]
                    step_log["chunk_mse"] = float(np.mean((prev_overlap - curr_overlap) ** 2))
            prev_chunks[env_idx] = full_chunk

            # 2. action_norm (ACM): RMS of the executed portion of the chunk
            executed = full_chunk[:n_action_steps]
            step_log["action_norm"] = float(np.sqrt(np.mean(executed ** 2)))

            # 3. action_jerk: 2nd-order finite difference of the mean executed action
            mean_action = executed.mean(axis=0)
            action_history[env_idx].append(mean_action)
            if len(action_history[env_idx]) >= 3:
                a = list(action_history[env_idx])
                jerk = a[-1] - 2 * a[-2] + a[-3]
                step_log["action_jerk"] = float(np.sqrt(np.mean(jerk ** 2)))

            # 4. gripper_oscillation: sign changes in a sliding window
            if gripper_val is not None:
                for gs in range(min(n_action_steps, len(gripper_val))):
                    g = float(gripper_val[gs]) if gripper_val.ndim > 0 else float(gripper_val)
                    gripper_history[env_idx].append(g)
                if len(gripper_history[env_idx]) >= 2:
                    gh = list(gripper_history[env_idx])
                    step_log["gripper_oscillation"] = sum(
                        1 for j in range(1, len(gh)) if (gh[j] >= 0.5) != (gh[j - 1] >= 0.5)
                    )

            # 5. physics_info (auxiliary): eef_pos + gripper_qpos
            physics = {}
            for obs_key in ("state.end_effector_position_absolute", "state.gripper_qpos"):
                if obs_key in observations:
                    val = observations[obs_key][env_idx]
                    if val.ndim > 1:
                        val = val[-1]
                    physics[obs_key] = val.tolist()
            if physics:
                step_log["physics_info"] = physics

            # 6. denoising_curvature: ODE-trajectory curvature from the policy server
            if isinstance(policy_info, dict) and "denoising_curvature" in policy_info:
                step_log["denoising_curvature"] = float(policy_info["denoising_curvature"])

            episode_step_logs[env_idx].append(step_log)

        # --- Episode bookkeeping ---
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, (list, np.ndarray)):
                    env_success = bool(np.any(env_success))
                current_successes[env_idx] |= bool(env_success)
            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                fs = env_infos["final_info"][env_idx]["success"]
                if isinstance(fs, (list, np.ndarray)):
                    fs = bool(np.any(fs))
                current_successes[env_idx] |= bool(fs)
            current_lengths[env_idx] += 1

            if terminations[env_idx] or truncations[env_idx]:
                all_episode_logs.append({
                    "episode_id": len(episode_successes),
                    "env_idx": env_idx,
                    "success": bool(current_successes[env_idx]),
                    "length": current_lengths[env_idx],
                    "steps": episode_step_logs[env_idx],
                })
                episode_lengths.append(current_lengths[env_idx])
                episode_successes.append(current_successes[env_idx])
                completed_episodes += 1
                pbar.update(1)
                # Reset trackers for this env
                current_successes[env_idx] = False
                current_lengths[env_idx] = 0
                prev_chunks[env_idx] = None
                action_history[env_idx].clear()
                gripper_history[env_idx].clear()
                episode_step_logs[env_idx] = []
        observations = next_obs
    pbar.close()

    # Write the metrics log
    if metrics_dir:
        log_dir = Path(metrics_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "metrics_log.jsonl"
        with open(log_path, "w") as f:
            for ep_log in all_episode_logs:
                f.write(json.dumps(ep_log) + "\n")
        print(f"Metrics log saved to {log_path}")

    env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time:.1f}s")
    return env_name, episode_successes, dict(episode_infos)


def create_gr00t_sim_policy(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    policy_client_host: str = "",
    policy_client_port: int | None = None,
) -> BasePolicy:
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    if policy_client_host and policy_client_port:
        from gr00t.policy.server_client import PolicyClient

        return PolicyClient(host=policy_client_host, port=policy_client_port)
    return Gr00tSimPolicyWrapper(
        Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path, device=0)
    )


def run_gr00t_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    metrics_dir: str = "",
    video_dir: str = "",
):
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)
    if not video_dir:
        video_dir = f"/tmp/sim_eval_videos_{env_name}_ac{n_action_steps}_{uuid.uuid4()}"

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=video_dir, max_episode_steps=max_episode_steps),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )
    policy = create_gr00t_sim_policy(
        model_path, embodiment_tag, policy_client_host, policy_client_port
    )
    return run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=n_episodes,
        n_envs=n_envs,
        metrics_dir=metrics_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env_name", type=str, required=True,
                        help="GR00T RoboCasa env name")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--max_episode_steps", type=int, default=504)
    parser.add_argument("--model_path", type=str, default="",
                        help="Local GR00T checkpoint (mutually exclusive with a policy server)")
    parser.add_argument("--policy_client_host", type=str, default="")
    parser.add_argument("--policy_client_port", type=int, default=None)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--metrics_dir", type=str, default="",
                        help="Directory to write metrics_log.jsonl")
    parser.add_argument("--video_dir", type=str, default="")
    args = parser.parse_args()

    assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
        not args.model_path and args.policy_client_host and args.policy_client_port is not None
    ), "Provide EITHER --model_path OR (--policy_client_host & --policy_client_port), not both."

    results = run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        metrics_dir=args.metrics_dir,
        video_dir=args.video_dir,
    )
    print("success rate:", float(np.mean(results[1])))
