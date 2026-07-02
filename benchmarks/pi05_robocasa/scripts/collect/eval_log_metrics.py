#!/usr/bin/env python3
"""
Pi0.5 RoboCasa rollout with step-level action-space metric logging  [REFERENCE].

Like the GR00T benchmark, Pi0.5 data collection is *model-coupled*: features are
logged online during evaluation, because the denoising / ODE-trajectory curvature
is returned by the policy itself. The script connects to an openpi Pi0.5 policy
server over WebSocket and rolls out RoboCasa tasks while logging per-step features.

It is provided as **reference** — it documents how the shipped
`data/metrics_logs/*.jsonl` were produced. It is NOT runnable from this repository
alone: it requires an openpi Pi0.5 policy server (`openpi_client`), RoboCasa and
robosuite. Point `--host/--port` at a running server to reproduce the rollouts.

The same harness is used for both the single-stage benchmark (the 24 tasks below)
and the multi-stage benchmark (pass the 5 composite task names via `--tasks`).

Per-step features (GR00T-compatible JSONL):
    chunk_mse           -> TCE (Temporal Consistency Error), used by ActProbe
    action_norm         -> ACM (Action Chunk Magnitude),     used by ActProbe
    action_jerk         -> 2nd-order action difference RMS
    gripper_oscillation -> gripper sign-change count in a sliding window
    physics_info        -> eef_pos + gripper_qpos (auxiliary, not an ActProbe input)
    denoising_curvature -> ODE-trajectory curvature returned by the policy
"""

import argparse
import collections
import json
import logging
import math
import pathlib

import imageio
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 24 single-stage RoboCasa task names
ALL_TASKS = [
    "PnPCounterToCab", "PnPCabToCounter", "PnPCounterToSink", "PnPSinkToCounter",
    "PnPCounterToMicrowave", "PnPMicrowaveToCounter", "PnPCounterToStove", "PnPStoveToCounter",
    "OpenSingleDoor", "CloseSingleDoor", "OpenDoubleDoor", "CloseDoubleDoor",
    "OpenDrawer", "CloseDrawer",
    "TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout",
    "TurnOnStove", "TurnOffStove",
    "CoffeeSetupMug", "CoffeeServeMug", "CoffeePressButton",
    "TurnOnMicrowave", "TurnOffMicrowave",
]

TASK_DESCRIPTIONS = {
    "PnPCounterToCab": "pick the object from the counter and place it in the cabinet",
    "PnPCabToCounter": "pick the object from the cabinet and place it on the counter",
    "PnPCounterToSink": "pick the object from the counter and place it in the sink",
    "PnPSinkToCounter": "pick the object from the sink and place it on the counter",
    "PnPCounterToMicrowave": "pick the object from the counter and place it in the microwave",
    "PnPMicrowaveToCounter": "pick the object from the microwave and place it on the counter",
    "PnPCounterToStove": "pick the object from the counter and place it on the stove",
    "PnPStoveToCounter": "pick the object from the stove and place it on the counter",
    "OpenSingleDoor": "open the single door cabinet",
    "CloseSingleDoor": "close the single door cabinet",
    "OpenDoubleDoor": "open the double door cabinet",
    "CloseDoubleDoor": "close the double door cabinet",
    "OpenDrawer": "open the drawer",
    "CloseDrawer": "close the drawer",
    "TurnOnSinkFaucet": "turn on the sink faucet",
    "TurnOffSinkFaucet": "turn off the sink faucet",
    "TurnSinkSpout": "turn the sink spout",
    "TurnOnStove": "turn on the stove",
    "TurnOffStove": "turn off the stove",
    "CoffeeSetupMug": "set up the mug on the coffee machine",
    "CoffeeServeMug": "serve the coffee mug",
    "CoffeePressButton": "press the coffee machine button",
    "TurnOnMicrowave": "turn on the microwave",
    "TurnOffMicrowave": "turn off the microwave",
}

MAX_STEPS = 500
DUMMY_ACTION_DIM = 12
REPLAN_STEPS = 5
IMAGE_SIZE = 256


def quat_to_axis_angle(quat):
    """Convert quaternion [x, y, z, w] to axis-angle (3D)."""
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(w)) / den


def extract_state_from_obs(raw_obs):
    """10-D state: eef_pos(3) + eef_rot(3) + gripper_qpos(2) + base_xy(2)."""
    eef_pos = raw_obs["robot0_eef_pos"]
    eef_rot = quat_to_axis_angle(raw_obs["robot0_eef_quat"])
    gripper_qpos = raw_obs["robot0_gripper_qpos"]
    base_pos = raw_obs.get("robot0_base_pos", np.zeros(3))
    return np.concatenate([eef_pos, eef_rot, gripper_qpos, base_pos[:2]]).astype(np.float32)


def make_robocasa_env(task_name, image_size=IMAGE_SIZE, seed=0):
    import robocasa  # noqa: F401
    import robosuite as suite

    env = suite.make(
        env_name=task_name,
        robots="PandaMobile",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_heights=image_size,
        camera_widths=image_size,
        ignore_done=True,
        reward_shaping=False,
    )
    env.seed = seed
    return env


def get_observation_for_policy(raw_obs, task_description, resize_size=224):
    """Convert raw RoboCasa obs to the policy server's input format."""
    from openpi_client import image_tools

    side_0 = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(raw_obs["robot0_agentview_left_image"], resize_size, resize_size))
    side_1 = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(raw_obs["robot0_agentview_right_image"], resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(raw_obs["robot0_eye_in_hand_image"], resize_size, resize_size))

    return {
        "observation/image_side_0": side_0,
        "observation/image_side_1": side_1,
        "observation/wrist_image": wrist,
        "observation/state": extract_state_from_obs(raw_obs),
        "prompt": task_description,
    }


def action_model_to_env(action):
    """Convert 12-D model action to robosuite action format.

    Model:     [gripper_close(1), ee_pos(3), ee_rot(3), base_motion(4), control_mode(1)]
    Robosuite: [ee_pos(3), ee_rot(3), gripper(1), base(3), torso(1), base_mode(1)]
    """
    gripper = 1.0 if action[0] > 0.5 else -1.0
    base_mode = 1.0 if action[11] > 0.5 else -1.0
    ee_pos, ee_rot, base_motion = action[1:4], action[4:7], action[7:11]
    return np.concatenate([ee_pos, ee_rot, [gripper], base_motion[:3], base_motion[3:4], [base_mode]])


# ── Metric computation helpers ──────────────────────────────────────────────

def compute_action_norm(action_chunk, replan_steps):
    """ACM: RMS of the executed portion of the action chunk."""
    executed = np.array(action_chunk[:replan_steps])
    return float(np.sqrt(np.mean(executed ** 2)))


def compute_action_jerk(action_history):
    if len(action_history) < 3:
        return None
    a = action_history
    jerk = a[-1] - 2 * a[-2] + a[-3]
    return float(np.sqrt(np.mean(jerk ** 2)))


def compute_chunk_mse(prev_chunk, curr_chunk, replan_steps):
    """TCE: MSE between overlapping portions of consecutive action chunks."""
    if prev_chunk is None:
        return None
    prev_tail = np.array(prev_chunk[replan_steps:])
    curr_head = np.array(curr_chunk[:len(prev_tail)])
    if len(curr_head) == 0 or len(prev_tail) == 0:
        return None
    min_len = min(len(prev_tail), len(curr_head))
    return float(np.mean((prev_tail[:min_len] - curr_head[:min_len]) ** 2))


def compute_gripper_oscillation(gripper_history, window=10):
    if len(gripper_history) < 2:
        return 0
    recent = gripper_history[-window:]
    binary = [1 if g >= 0.5 else 0 for g in recent]
    return sum(1 for i in range(1, len(binary)) if binary[i] != binary[i - 1])


# ── Main evaluation loop ────────────────────────────────────────────────────

def eval_single_task(client, task_name, num_trials, metrics_dir, video_out_path,
                     seed=0, resize_size=224, replan_steps=REPLAN_STEPS,
                     num_steps_wait=10, episode_id_offset=0):
    task_desc = TASK_DESCRIPTIONS.get(task_name, task_name)
    logger.info(f"Evaluating: {task_name} ({task_desc})")

    env = make_robocasa_env(task_name, seed=seed)
    metrics_file = metrics_dir / f"{task_name}_PandaOmron_Env.jsonl"
    fh = open(metrics_file, "a")

    successes = 0
    for trial in range(num_trials):
        raw_obs = env.reset()
        action_plan = collections.deque()
        replay_images = []
        done = False

        steps_log = []
        action_mean_history = []
        prev_chunk = None
        gripper_history = []
        step_idx = 0

        for t in range(MAX_STEPS + num_steps_wait):
            if t < num_steps_wait:
                dummy_action = np.zeros(DUMMY_ACTION_DIM)
                dummy_action[6] = -1.0
                dummy_action[11] = -1.0
                raw_obs, reward, done_flag, info = env.step(dummy_action)
                continue

            obs_dict = get_observation_for_policy(raw_obs, task_desc, resize_size)
            replay_images.append(obs_dict["observation/image_side_0"])

            denoising_curvature = None
            curr_chunk = None
            is_replan = False
            if not action_plan:
                is_replan = True
                result = client.infer(obs_dict)
                action_chunk = result["actions"]
                curr_chunk = [a.tolist() if hasattr(a, "tolist") else list(a) for a in action_chunk]
                assert len(action_chunk) >= replan_steps, (
                    f"Policy predicts {len(action_chunk)} steps, need >= {replan_steps}")
                action_plan.extend(action_chunk[:replan_steps])
                denoising_curvature = result.get("denoising_curvature", None)

            action = action_plan.popleft()
            action_np = np.array(action) if not isinstance(action, np.ndarray) else action
            env_action = action_model_to_env(action_np)

            step_data = {"t": step_idx, "is_replan": is_replan}

            step_data["physics_info"] = {
                "state.end_effector_position_absolute": raw_obs["robot0_eef_pos"].tolist(),
                "state.gripper_qpos": raw_obs["robot0_gripper_qpos"].tolist(),
            }

            # action_norm (ACM)
            if is_replan and curr_chunk is not None:
                step_data["action_norm"] = compute_action_norm(curr_chunk, replan_steps)
            else:
                step_data["action_norm"] = float(np.sqrt(np.mean(action_np ** 2)))

            action_mean_history.append(action_np.mean(axis=0) if action_np.ndim > 1 else action_np)
            jerk = compute_action_jerk(action_mean_history)
            if jerk is not None:
                step_data["action_jerk"] = jerk

            # chunk_mse (TCE)
            if is_replan and curr_chunk is not None:
                mse = compute_chunk_mse(prev_chunk, curr_chunk, replan_steps)
                if mse is not None:
                    step_data["chunk_mse"] = mse
                prev_chunk = curr_chunk

            gripper_val = action_np[0] if len(action_np) >= 1 else 0.0
            gripper_history.append(float(gripper_val))
            step_data["gripper_oscillation"] = compute_gripper_oscillation(gripper_history)

            if denoising_curvature is not None:
                step_data["denoising_curvature"] = float(denoising_curvature)

            steps_log.append(step_data)
            step_idx += 1

            raw_obs, reward, done_flag, info = env.step(env_action.tolist())

            if env._check_success():
                done = True
                successes += 1
                break

        fh.write(json.dumps({
            "episode_id": episode_id_offset + trial,
            "env_idx": 0,
            "success": done,
            "length": step_idx,
            "steps": steps_log,
        }) + "\n")
        fh.flush()

        # Save a sample of failure videos
        if replay_images and video_out_path and not done and np.random.random() < 0.3:
            try:
                video_file = video_out_path / f"{task_name}_trial{trial}_failure.mp4"
                writer = imageio.get_writer(str(video_file), fps=10, codec="libx264", quality=5)
                for frame in replay_images:
                    writer.append_data(frame)
                writer.close()
            except Exception as e:
                logger.warning(f"Failed to save video: {e}")

        logger.info(f"  {task_name} trial {trial + 1}/{num_trials}: "
                    f"{'SUCCESS' if done else 'FAIL'} "
                    f"(cumulative: {successes}/{trial + 1} = {successes / (trial + 1) * 100:.1f}%)")

    fh.close()
    env.close()
    return successes, num_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="openpi policy server host")
    parser.add_argument("--port", type=int, default=8000, help="openpi policy server port")
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--num_trials", type=int, default=50, help="Trials per task")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Specific tasks (default: all 24 single-stage; "
                             "pass the composite task names for the multi-stage benchmark)")
    parser.add_argument("--video_out_path", type=str, default=None)
    parser.add_argument("--metrics_dir", type=str, required=True,
                        help="Directory for JSONL metrics output")
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from openpi_client import websocket_client_policy as wcp

    np.random.seed(args.seed)
    client = wcp.WebsocketClientPolicy(args.host, args.port)

    tasks = args.tasks if args.tasks else ALL_TASKS
    metrics_dir = pathlib.Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    video_out_path = None
    if args.video_out_path:
        video_out_path = pathlib.Path(args.video_out_path)
        video_out_path.mkdir(parents=True, exist_ok=True)

    results = {}
    total_successes = 0
    total_trials = 0
    for task_name in tasks:
        successes, trials = eval_single_task(
            client=client, task_name=task_name, num_trials=args.num_trials,
            metrics_dir=metrics_dir, video_out_path=video_out_path, seed=args.seed,
            resize_size=args.resize_size, replan_steps=args.replan_steps,
            num_steps_wait=args.num_steps_wait, episode_id_offset=total_trials,
        )
        rate = successes / trials * 100
        results[task_name] = {"successes": successes, "trials": trials, "rate": rate}
        total_successes += successes
        total_trials += trials
        logger.info(f"{task_name}: {rate:.1f}% ({successes}/{trials})")

    overall_rate = total_successes / total_trials * 100 if total_trials > 0 else 0
    results["_overall"] = {"successes": total_successes, "trials": total_trials, "rate": overall_rate}
    logger.info(f"Overall: {overall_rate:.1f}%  ({total_successes}/{total_trials})")

    results_path = args.results_path or str(metrics_dir / "results_summary.json")
    results_file = pathlib.Path(results_path)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
