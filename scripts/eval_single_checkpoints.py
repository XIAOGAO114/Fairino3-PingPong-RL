"""Evaluate Fairino3 single-arm ping-pong checkpoints.

Launches Isaac once, then evaluates multiple RSL-RL checkpoints in the same
standard-table single-arm environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ISAAC_ROOT = Path("/home/glq/isaac_ws/test_isaac")
RSL_RL_SCRIPT_DIR = ISAAC_ROOT / "scripts" / "rsl_rl"
if str(RSL_RL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate Fairino3-PingPong checkpoints.")
parser.add_argument("--task", type=str, default="Fairino3-PingPong-Rally-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--json_out", type=str, default=None)
parser.add_argument("--trace_episodes", type=int, default=0, help="Print per-episode clean-return traces.")
parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.managers import SceneEntityCfg
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import importlib.metadata as metadata

import isaaclab_tasks  # noqa: F401
import test_isaac.tasks  # noqa: F401
import test_isaac_rail.tasks  # noqa: F401


INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")


def _runner_cfg_dict(agent_cfg) -> dict:
    """Return an rsl-rl >=4 compatible runner config."""
    runner_cfg = agent_cfg.to_dict()
    if not runner_cfg.get("actor"):
        policy_cfg = runner_cfg.pop("policy", {})
        init_std = policy_cfg.pop("init_noise_std", 0.8)
        runner_cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("actor_hidden_dims", [512, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("actor_obs_normalization", False),
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": init_std},
        }
        runner_cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("critic_obs_normalization", False),
        }
        runner_cfg.pop("distribution_cfg", None)
    return runner_cfg


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _pct(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _new_second_hit_state(env, num_envs: int) -> dict[str, torch.Tensor]:
    device = env.unwrapped.device
    return {
        "prev_contact": torch.zeros(num_envs, device=device, dtype=torch.bool),
        "prev_hit": torch.zeros(num_envs, device=device, dtype=torch.bool),
        "steps_since_hit": torch.zeros(num_envs, device=device, dtype=torch.long),
        "seen_true_second": torch.zeros(num_envs, device=device, dtype=torch.bool),
        "prev_ball_vel": torch.zeros((num_envs, 3), device=device),
    }


def _true_second_hit_step(
    env,
    state: dict[str, torch.Tensor],
    dones: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.10,
    grace_steps: int = 4,
    window_steps: int = 60,
    min_return_speed: float = 0.10,
    min_delta_speed: float = 0.03,
) -> dict[str, torch.Tensor | int | float]:
    """Track likely true second hits from kinematics, independent of reward terms."""
    dones = dones.bool()
    robot = env.unwrapped.scene[robot_cfg.name]
    ball = env.unwrapped.scene[ball_cfg.name]
    body_ids = robot_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids, _ = robot.find_bodies(robot_cfg.body_names, preserve_order=True)
    paddle_pos = robot.data.body_pos_w[:, body_ids[0], :] - env.unwrapped.scene.env_origins
    ball_pos = ball.data.root_pos_w[:, :3] - env.unwrapped.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w[:, :3]

    contact_now = torch.linalg.norm(paddle_pos - ball_pos, dim=1) <= contact_distance
    contact_event = contact_now & (~state["prev_contact"])

    hit_done = getattr(env.unwrapped, "_fairino_return_hit_state", None)
    if hit_done is None or hit_done.shape[0] != contact_now.shape[0]:
        hit_done = torch.zeros_like(contact_now)

    new_hit = hit_done & (~state["prev_hit"])
    steps_since_hit = torch.where(
        new_hit,
        torch.zeros_like(state["steps_since_hit"]),
        torch.where(hit_done, state["steps_since_hit"] + 1, torch.zeros_like(state["steps_since_hit"])),
    )

    delta_speed = torch.linalg.norm(ball_vel - state["prev_ball_vel"], dim=1)
    true_second = (
        contact_event
        & hit_done
        & (~new_hit)
        & (~state["seen_true_second"])
        & (steps_since_hit >= int(grace_steps))
        & (steps_since_hit <= int(window_steps))
        & (ball_vel[:, 0] > min_return_speed)
        & (delta_speed > min_delta_speed)
    )

    if torch.any(dones):
        state["prev_contact"][dones] = False
        state["prev_hit"][dones] = False
        state["steps_since_hit"][dones] = 0
        state["seen_true_second"][dones] = False
        state["prev_ball_vel"][dones] = ball_vel[dones]

    active = ~dones
    state["prev_contact"][active] = contact_now[active]
    state["prev_hit"][active] = hit_done[active]
    state["steps_since_hit"][active] = steps_since_hit[active]
    state["seen_true_second"][active] |= true_second[active]
    state["prev_ball_vel"][active] = ball_vel[active]

    return {
        "event": true_second,
        "count": int(true_second.sum().item()),
        "steps": steps_since_hit[true_second].detach().cpu().tolist(),
        "delta_speed": delta_speed[true_second].detach().cpu().tolist(),
        "vx": ball_vel[true_second, 0].detach().cpu().tolist(),
    }


def _new_clean_return_state(env, num_envs: int) -> dict[str, torch.Tensor]:
    device = env.unwrapped.device
    return {
        "prev_hit": torch.zeros(num_envs, device=device, dtype=torch.bool),
        "prev_ball_x": torch.zeros(num_envs, device=device),
        "steps_since_hit": torch.zeros(num_envs, device=device, dtype=torch.long),
        "seen_hit": torch.zeros(num_envs, device=device, dtype=torch.bool),
        "seen_illegal_before_net": torch.zeros(num_envs, device=device, dtype=torch.bool),
        "seen_clean_over_net": torch.zeros(num_envs, device=device, dtype=torch.bool),
    }


def _clean_return_step(
    env,
    state: dict[str, torch.Tensor],
    dones: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_height: float = 0.9125,
    ball_radius: float = 0.02,
    max_over_net_steps: int = 90,
    min_return_speed: float = 0.10,
) -> dict[str, torch.Tensor | int]:
    """Track clean one-hit returns that visibly cross the net after the first hit.

    This metric is intentionally stricter than the training reward.  It counts
    a clean over-net only if the ball crosses the net plane from left to right
    after the first valid return hit, above net height, and before any detected
    illegal second hit.
    """
    ball = env.unwrapped.scene[ball_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.unwrapped.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]
    num_envs = pos.shape[0]
    dones = dones.bool()

    hit_done = getattr(env.unwrapped, "_fairino_return_hit_state", None)
    if hit_done is None or hit_done.shape[0] != num_envs:
        hit_done = torch.zeros(num_envs, device=env.unwrapped.device, dtype=torch.bool)
    illegal_done = getattr(env.unwrapped, "_fairino_illegal_second_seen", None)
    if illegal_done is None or illegal_done.shape[0] != num_envs:
        illegal_done = torch.zeros(num_envs, device=env.unwrapped.device, dtype=torch.bool)

    episode_length_buf = getattr(env.unwrapped, "episode_length_buf", None)
    fresh_reset = (
        episode_length_buf <= 1
        if episode_length_buf is not None
        else torch.zeros(num_envs, device=env.unwrapped.device, dtype=torch.bool)
    )

    new_hit = hit_done & (~state["seen_hit"]) & (~fresh_reset) & (~dones)
    steps_since_hit = torch.where(
        new_hit,
        torch.zeros_like(state["steps_since_hit"]),
        torch.where(hit_done, state["steps_since_hit"] + 1, torch.zeros_like(state["steps_since_hit"])),
    )
    seen_hit = state["seen_hit"] | new_hit
    seen_illegal_before_net = state["seen_illegal_before_net"] | (illegal_done & (~state["seen_clean_over_net"]))

    crossed_net = (state["prev_ball_x"] < net_x) & (pos[:, 0] >= net_x)
    on_or_past_net = pos[:, 0] >= net_x
    above_net = pos[:, 2] > (net_height + ball_radius)
    in_window = (steps_since_hit > 0) & (steps_since_hit <= int(max_over_net_steps))
    clean_over_net = (
        (crossed_net | on_or_past_net)
        & above_net
        & in_window
        & (vel[:, 0] > min_return_speed)
        & hit_done
        & (~fresh_reset)
        & (~dones)
        & (~seen_illegal_before_net)
        & (~state["seen_clean_over_net"])
    )
    seen_clean_over_net = state["seen_clean_over_net"] | clean_over_net

    clean_right = torch.zeros(num_envs, device=env.unwrapped.device, dtype=torch.bool)
    term_names = list(env.unwrapped.termination_manager.active_terms)
    if "right_table_bounce" in term_names:
        right_idx = term_names.index("right_table_bounce")
        right_now = env.unwrapped.termination_manager._term_dones[:, right_idx]
        clean_right = right_now & seen_clean_over_net & (~illegal_done)

    reset_state = dones | fresh_reset
    if torch.any(reset_state):
        state["prev_hit"][reset_state] = False
        state["prev_ball_x"][reset_state] = pos[reset_state, 0]
        state["steps_since_hit"][reset_state] = 0
        state["seen_hit"][reset_state] = False
        state["seen_illegal_before_net"][reset_state] = False
        state["seen_clean_over_net"][reset_state] = False

    active = ~reset_state
    state["prev_hit"][active] = hit_done[active]
    state["prev_ball_x"][active] = pos[active, 0]
    state["steps_since_hit"][active] = steps_since_hit[active]
    state["seen_hit"][active] = seen_hit[active]
    state["seen_illegal_before_net"][active] = seen_illegal_before_net[active]
    state["seen_clean_over_net"][active] = seen_clean_over_net[active]

    return {
        "new_hit": new_hit,
        "clean_over_net": clean_over_net,
        "clean_right": clean_right,
        "hit_done": hit_done,
        "illegal_before_net": seen_illegal_before_net,
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL_VERSION)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg.log_dir = os.path.abspath(os.path.dirname(args_cli.checkpoints[0]))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cfg = _runner_cfg_dict(agent_cfg)
    device = agent_cfg.device
    env_device = env.unwrapped.device
    reward_manager = env.unwrapped.reward_manager
    termination_manager = env.unwrapped.termination_manager
    reward_names = list(reward_manager.active_terms)
    termination_names = list(termination_manager.active_terms)
    reward_index = {name: i for i, name in enumerate(reward_names)}
    right_term_index = termination_names.index("right_table_bounce") if "right_table_bounce" in termination_names else None

    key_rewards = [
        "right_table_bounce",
        "target_landing",
        "predicted_landing",
        "return_ball",
        "first_return_hit",
        "second_paddle_contact",
        "post_hit_paddle_clearance",
        "post_hit_paddle_retreat",
        "legal_return_separation",
        "rally_return",
        "net_direction",
        "net_height",
        "paddle_to_intercept",
        "post_return_idle",
        "action_rate",
        "joint_vel",
        "joint_limit",
        "table_collision",
    ]

    results = []
    for checkpoint in args_cli.checkpoints:
        checkpoint = os.path.abspath(checkpoint)
        if not os.path.exists(checkpoint):
            print(f"[WARN] Missing checkpoint, skipping: {checkpoint}", flush=True)
            continue
        print(f"[EVAL] Loading {checkpoint}", flush=True)
        runner = OnPolicyRunner(env, copy.deepcopy(runner_cfg), log_dir=None, device=device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env_device)

        obs, _ = env.reset()
        second_hit_state = _new_second_hit_state(env, args_cli.num_envs)
        clean_return_state = _new_clean_return_state(env, args_cli.num_envs)
        termination_counts = {name: 0 for name in termination_names}
        reward_sums = {name: 0.0 for name in key_rewards if name in reward_index}
        reward_peaks = {name: 0.0 for name in key_rewards if name in reward_index}
        log_values = defaultdict(list)
        right_bounce_x_values = []
        right_bounce_vx_values = []
        near_net_right_bounce_count = 0
        right_bounce_legal_count = 0
        right_bounce_illegal_count = 0
        true_second_hit_count = 0
        true_second_hit_steps = []
        true_second_hit_delta_speed = []
        true_second_hit_vx = []
        true_second_before_right_count = 0
        true_second_before_out_count = 0
        clean_hit_count = 0
        clean_over_net_count = 0
        clean_right_count = 0
        illegal_before_net_done_count = 0
        trace_seen_hit = torch.zeros(args_cli.num_envs, device=env_device, dtype=torch.bool)
        trace_seen_over_net = torch.zeros(args_cli.num_envs, device=env_device, dtype=torch.bool)
        trace_seen_clean_right = torch.zeros(args_cli.num_envs, device=env_device, dtype=torch.bool)
        trace_seen_illegal = torch.zeros(args_cli.num_envs, device=env_device, dtype=torch.bool)
        trace_episode_steps = torch.zeros(args_cli.num_envs, device=env_device, dtype=torch.long)
        trace_printed = 0
        total_reward = 0.0
        done_count = 0
        env_steps = args_cli.steps * args_cli.num_envs

        for _ in range(args_cli.steps):
            with torch.no_grad():
                actions = policy(obs)
                obs, rewards, dones, extras = env.step(actions)
                dones = dones.bool()
                policy.reset(dones)

            total_reward += float(rewards.sum().item())
            done_count += int(dones.sum().item())

            term_dones = termination_manager._term_dones
            for idx, name in enumerate(termination_names):
                termination_counts[name] += int(term_dones[:, idx].sum().item())
            if right_term_index is not None:
                right_mask = term_dones[:, right_term_index]
                right_count = int(right_mask.sum().item())
                if right_count:
                    bounce_x = env.unwrapped._right_bounce_x_at_event[right_mask]
                    bounce_y = env.unwrapped._right_bounce_y_at_event[right_mask]
                    bounce_vx = env.unwrapped._right_bounce_vx_at_event[right_mask]
                    illegal_flags = getattr(env.unwrapped, "_right_bounce_illegal_at_event", None)
                    if illegal_flags is None or illegal_flags.shape[0] != right_mask.shape[0]:
                        right_bounce_legal_count += right_count
                    else:
                        illegal_at_bounce = illegal_flags[right_mask]
                        illegal_count = int(illegal_at_bounce.sum().item())
                        right_bounce_illegal_count += illegal_count
                        right_bounce_legal_count += right_count - illegal_count
                    right_bounce_x_values.extend(float(value) for value in bounce_x.detach().cpu())
                    right_bounce_vx_values.extend(float(value) for value in bounce_vx.detach().cpu())
                    near_net = (bounce_x > 0.55) & (bounce_x < 1.00) & (torch.abs(bounce_y) < 0.45)
                    enough_speed = bounce_vx > 0.75
                    near_net_right_bounce_count += int((near_net & enough_speed).sum().item())

            second_diag = _true_second_hit_step(env, second_hit_state, dones)
            true_second_hit_count += int(second_diag["count"])
            true_second_hit_steps.extend(float(value) for value in second_diag["steps"])
            true_second_hit_delta_speed.extend(float(value) for value in second_diag["delta_speed"])
            true_second_hit_vx.extend(float(value) for value in second_diag["vx"])
            if int(second_diag["count"]):
                event_mask = second_diag["event"]
                if right_term_index is not None:
                    true_second_before_right_count += int((event_mask & term_dones[:, right_term_index]).sum().item())
                if "ball_out_of_bounds" in termination_names:
                    out_idx = termination_names.index("ball_out_of_bounds")
                    true_second_before_out_count += int((event_mask & term_dones[:, out_idx]).sum().item())

            clean_diag = _clean_return_step(env, clean_return_state, dones)
            clean_hit_count += int(clean_diag["new_hit"].sum().item())
            clean_over_net_count += int(clean_diag["clean_over_net"].sum().item())
            clean_right_count += int(clean_diag["clean_right"].sum().item())
            illegal_before_net_done_count += int((dones & clean_diag["illegal_before_net"]).sum().item())
            if args_cli.trace_episodes > 0:
                illegal_done = getattr(env.unwrapped, "_fairino_illegal_second_seen", None)
                if illegal_done is None or illegal_done.shape[0] != args_cli.num_envs:
                    illegal_done = torch.zeros(args_cli.num_envs, device=env_device, dtype=torch.bool)
                trace_episode_steps += 1
                trace_seen_hit |= clean_diag["new_hit"]
                trace_seen_over_net |= clean_diag["clean_over_net"]
                trace_seen_clean_right |= clean_diag["clean_right"]
                trace_seen_illegal |= illegal_done
                done_indices = torch.nonzero(dones, as_tuple=False).flatten().detach().cpu().tolist()
                for env_id in done_indices:
                    if trace_printed >= args_cli.trace_episodes:
                        break
                    reasons = [
                        name
                        for idx, name in enumerate(termination_names)
                        if bool(term_dones[env_id, idx].item())
                    ]
                    print(
                        "[TRACE] "
                        f"episode={trace_printed + 1} env={env_id} "
                        f"steps={int(trace_episode_steps[env_id].item())} "
                        f"hit={bool(trace_seen_hit[env_id].item())} "
                        f"clean_over_net={bool(trace_seen_over_net[env_id].item())} "
                        f"clean_right={bool(trace_seen_clean_right[env_id].item())} "
                        f"illegal={bool(trace_seen_illegal[env_id].item())} "
                        f"reasons={','.join(reasons) if reasons else 'none'}",
                        flush=True,
                    )
                    trace_printed += 1
                if torch.any(dones):
                    trace_seen_hit[dones] = False
                    trace_seen_over_net[dones] = False
                    trace_seen_clean_right[dones] = False
                    trace_seen_illegal[dones] = False
                    trace_episode_steps[dones] = 0

            step_reward = reward_manager._step_reward
            for name, idx in reward_index.items():
                if name in reward_sums:
                    value = float(step_reward[:, idx].mean().item())
                    reward_sums[name] += value
                    if value > reward_peaks[name]:
                        reward_peaks[name] = value

            for key, value in extras.get("log", {}).items():
                if key.startswith("Episode_Reward/") or key.startswith("Episode_Termination/"):
                    log_values[key].append(float(value))

        summary = {
            "checkpoint": checkpoint,
            "steps": args_cli.steps,
            "num_envs": args_cli.num_envs,
            "env_steps": env_steps,
            "done_count": done_count,
            "mean_reward_per_env_step": total_reward / env_steps,
            "termination_counts": termination_counts,
            "termination_rates": {name: count / env_steps for name, count in termination_counts.items()},
            "reward_step_means": {name: reward_sums[name] / args_cli.steps for name in reward_sums},
            "reward_step_peaks": reward_peaks,
            "episode_log_means": {key: _mean(values) for key, values in log_values.items()},
            "episode_log_peaks": {key: max(values) for key, values in log_values.items()},
            "right_bounce_quality": {
                "count": len(right_bounce_x_values),
                "legal_count": right_bounce_legal_count,
                "illegal_count": right_bounce_illegal_count,
                "legal_fraction": _pct(right_bounce_legal_count, len(right_bounce_x_values)),
                "near_net_speed_count": near_net_right_bounce_count,
                "near_net_speed_fraction": _pct(near_net_right_bounce_count, len(right_bounce_x_values)),
                "mean_x": _mean(right_bounce_x_values),
                "mean_vx": _mean(right_bounce_vx_values),
            },
            "true_second_hit": {
                "count": true_second_hit_count,
                "event_rate_per_env_step": true_second_hit_count / env_steps,
                "event_rate_per_done": _pct(true_second_hit_count, done_count),
                "same_step_right_bounce_count": true_second_before_right_count,
                "same_step_out_count": true_second_before_out_count,
                "mean_steps_after_hit": _mean(true_second_hit_steps),
                "mean_delta_speed": _mean(true_second_hit_delta_speed),
                "mean_vx": _mean(true_second_hit_vx),
            },
            "clean_return": {
                "hit_count": clean_hit_count,
                "clean_over_net_count": clean_over_net_count,
                "clean_right_count": clean_right_count,
                "clean_over_net_per_hit": _pct(clean_over_net_count, clean_hit_count),
                "clean_right_per_hit": _pct(clean_right_count, clean_hit_count),
                "clean_right_per_over_net": _pct(clean_right_count, clean_over_net_count),
                "illegal_before_net_done_count": illegal_before_net_done_count,
                "illegal_before_net_done_per_hit": _pct(illegal_before_net_done_count, clean_hit_count),
            },
        }
        results.append(summary)

        out_rate = summary["termination_rates"].get("ball_out_of_bounds", 0.0)
        joint_rate = summary["termination_rates"].get("robot_joint_limit", 0.0)
        table_rate = summary["termination_rates"].get("robot_table_collision", 0.0)
        right_mean = summary["reward_step_means"].get("right_table_bounce", 0.0)
        target_mean = summary["reward_step_means"].get("target_landing", 0.0)
        pred_mean = summary["reward_step_means"].get("predicted_landing", 0.0)
        hit_mean = summary["reward_step_means"].get("first_return_hit", 0.0)
        return_mean = summary["reward_step_means"].get("return_ball", 0.0)
        second_mean = summary["reward_step_means"].get("second_paddle_contact", 0.0)
        clearance_mean = summary["reward_step_means"].get("post_hit_paddle_clearance", 0.0)
        retreat_mean = summary["reward_step_means"].get("post_hit_paddle_retreat", 0.0)
        legal_sep_mean = summary["reward_step_means"].get("legal_return_separation", 0.0)
        quality = summary["right_bounce_quality"]
        true_second = summary["true_second_hit"]
        clean_return = summary["clean_return"]
        print(
            "[RESULT] "
            f"{os.path.basename(checkpoint)} "
            f"reward={summary['mean_reward_per_env_step']:.6g} "
            f"right={right_mean:.6g} target={target_mean:.6g} "
            f"predicted={pred_mean:.6g} return={return_mean:.6g} hit={hit_mean:.6g} "
            f"second={second_mean:.6g} clearance={clearance_mean:.6g} retreat={retreat_mean:.6g} "
            f"legal_sep={legal_sep_mean:.6g} "
            f"out_rate={out_rate:.8g} joint_rate={joint_rate:.8g} table_rate={table_rate:.8g} "
            f"right_count={quality['count']} near_net_speed={quality['near_net_speed_fraction']:.3f} "
            f"legal_right={quality['legal_fraction']:.3f} "
            f"mean_x={quality['mean_x']:.3f} mean_vx={quality['mean_vx']:.3f} "
            f"clean_hit={clean_return['hit_count']} "
            f"clean_over_net={clean_return['clean_over_net_count']} "
            f"clean_over_net_per_hit={clean_return['clean_over_net_per_hit']:.3f} "
            f"clean_right={clean_return['clean_right_count']} "
            f"clean_right_per_hit={clean_return['clean_right_per_hit']:.3f} "
            f"true_second_count={true_second['count']} "
            f"true_second_done_rate={true_second['event_rate_per_done']:.4f} "
            f"true_second_steps={true_second['mean_steps_after_hit']:.2f} "
            f"true_second_dv={true_second['mean_delta_speed']:.3f}",
            flush=True,
        )

    if args_cli.json_out:
        output_path = Path(args_cli.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote {output_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
