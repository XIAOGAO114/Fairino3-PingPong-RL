"""Custom reset events for the Fairino3 ping-pong task."""

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def reset_ball_valid_serve(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_height: float = 0.9125,
    table_half_width: float = 0.7625,
    net_clearance: float = 0.04,
    max_clearance_height: float = 1.42,
    gravity: float = 9.81,
    max_resamples: int = 32,
):
    """Reset the ball with a sampled serve trajectory that clears the net."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    num_envs = len(env_ids)

    pose_ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
        device=asset.device,
    )
    vel_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
        device=asset.device,
    )

    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (num_envs, 6), device=asset.device
    )
    vel_samples = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (num_envs, 6), device=asset.device)

    accepted = _serve_clears_net(
        root_states[:, 0:3] + pose_samples[:, 0:3],
        root_states[:, 7:13] + vel_samples,
        net_x,
        net_height,
        table_half_width,
        net_clearance,
        max_clearance_height,
        gravity,
    )

    for _ in range(max_resamples):
        if torch.all(accepted):
            break
        retry_ids = torch.nonzero(~accepted, as_tuple=False).squeeze(-1)
        pose_samples[retry_ids] = math_utils.sample_uniform(
            pose_ranges[:, 0], pose_ranges[:, 1], (len(retry_ids), 6), device=asset.device
        )
        vel_samples[retry_ids] = math_utils.sample_uniform(
            vel_ranges[:, 0], vel_ranges[:, 1], (len(retry_ids), 6), device=asset.device
        )
        accepted[retry_ids] = _serve_clears_net(
            root_states[retry_ids, 0:3] + pose_samples[retry_ids, 0:3],
            root_states[retry_ids, 7:13] + vel_samples[retry_ids],
            net_x,
            net_height,
            table_half_width,
            net_clearance,
            max_clearance_height,
            gravity,
        )

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pose_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    velocities = root_states[:, 7:13] + vel_samples

    # If the random sampler exhausts retries, raise vertical speed just enough to clear the net.
    if not torch.all(accepted):
        failed = torch.nonzero(~accepted, as_tuple=False).squeeze(-1)
        local_pos = root_states[failed, 0:3] + pose_samples[failed, 0:3]
        vx = torch.clamp(velocities[failed, 0], max=-0.05)
        t_net = (net_x - local_pos[:, 0]) / vx
        required_vz = (net_height + net_clearance - local_pos[:, 2] + 0.5 * gravity * t_net.square()) / t_net
        velocities[failed, 2] = torch.clamp(required_vz, min=vel_ranges[2, 0], max=vel_ranges[2, 1])

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def _serve_clears_net(
    local_pos: torch.Tensor,
    velocity: torch.Tensor,
    net_x: float,
    net_height: float,
    table_half_width: float,
    net_clearance: float,
    max_clearance_height: float,
    gravity: float,
) -> torch.Tensor:
    vx = velocity[:, 0]
    t_net = (net_x - local_pos[:, 0]) / torch.clamp(vx, max=-1.0e-4)
    y_net = local_pos[:, 1] + velocity[:, 1] * t_net
    z_net = local_pos[:, 2] + velocity[:, 2] * t_net - 0.5 * gravity * t_net.square()
    return (
        (local_pos[:, 0] > net_x)
        & (vx < -0.05)
        & (t_net > 0.0)
        & (torch.abs(y_net) < table_half_width)
        & (z_net > net_height + net_clearance)
        & (z_net < max_clearance_height)
    )


def reset_ball_from_crossing(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    pos_mean: tuple[float, float, float] = (0.53, 0.01, 1.00),
    pos_std: tuple[float, float, float] = (0.01, 0.17, 0.05),
    vel_mean: tuple[float, float, float] = (-2.55, -0.04, -1.12),
    vel_std: tuple[float, float, float] = (0.59, 0.32, 0.70),
    home_side: str = "left",
):
    """Reset the ball at the net-crossing instant using REALISTIC incoming-return state.

    Bypasses serve-legality (reset_ball_valid_serve requires a RISING ball). In real
    dual-play the incoming ball is DESCENDING (vz<0) after bouncing off the opponent's
    side — a trajectory the standard serve generator can never produce. Injecting the
    crossing state (pos ~ net_x, vz negative) lets the left arm learn to return
    downward-descending balls, closing the generalisation gap. pos/vel sampled from
    Gaussian(mean,std) collected via collect_returns_bidir (data-driven).

    home_side: left = ball flies -x (toward left arm), return goes +x.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    num_envs = len(env_ids)

    pos_m = torch.tensor(pos_mean, device=asset.device)
    pos_s = torch.tensor(pos_std, device=asset.device)
    vel_m = torch.tensor(vel_mean, device=asset.device)
    vel_s = torch.tensor(vel_std, device=asset.device)

    # Sample crossing-state position and velocity (Gaussian, negative vz allowed)
    pos_off = pos_m + pos_s * torch.randn(num_envs, 3, device=asset.device)
    vel = vel_m + vel_s * torch.randn(num_envs, 3, device=asset.device)

    # GLOBAL coords = env_origin + local (do NOT add default_root_state pos again —
    # that is the ball spawn (0.15,0,1.0); adding it double-counts to z~2.0 sky).
    positions = env.scene.env_origins[env_ids] + pos_off
    orientations = root_states[:, 3:7].clone()
    velocities = torch.cat([vel, torch.zeros(num_envs, 3, device=asset.device)], dim=-1)

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
