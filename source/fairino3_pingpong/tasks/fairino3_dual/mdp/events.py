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
    """Reset the ball with a sampled serve trajectory that clears the net.

    50/50 random direction: toward the left arm (ball starts right of net,
    vx < 0) or toward the right arm (ball starts left of net, vx > 0).

    Right-direction serves are created by mirroring the position around *net_x*
    and flipping the sign of vx — this keeps physics symmetric without
    hard-coding magic offsets.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    num_envs = len(env_ids)
    device = asset.device

    pose_ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
        device=device,
    )
    vel_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
        device=device,
    )

    # -- randomly choose serve direction (False = toward left, True = toward right) --
    serve_right_prob = getattr(env.unwrapped, '_serve_right_prob', 0.5) if hasattr(env.unwrapped, '_serve_right_prob') else 0.5
    serve_right = torch.rand(num_envs, device=device) < serve_right_prob

    # -- curriculum: scale ball speed (default 1.0 = normal) --
    ball_speed_scale = getattr(env.unwrapped, '_ball_speed_scale', 1.0) if hasattr(env.unwrapped, '_ball_speed_scale') else 1.0

    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (num_envs, 6), device=device
    )
    vel_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (num_envs, 6), device=device
    )
    # apply curriculum speed scaling
    vel_samples[:, :3] *= ball_speed_scale

    # build total local position & velocity, then mirror for right-direction serves
    local_pos = root_states[:, 0:3] + pose_samples[:, 0:3]
    velocities = root_states[:, 7:13] + vel_samples
    _mirror_serve_x(serve_right, local_pos, velocities, net_x)

    accepted = _serve_clears_net(
        local_pos, velocities, net_x, net_height, table_half_width,
        net_clearance, max_clearance_height, gravity,
    )

    for _ in range(max_resamples):
        if torch.all(accepted):
            break
        retry_ids = torch.nonzero(~accepted, as_tuple=False).squeeze(-1)
        pose_samples[retry_ids] = math_utils.sample_uniform(
            pose_ranges[:, 0], pose_ranges[:, 1], (len(retry_ids), 6), device=device
        )
        vel_samples[retry_ids] = math_utils.sample_uniform(
            vel_ranges[:, 0], vel_ranges[:, 1], (len(retry_ids), 6), device=device
        )
        vel_samples[retry_ids, :3] *= ball_speed_scale
        local_pos[retry_ids] = root_states[retry_ids, 0:3] + pose_samples[retry_ids, 0:3]
        velocities[retry_ids] = root_states[retry_ids, 7:13] + vel_samples[retry_ids]
        _mirror_serve_x(serve_right[retry_ids], local_pos[retry_ids], velocities[retry_ids], net_x)

        accepted[retry_ids] = _serve_clears_net(
            local_pos[retry_ids], velocities[retry_ids],
            net_x, net_height, table_half_width,
            net_clearance, max_clearance_height, gravity,
        )

    # positions in world frame
    positions = local_pos + env.scene.env_origins[env_ids]
    orientations_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # fallback: force vz to clear net
    if not torch.all(accepted):
        failed = torch.nonzero(~accepted, as_tuple=False).squeeze(-1)
        goes_right_f = velocities[failed, 0] > 0.05
        vx_clamped = torch.where(
            goes_right_f,
            torch.clamp(velocities[failed, 0], min=0.05),
            torch.clamp(velocities[failed, 0], max=-0.05),
        )
        t_net = (net_x - local_pos[failed, 0]) / vx_clamped
        required_vz = (net_height + net_clearance - local_pos[failed, 2] + 0.5 * gravity * t_net.square()) / t_net
        velocities[failed, 2] = torch.clamp(required_vz, min=vel_ranges[2, 0], max=vel_ranges[2, 1])

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def _mirror_serve_x(
    serve_right: torch.Tensor,
    local_pos: torch.Tensor,
    velocities: torch.Tensor,
    net_x: float,
) -> None:
    """Mirror serve x-position around *net_x* and flip vx for right-direction envs.

    The default serve starts right of the net and goes left.  Mirroring around
    the net flips it to start left of the net and go right.
    """
    if not torch.any(serve_right):
        return
    # mirror position: x' = net_x - (x - net_x) = 2*net_x - x
    local_pos[serve_right, 0] = 2.0 * net_x - local_pos[serve_right, 0]
    # flip velocity direction
    velocities[serve_right, 0] = -velocities[serve_right, 0]


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
    """Check that a serve trajectory clears the net.  Handles both directions."""
    vx = velocity[:, 0]
    px = local_pos[:, 0]

    goes_right = vx > 0.05
    goes_left = vx < -0.05

    # clamp vx away from zero in the correct direction for safe t_net division
    vx_clamped = torch.where(
        goes_right,
        torch.clamp(vx, min=1.0e-4),
        torch.clamp(vx, max=-1.0e-4),
    )
    t_net = (net_x - px) / vx_clamped

    y_net = local_pos[:, 1] + velocity[:, 1] * t_net
    z_net = local_pos[:, 2] + velocity[:, 2] * t_net - 0.5 * gravity * t_net.square()

    # ball must start on the opposite side of the net from where it's going
    valid_side = (goes_right & (px < net_x)) | (goes_left & (px > net_x))

    return (
        valid_side
        & (goes_right | goes_left)
        & (t_net > 0.0)
        & (torch.abs(y_net) < table_half_width)
        & (z_net > net_height + net_clearance)
        & (z_net < max_clearance_height)
    )
