"""Custom terminations for the Fairino3 ping-pong task."""

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from ._shared import (
    clean_over_net_event,
    has_clean_over_net,
    has_illegal_second_hit,
    has_return_hit,
    illegal_second_hit_event,
    paddle_contact_event,
)


def ball_out_of_bounds(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    x_bounds: tuple[float, float] = (-1.52, 2.62),
    y_bounds: tuple[float, float] = (-1.2125, 1.2125),
    z_min: float = 0.03,
) -> torch.Tensor:
    """Terminate when the ball leaves the useful training volume."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    return (
        (pos[:, 0] < x_bounds[0])
        | (pos[:, 0] > x_bounds[1])
        | (pos[:, 1] < y_bounds[0])
        | (pos[:, 1] > y_bounds[1])
        | (pos[:, 2] < z_min)
    )


def right_table_bounce(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    table_center: tuple[float, float, float] = (0.55, 0.0, 0.74),
    table_size: tuple[float, float, float] = (2.74, 1.525, 0.04),
    ball_radius: float = 0.02,
    net_x: float = 0.55,
    height_tolerance: float = 0.035,
    min_upward_velocity: float = 0.05,
    illegal_contact_distance: float = 0.10,
    allow_illegal_success: bool = False,
    require_clean_over_net: bool = False,
    home_side: str = "right",
) -> torch.Tensor:
    """Terminate on opponent-table bounces."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]
    center = torch.tensor(table_center, device=env.device)
    size = torch.tensor(table_size, device=env.device)
    half_size = 0.5 * size
    table_top = center[2] + half_size[2]
    ball_center_at_table = table_top + ball_radius

    if home_side == "right":
        in_opponent_half = (pos[:, 0] < net_x) & (pos[:, 0] > center[0] - half_size[0])
        returning = (vel[:, 0] < -0.05) & has_return_hit(env, pos.shape[0])
    else:
        in_opponent_half = (pos[:, 0] > net_x) & (pos[:, 0] < center[0] + half_size[0])
        returning = (vel[:, 0] > 0.05) & has_return_hit(env, pos.shape[0])
    in_table_y = torch.abs(pos[:, 1] - center[1]) < half_size[1]
    near_table_height = torch.abs(pos[:, 2] - ball_center_at_table) < height_tolerance
    bounced_up = vel[:, 2] > min_upward_velocity
    clean_over_net_event(
        env,
        ball_cfg=asset_cfg,
        robot_cfg=robot_cfg,
        net_x=net_x,
        net_height=table_top + 0.1525,
        ball_radius=ball_radius,
        illegal_contact_distance=illegal_contact_distance,
        home_side=home_side,
    )
    illegal_second_hit_event(
        env,
        robot_cfg=robot_cfg,
        ball_cfg=asset_cfg,
        contact_distance=illegal_contact_distance,
    )
    legal = ~has_illegal_second_hit(env, pos.shape[0])
    clean_gate = has_clean_over_net(env, pos.shape[0]) if require_clean_over_net else torch.ones_like(legal)
    raw_bounce = in_opponent_half & in_table_y & near_table_height & bounced_up & returning
    bounce = raw_bounce & clean_gate & (legal | bool(allow_illegal_success))
    if torch.any(bounce):
        num_envs = pos.shape[0]
        if not hasattr(env, "_right_bounce_x_at_event") or env._right_bounce_x_at_event.shape[0] != num_envs:
            env._right_bounce_x_at_event = torch.zeros(num_envs, device=env.device)
            env._right_bounce_y_at_event = torch.zeros(num_envs, device=env.device)
            env._right_bounce_vx_at_event = torch.zeros(num_envs, device=env.device)
            env._right_bounce_illegal_at_event = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        env._right_bounce_x_at_event[bounce] = pos[bounce, 0]
        env._right_bounce_y_at_event[bounce] = pos[bounce, 1]
        env._right_bounce_vx_at_event[bounce] = vel[bounce, 0]
        env._right_bounce_illegal_at_event[bounce] = ~legal[bounce]
    return bounce


def second_paddle_contact(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.10,
) -> torch.Tensor:
    """Terminate when the paddle touches the ball more than once in an episode."""
    _, _, second_contact = paddle_contact_event(
        env,
        robot_cfg=robot_cfg,
        ball_cfg=ball_cfg,
        contact_distance=contact_distance,
    )
    return second_contact


def robot_table_collision(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    table_center: tuple[float, float, float] = (0.55, 0.0, 0.74),
    table_size: tuple[float, float, float] = (2.74, 1.525, 0.04),
    clearance: float = 0.02,
) -> torch.Tensor:
    """Terminate when any robot body enters the table top safety volume."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_pos = robot.data.body_pos_w - env.scene.env_origins.unsqueeze(1)
    center = torch.tensor(table_center, device=env.device)
    size = torch.tensor(table_size, device=env.device)
    half_size = 0.5 * size
    min_xy = center[:2] - half_size[:2]
    max_xy = center[:2] + half_size[:2]
    table_top = center[2] + half_size[2]

    in_table_xy = (
        (body_pos[..., 0] > min_xy[0])
        & (body_pos[..., 0] < max_xy[0])
        & (body_pos[..., 1] > min_xy[1])
        & (body_pos[..., 1] < max_xy[1])
    )
    too_low = body_pos[..., 2] < table_top + clearance
    return torch.any(in_table_xy & too_low, dim=1)
