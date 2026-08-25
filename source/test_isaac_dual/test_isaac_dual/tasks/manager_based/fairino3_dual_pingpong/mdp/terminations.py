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


def _is_returning(vel_x: torch.Tensor, home_side: str, threshold: float = 0.05) -> torch.Tensor:
    if home_side == "left":
        return vel_x > threshold
    return vel_x < -threshold


def simple_ball_on_table(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    table_center: tuple[float, float, float] = (0.55, 0.0, 0.74),
    table_size: tuple[float, float, float] = (2.74, 1.525, 0.04),
    ball_radius: float = 0.02,
    height_tolerance: float = 0.08,
    side: str = "right",
    net_x: float = 0.55,
) -> torch.Tensor:
    """Simple termination: ball falls off the table (drops below table surface).

    ``side="right"`` → right half (x > net_x). ``side="left"`` → left half (x < net_x).
    Terminates when ball drops below the table top level anywhere in the correct half.
    """
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    center = torch.tensor(table_center, device=env.device)
    size = torch.tensor(table_size, device=env.device)
    half_size = 0.5 * size
    table_top = center[2] + half_size[2]  # 0.76

    if side == "right":
        in_half = pos[:, 0] > net_x
    else:
        in_half = pos[:, 0] < net_x

    # Ball center drops below table surface
    below_table = pos[:, 2] < table_top

    return in_half & below_table


def ball_fall_off_table(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    min_z: float = 0.54,
) -> torch.Tensor:
    """Terminate when the ball drops below *min_z* (i.e. falls off the table)."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    return pos[:, 2] < min_z


def ball_resting_on_table(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    table_top_z: float = 0.76,
    ball_radius: float = 0.02,
    height_tolerance: float = 0.04,
    min_steps: int = 45,
) -> torch.Tensor:
    """Terminate when the ball sits still on the table for *min_steps*.

    The ball centre rests at ``table_top_z + ball_radius``.  If it stays within
    *height_tolerance* of that height for *min_steps* consecutive environment
    steps the rally is dead.
    """
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    num_envs = pos.shape[0]

    rest_z = table_top_z + ball_radius
    at_rest = torch.abs(pos[:, 2] - rest_z) < height_tolerance

    attr = "_fairino_ball_resting_steps"
    steps = getattr(env, attr, None)
    if steps is None or steps.shape[0] != num_envs:
        steps = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        setattr(env, attr, steps)

    from ._shared import episode_reset_mask
    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        steps = getattr(env, attr).clone()
        steps[reset_mask] = 0
        setattr(env, attr, steps)

    steps = torch.where(at_rest, getattr(env, attr) + 1, torch.zeros_like(getattr(env, attr)))
    setattr(env, attr, steps)
    return steps >= int(min_steps)


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
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Terminate when the ball bounces on the opponent half.

    ``home_side="left"`` → opponent half is right side (x > net_x).
    ``home_side="right"`` → opponent half is left side (x < net_x).
    """
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]
    center = torch.tensor(table_center, device=env.device)
    size = torch.tensor(table_size, device=env.device)
    half_size = 0.5 * size
    table_top = center[2] + half_size[2]
    ball_center_at_table = table_top + ball_radius

    if home_side == "left":
        in_opponent_half = (pos[:, 0] > net_x) & (pos[:, 0] < center[0] + half_size[0])
    else:
        in_opponent_half = (pos[:, 0] > center[0] - half_size[0]) & (pos[:, 0] < net_x)

    in_table_y = torch.abs(pos[:, 1] - center[1]) < half_size[1]
    near_table_height = torch.abs(pos[:, 2] - ball_center_at_table) < height_tolerance
    bounced_up = vel[:, 2] > min_upward_velocity
    returning = _is_returning(vel[:, 0], home_side) & has_return_hit(env, pos.shape[0], state_prefix=state_prefix)
    clean_over_net_event(
        env,
        ball_cfg=asset_cfg,
        robot_cfg=robot_cfg,
        net_x=net_x,
        net_height=table_top + 0.1525,
        ball_radius=ball_radius,
        illegal_contact_distance=illegal_contact_distance,
        state_prefix=state_prefix,
        home_side=home_side,
    )
    illegal_second_hit_event(
        env,
        robot_cfg=robot_cfg,
        ball_cfg=asset_cfg,
        contact_distance=illegal_contact_distance,
        state_prefix=state_prefix,
        home_side=home_side,
    )
    legal = ~has_illegal_second_hit(env, pos.shape[0], state_prefix=state_prefix)
    clean_gate = has_clean_over_net(env, pos.shape[0], state_prefix=state_prefix) if require_clean_over_net else torch.ones_like(legal)
    raw_bounce = in_opponent_half & in_table_y & near_table_height & bounced_up & returning
    bounce = raw_bounce & clean_gate & (legal | bool(allow_illegal_success))
    if torch.any(bounce):
        num_envs = pos.shape[0]
        bx_attr = f"_fairino{state_prefix}_bounce_x_at_event"
        by_attr = f"_fairino{state_prefix}_bounce_y_at_event"
        bv_attr = f"_fairino{state_prefix}_bounce_vx_at_event"
        bi_attr = f"_fairino{state_prefix}_bounce_illegal_at_event"
        if not hasattr(env, bx_attr) or getattr(env, bx_attr).shape[0] != num_envs:
            setattr(env, bx_attr, torch.zeros(num_envs, device=env.device))
            setattr(env, by_attr, torch.zeros(num_envs, device=env.device))
            setattr(env, bv_attr, torch.zeros(num_envs, device=env.device))
            setattr(env, bi_attr, torch.zeros(num_envs, device=env.device, dtype=torch.bool))
        getattr(env, bx_attr)[bounce] = pos[bounce, 0]
        getattr(env, by_attr)[bounce] = pos[bounce, 1]
        getattr(env, bv_attr)[bounce] = vel[bounce, 0]
        getattr(env, bi_attr)[bounce] = ~legal[bounce]
    return bounce


def second_paddle_contact(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.10,
    state_prefix: str = "",
) -> torch.Tensor:
    """Terminate when the paddle touches the ball more than once in an episode."""
    _, _, second_contact = paddle_contact_event(
        env,
        robot_cfg=robot_cfg,
        ball_cfg=ball_cfg,
        contact_distance=contact_distance,
        state_prefix=state_prefix,
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
