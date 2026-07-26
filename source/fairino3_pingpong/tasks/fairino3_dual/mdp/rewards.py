"""Custom rewards for the Fairino3 ping-pong task.

Every reward function that queries state machines accepts ``state_prefix``
(default ``""`` for left arm, ``"_right"`` for right arm) and ``home_side``
(default ``"left"``) so the same code works symmetrically for both arms.
"""

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from ._shared import (
    clean_over_net_event,
    episode_reset_mask,
    first_return_hit_event,
    has_clean_over_net,
    has_home_bounce,
    has_illegal_second_hit,
    has_return_hit,
    home_table_bounce_event,
    illegal_second_hit_event,
    left_table_bounce_event,
    legal_post_hit_mask,
    paddle_contact_event,
    peak_post_hit_ball_z,
    rally_exchange_event,
)

# backward compat
has_left_bounce = has_home_bounce  # noqa


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_returning(vel_x: torch.Tensor, home_side: str, threshold: float = 0.05) -> torch.Tensor:
    """Ball is moving toward the opponent's side."""
    if home_side == "left":
        return vel_x > threshold
    return vel_x < -threshold


def _is_incoming(vel_x: torch.Tensor, home_side: str, threshold: float = 0.05) -> torch.Tensor:
    """Ball is moving toward the home side."""
    if home_side == "left":
        return vel_x < -threshold
    return vel_x > threshold


# ---------------------------------------------------------------------------
# pre-hit rewards
# ---------------------------------------------------------------------------

def paddle_to_ball_distance(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    std: float = 0.35,
    left_bounce_gate: bool = False,
    state_prefix: str = "",
) -> torch.Tensor:
    """Reward the paddle for getting close to the ball."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    distance = torch.linalg.norm(paddle_pos - ball.data.root_pos_w[:, :3], dim=1)
    reward = 1.0 - torch.tanh(distance / std)
    if left_bounce_gate:
        reward = reward * has_home_bounce(env, state_prefix=state_prefix).float()
    return reward


def paddle_to_intercept(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    intercept_x: float = -0.05,
    std: float = 0.35,
    home_side: str = "left",
) -> torch.Tensor:
    """Reward the paddle for moving toward the predicted incoming-ball intercept point."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] - env.scene.env_origins
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w[:, :3]

    incoming = _is_incoming(ball_vel[:, 0], home_side)
    vel_x = ball_vel[:, 0]
    if home_side == "left":
        clamped_vx = torch.clamp(vel_x, max=-0.05)
    else:
        clamped_vx = torch.clamp(vel_x, min=0.05)

    time_to_intercept = (intercept_x - ball_pos[:, 0]) / clamped_vx
    time_to_intercept = torch.clamp(time_to_intercept, min=0.0, max=1.0)
    intercept_pos = ball_pos + ball_vel * time_to_intercept.unsqueeze(-1)
    intercept_pos[:, 0] = intercept_x
    intercept_pos[:, 2] = torch.clamp(intercept_pos[:, 2], min=0.80, max=1.36)

    distance = torch.linalg.norm(paddle_pos - intercept_pos, dim=1)
    return (1.0 - torch.tanh(distance / std)) * incoming.float()


def paddle_to_bounce_zone(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    table_center: tuple[float, float, float] = (0.55, 0.0, 0.74),
    table_size: tuple[float, float, float] = (2.74, 1.525, 0.04),
    ball_radius: float = 0.02,
    std: float = 0.25,
    gravity: float = 9.81,
    home_side: str = "left",
    state_prefix: str = "",
) -> torch.Tensor:
    """Reward paddle for hovering near where the ball will bounce on the home table."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    num_envs = ball.data.root_pos_w.shape[0]
    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] - env.scene.env_origins
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w[:, :3]

    center = torch.tensor(table_center, device=env.device)
    size = torch.tensor(table_size, device=env.device)
    half_size = 0.5 * size
    table_top = center[2] + half_size[2]
    ball_z_at_table = table_top + ball_radius

    height_above = ball_pos[:, 2] - ball_z_at_table
    discriminant = height_above * 2.0 * gravity
    vz_sq = ball_vel[:, 2] ** 2
    time_to_table = torch.where(
        discriminant >= -vz_sq,
        (ball_vel[:, 2] + torch.sqrt(torch.clamp(vz_sq + discriminant, min=0.0))) / gravity,
        torch.full((num_envs,), 0.15, device=env.device),
    )
    time_to_table = torch.clamp(time_to_table, min=0.02, max=0.5)

    bounce_xy = ball_pos[:, :2] + ball_vel[:, :2] * time_to_table.unsqueeze(-1)
    ready_z = torch.full((num_envs,), ball_z_at_table + 0.18, device=env.device)

    ready_point = torch.stack([bounce_xy[:, 0], bounce_xy[:, 1], ready_z], dim=-1)

    incoming = _is_incoming(ball_vel[:, 0], home_side)
    home_table_bounce_event(env, ball_cfg=ball_cfg, side=home_side, state_prefix=state_prefix)
    pre_bounce = (~has_home_bounce(env, state_prefix=state_prefix)) & incoming

    net_x = center[0]
    if home_side == "left":
        home_half = (bounce_xy[:, 0] > center[0] - half_size[0]) & (bounce_xy[:, 0] < net_x)
    else:
        home_half = (bounce_xy[:, 0] > net_x) & (bounce_xy[:, 0] < center[0] + half_size[0])

    active = pre_bounce & home_half
    distance = torch.linalg.norm(paddle_pos - ready_point, dim=1)
    return (1.0 - torch.tanh(distance / std)) * active.float()


def incoming_ball_velocity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Reward a moderate incoming serve from the right half of the table toward the robot."""
    ball: RigidObject = env.scene[asset_cfg.name]
    return torch.clamp(-ball.data.root_lin_vel_w[:, 0], min=0.0, max=4.0) / 4.0


# ---------------------------------------------------------------------------
# hit detection rewards
# ---------------------------------------------------------------------------

def first_return_hit(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.14,
    min_incoming_speed: float = 0.05,
    min_return_speed: float = 0.05,
    contact_window_steps: int = 4,
    state_prefix: str = "",
    home_side: str = "left",
    other_state_prefix: str = "",
) -> torch.Tensor:
    """Reward the first paddle hit that turns the incoming ball into a return."""
    return first_return_hit_event(
        env,
        robot_cfg=robot_cfg,
        ball_cfg=ball_cfg,
        contact_distance=contact_distance,
        min_incoming_speed=min_incoming_speed,
        min_return_speed=min_return_speed,
        contact_window_steps=contact_window_steps,
        state_prefix=state_prefix,
        home_side=home_side,
        other_state_prefix=other_state_prefix,
    ).float()


def return_ball_velocity(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    hit_distance: float = 0.16,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Reward ball velocity back toward the opponent side after a useful hit."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    distance = torch.linalg.norm(paddle_pos - ball.data.root_pos_w[:, :3], dim=1)
    hit_gate = torch.clamp(1.0 - distance / hit_distance, min=0.0, max=1.0)
    return_speed = torch.clamp(torch.abs(ball.data.root_lin_vel_w[:, 0]), min=0.0, max=6.0) / 6.0
    legal = legal_post_hit_mask(
        env, ball.data.root_pos_w.shape[0],
        robot_cfg=robot_cfg, ball_cfg=ball_cfg,
        state_prefix=state_prefix, home_side=home_side,
    ).float()
    return legal * torch.maximum(hit_gate, return_speed) * return_speed


# ---------------------------------------------------------------------------
# penalty rewards
# ---------------------------------------------------------------------------

def second_paddle_contact_penalty(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.10,
    grace_steps: int = 4,
    window_steps: int = 24,
    min_return_speed: float = 0.10,
    min_delta_speed: float = 0.12,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """One-time penalty for an illegal second hit after the valid return."""
    return illegal_second_hit_event(
        env,
        robot_cfg=robot_cfg,
        ball_cfg=ball_cfg,
        contact_distance=contact_distance,
        grace_steps=grace_steps,
        window_steps=window_steps,
        min_return_speed=min_return_speed,
        min_delta_speed=min_delta_speed,
        state_prefix=state_prefix,
        home_side=home_side,
    ).float()


def post_hit_paddle_ball_clearance_penalty(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    min_distance: float = 0.22,
    grace_steps: int = 4,
    window_steps: int = 18,
    min_return_speed: float = 0.10,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Penalize keeping the paddle close shortly after a useful hit."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    num_envs = ball.data.root_pos_w.shape[0]

    hit_done = has_return_hit(env, num_envs, state_prefix=state_prefix)
    ph_attr = f"_fairino{state_prefix}_post_hit_clearance_prev_hit"
    ps_attr = f"_fairino{state_prefix}_post_hit_clearance_steps"
    prev_hit = getattr(env, ph_attr, None)
    post_hit_steps = getattr(env, ps_attr, None)
    if prev_hit is None or prev_hit.shape[0] != num_envs:
        prev_hit = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        post_hit_steps = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        setattr(env, ph_attr, prev_hit)
        setattr(env, ps_attr, post_hit_steps)

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_hit = getattr(env, ph_attr).clone()
        post_hit_steps = getattr(env, ps_attr).clone()
        prev_hit[reset_mask] = False
        post_hit_steps[reset_mask] = 0
        setattr(env, ph_attr, prev_hit)
        setattr(env, ps_attr, post_hit_steps)

    new_hit = hit_done & (~getattr(env, ph_attr))
    post_hit_steps = torch.where(
        new_hit,
        torch.zeros_like(getattr(env, ps_attr)),
        torch.where(hit_done, getattr(env, ps_attr) + 1, torch.zeros_like(getattr(env, ps_attr))),
    )

    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    ball_pos = ball.data.root_pos_w[:, :3]
    distance = torch.linalg.norm(paddle_pos - ball_pos, dim=1)
    violation = torch.clamp(min_distance - distance, min=0.0) / min_distance
    returning = _is_returning(ball.data.root_lin_vel_w[:, 0], home_side, min_return_speed)
    active_window = (post_hit_steps >= int(grace_steps)) & (post_hit_steps <= int(window_steps))

    setattr(env, ps_attr, post_hit_steps)
    setattr(env, ph_attr, hit_done.clone())
    return violation.square() * (hit_done & returning & active_window).float()


def post_hit_paddle_retreat_penalty(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    min_behind_x: float = 0.12,
    near_distance: float = 0.34,
    grace_steps: int = 4,
    window_steps: int = 22,
    min_return_speed: float = 0.10,
    chase_speed: float = 0.05,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Penalize the paddle chasing the ball after a valid return hit."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    num_envs = ball.data.root_pos_w.shape[0]

    hit_done = has_return_hit(env, num_envs, state_prefix=state_prefix)
    ph_attr = f"_fairino{state_prefix}_post_hit_retreat_prev_hit"
    ps_attr = f"_fairino{state_prefix}_post_hit_retreat_steps"
    prev_hit = getattr(env, ph_attr, None)
    post_hit_steps = getattr(env, ps_attr, None)
    if prev_hit is None or prev_hit.shape[0] != num_envs:
        prev_hit = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        post_hit_steps = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        setattr(env, ph_attr, prev_hit)
        setattr(env, ps_attr, post_hit_steps)

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_hit = getattr(env, ph_attr).clone()
        post_hit_steps = getattr(env, ps_attr).clone()
        prev_hit[reset_mask] = False
        post_hit_steps[reset_mask] = 0
        setattr(env, ph_attr, prev_hit)
        setattr(env, ps_attr, post_hit_steps)

    new_hit = hit_done & (~getattr(env, ph_attr))
    post_hit_steps = torch.where(
        new_hit,
        torch.zeros_like(getattr(env, ps_attr)),
        torch.where(hit_done, getattr(env, ps_attr) + 1, torch.zeros_like(getattr(env, ps_attr))),
    )

    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    paddle_vel = robot.data.body_lin_vel_w[:, robot_cfg.body_ids[0], :]
    ball_vel = ball.data.root_lin_vel_w[:, :3]
    ball_pos = ball.data.root_pos_w[:, :3]

    distance = torch.linalg.norm(paddle_pos - ball_pos, dim=1)

    # behind check depends on home_side
    if home_side == "left":
        behind_margin = paddle_pos[:, 0] - ball_pos[:, 0] + min_behind_x
    else:
        behind_margin = ball_pos[:, 0] - paddle_pos[:, 0] + min_behind_x

    not_behind = torch.clamp(behind_margin / min_behind_x, min=0.0, max=2.0)
    close_gate = torch.clamp((near_distance - distance) / near_distance, min=0.0, max=1.0)
    chasing_right = torch.clamp((paddle_vel[:, 0] - chase_speed) / 0.60, min=0.0, max=1.0)
    active_window = (post_hit_steps >= int(grace_steps)) & (post_hit_steps <= int(window_steps))
    returning = _is_returning(ball_vel[:, 0], home_side, min_return_speed)

    setattr(env, ps_attr, post_hit_steps)
    setattr(env, ph_attr, hit_done.clone())
    return (not_behind.square() * close_gate * (0.5 + 0.5 * chasing_right)) * (hit_done & returning & active_window).float()


def early_hit_penalty(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.14,
    state_prefix: str = "",
) -> torch.Tensor:
    """Penalize paddle-ball contact before the ball has bounced on the home table."""
    contact_now, _, _ = paddle_contact_event(
        env, robot_cfg=robot_cfg, ball_cfg=ball_cfg,
        contact_distance=contact_distance, state_prefix=state_prefix,
    )
    pre_bounce = ~has_home_bounce(env, state_prefix=state_prefix)
    return (contact_now & pre_bounce).float()


# ---------------------------------------------------------------------------
# post-hit quality rewards
# ---------------------------------------------------------------------------

def legal_return_separation_reward(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    table_center: tuple[float, float, float] = (0.55, 0.0, 0.74),
    table_size: tuple[float, float, float] = (2.74, 1.525, 0.04),
    ball_radius: float = 0.02,
    net_x: float = 0.55,
    target_xy: tuple[float, float] = (1.235, 0.0),
    min_behind_x: float = 0.14,
    behind_std: float = 0.18,
    min_return_speed: float = 0.25,
    target_return_speed: float = 1.8,
    grace_steps: int = 3,
    window_steps: int = 34,
    landing_std: float = 0.42,
    max_prediction_time: float = 1.2,
    gravity: float = 9.81,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Reward legal post-hit separation that still predicts a valid table return."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    num_envs = ball.data.root_pos_w.shape[0]

    hit_done = has_return_hit(env, num_envs, state_prefix=state_prefix)
    ph_attr = f"_fairino{state_prefix}_legal_sep_prev_hit"
    ps_attr = f"_fairino{state_prefix}_legal_sep_steps"
    prev_hit = getattr(env, ph_attr, None)
    post_hit_steps = getattr(env, ps_attr, None)
    if prev_hit is None or prev_hit.shape[0] != num_envs:
        prev_hit = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        post_hit_steps = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        setattr(env, ph_attr, prev_hit)
        setattr(env, ps_attr, post_hit_steps)

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_hit = getattr(env, ph_attr).clone()
        post_hit_steps = getattr(env, ps_attr).clone()
        prev_hit[reset_mask] = False
        post_hit_steps[reset_mask] = 0
        setattr(env, ph_attr, prev_hit)
        setattr(env, ps_attr, post_hit_steps)

    new_hit = hit_done & (~getattr(env, ph_attr))
    post_hit_steps = torch.where(
        new_hit,
        torch.zeros_like(getattr(env, ps_attr)),
        torch.where(hit_done, getattr(env, ps_attr) + 1, torch.zeros_like(getattr(env, ps_attr))),
    )

    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] - env.scene.env_origins
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w[:, :3]

    center = torch.tensor(table_center, device=env.device)
    size = torch.tensor(table_size, device=env.device)
    half_size = 0.5 * size
    table_top = center[2] + half_size[2]
    ball_center_at_table = table_top + ball_radius

    height_above_table = torch.clamp(ball_pos[:, 2] - ball_center_at_table, min=0.0)
    discriminant = torch.clamp(ball_vel[:, 2] ** 2 + 2.0 * gravity * height_above_table, min=0.0)
    time_to_table = (ball_vel[:, 2] + torch.sqrt(discriminant)) / gravity
    time_to_table = torch.clamp(time_to_table, min=0.0, max=max_prediction_time)
    landing_xy = ball_pos[:, :2] + ball_vel[:, :2] * time_to_table.unsqueeze(-1)

    target = torch.tensor(target_xy, device=env.device)
    target_score = 1.0 - torch.tanh(torch.linalg.norm(landing_xy - target, dim=1) / landing_std)

    if home_side == "left":
        in_opponent_half = (landing_xy[:, 0] > net_x) & (landing_xy[:, 0] < center[0] + half_size[0])
    else:
        in_opponent_half = (landing_xy[:, 0] > center[0] - half_size[0]) & (landing_xy[:, 0] < net_x)

    in_table_y = torch.abs(landing_xy[:, 1] - center[1]) < half_size[1]
    landing_score = torch.maximum((in_opponent_half & in_table_y).float(), target_score)

    if home_side == "left":
        behind_margin = ball_pos[:, 0] - paddle_pos[:, 0] - min_behind_x
    else:
        behind_margin = paddle_pos[:, 0] - ball_pos[:, 0] - min_behind_x
    behind_score = torch.clamp(behind_margin / behind_std, min=0.0, max=1.0)
    speed_score = torch.clamp(
        (torch.abs(ball_vel[:, 0]) - min_return_speed) / (target_return_speed - min_return_speed),
        min=0.0, max=1.0,
    )
    active_window = (post_hit_steps >= int(grace_steps)) & (post_hit_steps <= int(window_steps))
    legal = ~has_illegal_second_hit(env, num_envs, state_prefix=state_prefix)
    plausible_height = ball_pos[:, 2] > ball_center_at_table

    setattr(env, ps_attr, post_hit_steps)
    setattr(env, ph_attr, hit_done.clone())
    return (behind_score * speed_score * landing_score) * (hit_done & legal & active_window & plausible_height).float()


# ---------------------------------------------------------------------------
# net-crossing rewards
# ---------------------------------------------------------------------------

def ball_over_net(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_height: float = 0.9125,
    table_half_width: float = 0.7625,
    net_window: float = 0.18,
    max_clearance_height: float = 1.41,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Reward returned balls while they cross the net corridor."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel_x = ball.data.root_lin_vel_w[:, 0]
    returning = _is_returning(vel_x, home_side) & legal_post_hit_mask(
        env, pos.shape[0], ball_cfg=asset_cfg, state_prefix=state_prefix, home_side=home_side,
    )

    near_net = torch.abs(pos[:, 0] - net_x) < net_window
    in_table_y = torch.abs(pos[:, 1]) < table_half_width
    below_flyaway_height = pos[:, 2] < max_clearance_height
    height_score = torch.clamp((pos[:, 2] - net_height) / (max_clearance_height - net_height), min=0.0, max=1.0)
    corridor_gate = near_net & in_table_y & below_flyaway_height

    if home_side == "left":
        rightward_score = torch.clamp(vel_x / 2.5, min=0.0, max=1.0)
        crossed = corridor_gate & (pos[:, 0] > net_x) & (pos[:, 2] > net_height)
    else:
        rightward_score = torch.clamp(-vel_x / 2.5, min=0.0, max=1.0)
        crossed = corridor_gate & (pos[:, 0] < net_x) & (pos[:, 2] > net_height)

    crossed_score = crossed.float()
    approach_score = height_score * rightward_score
    return torch.maximum(crossed_score, approach_score * corridor_gate.float()) * returning.float()


def ball_to_target(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target: tuple[float, float, float] = (1.235, 0.0, 0.76),
    std: float = 0.45,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Reward ball proximity to the target landing area."""
    ball: RigidObject = env.scene[asset_cfg.name]
    target_pos = torch.tensor(target, device=env.device)
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    distance = torch.linalg.norm(ball_pos - target_pos, dim=1)
    vel_x = ball.data.root_lin_vel_w[:, 0]
    returning = (_is_returning(vel_x, home_side) & legal_post_hit_mask(
        env, ball_pos.shape[0], ball_cfg=asset_cfg, state_prefix=state_prefix, home_side=home_side,
    )).float()
    return (1.0 - torch.tanh(distance / std)) * returning


def predicted_right_table_landing(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target_xy: tuple[float, float] = (1.235, 0.0),
    table_center: tuple[float, float, float] = (0.55, 0.0, 0.74),
    table_size: tuple[float, float, float] = (2.74, 1.525, 0.04),
    ball_radius: float = 0.02,
    net_x: float = 0.55,
    gravity: float = 9.81,
    std: float = 0.35,
    max_prediction_time: float = 1.2,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Reward post-hit balls whose ballistic landing estimate is on the opponent table."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]

    center = torch.tensor(table_center, device=env.device)
    size = torch.tensor(table_size, device=env.device)
    half_size = 0.5 * size
    table_top = center[2] + half_size[2]
    ball_center_at_table = table_top + ball_radius

    height_above_table = torch.clamp(pos[:, 2] - ball_center_at_table, min=0.0)
    discriminant = torch.clamp(vel[:, 2] ** 2 + 2.0 * gravity * height_above_table, min=0.0)
    time_to_table = (vel[:, 2] + torch.sqrt(discriminant)) / gravity
    time_to_table = torch.clamp(time_to_table, min=0.0, max=max_prediction_time)
    landing_xy = pos[:, :2] + vel[:, :2] * time_to_table.unsqueeze(-1)

    target = torch.tensor(target_xy, device=env.device)
    target_score = 1.0 - torch.tanh(torch.linalg.norm(landing_xy - target, dim=1) / std)

    if home_side == "left":
        in_opponent_half = (landing_xy[:, 0] > net_x) & (landing_xy[:, 0] < center[0] + half_size[0])
    else:
        in_opponent_half = (landing_xy[:, 0] > center[0] - half_size[0]) & (landing_xy[:, 0] < net_x)

    in_table_y = torch.abs(landing_xy[:, 1] - center[1]) < half_size[1]
    table_score = (in_opponent_half & in_table_y).float()
    returning = _is_returning(vel[:, 0], home_side) & legal_post_hit_mask(
        env, pos.shape[0], ball_cfg=asset_cfg, state_prefix=state_prefix, home_side=home_side,
    )
    plausible_height = pos[:, 2] > ball_center_at_table
    base = torch.maximum(table_score, target_score)

    max_good_z = table_top + 0.50
    height_factor = torch.clamp(max_good_z / torch.clamp(pos[:, 2], min=max_good_z), 0.4, 1.0)

    return base * height_factor * (returning & plausible_height).float()


def predicted_right_near_net_landing_speed(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target_xy: tuple[float, float] = (0.85, 0.0),
    table_center: tuple[float, float, float] = (0.55, 0.0, 0.74),
    table_size: tuple[float, float, float] = (2.74, 1.525, 0.04),
    ball_radius: float = 0.02,
    net_x: float = 0.55,
    gravity: float = 9.81,
    std_x: float = 0.18,
    std_y: float = 0.22,
    min_right_speed: float = 0.8,
    target_right_speed: float = 1.8,
    max_prediction_time: float = 1.2,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Reward predicted opponent-table landings near the net with enough speed."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]

    center = torch.tensor(table_center, device=env.device)
    size = torch.tensor(table_size, device=env.device)
    half_size = 0.5 * size
    table_top = center[2] + half_size[2]
    ball_center_at_table = table_top + ball_radius

    height_above_table = torch.clamp(pos[:, 2] - ball_center_at_table, min=0.0)
    discriminant = torch.clamp(vel[:, 2] ** 2 + 2.0 * gravity * height_above_table, min=0.0)
    time_to_table = (vel[:, 2] + torch.sqrt(discriminant)) / gravity
    time_to_table = torch.clamp(time_to_table, min=0.0, max=max_prediction_time)
    landing_xy = pos[:, :2] + vel[:, :2] * time_to_table.unsqueeze(-1)

    target = torch.tensor(target_xy, device=env.device)
    x_score = 1.0 - torch.tanh(torch.abs(landing_xy[:, 0] - target[0]) / std_x)
    y_score = 1.0 - torch.tanh(torch.abs(landing_xy[:, 1] - target[1]) / std_y)
    return_speed = torch.clamp(torch.abs(vel[:, 0]), min=0.0)
    speed_score = torch.clamp(return_speed / target_right_speed, 0.0, 1.0)
    speed_floor = torch.clamp((return_speed - 0.1) / max(min_right_speed - 0.1, 1.0e-6), 0.0, 1.0)

    if home_side == "left":
        in_opponent_half = (landing_xy[:, 0] > net_x) & (landing_xy[:, 0] < center[0] + half_size[0])
    else:
        in_opponent_half = (landing_xy[:, 0] > center[0] - half_size[0]) & (landing_xy[:, 0] < net_x)

    in_table_y = torch.abs(landing_xy[:, 1] - center[1]) < half_size[1]
    returning = _is_returning(vel[:, 0], home_side) & legal_post_hit_mask(
        env, pos.shape[0], ball_cfg=asset_cfg, state_prefix=state_prefix, home_side=home_side,
    )
    plausible_height = pos[:, 2] > ball_center_at_table
    quality = x_score * y_score * (0.25 + 0.75 * speed_score) * speed_floor
    return quality * (in_opponent_half & in_table_y & returning & plausible_height).float()


# ---------------------------------------------------------------------------
# sparse success / table bounce
# ---------------------------------------------------------------------------

def right_table_bounce_reward(
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
    illegal_reward_scale: float = 0.0,
    require_clean_over_net: bool = False,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Reward a returned ball that bounces on the opponent half.

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
    bounce = in_opponent_half & in_table_y & near_table_height & bounced_up & returning & clean_gate
    scale = torch.where(
        legal,
        torch.ones_like(pos[:, 0]),
        torch.full_like(pos[:, 0], float(illegal_reward_scale)),
    )

    peak_z = peak_post_hit_ball_z(env, ball_cfg=asset_cfg, state_prefix=state_prefix)
    max_good_z = table_top + 0.50
    height_factor = torch.clamp(max_good_z / torch.clamp(peak_z, min=max_good_z), 0.4, 1.0)

    return bounce.float() * scale * height_factor


# ---------------------------------------------------------------------------
# generic penalties (already cfg-driven, no state machines)
# ---------------------------------------------------------------------------

def ball_out_of_bounds_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    x_bounds: tuple[float, float] = (-1.52, 2.62),
    y_bounds: tuple[float, float] = (-1.2125, 1.2125),
    z_min: float = 0.03,
) -> torch.Tensor:
    """Penalize only ball-out failures, separate from successful terminal bounces."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    out = (
        (pos[:, 0] < x_bounds[0])
        | (pos[:, 0] > x_bounds[1])
        | (pos[:, 1] < y_bounds[0])
        | (pos[:, 1] > y_bounds[1])
        | (pos[:, 2] < z_min)
    )
    return out.float()


def paddle_body_clearance_penalty(
    env,
    paddle_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["forearm_link", "wrist1_link", "wrist2_link", "wrist3_link", "tool_clamp_link"],
    ),
    min_distance: float = 0.13,
) -> torch.Tensor:
    """Penalize paddle poses that bring the blade too close to the robot body."""
    robot: Articulation = env.scene[paddle_cfg.name]
    paddle_pos = robot.data.body_pos_w[:, paddle_cfg.body_ids[0], :]
    body_pos = robot.data.body_pos_w[:, body_cfg.body_ids, :]
    distances = torch.linalg.norm(body_pos - paddle_pos.unsqueeze(1), dim=-1)
    closest_distance = torch.min(distances, dim=1).values
    violation = torch.clamp(min_distance - closest_distance, min=0.0)
    return (violation / min_distance).square()


def joint_limit_margin_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.12,
) -> torch.Tensor:
    """Penalize joints that move too close to their soft limits."""
    robot: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    lower_dist = joint_pos - limits[..., 0]
    upper_dist = limits[..., 1] - joint_pos
    limit_distance = torch.minimum(lower_dist, upper_dist)
    return torch.sum(torch.clamp(margin - limit_distance, min=0.0), dim=1)


def post_return_idle_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    state_prefix: str = "",
) -> torch.Tensor:
    """Penalize ALL arm movement after a successful hit."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    gate = has_return_hit(env, ball.data.root_pos_w.shape[0], state_prefix=state_prefix)
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(joint_vel ** 2, dim=1) * gate.float()


def robot_table_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    table_center: tuple[float, float, float] = (0.55, 0.0, 0.74),
    table_size: tuple[float, float, float] = (2.74, 1.525, 0.04),
    clearance: float = 0.03,
) -> torch.Tensor:
    """Penalize robot links entering the table top volume plus a small clearance band."""
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
    return torch.any(in_table_xy & too_low, dim=1).float()


# ---------------------------------------------------------------------------
# net-crossing quality rewards
# ---------------------------------------------------------------------------

_SERVE_SPEED = (1.25 ** 2 + 0.12 ** 2) ** 0.5


def net_direction(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_window: float = 0.18,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Reward ball velocity toward opponent side at the net."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel_x = ball.data.root_lin_vel_w[:, 0]
    near_net = torch.abs(pos[:, 0] - net_x) < net_window
    returning = _is_returning(vel_x, home_side) & legal_post_hit_mask(
        env, pos.shape[0], ball_cfg=asset_cfg, state_prefix=state_prefix, home_side=home_side,
    )
    score = torch.clamp(torch.abs(vel_x), min=0.0, max=4.0) / 4.0
    return score * near_net.float() * returning.float()


def net_height(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_window: float = 0.18,
    ideal_z: float = 0.56,
    std: float = 0.08,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Reward ball height near serve height when crossing the net."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel_x = ball.data.root_lin_vel_w[:, 0]
    near_net = torch.abs(pos[:, 0] - net_x) < net_window
    returning = _is_returning(vel_x, home_side) & legal_post_hit_mask(
        env, pos.shape[0], ball_cfg=asset_cfg, state_prefix=state_prefix, home_side=home_side,
    )
    score = 1.0 - torch.tanh(torch.abs(pos[:, 2] - ideal_z) / std)
    return score * near_net.float() * returning.float()


def net_speed(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_window: float = 0.18,
    ideal_speed: float = _SERVE_SPEED,
    std: float = 0.3,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Reward ball speed close to initial serve speed when crossing the net."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]
    vel_x = vel[:, 0]
    near_net = torch.abs(pos[:, 0] - net_x) < net_window
    returning = _is_returning(vel_x, home_side) & legal_post_hit_mask(
        env, pos.shape[0], ball_cfg=asset_cfg, state_prefix=state_prefix, home_side=home_side,
    )
    speed = torch.linalg.norm(vel, dim=1)
    score = 1.0 - torch.tanh(torch.abs(speed - ideal_speed) / std)
    return score * near_net.float() * returning.float()


def centerline_x(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    y_window: float = 0.10,
    std: float = 0.25,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Reward ball x close to 0 when crossing the centerline (y≈0)."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel_x = ball.data.root_lin_vel_w[:, 0]
    near_center = torch.abs(pos[:, 1]) < y_window
    returning = _is_returning(vel_x, home_side) & legal_post_hit_mask(
        env, pos.shape[0], ball_cfg=asset_cfg, state_prefix=state_prefix, home_side=home_side,
    )
    score = 1.0 - torch.tanh(torch.abs(pos[:, 0]) / std)
    return score * near_center.float() * returning.float()


def rally_return(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """One-time reward: ball bounced on home table, then robot hit it back."""
    ball: RigidObject = env.scene[ball_cfg.name]
    num_envs = ball.data.root_pos_w.shape[0]
    vel_x = ball.data.root_lin_vel_w[:, 0]

    bounced_home = has_home_bounce(env, num_envs, state_prefix=state_prefix)
    hit_done = has_return_hit(env, num_envs, state_prefix=state_prefix)
    legal_hit = legal_post_hit_mask(
        env, num_envs, ball_cfg=ball_cfg, state_prefix=state_prefix, home_side=home_side,
    )
    going_opponent = _is_returning(vel_x, home_side)

    triggered = bounced_home & legal_hit & going_opponent
    trig_attr = f"_fairino{state_prefix}_rally_reward_triggered"
    was_triggered = getattr(env, trig_attr, None)
    if was_triggered is None or was_triggered.shape[0] != num_envs:
        was_triggered = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        setattr(env, trig_attr, was_triggered)

    ep_len = getattr(env, "episode_length_buf", None)
    if ep_len is not None:
        reset_mask = ep_len <= 1
        if torch.any(reset_mask):
            getattr(env, trig_attr)[reset_mask] = False

    new_trigger = triggered & (~getattr(env, trig_attr))
    if torch.any(new_trigger):
        getattr(env, trig_attr)[new_trigger] = True

    return new_trigger.float()


def left_table_bounce(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """One-time reward for the ball bouncing on the home table."""
    return home_table_bounce_event(
        env, ball_cfg=ball_cfg, side=home_side, state_prefix=state_prefix,
    ).float()


def high_return_height_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_window: float = 0.18,
    net_height: float = 0.9125,
    max_height: float = 1.2175,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Penalize returns that cross the net above max_height."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel_x = ball.data.root_lin_vel_w[:, 0]

    near_net = torch.abs(pos[:, 0] - net_x) < net_window
    returning = _is_returning(vel_x, home_side) & legal_post_hit_mask(
        env, pos.shape[0], ball_cfg=asset_cfg, state_prefix=state_prefix, home_side=home_side,
    )
    too_high = pos[:, 2] > max_height
    violation = torch.clamp((pos[:, 2] - max_height) / 0.3, 0.0, 1.0)

    return violation * near_net.float() * returning.float() * too_high.float()


def rally_exchange_reward(
    env,
    state_prefix: str = "",
) -> torch.Tensor:
    """Reward each rally exchange (left→right or right→left alternating hit).

    Fires **once per exchange** via rising-edge detection — not continuously.
    """
    num_envs = env.num_envs
    # keep state-machine side-effects current
    exchange = rally_exchange_event(env, num_envs)
    return exchange.float()
