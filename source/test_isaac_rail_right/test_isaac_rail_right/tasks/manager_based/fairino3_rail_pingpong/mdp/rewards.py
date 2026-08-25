"""Custom rewards for the Fairino3 ping-pong task."""

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from ._shared import (
    clean_over_net_event,
    episode_reset_mask,
    first_return_hit_event,
    has_clean_over_net,
    has_illegal_second_hit,
    has_left_bounce,
    has_return_hit,
    illegal_second_hit_event,
    left_table_bounce_event,
    legal_post_hit_mask,
    paddle_contact_event,
    peak_post_hit_ball_z,
)


def _is_returning(vel_x: torch.Tensor, home_side: str, threshold: float = 0.05) -> torch.Tensor:
    """Ball is moving toward the opponent's side."""
    if home_side == "left":
        return vel_x > threshold
    return vel_x < -threshold


def paddle_to_ball_distance(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    std: float = 0.35,
    left_bounce_gate: bool = False,
) -> torch.Tensor:
    """Reward the paddle for getting close to the ball.

    When left_bounce_gate is True, the reward is only active after the ball
    has bounced on the left table, which incentivises the paddle to actually
    approach for a hit rather than just hovering nearby.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    distance = torch.linalg.norm(paddle_pos - ball.data.root_pos_w[:, :3], dim=1)
    reward = 1.0 - torch.tanh(distance / std)
    if left_bounce_gate:
        reward = reward * has_left_bounce(env).float()
    return reward


def paddle_to_intercept(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    intercept_x: float = -0.05,
    std: float = 0.35,
) -> torch.Tensor:
    """Reward the paddle for moving toward the predicted incoming-ball intercept point."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] - env.scene.env_origins
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w[:, :3]

    incoming = ball_vel[:, 0] < -0.05
    time_to_intercept = (intercept_x - ball_pos[:, 0]) / torch.clamp(ball_vel[:, 0], max=-0.05)
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
) -> torch.Tensor:
    """Reward paddle for hovering near where the ball will bounce on the left table.

    Computes the predicted bounce point on the left half and rewards the paddle
    for staying near a ready position just above it.  Only active BEFORE the
    ball has bounced (incoming phase).
    """
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

    # Ballistic time to reach table level
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

    ready_point = torch.stack([
        bounce_xy[:, 0],
        bounce_xy[:, 1],
        ready_z,
    ], dim=-1)

    incoming = ball_vel[:, 0] < -0.05
    left_table_bounce_event(env, ball_cfg=ball_cfg)
    pre_bounce = (~has_left_bounce(env)) & incoming
    left_half = (bounce_xy[:, 0] > center[0] - half_size[0]) & (bounce_xy[:, 0] < center[0])
    active = pre_bounce & left_half

    distance = torch.linalg.norm(paddle_pos - ready_point, dim=1)
    return (1.0 - torch.tanh(distance / std)) * active.float()


def incoming_ball_velocity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Reward a moderate incoming serve from the right half of the table toward the robot."""
    ball: RigidObject = env.scene[asset_cfg.name]
    return torch.clamp(-ball.data.root_lin_vel_w[:, 0], min=0.0, max=4.0) / 4.0


def first_return_hit(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.14,
    min_incoming_speed: float = 0.05,
    min_return_speed: float = 0.05,
    contact_window_steps: int = 4,
    home_side: str = "right",
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
        home_side=home_side,
    ).float()


def return_ball_velocity(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    hit_distance: float = 0.16,
) -> torch.Tensor:
    """Reward ball velocity back toward the right side after a useful hit."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    distance = torch.linalg.norm(paddle_pos - ball.data.root_pos_w[:, :3], dim=1)
    hit_gate = torch.clamp(1.0 - distance / hit_distance, min=0.0, max=1.0)
    return_speed = torch.clamp(torch.abs(ball.data.root_lin_vel_w[:, 0]), min=0.0, max=6.0) / 6.0
    return legal_post_hit_mask(env, ball.data.root_pos_w.shape[0], robot_cfg=robot_cfg, ball_cfg=ball_cfg).float() * torch.maximum(hit_gate, return_speed) * return_speed


def second_paddle_contact_penalty(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.10,
    grace_steps: int = 4,
    window_steps: int = 24,
    min_return_speed: float = 0.10,
    min_delta_speed: float = 0.12,
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
    ).float()


def post_hit_paddle_ball_clearance_penalty(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    min_distance: float = 0.22,
    grace_steps: int = 4,
    window_steps: int = 18,
    min_return_speed: float = 0.10,
) -> torch.Tensor:
    """Penalize keeping the paddle close shortly after a useful hit.

    The first few frames after contact are normal follow-through.  After that
    grace period, a short window catches the failure mode where the paddle keeps
    chasing the returned ball and taps it again.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    num_envs = ball.data.root_pos_w.shape[0]

    hit_done = has_return_hit(env, num_envs)
    prev_hit = getattr(env, "_fairino_post_hit_clearance_prev_hit", None)
    post_hit_steps = getattr(env, "_fairino_post_hit_clearance_steps", None)
    if prev_hit is None or prev_hit.shape[0] != num_envs:
        prev_hit = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        env._fairino_post_hit_clearance_prev_hit = prev_hit
    if post_hit_steps is None or post_hit_steps.shape[0] != num_envs:
        post_hit_steps = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        env._fairino_post_hit_clearance_steps = post_hit_steps

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_hit = env._fairino_post_hit_clearance_prev_hit.clone()
        post_hit_steps = env._fairino_post_hit_clearance_steps.clone()
        prev_hit[reset_mask] = False
        post_hit_steps[reset_mask] = 0
        env._fairino_post_hit_clearance_prev_hit = prev_hit
        env._fairino_post_hit_clearance_steps = post_hit_steps

    new_hit = hit_done & (~env._fairino_post_hit_clearance_prev_hit)
    post_hit_steps = torch.where(
        new_hit,
        torch.zeros_like(env._fairino_post_hit_clearance_steps),
        torch.where(hit_done, env._fairino_post_hit_clearance_steps + 1, torch.zeros_like(env._fairino_post_hit_clearance_steps)),
    )

    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    ball_pos = ball.data.root_pos_w[:, :3]
    distance = torch.linalg.norm(paddle_pos - ball_pos, dim=1)
    violation = torch.clamp(min_distance - distance, min=0.0) / min_distance
    returning = ball.data.root_lin_vel_w[:, 0] > min_return_speed
    active_window = (post_hit_steps >= int(grace_steps)) & (post_hit_steps <= int(window_steps))

    env._fairino_post_hit_clearance_steps = post_hit_steps
    env._fairino_post_hit_clearance_prev_hit = hit_done.clone()
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
) -> torch.Tensor:
    """Penalize the paddle chasing the ball after a valid return hit.

    This targets the visual failure mode where the paddle keeps following the
    returned ball and stays close enough to tap or carry it.  After a short
    follow-through grace period, the paddle should lag behind the ball on x.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    num_envs = ball.data.root_pos_w.shape[0]

    hit_done = has_return_hit(env, num_envs)
    prev_hit = getattr(env, "_fairino_post_hit_retreat_prev_hit", None)
    post_hit_steps = getattr(env, "_fairino_post_hit_retreat_steps", None)
    if prev_hit is None or prev_hit.shape[0] != num_envs:
        prev_hit = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        env._fairino_post_hit_retreat_prev_hit = prev_hit
    if post_hit_steps is None or post_hit_steps.shape[0] != num_envs:
        post_hit_steps = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        env._fairino_post_hit_retreat_steps = post_hit_steps

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_hit = env._fairino_post_hit_retreat_prev_hit.clone()
        post_hit_steps = env._fairino_post_hit_retreat_steps.clone()
        prev_hit[reset_mask] = False
        post_hit_steps[reset_mask] = 0
        env._fairino_post_hit_retreat_prev_hit = prev_hit
        env._fairino_post_hit_retreat_steps = post_hit_steps

    new_hit = hit_done & (~env._fairino_post_hit_retreat_prev_hit)
    post_hit_steps = torch.where(
        new_hit,
        torch.zeros_like(env._fairino_post_hit_retreat_steps),
        torch.where(hit_done, env._fairino_post_hit_retreat_steps + 1, torch.zeros_like(env._fairino_post_hit_retreat_steps)),
    )

    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    paddle_vel = robot.data.body_lin_vel_w[:, robot_cfg.body_ids[0], :]
    ball_pos = ball.data.root_pos_w[:, :3]
    ball_vel = ball.data.root_lin_vel_w[:, :3]
    distance = torch.linalg.norm(paddle_pos - ball_pos, dim=1)

    not_behind = torch.clamp((paddle_pos[:, 0] - ball_pos[:, 0] + min_behind_x) / min_behind_x, min=0.0, max=2.0)
    close_gate = torch.clamp((near_distance - distance) / near_distance, min=0.0, max=1.0)
    chasing_right = torch.clamp((paddle_vel[:, 0] - chase_speed) / 0.60, min=0.0, max=1.0)
    active_window = (post_hit_steps >= int(grace_steps)) & (post_hit_steps <= int(window_steps))
    returning = ball_vel[:, 0] > min_return_speed

    env._fairino_post_hit_retreat_steps = post_hit_steps
    env._fairino_post_hit_retreat_prev_hit = hit_done.clone()
    return (not_behind.square() * close_gate * (0.5 + 0.5 * chasing_right)) * (hit_done & returning & active_window).float()


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
) -> torch.Tensor:
    """Reward legal post-hit separation that still predicts a right-table return.

    This is the positive counterpart to the second-hit penalties: after the
    useful hit, the paddle should fall behind the ball on x while the ball keeps
    moving toward a plausible right-table landing.  It is gated off as soon as an
    illegal second hit is observed.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    num_envs = ball.data.root_pos_w.shape[0]

    hit_done = has_return_hit(env, num_envs)
    prev_hit = getattr(env, "_fairino_legal_sep_prev_hit", None)
    post_hit_steps = getattr(env, "_fairino_legal_sep_steps", None)
    if prev_hit is None or prev_hit.shape[0] != num_envs:
        prev_hit = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        env._fairino_legal_sep_prev_hit = prev_hit
    if post_hit_steps is None or post_hit_steps.shape[0] != num_envs:
        post_hit_steps = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        env._fairino_legal_sep_steps = post_hit_steps

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_hit = env._fairino_legal_sep_prev_hit.clone()
        post_hit_steps = env._fairino_legal_sep_steps.clone()
        prev_hit[reset_mask] = False
        post_hit_steps[reset_mask] = 0
        env._fairino_legal_sep_prev_hit = prev_hit
        env._fairino_legal_sep_steps = post_hit_steps

    new_hit = hit_done & (~env._fairino_legal_sep_prev_hit)
    post_hit_steps = torch.where(
        new_hit,
        torch.zeros_like(env._fairino_legal_sep_steps),
        torch.where(hit_done, env._fairino_legal_sep_steps + 1, torch.zeros_like(env._fairino_legal_sep_steps)),
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
    in_right_half = (landing_xy[:, 0] > net_x) & (landing_xy[:, 0] < center[0] + half_size[0])
    in_table_y = torch.abs(landing_xy[:, 1] - center[1]) < half_size[1]
    landing_score = torch.maximum((in_right_half & in_table_y).float(), target_score)

    behind_margin = ball_pos[:, 0] - paddle_pos[:, 0] - min_behind_x
    behind_score = torch.clamp(behind_margin / behind_std, min=0.0, max=1.0)
    speed_score = torch.clamp((ball_vel[:, 0] - min_return_speed) / (target_return_speed - min_return_speed), min=0.0, max=1.0)
    active_window = (post_hit_steps >= int(grace_steps)) & (post_hit_steps <= int(window_steps))
    legal = ~has_illegal_second_hit(env, num_envs)
    plausible_height = ball_pos[:, 2] > ball_center_at_table

    env._fairino_legal_sep_steps = post_hit_steps
    env._fairino_legal_sep_prev_hit = hit_done.clone()
    return (behind_score * speed_score * landing_score) * (hit_done & legal & active_window & plausible_height).float()


def ball_over_net(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_height: float = 0.9125,
    table_half_width: float = 0.7625,
    net_window: float = 0.18,
    max_clearance_height: float = 1.41,
) -> torch.Tensor:
    """Reward returned balls while they cross the net corridor."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel_x = ball.data.root_lin_vel_w[:, 0]
    returning = (vel_x > 0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)

    near_net = torch.abs(pos[:, 0] - net_x) < net_window
    in_table_y = torch.abs(pos[:, 1]) < table_half_width
    below_flyaway_height = pos[:, 2] < max_clearance_height
    height_score = torch.clamp((pos[:, 2] - net_height) / (max_clearance_height - net_height), min=0.0, max=1.0)
    corridor_gate = near_net & in_table_y & below_flyaway_height
    rightward_score = torch.clamp(vel_x / 2.5, min=0.0, max=1.0)
    crossed_score = (corridor_gate & (pos[:, 0] > net_x) & (pos[:, 2] > net_height)).float()
    approach_score = height_score * rightward_score
    return torch.maximum(crossed_score, approach_score * corridor_gate.float()) * returning.float()


def ball_to_target(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target: tuple[float, float, float] = (1.235, 0.0, 0.76),
    std: float = 0.45,
) -> torch.Tensor:
    """Reward ball proximity to the target landing area."""
    ball: RigidObject = env.scene[asset_cfg.name]
    target_pos = torch.tensor(target, device=env.device)
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    distance = torch.linalg.norm(ball_pos - target_pos, dim=1)
    returning = ((ball.data.root_lin_vel_w[:, 0] > 0.05) & legal_post_hit_mask(env, ball_pos.shape[0], ball_cfg=asset_cfg)).float()
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
    home_side: str = "right",
) -> torch.Tensor:
    """Reward post-hit balls whose ballistic landing estimate is on the opponent's table."""
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

    if home_side == "right":
        in_opponent_half = (landing_xy[:, 0] < net_x) & (landing_xy[:, 0] > center[0] - half_size[0])
        returning = (vel[:, 0] < -0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
    else:
        in_opponent_half = (landing_xy[:, 0] > net_x) & (landing_xy[:, 0] < center[0] + half_size[0])
        returning = (vel[:, 0] > 0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
    in_table_y = torch.abs(landing_xy[:, 1] - center[1]) < half_size[1]
    table_score = (in_opponent_half & in_table_y).float()
    plausible_height = pos[:, 2] > ball_center_at_table
    base = torch.maximum(table_score, target_score)

    max_good_z = table_top + 0.35
    height_factor = torch.clamp(max_good_z / torch.clamp(pos[:, 2], min=max_good_z), 0.2, 1.0)

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
    home_side: str = "right",
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

    if home_side == "right":
        abs_speed = torch.clamp(-vel[:, 0], min=0.0)
        in_opponent_half = (landing_xy[:, 0] < net_x) & (landing_xy[:, 0] > center[0] - half_size[0])
        returning = (vel[:, 0] < -0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
    else:
        abs_speed = torch.clamp(vel[:, 0], min=0.0)
        in_opponent_half = (landing_xy[:, 0] > net_x) & (landing_xy[:, 0] < center[0] + half_size[0])
        returning = (vel[:, 0] > 0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
    speed_score = torch.clamp(abs_speed / target_right_speed, 0.0, 1.0)
    speed_floor = torch.clamp((abs_speed - 0.1) / max(min_right_speed - 0.1, 1.0e-6), 0.0, 1.0)

    in_table_y = torch.abs(landing_xy[:, 1] - center[1]) < half_size[1]
    plausible_height = pos[:, 2] > ball_center_at_table
    quality = x_score * y_score * (0.25 + 0.75 * speed_score) * speed_floor
    return quality * (in_opponent_half & in_table_y & returning & plausible_height).float()


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
    home_side: str = "right",
) -> torch.Tensor:
    """Reward a returned ball that bounces on the opponent's half."""
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
        env, ball_cfg=asset_cfg, robot_cfg=robot_cfg, net_x=net_x,
        net_height=table_top + 0.1525, ball_radius=ball_radius,
        illegal_contact_distance=illegal_contact_distance,
        home_side=home_side,
    )
    illegal_second_hit_event(
        env, robot_cfg=robot_cfg, ball_cfg=asset_cfg, contact_distance=illegal_contact_distance,
    )
    legal = ~has_illegal_second_hit(env, pos.shape[0])
    clean_gate = has_clean_over_net(env, pos.shape[0]) if require_clean_over_net else torch.ones_like(legal)
    bounce = in_opponent_half & in_table_y & near_table_height & bounced_up & returning & clean_gate
    scale = torch.where(
        legal,
        torch.ones_like(pos[:, 0]),
        torch.full_like(pos[:, 0], float(illegal_reward_scale)),
    )
    peak_z = peak_post_hit_ball_z(env, ball_cfg=asset_cfg)
    max_good_z = table_top + 0.50
    height_factor = torch.clamp(max_good_z / torch.clamp(peak_z, min=max_good_z), 0.4, 1.0)
    return bounce.float() * scale * height_factor


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
) -> torch.Tensor:
    """Penalize ALL arm movement after a successful hit.

    Once has_return_hit is true, the arm can no longer affect the ball.
    Any further joint motion is pointless and should be penalized.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    gate = has_return_hit(env, ball.data.root_pos_w.shape[0])
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


# ===================================================================
#  Net-crossing quality rewards (BEST_V1)
# ===================================================================

_SERVE_SPEED = (1.25**2 + 0.12**2) ** 0.5  # ~1.256 m/s


def net_direction(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_window: float = 0.18,
    home_side: str = "right",
) -> torch.Tensor:
    """Reward ball velocity away from hitter at the net."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel_x = ball.data.root_lin_vel_w[:, 0]
    near_net = torch.abs(pos[:, 0] - net_x) < net_window
    if home_side == "right":
        returning = (vel_x < -0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
        score = torch.clamp(-vel_x, min=0.0, max=4.0) / 4.0
    else:
        returning = (vel_x > 0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
        score = torch.clamp(vel_x, min=0.0, max=4.0) / 4.0
    return score * near_net.float() * returning.float()


def net_height(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_window: float = 0.18,
    ideal_z: float = 0.56,
    std: float = 0.08,
    home_side: str = "right",
) -> torch.Tensor:
    """Reward ball height near serve height when crossing the net."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel_x = ball.data.root_lin_vel_w[:, 0]
    near_net = torch.abs(pos[:, 0] - net_x) < net_window
    if home_side == "right":
        returning = (vel_x < -0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
    else:
        returning = (vel_x > 0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
    score = 1.0 - torch.tanh(torch.abs(pos[:, 2] - ideal_z) / std)
    return score * near_net.float() * returning.float()


def net_speed(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_window: float = 0.18,
    ideal_speed: float = _SERVE_SPEED,
    std: float = 0.3,
    home_side: str = "right",
) -> torch.Tensor:
    """Reward ball speed close to initial serve speed when crossing the net."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]
    vel_x = vel[:, 0]
    near_net = torch.abs(pos[:, 0] - net_x) < net_window
    if home_side == "right":
        returning = (vel_x < -0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
    else:
        returning = (vel_x > 0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
    speed = torch.linalg.norm(vel, dim=1)
    score = 1.0 - torch.tanh(torch.abs(speed - ideal_speed) / std)
    return score * near_net.float() * returning.float()


def opponent_compatible_return(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_window: float = 0.18,
    net_height: float = 0.9125,
    # — serve-comfort zone at net crossing —
    min_y: float = -0.55,
    max_y: float = 0.55,
    max_z: float = 1.20,
    min_vx: float = 0.3,
    max_vx: float = 3.0,
    min_vy: float = -0.80,
    max_vy: float = 0.80,
    min_vz: float = -0.10,
    max_vz: float = 0.70,
    home_side: str = "right",
) -> torch.Tensor:
    """Reward returns that cross the net inside the opponent's serve comfort zone."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]
    vel_x = vel[:, 0]

    if home_side == "right":
        returning = (vel_x < -0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
        abs_vx = -vel_x  # positive away from hitter
    else:
        returning = (vel_x > 0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
        abs_vx = vel_x
    near_net = torch.abs(pos[:, 0] - net_x) < net_window

    y_ok = (pos[:, 1] > min_y) & (pos[:, 1] < max_y)
    z_ok = (pos[:, 2] > net_height) & (pos[:, 2] < max_z)
    vx_ok = (abs_vx > min_vx) & (abs_vx < max_vx)
    vy_ok = (vel[:, 1] > min_vy) & (vel[:, 1] < max_vy)
    vz_ok = (vel[:, 2] > min_vz) & (vel[:, 2] < max_vz)

    all_ok = near_net & y_ok & z_ok & vx_ok & vy_ok & vz_ok
    compatible_mask = all_ok.float() * returning.float()
    env._opponent_compatible_mask = compatible_mask
    return compatible_mask


def opponent_compatible_return_v2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_window: float = 0.18,
    net_height: float = 0.9125,
    # Opponent serve distribution (abs values, symmetric)
    ideal_y: float = 0.0,       # center of table
    y_std: float = 0.30,        # half-width comfort zone
    ideal_z: float = 0.98,      # just above net
    z_std: float = 0.15,        # height tolerance
    ideal_vx: float = 1.2,      # moderate serve speed
    vx_std: float = 0.55,       # speed tolerance
    ideal_vy: float = 0.0,      # no sidespin
    vy_std: float = 0.35,
    ideal_vz: float = 0.15,     # slight topspin
    vz_std: float = 0.30,
    home_side: str = "right",
) -> torch.Tensor:
    """Dense reward for returns that look like opponent's serve distribution.

    Instead of a binary gate, this computes per-dimension Gaussian scores
    and multiplies them, giving continuous gradient even when only some
    dimensions are close to the comfort zone.
    """
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]
    vel_x = vel[:, 0]

    if home_side == "right":
        returning = (vel_x < -0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
        abs_vx = -vel_x
    else:
        returning = (vel_x > 0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
        abs_vx = vel_x

    near_net = torch.abs(pos[:, 0] - net_x) < net_window
    active = near_net & returning

    # Per-dimension Gaussian scores (0-1)
    score_y = torch.exp(-0.5 * ((pos[:, 1] - ideal_y) / y_std) ** 2)
    score_z = torch.exp(-0.5 * ((pos[:, 2] - ideal_z) / z_std) ** 2)
    score_vx = torch.exp(-0.5 * ((abs_vx - ideal_vx) / vx_std) ** 2)
    score_vy = torch.exp(-0.5 * ((vel[:, 1] - ideal_vy) / vy_std) ** 2)
    score_vz = torch.exp(-0.5 * ((vel[:, 2] - ideal_vz) / vz_std) ** 2)

    # Height penalty: z above ideal gets linear penalty baked in
    above_ideal = torch.clamp(pos[:, 2] - ideal_z, min=0.0)
    height_penalty = torch.exp(-above_ideal / 0.25)  # decays above ideal_z

    dense_score = score_y * score_z * score_vx * score_vy * score_vz * height_penalty

    # Still store binary mask for right_table_bounce scaling
    y_ok = (pos[:, 1] > -0.55) & (pos[:, 1] < 0.55)
    z_ok = (pos[:, 2] > net_height) & (pos[:, 2] < 1.20)
    vx_ok = (abs_vx > 0.2) & (abs_vx < 1.5)
    vy_ok = (vel[:, 1] > -0.80) & (vel[:, 1] < 0.80)
    vz_ok = (vel[:, 2] > -0.10) & (vel[:, 2] < 0.70)
    all_ok = near_net & y_ok & z_ok & vx_ok & vy_ok & vz_ok
    env._opponent_compatible_mask = all_ok.float() * returning.float()

    return dense_score * active.float()


def opponent_vz_quality(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_window: float = 0.18,
    ideal_vz: float = 0.15,
    vz_std: float = 0.12,
    home_side: str = "right",
) -> torch.Tensor:
    """Narrow vz-only Gaussian reward at net crossing — targets vertical velocity bottleneck."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]

    returning = _is_returning(vel[:, 0], home_side)
    returning = returning & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
    near_net = torch.abs(pos[:, 0] - net_x) < net_window
    active = near_net & returning

    score_vz = torch.exp(-0.5 * ((vel[:, 2] - ideal_vz) / vz_std) ** 2)
    return score_vz * active.float()


def right_table_bounce_opponent_scaled(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    table_center: tuple[float, float, float] = (0.55, 0.0, 0.74),
    table_size: tuple[float, float, float] = (2.74, 1.525, 0.04),
    ball_radius: float = 0.02,
    net_x: float = 0.55,
    illegal_contact_distance: float = 0.10,
    illegal_reward_scale: float = 0.0,
    require_clean_over_net: bool = True,
    compatible_base: float = 0.3,
    home_side: str = "right",
) -> torch.Tensor:
    """right_table_bounce_reward scaled by opponent-compatible factor."""
    base = right_table_bounce_reward(
        env, asset_cfg=asset_cfg, robot_cfg=robot_cfg,
        table_center=table_center, table_size=table_size,
        ball_radius=ball_radius, net_x=net_x,
        illegal_contact_distance=illegal_contact_distance,
        illegal_reward_scale=illegal_reward_scale,
        require_clean_over_net=require_clean_over_net,
        home_side=home_side,
    )
    compat = getattr(env, '_opponent_compatible_mask', torch.ones_like(base))
    scale = compat * 1.0 + (1.0 - compat) * compatible_base
    return base * scale


def centerline_x(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    y_window: float = 0.10,
    std: float = 0.25,
) -> torch.Tensor:
    """Reward ball x close to 0 when crossing the centerline (y≈0)."""
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel_x = ball.data.root_lin_vel_w[:, 0]
    near_center = torch.abs(pos[:, 1]) < y_window
    returning = (vel_x > 0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
    score = 1.0 - torch.tanh(torch.abs(pos[:, 0]) / std)
    return score * near_center.float() * returning.float()


def rally_return(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """One-time reward: ball bounced on left table, then robot hit it back over the net."""
    ball: RigidObject = env.scene[ball_cfg.name]
    num_envs = ball.data.root_pos_w.shape[0]
    vel_x = ball.data.root_lin_vel_w[:, 0]

    bounced_left = has_left_bounce(env, num_envs)
    hit_done = has_return_hit(env, num_envs)
    legal_hit = legal_post_hit_mask(env, num_envs, ball_cfg=ball_cfg)
    going_right = vel_x > 0.05

    # Rising edge: first step where all three conditions are met
    triggered = bounced_left & legal_hit & going_right
    was_triggered = getattr(env, "_rally_reward_triggered", None)
    if was_triggered is None or was_triggered.shape[0] != num_envs:
        was_triggered = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        env._rally_reward_triggered = was_triggered

    # Reset on episode start
    ep_len = getattr(env, "episode_length_buf", None)
    if ep_len is not None:
        reset_mask = ep_len <= 1
        if torch.any(reset_mask):
            env._rally_reward_triggered[reset_mask] = False

    new_trigger = triggered & (~env._rally_reward_triggered)
    if torch.any(new_trigger):
        env._rally_reward_triggered[new_trigger] = True

    return new_trigger.float()


def left_table_bounce(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """One-time reward for the ball bouncing on the left table (robot side)."""
    return left_table_bounce_event(env, ball_cfg=ball_cfg).float()


def high_return_height_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    net_x: float = 0.55,
    net_window: float = 0.18,
    net_height: float = 0.9125,
    max_height: float = 1.2175,
) -> torch.Tensor:
    """Penalize returns that cross the net above max_height (3x net).

    Zero below max_height, linearly increasing penalty above.
    Discourages high lobs for future two-arm rally.
    """
    ball: RigidObject = env.scene[asset_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel_x = ball.data.root_lin_vel_w[:, 0]

    near_net = torch.abs(pos[:, 0] - net_x) < net_window
    returning = (vel_x > 0.05) & legal_post_hit_mask(env, pos.shape[0], ball_cfg=asset_cfg)
    too_high = pos[:, 2] > max_height

    violation = torch.clamp((pos[:, 2] - max_height) / 0.3, 0.0, 1.0)

    return violation * near_net.float() * returning.float() * too_high.float()


def early_hit_penalty(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.14,
) -> torch.Tensor:
    """Penalize paddle-ball contact before the ball has bounced on the left table."""
    contact_now, _, _ = paddle_contact_event(
        env, robot_cfg=robot_cfg, ball_cfg=ball_cfg, contact_distance=contact_distance
    )
    pre_bounce = ~has_left_bounce(env)
    return (contact_now & pre_bounce).float()
