"""Shared helpers for the Fairino3 ping-pong MDP terms."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def episode_reset_mask(env, num_envs: int) -> torch.Tensor:
    """Return environments that are at the beginning of a new episode."""
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is None:
        return torch.zeros(num_envs, device=env.device, dtype=torch.bool)
    return episode_length_buf <= 1


def paddle_contact_event(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Track paddle-ball contact and detect a second contact within the same episode.

    Returns:
        contact_now: current distance-based contact gate.
        contact_event: rising edge of the contact gate.
        second_contact: rising edge after the first contact in the episode.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] - env.scene.env_origins
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    distance = torch.linalg.norm(paddle_pos - ball_pos, dim=1)
    contact_now = distance <= contact_distance

    num_envs = contact_now.shape[0]
    prev_contact = getattr(env, "_fairino_paddle_contact_prev", None)
    contact_seen = getattr(env, "_fairino_paddle_contact_seen", None)
    if prev_contact is None or prev_contact.shape[0] != num_envs:
        prev_contact = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        contact_seen = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        env._fairino_paddle_contact_prev = prev_contact
        env._fairino_paddle_contact_seen = contact_seen
    elif contact_seen is None or contact_seen.shape[0] != num_envs:
        contact_seen = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        env._fairino_paddle_contact_seen = contact_seen

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_contact = env._fairino_paddle_contact_prev.clone()
        contact_seen = env._fairino_paddle_contact_seen.clone()
        prev_contact[reset_mask] = False
        contact_seen[reset_mask] = False
        env._fairino_paddle_contact_prev = prev_contact
        env._fairino_paddle_contact_seen = contact_seen

    contact_event = contact_now & (~env._fairino_paddle_contact_prev)
    second_contact = contact_event & env._fairino_paddle_contact_seen

    if torch.any(contact_event):
        contact_seen = env._fairino_paddle_contact_seen.clone()
        contact_seen[contact_event] = True
        env._fairino_paddle_contact_seen = contact_seen

    env._fairino_paddle_contact_prev = contact_now
    return contact_now, contact_event, second_contact


def first_return_hit_event(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.10,
    min_incoming_speed: float = 0.10,
    min_return_speed: float = 0.10,
    contact_window_steps: int = 4,
    home_side: str = "right",
) -> torch.Tensor:
    """Detect the first useful hit that changes the ball from incoming to returning.

    Isaac Sim may apply the paddle impulse inside physics substeps, so the
    manager step can observe contact and velocity reversal on different frames.
    Keep a short post-contact window to avoid missing those hits.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] - env.scene.env_origins
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    distance = torch.linalg.norm(paddle_pos - ball_pos, dim=1)
    contact_now = distance <= contact_distance
    ball_vel_x = ball.data.root_lin_vel_w[:, 0]

    num_envs = ball_vel_x.shape[0]

    # Home table bounce: LEFT for left arm, RIGHT for right arm
    if home_side == "right":
        right_table_bounce_event(env, ball_cfg=ball_cfg)
        bounced_home = has_right_bounce(env, num_envs)
        incoming = (ball_vel_x > min_incoming_speed) | (getattr(env, "_fairino_prev_ball_vel_x", ball_vel_x) > min_incoming_speed)
        returning = ball_vel_x < -min_return_speed
    else:
        left_table_bounce_event(env, ball_cfg=ball_cfg)
        bounced_home = has_left_bounce(env, num_envs)
        incoming = (ball_vel_x < -min_incoming_speed) | (getattr(env, "_fairino_prev_ball_vel_x", ball_vel_x) < -min_incoming_speed)
        returning = ball_vel_x > min_return_speed

    prev_ball_vel_x = getattr(env, "_fairino_prev_ball_vel_x", None)
    return_hit_state = getattr(env, "_fairino_return_hit_state", None)
    contact_window = getattr(env, "_fairino_return_contact_window", None)
    incoming_contact_seen = getattr(env, "_fairino_return_incoming_contact_seen", None)
    if prev_ball_vel_x is None or prev_ball_vel_x.shape[0] != num_envs:
        prev_ball_vel_x = ball_vel_x.clone()
        return_hit_state = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        contact_window = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        incoming_contact_seen = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        env._fairino_prev_ball_vel_x = prev_ball_vel_x
        env._fairino_return_hit_state = return_hit_state
        env._fairino_return_contact_window = contact_window
        env._fairino_return_incoming_contact_seen = incoming_contact_seen

    if return_hit_state is None or return_hit_state.shape[0] != num_envs:
        return_hit_state = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        env._fairino_return_hit_state = return_hit_state
    if contact_window is None or contact_window.shape[0] != num_envs:
        contact_window = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        env._fairino_return_contact_window = contact_window
    if incoming_contact_seen is None or incoming_contact_seen.shape[0] != num_envs:
        incoming_contact_seen = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        env._fairino_return_incoming_contact_seen = incoming_contact_seen

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_ball_vel_x = env._fairino_prev_ball_vel_x.clone()
        return_hit_state = env._fairino_return_hit_state.clone()
        contact_window = env._fairino_return_contact_window.clone()
        incoming_contact_seen = env._fairino_return_incoming_contact_seen.clone()
        prev_ball_vel_x[reset_mask] = ball_vel_x[reset_mask]
        return_hit_state[reset_mask] = False
        contact_window[reset_mask] = 0
        incoming_contact_seen[reset_mask] = False
        env._fairino_prev_ball_vel_x = prev_ball_vel_x
        env._fairino_return_hit_state = return_hit_state
        env._fairino_return_contact_window = contact_window
        env._fairino_return_incoming_contact_seen = incoming_contact_seen

    contact_window = torch.clamp(env._fairino_return_contact_window - 1, min=0)
    incoming_contact_seen = env._fairino_return_incoming_contact_seen & (contact_window > 0)

    incoming_now_or_prev = incoming
    useful_contact = contact_now & bounced_home & (~env._fairino_return_hit_state) & incoming_now_or_prev
    if torch.any(useful_contact):
        contact_window = contact_window.clone()
        incoming_contact_seen = incoming_contact_seen.clone()
        contact_window[useful_contact] = max(int(contact_window_steps), 1)
        incoming_contact_seen[useful_contact] = True

    valid_hit = (
        (~env._fairino_return_hit_state)
        & bounced_home
        & incoming_contact_seen
        & (contact_window > 0)
        & returning
    )
    if torch.any(valid_hit):
        return_hit_state = env._fairino_return_hit_state.clone()
        return_hit_state[valid_hit] = True
        env._fairino_return_hit_state = return_hit_state
        contact_window = contact_window.clone()
        incoming_contact_seen = incoming_contact_seen.clone()
        contact_window[valid_hit] = 0
        incoming_contact_seen[valid_hit] = False

    env._fairino_return_contact_window = contact_window
    env._fairino_return_incoming_contact_seen = incoming_contact_seen
    env._fairino_prev_ball_vel_x = ball_vel_x.clone()
    return valid_hit


def has_return_hit(env, num_envs: int | None = None) -> torch.Tensor:
    """Return whether each environment has already produced a useful hit."""
    return_hit_state = getattr(env, "_fairino_return_hit_state", None)
    if return_hit_state is None:
        if num_envs is None:
            num_envs = env.num_envs
        return torch.zeros(num_envs, device=env.device, dtype=torch.bool)
    return return_hit_state


def illegal_second_hit_event(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.10,
    grace_steps: int = 4,
    window_steps: int = 60,
    min_return_speed: float = 0.10,
    min_delta_speed: float = 0.03,
) -> torch.Tensor:
    """Detect an illegal second paddle hit after the first valid return.

    The event is stricter than a raw distance re-contact: it must happen after
    the useful return hit, outside the immediate follow-through grace period,
    while the ball is still moving rightward, and with a noticeable velocity
    change.  The result is cached per manager step so reward and termination
    terms can query it without double-updating the contact edge state.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = ball.data.root_pos_w.shape[0]
    step_id = int(getattr(env, "common_step_counter", 0))

    cached_step = getattr(env, "_fairino_illegal_second_step_id", None)
    cached_event = getattr(env, "_fairino_illegal_second_event", None)
    if cached_step == step_id and cached_event is not None and cached_event.shape[0] == num_envs:
        return cached_event

    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] - env.scene.env_origins
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w[:, :3]
    distance = torch.linalg.norm(paddle_pos - ball_pos, dim=1)
    contact_now = distance <= contact_distance

    prev_contact = getattr(env, "_fairino_illegal_second_prev_contact", None)
    prev_hit = getattr(env, "_fairino_illegal_second_prev_hit", None)
    post_hit_steps = getattr(env, "_fairino_illegal_second_steps", None)
    seen_illegal = getattr(env, "_fairino_illegal_second_seen", None)
    prev_ball_vel = getattr(env, "_fairino_illegal_second_prev_ball_vel", None)
    if prev_contact is None or prev_contact.shape[0] != num_envs:
        prev_contact = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        prev_hit = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        post_hit_steps = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        seen_illegal = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        prev_ball_vel = ball_vel.clone()
        env._fairino_illegal_second_prev_contact = prev_contact
        env._fairino_illegal_second_prev_hit = prev_hit
        env._fairino_illegal_second_steps = post_hit_steps
        env._fairino_illegal_second_seen = seen_illegal
        env._fairino_illegal_second_prev_ball_vel = prev_ball_vel

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_contact = env._fairino_illegal_second_prev_contact.clone()
        prev_hit = env._fairino_illegal_second_prev_hit.clone()
        post_hit_steps = env._fairino_illegal_second_steps.clone()
        seen_illegal = env._fairino_illegal_second_seen.clone()
        prev_ball_vel = env._fairino_illegal_second_prev_ball_vel.clone()
        prev_contact[reset_mask] = False
        prev_hit[reset_mask] = False
        post_hit_steps[reset_mask] = 0
        seen_illegal[reset_mask] = False
        prev_ball_vel[reset_mask] = ball_vel[reset_mask]
        env._fairino_illegal_second_prev_contact = prev_contact
        env._fairino_illegal_second_prev_hit = prev_hit
        env._fairino_illegal_second_steps = post_hit_steps
        env._fairino_illegal_second_seen = seen_illegal
        env._fairino_illegal_second_prev_ball_vel = prev_ball_vel

    hit_done = has_return_hit(env, num_envs)
    new_hit = hit_done & (~env._fairino_illegal_second_prev_hit)
    post_hit_steps = torch.where(
        new_hit,
        torch.zeros_like(env._fairino_illegal_second_steps),
        torch.where(hit_done, env._fairino_illegal_second_steps + 1, torch.zeros_like(env._fairino_illegal_second_steps)),
    )
    contact_event = contact_now & (~env._fairino_illegal_second_prev_contact)
    delta_speed = torch.linalg.norm(ball_vel - env._fairino_illegal_second_prev_ball_vel, dim=1)
    illegal_event = (
        contact_event
        & hit_done
        & (~new_hit)
        & (~env._fairino_illegal_second_seen)
        & (post_hit_steps >= int(grace_steps))
        & (post_hit_steps <= int(window_steps))
        & (delta_speed > min_delta_speed)
    )

    if torch.any(illegal_event):
        seen_illegal = env._fairino_illegal_second_seen.clone()
        seen_illegal[illegal_event] = True
        env._fairino_illegal_second_seen = seen_illegal

    env._fairino_illegal_second_prev_contact = contact_now
    env._fairino_illegal_second_prev_hit = hit_done.clone()
    env._fairino_illegal_second_steps = post_hit_steps
    env._fairino_illegal_second_prev_ball_vel = ball_vel.clone()
    env._fairino_illegal_second_event = illegal_event
    env._fairino_illegal_second_step_id = step_id
    return illegal_event


def has_illegal_second_hit(env, num_envs: int | None = None) -> torch.Tensor:
    """Return whether each environment has had an illegal second hit this episode."""
    seen = getattr(env, "_fairino_illegal_second_seen", None)
    if seen is None:
        if num_envs is None:
            num_envs = env.num_envs
        return torch.zeros(num_envs, device=env.device, dtype=torch.bool)
    return seen


def legal_post_hit_mask(
    env,
    num_envs: int,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.10,
) -> torch.Tensor:
    """Return a mask that is True only when a legal return hit is active.

    The mask is True after the first valid return hit AND before any illegal
    second hit is detected.  Once an illegal second hit is observed the mask
    stays False for the rest of the episode.

    This calls illegal_second_hit_event internally so the illegal-hit state
    is always up-to-date when the mask is queried.
    """
    illegal_second_hit_event(
        env,
        robot_cfg=robot_cfg,
        ball_cfg=ball_cfg,
        contact_distance=contact_distance,
    )
    return has_return_hit(env, num_envs) & (~has_illegal_second_hit(env, num_envs))


def clean_over_net_event(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    net_x: float = 0.55,
    net_height: float = 0.9125,
    ball_radius: float = 0.02,
    max_steps_after_hit: int = 90,
    min_return_speed: float = 0.10,
    illegal_contact_distance: float = 0.10,
    home_side: str = "right",
) -> torch.Tensor:
    """Detect a clean one-hit return crossing the net after the first valid hit."""
    ball: RigidObject = env.scene[ball_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]
    num_envs = pos.shape[0]
    device = env.device
    step_id = int(getattr(env, "common_step_counter", 0))

    cached_step = getattr(env, "_fairino_clean_over_net_step_id", None)
    cached_event = getattr(env, "_fairino_clean_over_net_event", None)
    if cached_step == step_id and cached_event is not None and cached_event.shape[0] == num_envs:
        return cached_event

    illegal_second_hit_event(
        env,
        robot_cfg=robot_cfg,
        ball_cfg=ball_cfg,
        contact_distance=illegal_contact_distance,
    )
    hit_done = has_return_hit(env, num_envs)
    illegal_done = has_illegal_second_hit(env, num_envs)

    prev_hit = getattr(env, "_fairino_clean_over_net_prev_hit", None)
    prev_ball_x = getattr(env, "_fairino_clean_over_net_prev_ball_x", None)
    steps_since_hit = getattr(env, "_fairino_clean_over_net_steps", None)
    seen_over_net = getattr(env, "_fairino_clean_over_net_seen", None)
    if prev_hit is None or prev_hit.shape[0] != num_envs:
        prev_hit = torch.zeros(num_envs, device=device, dtype=torch.bool)
        prev_ball_x = pos[:, 0].clone()
        steps_since_hit = torch.zeros(num_envs, device=device, dtype=torch.long)
        seen_over_net = torch.zeros(num_envs, device=device, dtype=torch.bool)
        env._fairino_clean_over_net_prev_hit = prev_hit
        env._fairino_clean_over_net_prev_ball_x = prev_ball_x
        env._fairino_clean_over_net_steps = steps_since_hit
        env._fairino_clean_over_net_seen = seen_over_net

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_hit = env._fairino_clean_over_net_prev_hit.clone()
        prev_ball_x = env._fairino_clean_over_net_prev_ball_x.clone()
        steps_since_hit = env._fairino_clean_over_net_steps.clone()
        seen_over_net = env._fairino_clean_over_net_seen.clone()
        prev_hit[reset_mask] = False
        prev_ball_x[reset_mask] = pos[reset_mask, 0]
        steps_since_hit[reset_mask] = 0
        seen_over_net[reset_mask] = False
        env._fairino_clean_over_net_prev_hit = prev_hit
        env._fairino_clean_over_net_prev_ball_x = prev_ball_x
        env._fairino_clean_over_net_steps = steps_since_hit
        env._fairino_clean_over_net_seen = seen_over_net

    new_hit = hit_done & (~env._fairino_clean_over_net_prev_hit)
    steps_since_hit = torch.where(
        new_hit,
        torch.zeros_like(env._fairino_clean_over_net_steps),
        torch.where(hit_done, env._fairino_clean_over_net_steps + 1, torch.zeros_like(env._fairino_clean_over_net_steps)),
    )
    if home_side == "right":
        crossed_net = (env._fairino_clean_over_net_prev_ball_x >= net_x) & (pos[:, 0] < net_x)
        on_or_past_net = pos[:, 0] <= net_x
        moving_toward_opponent = vel[:, 0] < -min_return_speed
    else:
        crossed_net = (env._fairino_clean_over_net_prev_ball_x < net_x) & (pos[:, 0] >= net_x)
        on_or_past_net = pos[:, 0] >= net_x
        moving_toward_opponent = vel[:, 0] > min_return_speed
    above_net = pos[:, 2] > (net_height + ball_radius)
    in_window = (steps_since_hit > 0) & (steps_since_hit <= int(max_steps_after_hit))
    event = (
        (crossed_net | on_or_past_net)
        & above_net
        & in_window
        & moving_toward_opponent
        & hit_done
        & (~illegal_done)
        & (~env._fairino_clean_over_net_seen)
    )
    if torch.any(event):
        seen_over_net = env._fairino_clean_over_net_seen.clone()
        seen_over_net[event] = True
        env._fairino_clean_over_net_seen = seen_over_net

    env._fairino_clean_over_net_prev_hit = hit_done.clone()
    env._fairino_clean_over_net_prev_ball_x = pos[:, 0].clone()
    env._fairino_clean_over_net_steps = steps_since_hit
    env._fairino_clean_over_net_event = event
    env._fairino_clean_over_net_step_id = step_id
    return event


def has_clean_over_net(env, num_envs: int | None = None) -> torch.Tensor:
    """Return whether each environment has produced a clean over-net return."""
    seen = getattr(env, "_fairino_clean_over_net_seen", None)
    if seen is None:
        if num_envs is None:
            num_envs = env.num_envs
        return torch.zeros(num_envs, device=env.device, dtype=torch.bool)
    return seen


def left_table_bounce_event(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    table_center: tuple[float, float, float] = (0.55, 0.0, 0.74),
    table_size: tuple[float, float, float] = (2.74, 1.525, 0.04),
    ball_radius: float = 0.02,
    net_x: float = 0.55,
    height_tolerance: float = 0.06,
    min_upward_velocity: float = 0.05,
) -> torch.Tensor:
    """Detect ball bounce on the left half of the table (before the robot hits it).

    Returns a rising-edge signal: True exactly on the step the bounce is first detected.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]
    num_envs = pos.shape[0]
    device = env.device

    center = torch.tensor(table_center, device=device)
    size = torch.tensor(table_size, device=device)
    half_size = 0.5 * size
    table_top = center[2] + half_size[2]
    ball_center_at_table = table_top + ball_radius

    on_left = (pos[:, 0] > center[0] - half_size[0]) & (pos[:, 0] < net_x)
    in_y = torch.abs(pos[:, 1] - center[1]) < half_size[1]
    near_surface = torch.abs(pos[:, 2] - ball_center_at_table) < height_tolerance
    bouncing_up = vel[:, 2] > min_upward_velocity
    bounce_now = on_left & in_y & near_surface & bouncing_up

    # State tracking
    prev_bounce = getattr(env, "_fairino_left_bounce_prev", None)
    bounce_seen = getattr(env, "_fairino_left_bounce_seen", None)
    if prev_bounce is None or prev_bounce.shape[0] != num_envs:
        prev_bounce = torch.zeros(num_envs, device=device, dtype=torch.bool)
        bounce_seen = torch.zeros(num_envs, device=device, dtype=torch.bool)
        env._fairino_left_bounce_prev = prev_bounce
        env._fairino_left_bounce_seen = bounce_seen

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_bounce = env._fairino_left_bounce_prev.clone()
        bounce_seen = env._fairino_left_bounce_seen.clone()
        prev_bounce[reset_mask] = False
        bounce_seen[reset_mask] = False
        env._fairino_left_bounce_prev = prev_bounce
        env._fairino_left_bounce_seen = bounce_seen

    bounce_event = bounce_now & (~env._fairino_left_bounce_prev) & (~bounce_seen)

    if torch.any(bounce_event):
        bs = env._fairino_left_bounce_seen.clone()
        bs[bounce_event] = True
        env._fairino_left_bounce_seen = bs

    env._fairino_left_bounce_prev = bounce_now
    return bounce_event


def has_left_bounce(env, num_envs: int | None = None) -> torch.Tensor:
    """Return whether the ball has bounced on the left table this episode."""
    buf = getattr(env, "_fairino_left_bounce_seen", None)
    if buf is None:
        if num_envs is None:
            num_envs = env.num_envs
        return torch.zeros(num_envs, device=env.device, dtype=torch.bool)
    return buf


# ---- right-table bounce (mirror for right-arm training) ----

def right_table_bounce_event(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    table_center: tuple[float, float, float] = (0.55, 0.0, 0.74),
    table_size: tuple[float, float, float] = (2.74, 1.525, 0.04),
    ball_radius: float = 0.02,
    net_x: float = 0.55,
    height_tolerance: float = 0.06,
    min_upward_velocity: float = 0.05,
) -> torch.Tensor:
    """Detect ball bounce on the RIGHT half of the table."""
    ball: RigidObject = env.scene[ball_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]
    num_envs = pos.shape[0]
    device = env.device

    center = torch.tensor(table_center, device=device)
    size = torch.tensor(table_size, device=device)
    half_size = 0.5 * size
    table_top = center[2] + half_size[2]
    ball_center_at_table = table_top + ball_radius

    on_right = (pos[:, 0] > net_x) & (pos[:, 0] < center[0] + half_size[0])
    in_y = torch.abs(pos[:, 1] - center[1]) < half_size[1]
    near_surface = torch.abs(pos[:, 2] - ball_center_at_table) < height_tolerance
    bouncing_up = vel[:, 2] > min_upward_velocity
    bounce_now = on_right & in_y & near_surface & bouncing_up

    prev_bounce = getattr(env, "_fairino_right_bounce_prev", None)
    bounce_seen = getattr(env, "_fairino_right_bounce_seen", None)
    if prev_bounce is None or prev_bounce.shape[0] != num_envs:
        prev_bounce = torch.zeros(num_envs, device=device, dtype=torch.bool)
        bounce_seen = torch.zeros(num_envs, device=device, dtype=torch.bool)
        env._fairino_right_bounce_prev = prev_bounce
        env._fairino_right_bounce_seen = bounce_seen

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_bounce = env._fairino_right_bounce_prev.clone()
        bounce_seen = env._fairino_right_bounce_seen.clone()
        prev_bounce[reset_mask] = False
        bounce_seen[reset_mask] = False
        env._fairino_right_bounce_prev = prev_bounce
        env._fairino_right_bounce_seen = bounce_seen

    bounce_event = bounce_now & (~env._fairino_right_bounce_prev) & (~bounce_seen)

    if torch.any(bounce_event):
        bs = env._fairino_right_bounce_seen.clone()
        bs[bounce_event] = True
        env._fairino_right_bounce_seen = bs

    env._fairino_right_bounce_prev = bounce_now
    return bounce_event


def has_right_bounce(env, num_envs: int | None = None) -> torch.Tensor:
    """Return whether the ball has bounced on the right table this episode."""
    buf = getattr(env, "_fairino_right_bounce_seen", None)
    if buf is None:
        if num_envs is None:
            num_envs = env.num_envs
        return torch.zeros(num_envs, device=env.device, dtype=torch.bool)
    return buf


def peak_post_hit_ball_z(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Track the maximum ball Z observed after a return hit, per environment.

    Resets on episode boundaries. Used to scale success rewards by trajectory height.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    num_envs = pos.shape[0]

    peak = getattr(env, "_fairino_peak_ball_z", None)
    if peak is None or peak.shape[0] != num_envs:
        peak = torch.zeros(num_envs, device=env.device)
        env._fairino_peak_ball_z = peak

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        peak = env._fairino_peak_ball_z.clone()
        peak[reset_mask] = 0.0
        env._fairino_peak_ball_z = peak

    hit_done = has_return_hit(env, num_envs)
    env._fairino_peak_ball_z = torch.where(
        hit_done,
        torch.maximum(env._fairino_peak_ball_z, pos[:, 2]),
        env._fairino_peak_ball_z,
    )

    return env._fairino_peak_ball_z
