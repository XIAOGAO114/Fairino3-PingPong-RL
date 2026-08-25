"""Shared helpers for the Fairino3 ping-pong MDP terms.

All state-machine functions accept a ``state_prefix`` parameter so the same
logic can track left-arm and right-arm events independently.  Pass the
default empty string for the left arm and ``"_right"`` for the right arm.
"""

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


# ---------------------------------------------------------------------------
# paddle contact event
# ---------------------------------------------------------------------------

def paddle_contact_event(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.10,
    state_prefix: str = "",
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
    prev_attr = f"_fairino{state_prefix}_paddle_contact_prev"
    seen_attr = f"_fairino{state_prefix}_paddle_contact_seen"
    prev_contact = getattr(env, prev_attr, None)
    contact_seen = getattr(env, seen_attr, None)
    if prev_contact is None or prev_contact.shape[0] != num_envs:
        prev_contact = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        contact_seen = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        setattr(env, prev_attr, prev_contact)
        setattr(env, seen_attr, contact_seen)
    elif contact_seen is None or contact_seen.shape[0] != num_envs:
        contact_seen = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        setattr(env, seen_attr, contact_seen)

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_contact = getattr(env, prev_attr).clone()
        contact_seen = getattr(env, seen_attr).clone()
        prev_contact[reset_mask] = False
        contact_seen[reset_mask] = False
        setattr(env, prev_attr, prev_contact)
        setattr(env, seen_attr, contact_seen)

    contact_event = contact_now & (~getattr(env, prev_attr))
    second_contact = contact_event & getattr(env, seen_attr)

    if torch.any(contact_event):
        contact_seen = getattr(env, seen_attr).clone()
        contact_seen[contact_event] = True
        setattr(env, seen_attr, contact_seen)

    setattr(env, prev_attr, contact_now)
    return contact_now, contact_event, second_contact


# ---------------------------------------------------------------------------
# home table bounce event  (replaces left_table_bounce_event)
# ---------------------------------------------------------------------------

def home_table_bounce_event(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    table_center: tuple[float, float, float] = (0.55, 0.0, 0.74),
    table_size: tuple[float, float, float] = (2.74, 1.525, 0.04),
    ball_radius: float = 0.02,
    net_x: float = 0.55,
    height_tolerance: float = 0.06,
    min_upward_velocity: float = 0.05,
    side: str = "left",
    state_prefix: str = "",
) -> torch.Tensor:
    """Detect ball bounce on the home half of the table.

    ``side="left"`` checks x in (table_left_edge, net_x).
    ``side="right"`` checks x in (net_x, table_right_edge).
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

    if side == "left":
        on_home = (pos[:, 0] > center[0] - half_size[0]) & (pos[:, 0] < net_x)
    else:
        on_home = (pos[:, 0] > net_x) & (pos[:, 0] < center[0] + half_size[0])

    in_y = torch.abs(pos[:, 1] - center[1]) < half_size[1]
    near_surface = torch.abs(pos[:, 2] - ball_center_at_table) < height_tolerance
    bouncing_up = vel[:, 2] > min_upward_velocity
    bounce_now = on_home & in_y & near_surface & bouncing_up

    prev_attr = f"_fairino{state_prefix}_home_bounce_prev"
    seen_attr = f"_fairino{state_prefix}_home_bounce_seen"
    prev_bounce = getattr(env, prev_attr, None)
    bounce_seen = getattr(env, seen_attr, None)
    if prev_bounce is None or prev_bounce.shape[0] != num_envs:
        prev_bounce = torch.zeros(num_envs, device=device, dtype=torch.bool)
        bounce_seen = torch.zeros(num_envs, device=device, dtype=torch.bool)
        setattr(env, prev_attr, prev_bounce)
        setattr(env, seen_attr, bounce_seen)

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_bounce = getattr(env, prev_attr).clone()
        bounce_seen = getattr(env, seen_attr).clone()
        prev_bounce[reset_mask] = False
        bounce_seen[reset_mask] = False
        setattr(env, prev_attr, prev_bounce)
        setattr(env, seen_attr, bounce_seen)

    bounce_event = bounce_now & (~getattr(env, prev_attr)) & (~getattr(env, seen_attr))

    if torch.any(bounce_event):
        bs = getattr(env, seen_attr).clone()
        bs[bounce_event] = True
        setattr(env, seen_attr, bs)

    setattr(env, prev_attr, bounce_now)
    return bounce_event


def has_home_bounce(env, num_envs: int | None = None, state_prefix: str = "") -> torch.Tensor:
    """Return whether the ball has bounced on the home table half this episode."""
    buf = getattr(env, f"_fairino{state_prefix}_home_bounce_seen", None)
    if buf is None:
        if num_envs is None:
            num_envs = env.num_envs
        return torch.zeros(num_envs, device=env.device, dtype=torch.bool)
    return buf


# backward-compatible aliases
def left_table_bounce_event(env, **kwargs):
    """Legacy alias for home_table_bounce_event(side='left')."""
    return home_table_bounce_event(env, side="left", **kwargs)


def has_left_bounce(env, num_envs: int | None = None) -> torch.Tensor:
    """Legacy alias for has_home_bounce with default prefix."""
    return has_home_bounce(env, num_envs, state_prefix="")


# ---------------------------------------------------------------------------
# first return hit event
# ---------------------------------------------------------------------------

def first_return_hit_event(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.10,
    min_incoming_speed: float = 0.10,
    min_return_speed: float = 0.10,
    contact_window_steps: int = 4,
    state_prefix: str = "",
    home_side: str = "left",
    other_state_prefix: str = "",
) -> torch.Tensor:
    """Detect the first useful hit that changes the ball from incoming to returning.

    Isaac Sim may apply the paddle impulse inside physics substeps, so the
    manager step can observe contact and velocity reversal on different frames.
    Keep a short post-contact window to avoid missing those hits.

    When ``other_state_prefix`` is non-empty, a valid hit resets the other
    arm's episode state so it can re-hit in a rally.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] - env.scene.env_origins
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    distance = torch.linalg.norm(paddle_pos - ball_pos, dim=1)
    contact_now = distance <= contact_distance
    ball_vel_x = ball.data.root_lin_vel_w[:, 0]

    num_envs = ball_vel_x.shape[0]

    home_table_bounce_event(env, ball_cfg=ball_cfg, side=home_side, state_prefix=state_prefix)
    bounced_home = has_home_bounce(env, num_envs, state_prefix=state_prefix)

    pv_attr = f"_fairino{state_prefix}_prev_ball_vel_x"
    rh_attr = f"_fairino{state_prefix}_return_hit_state"
    cw_attr = f"_fairino{state_prefix}_return_contact_window"
    ic_attr = f"_fairino{state_prefix}_return_incoming_contact_seen"

    prev_ball_vel_x = getattr(env, pv_attr, None)
    return_hit_state = getattr(env, rh_attr, None)
    contact_window = getattr(env, cw_attr, None)
    incoming_contact_seen = getattr(env, ic_attr, None)

    if prev_ball_vel_x is None or prev_ball_vel_x.shape[0] != num_envs:
        prev_ball_vel_x = ball_vel_x.clone()
        return_hit_state = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        contact_window = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        incoming_contact_seen = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        setattr(env, pv_attr, prev_ball_vel_x)
        setattr(env, rh_attr, return_hit_state)
        setattr(env, cw_attr, contact_window)
        setattr(env, ic_attr, incoming_contact_seen)

    if return_hit_state is None or return_hit_state.shape[0] != num_envs:
        return_hit_state = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        setattr(env, rh_attr, return_hit_state)
    if contact_window is None or contact_window.shape[0] != num_envs:
        contact_window = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        setattr(env, cw_attr, contact_window)
    if incoming_contact_seen is None or incoming_contact_seen.shape[0] != num_envs:
        incoming_contact_seen = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        setattr(env, ic_attr, incoming_contact_seen)

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_ball_vel_x = getattr(env, pv_attr).clone()
        return_hit_state = getattr(env, rh_attr).clone()
        contact_window = getattr(env, cw_attr).clone()
        incoming_contact_seen = getattr(env, ic_attr).clone()
        prev_ball_vel_x[reset_mask] = ball_vel_x[reset_mask]
        return_hit_state[reset_mask] = False
        contact_window[reset_mask] = 0
        incoming_contact_seen[reset_mask] = False
        setattr(env, pv_attr, prev_ball_vel_x)
        setattr(env, rh_attr, return_hit_state)
        setattr(env, cw_attr, contact_window)
        setattr(env, ic_attr, incoming_contact_seen)

    contact_window = torch.clamp(getattr(env, cw_attr) - 1, min=0)
    incoming_contact_seen = getattr(env, ic_attr) & (contact_window > 0)

    # incoming direction depends on home_side
    if home_side == "left":
        incoming_now_or_prev = (ball_vel_x < -min_incoming_speed) | (getattr(env, pv_attr) < -min_incoming_speed)
    else:
        incoming_now_or_prev = (ball_vel_x > min_incoming_speed) | (getattr(env, pv_attr) > min_incoming_speed)

    useful_contact = contact_now & bounced_home & (~getattr(env, rh_attr)) & incoming_now_or_prev
    if torch.any(useful_contact):
        contact_window = contact_window.clone()
        incoming_contact_seen = incoming_contact_seen.clone()
        contact_window[useful_contact] = max(int(contact_window_steps), 1)
        incoming_contact_seen[useful_contact] = True

    # return direction depends on home_side
    if home_side == "left":
        return_condition = ball_vel_x > min_return_speed
    else:
        return_condition = ball_vel_x < -min_return_speed

    valid_hit = (
        (~getattr(env, rh_attr))
        & bounced_home
        & incoming_contact_seen
        & (contact_window > 0)
        & return_condition
    )
    if torch.any(valid_hit):
        return_hit_state = getattr(env, rh_attr).clone()
        return_hit_state[valid_hit] = True
        setattr(env, rh_attr, return_hit_state)
        contact_window = contact_window.clone()
        incoming_contact_seen = incoming_contact_seen.clone()
        contact_window[valid_hit] = 0
        incoming_contact_seen[valid_hit] = False
        # rally: track alternating hits + reset the other arm
        _increment_rally_count(env, num_envs, valid_hit, arm_id=1 if home_side == "left" else 2)
        if other_state_prefix:
            reset_arm_state(env, num_envs, other_state_prefix)

    setattr(env, cw_attr, contact_window)
    setattr(env, ic_attr, incoming_contact_seen)
    setattr(env, pv_attr, ball_vel_x.clone())
    return valid_hit


def has_return_hit(env, num_envs: int | None = None, state_prefix: str = "") -> torch.Tensor:
    """Return whether each environment has already produced a useful hit."""
    return_hit_state = getattr(env, f"_fairino{state_prefix}_return_hit_state", None)
    if return_hit_state is None:
        if num_envs is None:
            num_envs = env.num_envs
        return torch.zeros(num_envs, device=env.device, dtype=torch.bool)
    return return_hit_state


# ---------------------------------------------------------------------------
# rally exchange counter  (global — survives per-arm state reset)
# ---------------------------------------------------------------------------

def _init_rally_state(env, num_envs: int) -> None:
    """Initialise rally bookkeeping tensors if not already present."""
    for attr in ("_fairino_last_hitter", "_fairino_rally_exchange_count",
                 "_fairino_rally_exchange_prev"):
        val = getattr(env, attr, None)
        if val is None or val.shape[0] != num_envs:
            setattr(env, attr, torch.zeros(num_envs, device=env.device, dtype=torch.long))


def _increment_rally_count(env, num_envs: int, mask: torch.Tensor, arm_id: int) -> None:
    """Track alternating hits across both arms.

    *arm_id*: 1 = left,  2 = right.

    The rally exchange count only increments when **the other arm** was the
    last hitter, preventing a single arm from racking up exchanges on its own.
    """
    _init_rally_state(env, num_envs)

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        for attr in ("_fairino_last_hitter", "_fairino_rally_exchange_count",
                     "_fairino_rally_exchange_prev"):
            val = getattr(env, attr).clone()
            val[reset_mask] = 0
            setattr(env, attr, val)

    if not torch.any(mask):
        return

    last_hitter = getattr(env, "_fairino_last_hitter")
    exchange_count = getattr(env, "_fairino_rally_exchange_count")

    # rally exchange: other arm hit last (and not ourselves)
    other_hit_last = (last_hitter > 0) & (last_hitter != arm_id)
    rally_mask = mask & other_hit_last

    if torch.any(rally_mask):
        exchange_count = exchange_count.clone()
        exchange_count[rally_mask] += 1
        setattr(env, "_fairino_rally_exchange_count", exchange_count)

    # update last hitter for environments where this arm just hit
    last_hitter = last_hitter.clone()
    last_hitter[mask] = arm_id
    setattr(env, "_fairino_last_hitter", last_hitter)


def rally_exchange_event(env, num_envs: int) -> torch.Tensor:
    """Return True **only on the step** the rally exchange count increments.

    This is a rising-edge detector — each exchange fires exactly once.
    """
    _init_rally_state(env, num_envs)

    step_id = int(getattr(env, "common_step_counter", 0))
    cached_attr = "_fairino_rally_exchange_step_id"
    event_attr = "_fairino_rally_exchange_event"
    cached_step = getattr(env, cached_attr, None)
    cached_event = getattr(env, event_attr, None)
    if cached_step == step_id and cached_event is not None and cached_event.shape[0] == num_envs:
        return cached_event

    cur = getattr(env, "_fairino_rally_exchange_count")
    prev = getattr(env, "_fairino_rally_exchange_prev")
    event = cur > prev

    setattr(env, "_fairino_rally_exchange_prev", cur.clone())
    setattr(env, event_attr, event)
    setattr(env, cached_attr, step_id)
    return event


# ---------------------------------------------------------------------------
# rally support: reset one arm's episode state
# ---------------------------------------------------------------------------

_RALLY_RESET_ATTRS = [
    "_return_hit_state",
    "_home_bounce_prev",
    "_home_bounce_seen",
    "_illegal_second_seen",
    "_illegal_second_prev_contact",
    "_illegal_second_prev_hit",
    "_illegal_second_steps",
    "_illegal_second_prev_ball_vel",
    "_clean_over_net_seen",
    "_clean_over_net_prev_hit",
    "_clean_over_net_prev_ball_x",
    "_clean_over_net_steps",
    "_paddle_contact_prev",
    "_paddle_contact_seen",
    "_post_hit_clearance_prev_hit",
    "_post_hit_clearance_steps",
    "_post_hit_retreat_prev_hit",
    "_post_hit_retreat_steps",
    "_legal_sep_prev_hit",
    "_legal_sep_steps",
]


def reset_arm_state(env, num_envs: int, state_prefix: str) -> None:
    """Zero out all episode state for one arm so it can re-hit in a rally."""
    for suffix in _RALLY_RESET_ATTRS:
        attr = f"_fairino{state_prefix}{suffix}"
        val = getattr(env, attr, None)
        if val is not None and isinstance(val, torch.Tensor) and val.shape[0] == num_envs:
            val.zero_()


# ---------------------------------------------------------------------------
# illegal second hit event
# ---------------------------------------------------------------------------

def illegal_second_hit_event(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    contact_distance: float = 0.10,
    grace_steps: int = 4,
    window_steps: int = 60,
    min_return_speed: float = 0.10,
    min_delta_speed: float = 0.03,
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Detect an illegal second paddle hit after the first valid return.

    The event is stricter than a raw distance re-contact: it must happen after
    the useful return hit, outside the immediate follow-through grace period,
    while the ball is still moving in the return direction, and with a
    noticeable velocity change.  The result is cached per manager step so
    reward and termination terms can query it without double-updating the
    contact edge state.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = ball.data.root_pos_w.shape[0]
    step_id = int(getattr(env, "common_step_counter", 0))

    cached_attr = f"_fairino{state_prefix}_illegal_second_step_id"
    event_attr = f"_fairino{state_prefix}_illegal_second_event"
    cached_step = getattr(env, cached_attr, None)
    cached_event = getattr(env, event_attr, None)
    if cached_step == step_id and cached_event is not None and cached_event.shape[0] == num_envs:
        return cached_event

    paddle_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] - env.scene.env_origins
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w[:, :3]
    distance = torch.linalg.norm(paddle_pos - ball_pos, dim=1)
    contact_now = distance <= contact_distance

    pc_attr = f"_fairino{state_prefix}_illegal_second_prev_contact"
    ph_attr = f"_fairino{state_prefix}_illegal_second_prev_hit"
    ps_attr = f"_fairino{state_prefix}_illegal_second_steps"
    si_attr = f"_fairino{state_prefix}_illegal_second_seen"
    pv_attr = f"_fairino{state_prefix}_illegal_second_prev_ball_vel"

    prev_contact = getattr(env, pc_attr, None)
    prev_hit = getattr(env, ph_attr, None)
    post_hit_steps = getattr(env, ps_attr, None)
    seen_illegal = getattr(env, si_attr, None)
    prev_ball_vel = getattr(env, pv_attr, None)

    if prev_contact is None or prev_contact.shape[0] != num_envs:
        prev_contact = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        prev_hit = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        post_hit_steps = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        seen_illegal = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        prev_ball_vel = ball_vel.clone()
        setattr(env, pc_attr, prev_contact)
        setattr(env, ph_attr, prev_hit)
        setattr(env, ps_attr, post_hit_steps)
        setattr(env, si_attr, seen_illegal)
        setattr(env, pv_attr, prev_ball_vel)

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_contact = getattr(env, pc_attr).clone()
        prev_hit = getattr(env, ph_attr).clone()
        post_hit_steps = getattr(env, ps_attr).clone()
        seen_illegal = getattr(env, si_attr).clone()
        prev_ball_vel = getattr(env, pv_attr).clone()
        prev_contact[reset_mask] = False
        prev_hit[reset_mask] = False
        post_hit_steps[reset_mask] = 0
        seen_illegal[reset_mask] = False
        prev_ball_vel[reset_mask] = ball_vel[reset_mask]
        setattr(env, pc_attr, prev_contact)
        setattr(env, ph_attr, prev_hit)
        setattr(env, ps_attr, post_hit_steps)
        setattr(env, si_attr, seen_illegal)
        setattr(env, pv_attr, prev_ball_vel)

    hit_done = has_return_hit(env, num_envs, state_prefix=state_prefix)
    new_hit = hit_done & (~getattr(env, ph_attr))
    post_hit_steps = torch.where(
        new_hit,
        torch.zeros_like(getattr(env, ps_attr)),
        torch.where(hit_done, getattr(env, ps_attr) + 1, torch.zeros_like(getattr(env, ps_attr))),
    )
    contact_event = contact_now & (~getattr(env, pc_attr))
    delta_speed = torch.linalg.norm(ball_vel - getattr(env, pv_attr), dim=1)

    # return-direction check depends on home_side
    if home_side == "left":
        return_direction_ok = ball_vel[:, 0] > min_return_speed
    else:
        return_direction_ok = ball_vel[:, 0] < -min_return_speed

    illegal_event = (
        contact_event
        & hit_done
        & (~new_hit)
        & (~getattr(env, si_attr))
        & (post_hit_steps >= int(grace_steps))
        & (post_hit_steps <= int(window_steps))
        & return_direction_ok
        & (delta_speed > min_delta_speed)
    )

    if torch.any(illegal_event):
        seen_illegal = getattr(env, si_attr).clone()
        seen_illegal[illegal_event] = True
        setattr(env, si_attr, seen_illegal)

    setattr(env, pc_attr, contact_now)
    setattr(env, ph_attr, hit_done.clone())
    setattr(env, ps_attr, post_hit_steps)
    setattr(env, pv_attr, ball_vel.clone())
    setattr(env, event_attr, illegal_event)
    setattr(env, cached_attr, step_id)
    return illegal_event


def has_illegal_second_hit(env, num_envs: int | None = None, state_prefix: str = "") -> torch.Tensor:
    """Return whether each environment has had an illegal second hit this episode."""
    seen = getattr(env, f"_fairino{state_prefix}_illegal_second_seen", None)
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
    state_prefix: str = "",
    home_side: str = "left",
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
        state_prefix=state_prefix,
        home_side=home_side,
    )
    return has_return_hit(env, num_envs, state_prefix=state_prefix) & (
        ~has_illegal_second_hit(env, num_envs, state_prefix=state_prefix)
    )


# ---------------------------------------------------------------------------
# clean over-net event
# ---------------------------------------------------------------------------

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
    state_prefix: str = "",
    home_side: str = "left",
) -> torch.Tensor:
    """Detect a clean one-hit return crossing the net after the first valid hit."""
    ball: RigidObject = env.scene[ball_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = ball.data.root_lin_vel_w[:, :3]
    num_envs = pos.shape[0]
    device = env.device
    step_id = int(getattr(env, "common_step_counter", 0))

    cached_attr = f"_fairino{state_prefix}_clean_over_net_step_id"
    event_attr = f"_fairino{state_prefix}_clean_over_net_event"
    cached_step = getattr(env, cached_attr, None)
    cached_event = getattr(env, event_attr, None)
    if cached_step == step_id and cached_event is not None and cached_event.shape[0] == num_envs:
        return cached_event

    illegal_second_hit_event(
        env,
        robot_cfg=robot_cfg,
        ball_cfg=ball_cfg,
        contact_distance=illegal_contact_distance,
        state_prefix=state_prefix,
        home_side=home_side,
    )
    hit_done = has_return_hit(env, num_envs, state_prefix=state_prefix)
    illegal_done = has_illegal_second_hit(env, num_envs, state_prefix=state_prefix)

    ph_attr = f"_fairino{state_prefix}_clean_over_net_prev_hit"
    px_attr = f"_fairino{state_prefix}_clean_over_net_prev_ball_x"
    ss_attr = f"_fairino{state_prefix}_clean_over_net_steps"
    so_attr = f"_fairino{state_prefix}_clean_over_net_seen"

    prev_hit = getattr(env, ph_attr, None)
    prev_ball_x = getattr(env, px_attr, None)
    steps_since_hit = getattr(env, ss_attr, None)
    seen_over_net = getattr(env, so_attr, None)

    if prev_hit is None or prev_hit.shape[0] != num_envs:
        prev_hit = torch.zeros(num_envs, device=device, dtype=torch.bool)
        prev_ball_x = pos[:, 0].clone()
        steps_since_hit = torch.zeros(num_envs, device=device, dtype=torch.long)
        seen_over_net = torch.zeros(num_envs, device=device, dtype=torch.bool)
        setattr(env, ph_attr, prev_hit)
        setattr(env, px_attr, prev_ball_x)
        setattr(env, ss_attr, steps_since_hit)
        setattr(env, so_attr, seen_over_net)

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        prev_hit = getattr(env, ph_attr).clone()
        prev_ball_x = getattr(env, px_attr).clone()
        steps_since_hit = getattr(env, ss_attr).clone()
        seen_over_net = getattr(env, so_attr).clone()
        prev_hit[reset_mask] = False
        prev_ball_x[reset_mask] = pos[reset_mask, 0]
        steps_since_hit[reset_mask] = 0
        seen_over_net[reset_mask] = False
        setattr(env, ph_attr, prev_hit)
        setattr(env, px_attr, prev_ball_x)
        setattr(env, ss_attr, steps_since_hit)
        setattr(env, so_attr, seen_over_net)

    new_hit = hit_done & (~getattr(env, ph_attr))
    steps_since_hit = torch.where(
        new_hit,
        torch.zeros_like(getattr(env, ss_attr)),
        torch.where(hit_done, getattr(env, ss_attr) + 1, torch.zeros_like(getattr(env, ss_attr))),
    )

    # net-crossing and return-direction logic depends on home_side
    if home_side == "left":
        crossed_net = (getattr(env, px_attr) < net_x) & (pos[:, 0] >= net_x)
        on_or_past_net = pos[:, 0] >= net_x
        return_direction_ok = vel[:, 0] > min_return_speed
    else:
        crossed_net = (getattr(env, px_attr) > net_x) & (pos[:, 0] <= net_x)
        on_or_past_net = pos[:, 0] <= net_x
        return_direction_ok = vel[:, 0] < -min_return_speed

    above_net = pos[:, 2] > (net_height + ball_radius)
    in_window = (steps_since_hit > 0) & (steps_since_hit <= int(max_steps_after_hit))
    event = (
        (crossed_net | on_or_past_net)
        & above_net
        & in_window
        & return_direction_ok
        & hit_done
        & (~illegal_done)
        & (~getattr(env, so_attr))
    )
    if torch.any(event):
        seen_over_net = getattr(env, so_attr).clone()
        seen_over_net[event] = True
        setattr(env, so_attr, seen_over_net)

    setattr(env, ph_attr, hit_done.clone())
    setattr(env, px_attr, pos[:, 0].clone())
    setattr(env, ss_attr, steps_since_hit)
    setattr(env, event_attr, event)
    setattr(env, cached_attr, step_id)
    return event


def has_clean_over_net(env, num_envs: int | None = None, state_prefix: str = "") -> torch.Tensor:
    """Return whether each environment has produced a clean over-net return."""
    seen = getattr(env, f"_fairino{state_prefix}_clean_over_net_seen", None)
    if seen is None:
        if num_envs is None:
            num_envs = env.num_envs
        return torch.zeros(num_envs, device=env.device, dtype=torch.bool)
    return seen


# ---------------------------------------------------------------------------
# peak post-hit ball z
# ---------------------------------------------------------------------------

def peak_post_hit_ball_z(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    state_prefix: str = "",
) -> torch.Tensor:
    """Track the maximum ball Z observed after a return hit, per environment.

    Resets on episode boundaries. Used to scale success rewards by trajectory height.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    num_envs = pos.shape[0]

    peak_attr = f"_fairino{state_prefix}_peak_ball_z"
    peak = getattr(env, peak_attr, None)
    if peak is None or peak.shape[0] != num_envs:
        peak = torch.zeros(num_envs, device=env.device)
        setattr(env, peak_attr, peak)

    reset_mask = episode_reset_mask(env, num_envs)
    if torch.any(reset_mask):
        peak = getattr(env, peak_attr).clone()
        peak[reset_mask] = 0.0
        setattr(env, peak_attr, peak)

    hit_done = has_return_hit(env, num_envs, state_prefix=state_prefix)
    setattr(
        env,
        peak_attr,
        torch.where(
            hit_done,
            torch.maximum(getattr(env, peak_attr), pos[:, 2]),
            getattr(env, peak_attr),
        ),
    )

    return getattr(env, peak_attr)
