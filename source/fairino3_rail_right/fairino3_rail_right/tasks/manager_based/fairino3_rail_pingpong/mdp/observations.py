"""Custom observations for the Fairino3 ping-pong task."""

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


def _resolve_body_id(asset: Articulation, cfg: SceneEntityCfg, index: int = 0):
    """Resolve a single body index from a SceneEntityCfg, handling slice defaults."""
    idx = cfg.body_ids
    if isinstance(idx, slice):
        idx, _ = asset.find_bodies(cfg.body_names, preserve_order=cfg.preserve_order)
    return idx[index]


def body_pos_w(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return selected articulation body positions in the environment frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    if pos.ndim == 3:
        pos = pos.reshape(pos.shape[0], -1)
    return pos - env.scene.env_origins


def ball_to_paddle_pos(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Return ball position relative to the paddle in the environment frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    paddle_pos = robot.data.body_pos_w[:, _resolve_body_id(robot, robot_cfg), :]
    ball_pos = ball.data.root_pos_w[:, :3]
    return ball_pos - paddle_pos


def ball_to_paddle_vel(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Return ball linear velocity relative to the paddle."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    paddle_vel = robot.data.body_lin_vel_w[:, _resolve_body_id(robot, robot_cfg), :]
    ball_vel = ball.data.root_lin_vel_w[:, :3]
    return ball_vel - paddle_vel


def target_position(env, target: tuple[float, float, float] = (1.235, 0.0, 0.76)) -> torch.Tensor:
    """Return the fixed target position in each environment frame."""
    return torch.tensor(target, device=env.device).repeat(env.num_envs, 1)


def body_pos_robot_frame(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    root_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Return selected body positions in the robot base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_asset: Articulation = env.scene[root_cfg.name] if root_cfg is not None else asset
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos = root_asset.data.root_pos_w[:, :3].unsqueeze(1)
    root_quat = root_asset.data.root_quat_w[:, :4].unsqueeze(1).expand(-1, body_pos.shape[1], -1)
    local_pos = quat_apply_inverse(root_quat.reshape(-1, 4), (body_pos - root_pos).reshape(-1, 3))
    return local_pos.reshape(body_pos.shape[0], -1)


def root_pos_robot_frame(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return a root position in the robot base frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    rel_pos = asset.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    return quat_apply_inverse(robot.data.root_quat_w[:, :4], rel_pos)


def root_lin_vel_robot_frame(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return root linear velocity in the robot base frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w[:, :4], asset.data.root_lin_vel_w[:, :3])


def ball_to_paddle_pos_robot_frame(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Return ball position relative to the paddle in the robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    paddle_pos = robot.data.body_pos_w[:, _resolve_body_id(robot, robot_cfg), :]
    rel_pos = ball.data.root_pos_w[:, :3] - paddle_pos
    return quat_apply_inverse(robot.data.root_quat_w[:, :4], rel_pos)


def ball_to_paddle_vel_robot_frame(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Return ball velocity relative to the paddle in the robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    paddle_vel = robot.data.body_lin_vel_w[:, _resolve_body_id(robot, robot_cfg), :]
    rel_vel = ball.data.root_lin_vel_w[:, :3] - paddle_vel
    return quat_apply_inverse(robot.data.root_quat_w[:, :4], rel_vel)


def target_position_robot_frame(
    env,
    target: tuple[float, float, float] = (1.235, 0.0, 0.76),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the fixed target position in the robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    target_pos = torch.tensor(target, device=env.device).repeat(env.num_envs, 1)
    root_pos_env = robot.data.root_pos_w[:, :3] - env.scene.env_origins
    return quat_apply_inverse(robot.data.root_quat_w[:, :4], target_pos - root_pos_env)

def paddle_lin_vel_w(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["paddle_link"])
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    paddle_vel = robot.data.body_lin_vel_w[:, _resolve_body_id(robot, robot_cfg), :]
    return paddle_vel