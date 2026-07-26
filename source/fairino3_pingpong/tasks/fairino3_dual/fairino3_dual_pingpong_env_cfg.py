"""Manager-based dual Fairino3 ping-pong environment with Y-axis rails.

Symmetric setup: left and right robots each get a 32-dim robot-frame observation.
A shared-weight MLP (SharedActorMLPModel) processes both halves identically.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from test_isaac_dual.assets.fairino3_v6_rail import FAIRINO3_V6_RAIL_PADDLE_CFG

from . import mdp

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

TABLE_CENTER = (0.55, 0.0, 0.74)
TABLE_SIZE = (2.74, 1.525, 0.04)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] * 0.5
TABLE_HALF_LENGTH = TABLE_SIZE[0] * 0.5
TABLE_HALF_WIDTH = TABLE_SIZE[1] * 0.5
NET_X = TABLE_CENTER[0]
NET_HEIGHT = TABLE_TOP_Z + 0.1525
RIGHT_MID_TARGET_X = NET_X + TABLE_HALF_LENGTH * 0.5
LEFT_MID_TARGET_X = NET_X - TABLE_HALF_LENGTH * 0.5
RIGHT_NEAR_NET_TARGET_X = NET_X + TABLE_HALF_LENGTH * 0.22
LEFT_NEAR_NET_TARGET_X = NET_X - TABLE_HALF_LENGTH * 0.22
LEFT_INTERCEPT_X = NET_X - TABLE_HALF_LENGTH * 0.5
RIGHT_INTERCEPT_X = NET_X + TABLE_HALF_LENGTH * 0.5
RAIL_ROBOT_BASE_LIFT = 0.14
BALL_INIT_POS = (NET_X + 0.40, 0.0, TABLE_TOP_Z + 0.26)
TARGET_MARKER_Z = TABLE_TOP_Z + 0.005
X_BOUNDS = (NET_X - TABLE_HALF_LENGTH - 0.70, NET_X + TABLE_HALF_LENGTH + 0.70)
Y_BOUNDS = (-TABLE_HALF_WIDTH - 0.45, TABLE_HALF_WIDTH + 0.45)

_LEFT_ROBOT_X = -0.97
_RIGHT_ROBOT_X = NET_X + (NET_X - _LEFT_ROBOT_X)  # 2.07
_ALL_JOINTS = ["rail_y", "j1", "j2", "j3", "j4", "j5", "j6"]
_ARM_JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]

_LEFT_ROBOT_INIT = ArticulationCfg.InitialStateCfg(
    pos=(_LEFT_ROBOT_X, 0.0, TABLE_TOP_Z),
    rot=(0.7071068, 0.0, 0.0, 0.7071068),
    joint_pos={
        "rail_y": 0.0, "j1": -1.743, "j2": -2.94,
        "j3": 0.0, "j4": -1.847, "j5": 2.7, "j6": 1.054,
    },
)
_RIGHT_ROBOT_INIT = ArticulationCfg.InitialStateCfg(
    pos=(_RIGHT_ROBOT_X, 0.0, TABLE_TOP_Z),
    rot=(0.7071068, 0.0, 0.0, -0.7071068),  # 比左臂多绕 z 转 180°，互相对打
    joint_pos={
        "rail_y": 0.0, "j1": -1.743, "j2": -2.94,
        "j3": 0.0, "j4": -1.847, "j5": 2.7, "j6": 1.054,
    },
)

# ---------------------------------------------------------------------------
# shorthands
# ---------------------------------------------------------------------------

_L_ROBOT = SceneEntityCfg("robot")
_R_ROBOT = SceneEntityCfg("right_robot")
_L_PADDLE = SceneEntityCfg("robot", body_names=["paddle_link"])
_R_PADDLE = SceneEntityCfg("right_robot", body_names=["paddle_link"])
_L_TARGET = (LEFT_MID_TARGET_X, 0.0, TABLE_TOP_Z)
_R_TARGET = (RIGHT_MID_TARGET_X, 0.0, TABLE_TOP_Z)
_BALL = SceneEntityCfg("ball")

# body link names for clearance penalty
_BODY_LINKS = ["forearm_link", "wrist1_link", "wrist2_link", "wrist3_link", "tool_clamp_link"]


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------

@configclass
class Fairino3DualPingPongSceneCfg(InteractiveSceneCfg):
    """Scene with two rail-mounted Fairino3 arms, ball, table, and net."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(16.0, 12.0)),
    )

    robot: ArticulationCfg = FAIRINO3_V6_RAIL_PADDLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/LeftRobot",
        init_state=_LEFT_ROBOT_INIT,
    )

    right_robot: ArticulationCfg = FAIRINO3_V6_RAIL_PADDLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RightRobot",
        init_state=_RIGHT_ROBOT_INIT,
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=20.0,
                max_depenetration_velocity=10.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                angular_damping=0.05,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.4, restitution=0.88,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.05)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BALL_INIT_POS, lin_vel=(-1.25, 0.0, 0.12),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.65, dynamic_friction=0.55, restitution=0.85,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.35, 0.22)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_CENTER),
    )

    net = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Net",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, TABLE_SIZE[1] + 0.02, 0.1525),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(NET_X, 0.0, TABLE_TOP_Z + 0.1525 * 0.5)),
    )

    right_target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RightTarget",
        spawn=sim_utils.CuboidCfg(
            size=(0.18, 0.18, 0.005),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.45, 1.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(RIGHT_MID_TARGET_X, 0.0, TARGET_MARKER_Z)),
    )

    left_target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LeftTarget",
        spawn=sim_utils.CuboidCfg(
            size=(0.18, 0.18, 0.005),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.1)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(LEFT_MID_TARGET_X, 0.0, TARGET_MARKER_Z)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=800.0),
    )


# ---------------------------------------------------------------------------
# actions (unchanged: 14-DOF)
# ---------------------------------------------------------------------------

@configclass
class ActionsCfg:
    """14-DOF action: left(rail_y + j1-j6) + right(rail_y + j1-j6)."""

    left_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=_ARM_JOINTS,
        scale=0.28, use_default_offset=True, preserve_order=True,
    )
    left_rail_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["rail_y"],
        scale=0.04, use_default_offset=True,
    )
    right_arm_action = mdp.JointPositionActionCfg(
        asset_name="right_robot", joint_names=_ARM_JOINTS,
        scale=0.28, use_default_offset=True, preserve_order=True,
    )
    right_rail_action = mdp.JointPositionActionCfg(
        asset_name="right_robot", joint_names=["rail_y"],
        scale=0.04, use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# observations: 64-dim symmetric (32 left + 32 right, all robot-frame)
# ---------------------------------------------------------------------------

@configclass
class SymmetricObservationsCfg:
    """64-dim: 32 per arm, all in robot base frame."""

    @configclass
    class PolicyCfg(ObsGroup):
        # ---- left arm (32 dims, target = right table) ----
        left_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _L_ROBOT})
        left_joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2, params={"asset_cfg": _L_ROBOT})
        left_paddle_pos = ObsTerm(func=mdp.body_pos_robot_frame, params={"asset_cfg": _L_PADDLE})
        left_ball_pos = ObsTerm(func=mdp.root_pos_robot_frame, params={"asset_cfg": _BALL, "robot_cfg": _L_ROBOT})
        left_ball_vel = ObsTerm(func=mdp.root_lin_vel_robot_frame, scale=0.2, params={"asset_cfg": _BALL, "robot_cfg": _L_ROBOT})
        left_ball_to_paddle = ObsTerm(func=mdp.ball_to_paddle_pos_robot_frame, params={"robot_cfg": _L_PADDLE, "ball_cfg": _BALL})
        left_ball_to_paddle_vel = ObsTerm(func=mdp.ball_to_paddle_vel_robot_frame, scale=0.2, params={"robot_cfg": _L_PADDLE, "ball_cfg": _BALL})
        left_target = ObsTerm(func=mdp.target_position_robot_frame, params={"target": _R_TARGET, "robot_cfg": _L_ROBOT})

        # ---- right arm (32 dims, target = left table) ----
        right_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _R_ROBOT})
        right_joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2, params={"asset_cfg": _R_ROBOT})
        right_paddle_pos = ObsTerm(func=mdp.body_pos_robot_frame, params={"asset_cfg": _R_PADDLE})
        right_ball_pos = ObsTerm(func=mdp.root_pos_robot_frame, params={"asset_cfg": _BALL, "robot_cfg": _R_ROBOT})
        right_ball_vel = ObsTerm(func=mdp.root_lin_vel_robot_frame, scale=0.2, params={"asset_cfg": _BALL, "robot_cfg": _R_ROBOT})
        right_ball_to_paddle = ObsTerm(func=mdp.ball_to_paddle_pos_robot_frame, params={"robot_cfg": _R_PADDLE, "ball_cfg": _BALL})
        right_ball_to_paddle_vel = ObsTerm(func=mdp.ball_to_paddle_vel_robot_frame, scale=0.2, params={"robot_cfg": _R_PADDLE, "ball_cfg": _BALL})
        right_target = ObsTerm(func=mdp.target_position_robot_frame, params={"target": _L_TARGET, "robot_cfg": _R_ROBOT})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

@configclass
class EventCfg:
    """Reset randomization for both robots + random-direction serve."""

    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_ALL_JOINTS),
            "position_range": (-0.08, 0.08),
            "velocity_range": (-0.05, 0.05),
        },
    )
    reset_right_robot = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_robot", joint_names=_ALL_JOINTS),
            "position_range": (-0.08, 0.08),
            "velocity_range": (-0.05, 0.05),
        },
    )
    reset_ball = EventTerm(
        func=mdp.reset_ball_valid_serve,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {
                "x": (-0.20, 0.20), "y": (-0.35, 0.35), "z": (-0.08, 0.08),
            },
            "velocity_range": {
                "x": (-2.2, -0.7),       # toward left arm
                "y": (-0.45, 0.45), "z": (0.05, 0.55),
                "roll": (-1.2, 1.2), "pitch": (-1.2, 1.2), "yaw": (-1.2, 1.2),
            },
        },
    )


# ---------------------------------------------------------------------------
# reward factory — avoids duplicating the same 21-term block for each arm
# ---------------------------------------------------------------------------

def _make_arm_rewards(
    robot_name: str,
    paddle_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    target_xy: tuple[float, float],
    intercept_x: float,
    near_net_xy: tuple[float, float],
    home_side: str,
    state_prefix: str,
) -> dict[str, RewTerm]:
    """Return a dict of RewardTermCfg for one arm.

    ``home_side`` and ``state_prefix`` are the only parameters that differ
    between the left arm (``"left"`` / ``""``) and the right arm
    (``"right"`` / ``"_right"``).  Everything else is symmetric.
    """
    arm_joints = SceneEntityCfg(robot_name, joint_names=_ARM_JOINTS)
    all_joints = SceneEntityCfg(robot_name, joint_names=_ALL_JOINTS)
    body_cfg = SceneEntityCfg(robot_name, body_names=_BODY_LINKS)

    def _rw(func, weight, **kw):
        return RewTerm(func=func, weight=weight, params=dict(kw))

    hs = home_side
    sp = state_prefix

    return {
        # pre-hit
        "table_bounce":      _rw(mdp.left_table_bounce, 6.0, ball_cfg=_BALL, home_side=hs, state_prefix=sp),
        "paddle_to_ball":    _rw(mdp.paddle_to_ball_distance, 1.0, robot_cfg=paddle_cfg, ball_cfg=_BALL, std=0.45, state_prefix=sp),
        "paddle_to_intercept": _rw(mdp.paddle_to_intercept, 1.2, robot_cfg=paddle_cfg, ball_cfg=_BALL, intercept_x=intercept_x, std=0.30, home_side=hs),
        # hit detection
        "first_return_hit":  _rw(mdp.first_return_hit, 20.0, robot_cfg=paddle_cfg, ball_cfg=_BALL, contact_distance=0.10, min_incoming_speed=0.03, min_return_speed=0.03, contact_window_steps=6, home_side=hs, state_prefix=sp, other_state_prefix=("_right" if hs == "left" else "")),
        "return_ball":       _rw(mdp.return_ball_velocity, 12.0, robot_cfg=paddle_cfg, ball_cfg=_BALL, hit_distance=0.18, home_side=hs, state_prefix=sp),
        # penalties
        "second_paddle_contact": _rw(mdp.second_paddle_contact_penalty, -16.0, robot_cfg=paddle_cfg, ball_cfg=_BALL, contact_distance=0.10, grace_steps=4, window_steps=60, min_return_speed=0.10, min_delta_speed=0.03, home_side=hs, state_prefix=sp),
        # early_hit_penalty REMOVED — caused reward hacking (arm dodges ball)
        "post_hit_paddle_clearance": _rw(mdp.post_hit_paddle_ball_clearance_penalty, -8.0, robot_cfg=paddle_cfg, ball_cfg=_BALL, min_distance=0.22, grace_steps=4, window_steps=18, min_return_speed=0.10, home_side=hs, state_prefix=sp),
        "post_hit_paddle_retreat": _rw(mdp.post_hit_paddle_retreat_penalty, -4.0, robot_cfg=paddle_cfg, ball_cfg=_BALL, min_behind_x=0.12, near_distance=0.34, grace_steps=4, window_steps=22, min_return_speed=0.10, chase_speed=0.05, home_side=hs, state_prefix=sp),
        # post-hit quality (gated by legal_post_hit_mask)
        "net_direction":     _rw(mdp.net_direction, 10.0, asset_cfg=_BALL, net_x=NET_X, net_window=0.18, home_side=hs, state_prefix=sp),
        "net_height":        _rw(mdp.net_height, 3.0, asset_cfg=_BALL, net_x=NET_X, net_window=0.18, ideal_z=NET_HEIGHT + 0.04, std=0.10, home_side=hs, state_prefix=sp),
        "net_speed":         _rw(mdp.net_speed, 6.0, asset_cfg=_BALL, net_x=NET_X, net_window=0.18, ideal_speed=1.9, std=0.45, home_side=hs, state_prefix=sp),
        "high_return_height": _rw(mdp.high_return_height_penalty, -18.0, asset_cfg=_BALL, net_x=NET_X, net_window=0.18, net_height=NET_HEIGHT, max_height=TABLE_TOP_Z + 0.1525 * 3, home_side=hs, state_prefix=sp),
        "predicted_landing": _rw(mdp.predicted_right_table_landing, 30.0, asset_cfg=_BALL, target_xy=target_xy, table_center=TABLE_CENTER, table_size=TABLE_SIZE, ball_radius=0.02, net_x=NET_X, std=0.44, home_side=hs, state_prefix=sp),
        "near_net_landing_speed": _rw(mdp.predicted_right_near_net_landing_speed, 12.0, asset_cfg=_BALL, target_xy=near_net_xy, table_center=TABLE_CENTER, table_size=TABLE_SIZE, ball_radius=0.02, net_x=NET_X, std_x=0.22, std_y=0.24, min_right_speed=0.75, target_right_speed=1.8, home_side=hs, state_prefix=sp),
        "legal_return_separation": _rw(mdp.legal_return_separation_reward, 12.0, robot_cfg=paddle_cfg, ball_cfg=_BALL, table_center=TABLE_CENTER, table_size=TABLE_SIZE, ball_radius=0.02, net_x=NET_X, target_xy=target_xy, min_behind_x=0.14, behind_std=0.18, min_return_speed=0.25, target_return_speed=1.8, grace_steps=3, window_steps=34, landing_std=0.42, home_side=hs, state_prefix=sp),
        # sparse terminal success (rally mode: reduced from 250 → 40)
        "right_table_bounce": _rw(mdp.right_table_bounce_reward, 40.0, asset_cfg=_BALL, robot_cfg=paddle_cfg, table_center=TABLE_CENTER, table_size=TABLE_SIZE, ball_radius=0.02, net_x=NET_X, illegal_contact_distance=0.10, illegal_reward_scale=0.0, require_clean_over_net=True, home_side=hs, state_prefix=sp),
        # idle / safety
        "post_return_idle":  _rw(mdp.post_return_idle_penalty, -0.02, asset_cfg=arm_joints, ball_cfg=_BALL, state_prefix=sp),
        "joint_limit":       _rw(mdp.joint_limit_margin_penalty, -2.0, asset_cfg=all_joints, margin=0.30),
        "table_collision":   _rw(mdp.robot_table_penalty, -20.0, asset_cfg=robot_cfg, table_center=TABLE_CENTER, table_size=TABLE_SIZE, clearance=0.05),
        "paddle_body_clearance": _rw(mdp.paddle_body_clearance_penalty, -6.0, paddle_cfg=paddle_cfg, body_cfg=body_cfg, min_distance=0.15),
    }


def _prefix_keys(d: dict[str, RewTerm], prefix: str) -> dict[str, RewTerm]:
    """Prepend *prefix* to every key in *d*."""
    return {f"{prefix}{k}": v for k, v in d.items()}


_LEFT_REWARDS = _make_arm_rewards(
    robot_name="robot", paddle_cfg=_L_PADDLE, robot_cfg=_L_ROBOT,
    target_xy=(RIGHT_MID_TARGET_X, 0.0), intercept_x=LEFT_INTERCEPT_X,
    near_net_xy=(RIGHT_NEAR_NET_TARGET_X, 0.0),
    home_side="left", state_prefix="",
)

_RIGHT_REWARDS = _make_arm_rewards(
    robot_name="right_robot", paddle_cfg=_R_PADDLE, robot_cfg=_R_ROBOT,
    target_xy=(LEFT_MID_TARGET_X, 0.0), intercept_x=RIGHT_INTERCEPT_X,
    near_net_xy=(LEFT_NEAR_NET_TARGET_X, 0.0),
    home_side="right", state_prefix="_right",
)

# ---- Right arm reward override: dense pre-hit, remove quality penalties ----
_RIGHT_REWARDS["paddle_to_ball"] = RewTerm(
    func=mdp.paddle_to_ball_distance, weight=1.0,  # was 20.0 — aligned with left
    params={"robot_cfg": _R_PADDLE, "ball_cfg": _BALL, "std": 0.45, "state_prefix": "_right"},
)
_RIGHT_REWARDS["paddle_to_intercept"] = RewTerm(
    func=mdp.paddle_to_intercept, weight=1.2,  # was 15.0 — aligned with left
    params={"robot_cfg": _R_PADDLE, "ball_cfg": _BALL, "intercept_x": RIGHT_INTERCEPT_X, "std": 0.30, "home_side": "right"},
)
# Curriculum: right arm contact_distance 0.10 → 0.30 (easier to trigger first hit)
_RIGHT_REWARDS["first_return_hit"] = RewTerm(
    func=mdp.first_return_hit, weight=20.0,
    params={"robot_cfg": _R_PADDLE, "ball_cfg": _BALL, "contact_distance": 0.30,
            "min_incoming_speed": 0.03, "min_return_speed": 0.03,
            "contact_window_steps": 6, "home_side": "right",
            "state_prefix": "_right", "other_state_prefix": ""},
)
# Remove right arm quality penalties — let it learn to hit first
_RIGHT_REWARDS["high_return_height"] = None
_RIGHT_REWARDS["table_collision"] = None
_RIGHT_REWARDS["joint_limit"] = None
_RIGHT_REWARDS["post_return_idle"] = None
_RIGHT_REWARDS["post_hit_paddle_clearance"] = None
_RIGHT_REWARDS["post_hit_paddle_retreat"] = None


# ---------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    """Reward terms for both arms, built from a single symmetric factory."""

    # ---- left arm ----
    left_table_bounce = _LEFT_REWARDS["table_bounce"]
    left_paddle_to_ball = _LEFT_REWARDS["paddle_to_ball"]
    left_paddle_to_intercept = _LEFT_REWARDS["paddle_to_intercept"]
    left_first_return_hit = _LEFT_REWARDS["first_return_hit"]
    left_return_ball = _LEFT_REWARDS["return_ball"]
    left_second_paddle_contact = _LEFT_REWARDS["second_paddle_contact"]

    left_post_hit_paddle_clearance = _LEFT_REWARDS["post_hit_paddle_clearance"]
    left_post_hit_paddle_retreat = _LEFT_REWARDS["post_hit_paddle_retreat"]
    left_net_direction = _LEFT_REWARDS["net_direction"]
    left_net_height = _LEFT_REWARDS["net_height"]
    left_net_speed = _LEFT_REWARDS["net_speed"]
    left_high_return_height = _LEFT_REWARDS["high_return_height"]
    left_predicted_landing = _LEFT_REWARDS["predicted_landing"]
    left_near_net_landing_speed = _LEFT_REWARDS["near_net_landing_speed"]
    left_legal_return_separation = _LEFT_REWARDS["legal_return_separation"]
    left_right_table_bounce = _LEFT_REWARDS["right_table_bounce"]
    left_post_return_idle = _LEFT_REWARDS["post_return_idle"]
    left_joint_limit = _LEFT_REWARDS["joint_limit"]
    left_table_collision = _LEFT_REWARDS["table_collision"]
    left_paddle_body_clearance = _LEFT_REWARDS["paddle_body_clearance"]

    # ---- right arm (mirrored targets, home_side="right") ----
    right_left_table_bounce = _RIGHT_REWARDS["table_bounce"]
    right_paddle_to_ball = _RIGHT_REWARDS["paddle_to_ball"]
    right_paddle_to_intercept = _RIGHT_REWARDS["paddle_to_intercept"]
    right_first_return_hit = _RIGHT_REWARDS["first_return_hit"]
    right_return_ball = _RIGHT_REWARDS["return_ball"]
    right_second_paddle_contact = _RIGHT_REWARDS["second_paddle_contact"]

    right_post_hit_paddle_clearance = _RIGHT_REWARDS["post_hit_paddle_clearance"]
    right_post_hit_paddle_retreat = _RIGHT_REWARDS["post_hit_paddle_retreat"]
    right_net_direction = _RIGHT_REWARDS["net_direction"]
    right_net_height = _RIGHT_REWARDS["net_height"]
    right_net_speed = _RIGHT_REWARDS["net_speed"]
    right_high_return_height = _RIGHT_REWARDS["high_return_height"]
    right_predicted_landing = _RIGHT_REWARDS["predicted_landing"]
    right_near_net_landing_speed = _RIGHT_REWARDS["near_net_landing_speed"]
    right_legal_return_separation = _RIGHT_REWARDS["legal_return_separation"]
    right_left_table_bounce_reward = _RIGHT_REWARDS["right_table_bounce"]
    right_post_return_idle = _RIGHT_REWARDS["post_return_idle"]
    right_joint_limit = _RIGHT_REWARDS["joint_limit"]
    right_table_collision = _RIGHT_REWARDS["table_collision"]
    right_paddle_body_clearance = _RIGHT_REWARDS["paddle_body_clearance"]

    # ---- shared ----
    rally_exchange = RewTerm(func=mdp.rally_exchange_reward, weight=8.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ALL_JOINTS)},
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("right_robot", joint_names=_ALL_JOINTS)},
    )
    ball_out_penalty = RewTerm(
        func=mdp.ball_out_of_bounds_penalty, weight=-8.0,
        params={"asset_cfg": _BALL, "x_bounds": X_BOUNDS, "y_bounds": Y_BOUNDS, "z_min": 0.03},
    )
    # terminating reward REMOVED — rally mode: time_out is expected, not penalised
    # terminating = RewTerm(func=mdp.is_terminated, weight=-4.0)


# ---------------------------------------------------------------------------
# terminations
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    """Termination terms — symmetric for both arms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # ball_out_of_bounds DISABLED — allow long rally observation
    # ball_out_of_bounds = DoneTerm(
    #     func=mdp.ball_out_of_bounds,
    #     params={"asset_cfg": _BALL, "x_bounds": X_BOUNDS, "y_bounds": Y_BOUNDS, "z_min": 0.03},
    # )
    # Rally mode: table-bounce terminations DISABLED so the ball can
    # bounce and the rally can continue. Only time_out + ball_fall + collision
    # end the episode.
    ball_fall_off_table = DoneTerm(
        func=mdp.ball_fall_off_table,
        params={"asset_cfg": _BALL, "min_z": TABLE_TOP_Z - 0.22},
    )
    ball_resting = DoneTerm(
        func=mdp.ball_resting_on_table,
        params={
            "asset_cfg": _BALL, "table_top_z": TABLE_TOP_Z,
            "ball_radius": 0.02, "height_tolerance": 0.04, "min_steps": 45,
        },
    )
    # right_table_bounce = DoneTerm(
    #     func=mdp.simple_ball_on_table,
    #     params={
    #         "asset_cfg": _BALL,
    #         "table_center": TABLE_CENTER, "table_size": TABLE_SIZE,
    #         "ball_radius": 0.02, "net_x": NET_X,
    #         "height_tolerance": 0.08,
    #         "side": "right",
    #     },
    # )
    # left_table_bounce_done = DoneTerm(
    #     func=mdp.simple_ball_on_table,
    #     params={
    #         "asset_cfg": _BALL,
    #         "table_center": TABLE_CENTER, "table_size": TABLE_SIZE,
    #         "ball_radius": 0.02, "net_x": NET_X,
    #         "height_tolerance": 0.08,
    #         "side": "left",
    #     },
    # )
    # joint_limit terminations DISABLED — for long rally observation
    # left_joint_limit = DoneTerm(
    #     func=mdp.joint_pos_out_of_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ALL_JOINTS)},
    # )
    # right_joint_limit = DoneTerm(
    #     func=mdp.joint_pos_out_of_limit,
    #     params={"asset_cfg": SceneEntityCfg("right_robot", joint_names=_ALL_JOINTS)},
    # )
    # table_collision terminations DISABLED — keep only the reward penalty
    # so the arm can recover from near-table movements
    # left_table_collision = DoneTerm(
    #     func=mdp.robot_table_collision,
    #     params={"asset_cfg": _L_ROBOT, "table_center": TABLE_CENTER,
    #             "table_size": TABLE_SIZE, "clearance": 0.02},
    # )
    # right_table_collision = DoneTerm(
    #     func=mdp.robot_table_collision,
    #     params={"asset_cfg": _R_ROBOT, "table_center": TABLE_CENTER,
    #             "table_size": TABLE_SIZE, "clearance": 0.02},
    # )


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

@configclass
class Fairino3DualPingPongCenterlineEnvCfg(ManagerBasedRLEnvCfg):
    """Dual-arm Fairino3 ping-pong RL environment config."""

    scene: Fairino3DualPingPongSceneCfg = Fairino3DualPingPongSceneCfg(num_envs=1024, env_spacing=5.5)
    observations: SymmetricObservationsCfg = SymmetricObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 60.0
        self.viewer.eye = (3.2, -4.5, 2.2)
        self.viewer.lookat = (NET_X, 0.0, TABLE_TOP_Z)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
