"""Right-arm variant: robot on RIGHT side, ball served from LEFT."""
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

from fairino3_rail.assets.fairino3_v6_rail import FAIRINO3_V6_RAIL_PADDLE_CFG

from . import mdp

# ---- constants (robot on RIGHT, ball from LEFT) ----
TABLE_CENTER = (0.55, 0.0, 0.74)
TABLE_SIZE = (2.74, 1.525, 0.04)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] * 0.5
TABLE_HALF_LENGTH = TABLE_SIZE[0] * 0.5
TABLE_HALF_WIDTH = TABLE_SIZE[1] * 0.5
NET_X = TABLE_CENTER[0]
NET_HEIGHT = TABLE_TOP_Z + 0.1525
RIGHT_RAIL_ROBOT_POS = (2.07, 0.0, TABLE_TOP_Z)
LEFT_MID_TARGET_X = NET_X - TABLE_HALF_LENGTH * 0.5       # opponent side
LEFT_NEAR_NET_TARGET_X = NET_X - TABLE_HALF_LENGTH * 0.22
RIGHT_INTERCEPT_X = NET_X + TABLE_HALF_LENGTH * 0.5         # own side intercept
BALL_INIT_POS = (NET_X - 0.40, 0.0, TABLE_TOP_Z + 0.26)    # left of net
TARGET_MARKER_Z = TABLE_TOP_Z + 0.005
X_BOUNDS = (NET_X - TABLE_HALF_LENGTH - 0.70, NET_X + TABLE_HALF_LENGTH + 0.70)
Y_BOUNDS = (-TABLE_HALF_WIDTH - 0.45, TABLE_HALF_WIDTH + 0.45)

_ALL_JOINTS = ["rail_y", "j1", "j2", "j3", "j4", "j5", "j6"]
_ARM_JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]

_RIGHT_RAIL_ROBOT_INIT = ArticulationCfg.InitialStateCfg(
    pos=RIGHT_RAIL_ROBOT_POS,
    rot=(0.7071068, 0.0, 0.0, -0.7071068),
    joint_pos={
        "rail_y": 0.0, "j1": -1.743, "j2": -2.940,
        "j3": 0.0, "j4": -1.847, "j5": 2.7, "j6": 1.054,
    },
)


@configclass
class Fairino3RailRightPingPongSceneCfg(InteractiveSceneCfg):
    """Scene with rail-mounted Fairino3 arm on the RIGHT side."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(12.0, 12.0)))

    robot: ArticulationCfg = FAIRINO3_V6_RAIL_PADDLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", init_state=_RIGHT_RAIL_ROBOT_INIT,
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_linear_velocity=20.0,
                max_depenetration_velocity=10.0, solver_position_iteration_count=8,
                solver_velocity_iteration_count=2, angular_damping=0.05,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.4, restitution=0.88),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.05)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_INIT_POS, lin_vel=(1.25, 0.0, 0.12)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.65, dynamic_friction=0.55, restitution=0.85),
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

    target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(size=(0.18, 0.18, 0.005),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.45, 1.0))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(LEFT_MID_TARGET_X, 0.0, TARGET_MARKER_Z)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=800.0),
    )


# ---- observations (same as left arm, 32-dim, robot-frame relative) ----
class CenterlineRightObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        paddle_pos = ObsTerm(func=mdp.body_pos_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["paddle_link"])})
        ball_pos = ObsTerm(func=mdp.root_pos_robot_frame,
            params={"asset_cfg": SceneEntityCfg("ball"), "robot_cfg": SceneEntityCfg("robot")})
        ball_vel = ObsTerm(func=mdp.root_lin_vel_robot_frame, scale=0.2,
            params={"asset_cfg": SceneEntityCfg("ball"), "robot_cfg": SceneEntityCfg("robot")})
        ball_to_paddle_pos = ObsTerm(func=mdp.ball_to_paddle_pos_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]), "ball_cfg": SceneEntityCfg("ball")})
        ball_to_paddle_vel = ObsTerm(func=mdp.ball_to_paddle_vel_robot_frame, scale=0.2,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]), "ball_cfg": SceneEntityCfg("ball")})
        target_pos = ObsTerm(func=mdp.target_position_robot_frame,
            params={"target": (LEFT_MID_TARGET_X, 0.0, TABLE_TOP_Z), "robot_cfg": SceneEntityCfg("robot")})
        def __post_init__(self):
            self.enable_corruption = False; self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()


# ---- actions ----
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=_ARM_JOINTS, scale=0.5)
    rail_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["rail_y"], scale=0.5)


# ---- events (serve toward right arm) ----
class EventCfg:
    reset_robot = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ALL_JOINTS),
                "position_range": (-0.08, 0.08), "velocity_range": (-0.05, 0.05)})
    reset_ball = EventTerm(func=mdp.reset_ball_valid_serve, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {"x": (-0.20, 0.20), "y": (-0.35, 0.35), "z": (-0.08, 0.08)},
            "velocity_range": {
                "x": (0.7, 2.2), "y": (-0.45, 0.45), "z": (0.05, 0.55),
                "roll": (-1.2, 1.2), "pitch": (-1.2, 1.2), "yaw": (-1.2, 1.2),
            },
            "max_resamples": 64, "gravity": 9.81,
        })


# ---- rewards (mirrored: left→right, right→left) ----
class RewardsCfg:
    left_table_bounce = RewTerm(func=mdp.left_table_bounce, weight=6.0,
        params={"ball_cfg": SceneEntityCfg("ball")})
    alive = None
    paddle_to_ball = RewTerm(func=mdp.paddle_to_ball_distance, weight=1.0,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]),
                "ball_cfg": SceneEntityCfg("ball"), "std": 0.45})
    paddle_to_intercept = RewTerm(func=mdp.paddle_to_intercept, weight=1.2,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]),
                "ball_cfg": SceneEntityCfg("ball"), "intercept_x": RIGHT_INTERCEPT_X, "std": 0.30})
    paddle_to_bounce_zone = None
    incoming_ball = None
    first_return_hit = RewTerm(func=mdp.first_return_hit, weight=10.0,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]),
                "ball_cfg": SceneEntityCfg("ball"),
                "contact_distance": 0.10, "min_incoming_speed": 0.03,
                "min_return_speed": 0.03, "contact_window_steps": 6})
    return_ball = RewTerm(func=mdp.return_ball_velocity, weight=12.0,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]),
                "ball_cfg": SceneEntityCfg("ball"), "hit_distance": 0.18})
    second_paddle_contact = RewTerm(func=mdp.second_paddle_contact_penalty, weight=-16.0,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]),
                "ball_cfg": SceneEntityCfg("ball"),
                "contact_distance": 0.10, "grace_steps": 4, "window_steps": 60,
                "min_return_speed": 0.10, "min_delta_speed": 0.03})
    over_net = None
    net_direction = RewTerm(func=mdp.net_direction, weight=10.0,
        params={"asset_cfg": SceneEntityCfg("ball"), "net_x": NET_X, "net_window": 0.18})
    net_height = RewTerm(func=mdp.net_height, weight=3.0,
        params={"asset_cfg": SceneEntityCfg("ball"), "net_x": NET_X,
                "net_window": 0.18, "ideal_z": NET_HEIGHT + 0.04, "std": 0.10})
    net_speed = RewTerm(func=mdp.net_speed, weight=6.0,
        params={"asset_cfg": SceneEntityCfg("ball"), "net_x": NET_X,
                "net_window": 0.18, "ideal_speed": 1.9, "std": 0.45})
    opponent_compatible = RewTerm(func=mdp.opponent_compatible_return, weight=80.0,
        params={"asset_cfg": SceneEntityCfg("ball"), "net_x": NET_X,
                "net_window": 0.18, "net_height": NET_HEIGHT,
                "min_y": -0.55, "max_y": 0.55, "max_z": 1.20,
                "min_vx": 0.3, "max_vx": 3.0,
                "min_vy": -0.80, "max_vy": 0.80,
                "min_vz": -0.10, "max_vz": 0.70})
    high_return_height = None
    centerline_x = None
    predicted_landing = RewTerm(func=mdp.predicted_right_table_landing, weight=30.0,
        params={"asset_cfg": SceneEntityCfg("ball"),
                "target_xy": (LEFT_MID_TARGET_X, 0.0),
                "table_center": TABLE_CENTER, "table_size": TABLE_SIZE,
                "ball_radius": 0.02, "net_x": NET_X, "std": 0.44})
    near_net_landing_speed = RewTerm(func=mdp.predicted_right_near_net_landing_speed, weight=12.0,
        params={"asset_cfg": SceneEntityCfg("ball"),
                "target_xy": (LEFT_NEAR_NET_TARGET_X, 0.0),
                "table_center": TABLE_CENTER, "table_size": TABLE_SIZE,
                "ball_radius": 0.02, "net_x": NET_X,
                "std_x": 0.22, "std_y": 0.24,
                "min_right_speed": 0.75, "target_right_speed": 1.8})
    target_landing = None
    rally_return = None
    early_hit_penalty = RewTerm(func=mdp.early_hit_penalty, weight=-3.0,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]),
                "ball_cfg": SceneEntityCfg("ball"), "contact_distance": 0.10})
    post_hit_paddle_clearance = RewTerm(func=mdp.post_hit_paddle_ball_clearance_penalty, weight=-8.0,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]),
                "ball_cfg": SceneEntityCfg("ball"),
                "min_distance": 0.22, "grace_steps": 4, "window_steps": 18, "min_return_speed": 0.10})
    post_hit_paddle_retreat = RewTerm(func=mdp.post_hit_paddle_retreat_penalty, weight=-4.0,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]),
                "ball_cfg": SceneEntityCfg("ball"),
                "min_behind_x": 0.12, "near_distance": 0.34,
                "grace_steps": 4, "window_steps": 22, "min_return_speed": 0.10, "chase_speed": 0.05})
    legal_return_separation = RewTerm(func=mdp.legal_return_separation_reward, weight=12.0,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]),
                "ball_cfg": SceneEntityCfg("ball"),
                "table_center": TABLE_CENTER, "table_size": TABLE_SIZE,
                "ball_radius": 0.02, "net_x": NET_X,
                "target_xy": (LEFT_MID_TARGET_X, 0.0),
                "min_behind_x": 0.14, "behind_std": 0.18,
                "min_return_speed": 0.25, "target_return_speed": 1.8,
                "grace_steps": 3, "window_steps": 34, "landing_std": 0.42})
    right_table_bounce = RewTerm(func=mdp.right_table_bounce_opponent_scaled, weight=250.0,
        params={"asset_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]),
                "table_center": TABLE_CENTER, "table_size": TABLE_SIZE,
                "ball_radius": 0.02, "net_x": NET_X,
                "illegal_contact_distance": 0.10, "illegal_reward_scale": 0.0, "require_clean_over_net": True})
    post_return_idle = None
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ALL_JOINTS)})
    joint_limit = None
    table_collision = None
    ball_out_penalty = RewTerm(func=mdp.ball_out_of_bounds_penalty, weight=-8.0,
        params={"asset_cfg": SceneEntityCfg("ball"), "x_bounds": X_BOUNDS, "y_bounds": Y_BOUNDS, "z_min": 0.03})
    paddle_body_clearance = RewTerm(func=mdp.paddle_body_clearance_penalty, weight=-6.0,
        params={"paddle_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]),
                "body_cfg": SceneEntityCfg("robot",
                    body_names=["forearm_link", "wrist1_link", "wrist2_link", "wrist3_link", "tool_clamp_link"]),
                "min_distance": 0.15})
    terminating = RewTerm(func=mdp.is_terminated, weight=-4.0)


# ---- terminations (mirrored) ----
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    ball_out_of_bounds = DoneTerm(func=mdp.ball_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("ball"), "x_bounds": X_BOUNDS, "y_bounds": Y_BOUNDS, "z_min": 0.03})
    left_table_bounce = DoneTerm(func=mdp.right_table_bounce,  # ball goes to LEFT (opponent) side
        params={"asset_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot", body_names=["paddle_link"]),
                "table_center": TABLE_CENTER, "table_size": TABLE_SIZE,
                "ball_radius": 0.02, "net_x": NET_X,
                "illegal_contact_distance": 0.10,
                "allow_illegal_success": False, "require_clean_over_net": True})
    robot_joint_limit = DoneTerm(func=mdp.joint_pos_out_of_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ALL_JOINTS)})
    robot_table_collision = DoneTerm(func=mdp.robot_table_collision,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "table_center": TABLE_CENTER, "table_size": TABLE_SIZE, "clearance": 0.04})


@configclass
class Fairino3RailRightPingPongEnvCfg(ManagerBasedRLEnvCfg):
    """Right-arm rail-mounted Fairino3 ping-pong RL environment config."""

    scene: Fairino3RailRightPingPongSceneCfg = Fairino3RailRightPingPongSceneCfg(num_envs=1024, env_spacing=4.5)
    observations: CenterlineRightObservationsCfg = CenterlineRightObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 3.0
        self.viewer.eye = (3.2, 3.0, 1.8)
        self.viewer.lookat = (NET_X, 0.0, TABLE_TOP_Z)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
