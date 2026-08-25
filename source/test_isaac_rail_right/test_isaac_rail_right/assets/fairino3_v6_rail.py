"""Fairino3 V6 robot with Y-axis prismatic rail for single-arm ping-pong."""

from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ASSET_DIR = Path(__file__).resolve().parent / "fairino3_v6_rail"

FAIRINO3_V6_RAIL_PADDLE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ASSET_DIR / "fairino3_v6_rail_paddle_isaac.urdf"),
        usd_dir=str(ASSET_DIR / "usd"),
        usd_file_name="fairino3_v6_rail_paddle.usd",
        fix_base=True,
        merge_fixed_joints=False,
        self_collision=False,
        collider_type="convex_hull",
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=400.0, damping=40.0),
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.97, 0.0, 0.76),
        rot=(0.7071068, 0.0, 0.0, 0.7071068),
        joint_pos={
            "rail_y": 0.0,
            "j1": -1.743,
            "j2": -2.940,
            "j3": 0.0,
            "j4": -1.847,
            "j5": 2.7,
            "j6": 1.054,
        },
    ),
    actuators={
        "rail": ImplicitActuatorCfg(
            joint_names_expr=["rail_y"],
            effort_limit_sim=150.0,
            velocity_limit_sim=0.5,
            stiffness=400.0,
            damping=60.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["j[1-6]"],
            effort_limit_sim={
                "j[1-3]": 150.0,
                "j[4-6]": 28.0,
            },
            velocity_limit_sim={
                "j[1-3]": 3.15,
                "j[4-6]": 3.2,
            },
            stiffness=400.0,
            damping=40.0,
        ),
    },
)
