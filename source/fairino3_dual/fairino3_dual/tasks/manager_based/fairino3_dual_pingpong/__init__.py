import gymnasium as gym

from . import agents

gym.register(
    id="Fairino3-PingPong-Dual-Centerline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fairino3_dual_pingpong_env_cfg:Fairino3DualPingPongCenterlineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
