"""评估合并模型: 逐 episode 统计击球和回球"""
import argparse, torch, numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=5000)
parser.add_argument("--checkpoint", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import test_isaac_dual.tasks
from test_isaac_dual.tasks.manager_based.fairino3_dual_pingpong.fairino3_dual_pingpong_env_cfg import Fairino3DualPingPongCenterlineEnvCfg

env_cfg = Fairino3DualPingPongCenterlineEnvCfg()
env_cfg.seed = 42
env_cfg.scene.num_envs = args_cli.num_envs
env = gym.make("Fairino3-PingPong-Dual-Centerline-v0", cfg=env_cfg)
device = env.unwrapped.device

# Load model
ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
from test_isaac_dual.tasks.manager_based.fairino3_dual_pingpong.models.dual_arm_actor import DualArmActor
obs, _ = env.reset()
model = DualArmActor(
    obs=obs, obs_groups={'actor': ['policy'], 'critic': ['policy']}, obs_set='actor', output_dim=14,
    hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False,
    distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 0.4, 'std_type': 'scalar'},
).to(device)
model.load_state_dict(ckpt['actor_state_dict'], strict=True)
model.eval()
model.distribution.std_param.data.fill_(0.0)  # deterministic

obs, _ = env.reset()
left_hits = 0; right_hits = 0; left_bounces = 0; rallies = 0
total_steps = 0

for step in range(args_cli.num_steps):
    with torch.no_grad():
        actions = model.forward(obs)
    obs, reward, terminated, truncated, info = env.step(actions)
    total_steps += 1

    for eid in range(args_cli.num_envs):
        lhs = getattr(env.unwrapped, '_fairino_return_hit_state', None)
        rhs = getattr(env.unwrapped, '_fairino_right_return_hit_state', None)
        rally_s = getattr(env.unwrapped, '_rally_count', None)
        if lhs is not None and lhs[eid].item() > 0: left_hits += 1
        if rhs is not None and rhs[eid].item() > 0: right_hits += 1
        if rally_s is not None and rally_s[eid].item() > 0: rallies += 1

print(f"\n{'='*50}")
print(f"合并模型评估: {args_cli.num_steps} steps x {args_cli.num_envs} envs")
print(f"{'='*50}")
print(f"  Left hit events:     {left_hits}")
print(f"  Right hit events:    {right_hits}")
print(f"  Rally events:        {rallies}")
print(f"  Total steps:         {total_steps}")
env.close(); simulation_app.close()
