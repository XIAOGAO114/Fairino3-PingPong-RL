"""Measure rally exchange count per episode for dual-arm merged checkpoint.

Counts _fairino_rally_exchange_count at episode boundaries.
Reports distribution of 0/1/2+ exchanges, mean, max.
"""
import sys, os, torch, numpy as np, torch.nn as nn
from isaaclab.app import AppLauncher

CKPT = sys.argv[1] if len(sys.argv) > 1 else "<TRAIN_WORKSPACE>/fairino3_dual/logs/rsl_rl/fairino3_dual_centerline_v1/merged_2042_2042/model_0.pt"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
NUM_ENVS = 256

sys.path.insert(0, "<TRAIN_WORKSPACE>/fairino3_dual/source")
app = AppLauncher(headless=True, args=['']); sim = app.app
import gymnasium as gym
import fairino3_rail.tasks; import fairino3_dual.tasks
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

cfg = load_cfg_from_registry("Fairino3-PingPong-Dual-Centerline-v0", "env_cfg_entry_point")
cfg.scene.num_envs = NUM_ENVS; cfg.seed = 42
env = gym.make("Fairino3-PingPong-Dual-Centerline-v0", cfg=cfg)
obs_dict, _ = env.reset()

la = nn.Sequential(nn.Linear(32,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,7)).cuda().eval()
ra = nn.Sequential(nn.Linear(32,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,7)).cuda().eval()
ckpt = torch.load(CKPT, map_location="cuda:0", weights_only=False)
lsd, rsd = {}, {}
for k, v in ckpt["actor_state_dict"].items():
    if k.startswith("left_actor."): lsd[k[11:]] = v
    elif k.startswith("right_actor."): rsd[k[12:]] = v
la.load_state_dict(lsd); ra.load_state_dict(rsd)

total_episodes = 0
rally_counts = []

with torch.no_grad():
    for step in range(STEPS):
        ot = obs_dict["policy"]
        a = torch.cat([la(ot[:,:32]), ra(ot[:,32:64])], dim=-1)
        obs_dict, _, _, _, _ = env.step(a)
        ep_len = env.unwrapped.episode_length_buf
        just_reset = ep_len <= 1
        if just_reset.any():
            rally_vals = env.unwrapped._fairino_rally_exchange_count[just_reset].cpu().numpy()
            rally_counts.extend(rally_vals.tolist())
            total_episodes += just_reset.sum().item()
        if step % 200 == 0:
            sys.stdout.write(f"Step {step}, {total_episodes} episodes\n"); sys.stdout.flush()

rally_vals = env.unwrapped._fairino_rally_exchange_count.cpu().numpy()
rally_counts.extend(rally_vals.tolist())
total_episodes += NUM_ENVS

rc = np.array(rally_counts)
sys.stdout.write(f"\n=== Rally Stats ({total_episodes} episodes) ===\n")
for i in range(int(rc.max()) + 1):
    cnt = (rc == i).sum()
    sys.stdout.write(f"  {i} exchanges: {cnt:5d} ({cnt/len(rc)*100:5.1f}%)\n")
sys.stdout.write(f"  Max: {rc.max()}\n")
sys.stdout.write(f"  Mean: {rc.mean():.4f}\n")
sys.stdout.write(f"  >0 rate: {(rc>0).mean()*100:.2f}%\n")
sys.stdout.write(f"  >=2 rate: {(rc>=2).mean()*100:.2f}%\n")
sys.stdout.flush()
env.close(); sim.close()
