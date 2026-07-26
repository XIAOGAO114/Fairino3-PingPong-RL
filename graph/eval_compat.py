"""Measure compatible return rate for single-arm model.
Usage: python eval_compat.py <left|right>

Counts net-crossing returns that fall within opponent's serve comfort zone.
Comfort zone = opponent's training serve distribution (y, z, vx, vy, vz bounds).
"""
import sys, os, torch, numpy as np, torch.nn as nn
from isaaclab.app import AppLauncher

WORKSPACE = sys.argv[1]  # "left" or "right"

if WORKSPACE == "left":
    src = "/home/glq/isaac_ws/test_isaac_rail/source"
    task = "Fairino3-PingPong-Rail-Centerline-v0"
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "/home/glq/isaac_ws/test_isaac_rail/logs/rsl_rl/fairino3_rail_centerline_v1/2026-05-30_21-35-06_compat_v1/model_2042.pt"
    pkg = "test_isaac_rail"
else:
    src = "/home/glq/isaac_ws/test_isaac_rail_right/source"
    task = "Fairino3-PingPong-Rail-Right-Centerline-v0"
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "/home/glq/isaac_ws/test_isaac_rail_right/logs/rsl_rl/fairino3_rail_right_centerline_v1/2026-05-28_15-08-48_right_v6_angdamp_cont1_1525/model_2225.pt"
    pkg = "test_isaac_rail_right"

sys.path.insert(0, src)
app = AppLauncher(headless=True, args=['']); sim = app.app
exec(f"import {pkg}.tasks")
import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

cfg = load_cfg_from_registry(task.split(":")[-1], "env_cfg_entry_point")
cfg.scene.num_envs = 256; cfg.seed = 42
env = gym.make(task, cfg=cfg)
obs_dict, _ = env.reset()

m = nn.Sequential(nn.Linear(32,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,7)).cuda().eval()
ckpt = torch.load(ckpt_path, map_location="cuda:0", weights_only=False)
sd = ckpt["actor_state_dict"]
state = {k[4:]:v for k,v in sd.items() if k.startswith("mlp.")}
m.load_state_dict(state, strict=False)

NET_X = 0.55; total, compat = 0, 0
LEFTRETURN = (WORKSPACE == "left")

with torch.no_grad():
    for step in range(500):
        ot = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        a = m(ot); obs_dict, _, _, _, _ = env.step(a)
        bp = env.unwrapped.scene["ball"].data.root_pos_w[:,:3] - env.unwrapped.scene.env_origins
        bv = env.unwrapped.scene["ball"].data.root_lin_vel_w[:,:3]
        near_net = torch.abs(bp[:,0] - NET_X) < 0.18
        returning = bv[:,0] > 0.05 if LEFTRETURN else bv[:,0] < -0.05
        crossed = near_net & returning
        y_ok = (bp[:,1] > -0.55) & (bp[:,1] < 0.55)
        z_ok = (bp[:,2] > 0.9125) & (bp[:,2] < 1.20)
        abs_vx = torch.abs(bv[:,0])
        vx_ok = (abs_vx > 0.3) & (abs_vx < 3.0)
        vy_ok = (bv[:,1] > -0.80) & (bv[:,1] < 0.80)
        vz_ok = (bv[:,2] > -0.10) & (bv[:,2] < 0.70)
        c = crossed & y_ok & z_ok & vx_ok & vy_ok & vz_ok
        total += crossed.sum().item(); compat += c.sum().item()
        if step%100==0:
            r = compat/total*100 if total > 0 else 0
            sys.stdout.write(f"{WORKSPACE} {step}: {compat}/{total}={r:.1f}%\n"); sys.stdout.flush()

rate = compat/total*100 if total > 0 else 0
sys.stdout.write(f"\n{WORKSPACE}: {compat}/{total} = {rate:.1f}% compatible\n"); sys.stdout.flush()
env.close(); sim.close()
