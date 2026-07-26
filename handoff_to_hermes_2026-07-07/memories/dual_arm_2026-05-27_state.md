---
name: dual-arm-2026-05-27-state
description: Dual-arm project state after major restructuring on 2026-05-27
metadata: 
  node_type: memory
  type: project
  originSessionId: 155572ce-1598-4d0a-8337-afb576ede21b
---

## Dual-Arm Project — 2026-05-27 State

**Workspace:** `/home/glq/isaac_ws/test_isaac_dual/`
**Task ID:** `Fairino3-PingPong-Dual-Centerline-v0`

### Architecture Decisions Made

1. **Model**: Changed from `SharedActorMLPModel` (shared-weight MLP + j1/rail_y mirroring) to `DualArmActor` (two independent MLPs 32→7 + shared 14-dim distribution).
   - File: `models/dual_arm_actor.py`
   - Left actor loaded from single-arm `model_1543.pt`, right actor random init
   - **No j1/rail_y negation** — both arms have identical joint config, right robot rotated 180° around z

2. **Rewards**: Extracted `_make_arm_rewards()` factory in env_cfg.py. Both arms get identical 21-term reward structure, just with mirrored targets and home_side.
   - Left: `home_side="left"`, target = RIGHT table
   - Right: `home_side="right"`, target = LEFT table

3. **Serve**: Changed to left-only (matches single-arm). Removed `_serve_clears_net` validation — physics handles net collisions naturally.

### Critical Configuration

**Robot positions (from-scratch single-arm values):**
- Left: `pos=(-0.97, 0, 0.76)`, right: `pos=(2.07, 0, 0.76)`
- **Joint angles (SOFT values — essential for low joint limits):**
  - `j2 = -2.94` (NOT -3.368!)
  - `j5 = 2.7` (NOT 3.030!)
  - j1=-1.743, j3=0, j4=-1.847, j6=1.054
- All joints same for both arms
- Left rot: (0.707, 0, 0, 0.707), Right rot: (0.707, 0, 0, -0.707)

**Why soft angles matter:** j2=-3.368 is 0.13 rad from lower limit (~-3.5). One random action step pushes it past the limit, causing 55-78% joint limit termination rate and 1.2-step episodes. With j2=-2.94, joint limit rate is <1%.

**PPO config:**
- `init_std=0.15` (reduced from 0.7 to prevent joint limit violations from right arm)
- hidden_dims=[512, 256, 128], activation=elu

### Key Bug Fixes

1. `legal_return_separation_reward` behind_margin: now handles home_side (right arm was getting 0)
2. Removed `_serve_clears_net` validation (too strict, caused ball flickering near net)
3. Right arm initial joint angles = left arm (not mirrored j1)
4. Removed j1/rail_y negation from model (not needed with 180° base rotation)

### Last Working Checkpoint

**Path:** `/home/glq/isaac_ws/test_isaac_dual/logs/rsl_rl/fairino3_dual_centerline_v1/2026-05-27_02-32-32_init_023224/model_0_left1543.pt`

This has:
- `left_actor` weights from model_1543
- `distribution.std_param[:7]` from model_1543
- `right_actor` random init
- `distribution.std_param[7:]` = 0.15

### Training Results (25 iterations, 512 envs)

| Metric | Start | End |
|--------|-------|-----|
| left_first_return_hit | 0.27% | 2.6% |
| left_right_table_bounce (success) | 0% | **41-51%** |
| right_table_bounce termination | 0% | 51-56% |
| right_first_return_hit | 0% | 0.09% |
| right_left_table_bounce (home bounce) | 0.4% | 2.0% |
| left_joint_limit | 0.4% | 10.6% |
| right_joint_limit | 0% | 0% |
| mean_episode_length | 31 | 112 |
| mean_reward | 0.74 | 19.4 |

**Left arm is working** (left model_1543 frozen, 41-51% success rate after just 25 iters).
**Right arm barely starting** (0.09% first hit).
**Left joint limit creeping up** (0.4% → 10.6%) — model_1543's std (1.0-2.5) adds noise.

### Next Steps

1. Resume training from `model_0_left1543.pt` for 500+ iterations
2. Or: reduce left arm's std further to control joint limit
3. Watch for: right_first_return_hit > 5%, right_left_table_bounce_reward > 0
4. Once right arm shows signs of learning, consider unfreezing left arm for joint fine-tuning

### Training Command

```bash
cd /home/glq/isaac_ws/test_isaac_dual
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Fairino3-PingPong-Dual-Centerline-v0 \
  --num_envs 512 --max_iterations 500 --headless \
  --resume \
  --load_run 2026-05-27_02-32-32_init_023224 \
  --checkpoint model_0_left1543.pt \
  --run_name dual_continue_<tag>
```
