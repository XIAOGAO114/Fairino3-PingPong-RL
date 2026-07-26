---
name: best-single-arm-models
description: 2026-07-11 清理后保留的最佳单臂模型及合并模型
metadata: 
  node_type: memory
  type: project
  updated: 2026-07-11
---

## 清理后保留的模型（2026-07-11）

已删除 241 个无用训练 run，释放 8.8G。保留 9 个有效 run（共 519M）。

### 左臂 (test_isaac_rail) — 277M

| 模型 | Run | 指标 |
|------|-----|------|
| **model_1543** | `2026-05-26_11-34-16_v6_generalize_from1244_300` | 97.1% clean_right, 所有后续模型 base |
| **model_2042** | `2026-05-30_21-35-06_compat_v1` | 93.7% hit, +opponent_compatible, merged_2042_2042 左臂 |
| **model_1842** | `2026-05-26_12-15-48_v7b_soft_height_from1543_300` | 94.7%, height_factor 改善回球高度 |
| **model_448** | `2026-05-26_01-35-19_cont_v1_400iter` | 97%, 第一个 7-DOF clean checkpoint |

路径前缀: `/home/glq/isaac_ws/test_isaac_rail/logs/rsl_rl/fairino3_rail_centerline_v1/`

### 右臂 (test_isaac_rail_right) — 210M

| 模型 | Run | 指标 |
|------|-----|------|
| **model_2225** | `2026-05-28_15-08-48_right_v6_angdamp_cont1_1525` | 84.4%, 视觉完美 |

路径前缀: `/home/glq/isaac_ws/test_isaac_rail_right/logs/rsl_rl/fairino3_rail_right_centerline_v1/`

### 双臂合并 (test_isaac_dual) — 32M

| 合并 | 组成 | 指标 |
|------|------|------|
| **merged_2042_2042** | 左 model_2042 + 右 model_2042 | L=1729, R=865, rally=1107 (最佳) |
| **merged_1543_2225** | 左 model_1543 + 右 model_2225 | 保底合并 |
| **merged_1543_2225_final** | 同上（含 params） | 含训练参数 |
| **merged_compat** | compat 训练后合并 | rally 4x 提升 |

路径前缀: `/home/glq/isaac_ws/test_isaac_dual/logs/rsl_rl/fairino3_dual_centerline_v1/`

### 关键参数（所有模型通用）

- 架构: MLP [512, 256, 128], ELU
- PPO: lr=3e-4, γ=0.99, λ=0.95, entropy=0.0015, 5 epoch, 32 mini-batch
- 左臂位置: x=-0.97, y=0.0, z=0.76, rot=(0.707, 0, 0, 0.707)
- 右臂位置: x=2.07, y=0.0, z=0.76, rot=(0.707, 0, 0, -0.707)
- Ball: mass=0.0027, restitution=0.88, angular_damping=0.05
- init_std: 左臂 0.7, 右臂 0.2

### Play 命令

```bash
# 最佳双臂
cd /home/glq/isaac_ws/test_isaac_dual
python scripts/rsl_rl/play.py \
  --task Fairino3-PingPong-Dual-Centerline-v0 \
  --num_envs 1 --real-time \
  --checkpoint logs/rsl_rl/fairino3_dual_centerline_v1/merged_2042_2042/model_0.pt
```
