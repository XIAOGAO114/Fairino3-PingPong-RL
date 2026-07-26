---
name: dual-arm-2026-05-28-state
description: 双臂项目 2026-05-28 最新状态：左臂v7b高度压低版+右臂v6完成，终止条件简化，需微调
metadata:
  type: project
  originSessionId: bb875f67-e09f-491d-b1db-3776803f15b5
---

## 双臂项目 — 2026-05-28 最新状态

**工作区:** `/home/glq/isaac_ws/test_isaac_dual/`
**任务 ID:** `Fairino3-PingPong-Dual-Centerline-v0`

### 当前双臂配置

| 手臂 | 模型 | 成功率 | 特点 |
|------|------|------|------|
| 左臂 | model_2541.pt | 84% | v7b高度压低版(1000iter续训)，angular_damping=0.05 |
| 右臂 | model_2225.pt | 83% | v6 angular_damping，视觉完美 |

**合并 checkpoint:** `logs/rsl_rl/fairino3_dual_centerline_v1/merged_2541_2225/model_0.pt`

### 左臂续训历程

- model_1543 (v6, 97.1%) → 回球太高
- model_2042 (v7b第一轮500iter, 79.6%) → 高度惩罚降62%
- model_2541 (v7b第二轮500iter, 84.0%) → 高度惩罚卡在-0.22，成功率84%
- 高度惩罚已到瓶颈，需要更大力度（降低max_good_z或增加权重）

### 环境配置修改

- `episode_length_s = 60.0`
- 终止条件：time_out + ball_fall_off_table (z<0.76) + table_collision
- `simple_ball_on_table()` — 简化的落台终止（不要求has_return_hit等）
- ball_out_of_bounds + joint_limit 已注释

### Play 命令

```bash
cd /home/glq/isaac_ws/test_isaac_dual
python scripts/rsl_rl/play.py \
  --task Fairino3-PingPong-Dual-Centerline-v0 \
  --num_envs 1 --real-time \
  --checkpoint logs/rsl_rl/fairino3_dual_centerline_v1/merged_2541_2225/model_0.pt
```

### 待解决

- 左臂回球仍然偏高，84%成功率不够
- 可能需要：降低max_good_z、增加high_return_height权重、或尝试angular_damping增大
- 双臂对打需进一步微调
- 终止条件安装问题：需确保修改后pip install -e source/test_isaac_dual生效
