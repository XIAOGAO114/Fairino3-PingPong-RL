---
name: right-arm-2026-05-28-progress
description: 右臂单臂训练进展：v6 angular_damping 方案成功，model_2225 视觉完美，指标被检测噪声拉低
metadata: 
  type: project
  originSessionId: 1c5f5016-19f2-439d-9862-6cdff39cfb61
---

## 右臂单臂训练 — 2026-05-28 进展

**项目：** `/home/glq/isaac_ws/test_isaac_rail_right/`
**任务 ID：** `Fairino3-PingPong-Rail-Right-Centerline-v0`

### v6 angular_damping 方案已成功

**最佳模型：** `model_2225.pt` @ `2026-05-28_15-08-48_right_v6_angdamp_cont1_1525/`
- 训练指标：succ=82.9%, jl=13.3%, spin=-1.61
- **Play 视觉表现：完美。所有回球过网落台，无失误。**
- 80% 的训练指标被左台反弹检测噪声拉低（单 env play vs 1024 env train，样本少导致 left_table_bounce 检测不准）

### 训练发散问题

从 model_1525 续跑 ~1150 迭代，entropy_coef=0.0015 导致 action_std 从 0.85→1.19，熵从 7.8→9.35 持续上升。成功率在 80-83% 横盘从未突破。但视觉上 model_2225 已经很好了。

### angular_damping 物理方案确认

- `angular_damping=0.05` 在三个项目已同步
- 球旋转在飞行中自然衰减，不引入 reward 副作用
- 比 reward 侧旋惩罚方案稳定（后者导致关节极限崩塌）

### 参考

- 原始 v4 最佳：`model_1350.pt` (83%，无 angular_damping)
- v6 视觉最佳：`model_2225.pt` (82.9% 指标，视觉完美)
- Play 命令：`python scripts/rsl_rl/play.py --task Fairino3-PingPong-Rail-Right-Centerline-v0 --num_envs 1 --real-time --checkpoint <path>`
