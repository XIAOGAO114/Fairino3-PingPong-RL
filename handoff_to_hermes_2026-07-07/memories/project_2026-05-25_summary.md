---
name: 2026-05-25-summary
description: 当天工作进展：v2i/v2j gate、rail 项目创建、v2 基础训练
metadata: 
  node_type: memory
  type: project
  originSessionId: 2de03140-06d9-4131-82d0-23b8101f1508
---

# 2026-05-25 进展

## 完成的工作

### 1. v2i legal dense gate（单臂）
- `_shared.py` 新增 `legal_post_hit_mask()` — has_return_hit & ~has_illegal_second_hit
- `rewards.py` 10 个奖励函数 gate 替换
- v2h model_2946 训练两轮，最佳 model_3194.pt

### 2. v2j 严格检测（单臂）
- `illegal_second_hit_event` 参数收紧：window_steps 24→60, min_delta_speed 0.12→0.03, 移除 vx 方向限制
- 严格检测揭示了真实连击率远高于之前指标

### 3. test_isaac_rail 项目（七自由度带滑轨）
- 独立项目 `/home/glq/isaac_ws/test_isaac_rail/`
- URDF 含 X 轴 prismatic joint (rail_y, axis xyz="1 0 0")
- 7-DOF 动作空间 (rail + j1-j6)
- 机器人位置 x=-1.35, z=0.98
- right_table_bounce 权重 250
- v2j strict gate 完整启用
- v1 训练 2300+ iter（方向错误，作废）
- v2 训练 998 iter（方向正确），model_850 达 22.2% clean_right

### 4. v3 位姿调整（最新，未训练）
- j1=-1.743, j2=-3.368, j3=0.0, j4=-1.847, j5=3.030, j6=1.054
- 拍面朝 +X，拍杆平行 Y 轴
- 发球速度提升: vx=(-1.85,-0.90), vz=(0.08,0.42)

## 待继续
- 用户确认 v3 位姿后从头训练
- 右台反弹率目标：60%+
- 泛化性：扩大发球范围（左侧桌面反弹一次的合法发球）
- 过网速度限制 + 落点速度方向约束
