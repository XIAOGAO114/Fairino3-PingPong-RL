---
name: 2026-05-29-height-finetune
description: "Height penalty fine-tuning results: left arm h5 config (max_good_z=0.35, min_factor=0.2, height_weight=-18) and arm position +10cm adaptation"
metadata: 
  node_type: memory
  type: project
  originSessionId: b4204dcd-3d52-4262-90d7-c1952cfbc118
---

## 2026-05-29 高度微调与位置适配总结

### 高度奖励参数最终配置

经过 h2→h3→h4→h5 迭代，最终有效配置：

| 参数 | 原始 | 最终 |
|------|------|------|
| `height_factor` max_good_z | TABLE_TOP_Z + 0.50 (1.26m) | **TABLE_TOP_Z + 0.35 (1.11m)** |
| `height_factor` min_factor | 0.4 | **0.2** |
| `high_return_height_penalty` weight | -12.0 | **-18.0** |
| `high_return_height_penalty` max_height | TABLE_TOP_Z + 0.1525*3 (1.22m) | 不变 |

- h2: max_good_z=0.35, min_factor=0.2, weight=-12 → 高度惩罚 -0.22→-0.14, 成功率 84%
- h3: 加上 max_height 降到 1.07m → 成功率崩到 49%
- h4: weight 提到 -30 → 成功率崩到 74%
- **h5: weight=-18, max_height=1.22m（原始）→ 高度 -0.17, 成功率 81%，最佳平衡**

### 双臂位置 +10cm 适配

两个单臂项目的机器人位置都远离桌子 10cm：

| 臂 | 原始 | 改后 |
|----|------|------|
| 左 | x=-0.97 | **x=-1.07** |
| 右 | x=2.07 | **x=2.17** |

### 左臂 far10cm 训练结果

- 从 model_3139 (h5) 续训 300 iter
- 最终: `model_3438.pt` @ `2026-05-29_01-10-27_left_far10cm/`
- 成功率: **78%**, 关节极限: 5.2%, 高度: -0.15

### 右臂 far10cm 训练结果

- 从 model_2225 续训
- **关键发现:** ball_spin 惩罚 (weight=-0.3) 导致关节极限 25-35%，去掉后恢复正常
- 最终: `model_3872.pt` @ `2026-05-29_02-37-40_right_far10cm_v3/`
- 成功率: **72%**, 关节极限: 5.4%, 高度: -0.11
- v4 续训成功率在 65-70% 横盘，未突破

### 左右臂配置同步化

右臂原本缺少左臂的高度收紧参数，现已同步：
- rewards.py: max_good_z=0.35, min_factor=0.2
- env_cfg: high_return_height weight=-18
- env_cfg: 移除 ball_spin 惩罚（物理 angular_damping=0.05 已足够）
