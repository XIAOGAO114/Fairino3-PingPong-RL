---
name: right-arm-project
description: test_isaac_rail_right — 右臂单臂训练项目，镜像左臂配置，v4达83%成功率
metadata: 
  node_type: memory
  type: project
  originSessionId: 1c5f5016-19f2-439d-9862-6cdff39cfb61
---

## 右臂单臂训练项目

**Workspace:** `/home/glq/isaac_ws/test_isaac_rail_right/`
**Task ID:** `Fairino3-PingPong-Rail-Right-Centerline-v0`
**最佳模型:** `model_1325.pt` @ `2026-05-28_02-25-56_right_v4_spin03_0528_0228/` — 83% 成功率

### 配置镜像

| 项目 | 左臂 | 右臂 |
|------|------|------|
| 机器人位置 | (-0.97, 0, 0.76) | (2.07, 0, 0.76) |
| 机器人旋转 | (0.707, 0, 0, 0.707) | (0.707, 0, 0, **-0.707**) |
| 发球方向 | vx 负向 | vx 正向 |
| 发球起点 | NET_X + 0.40 | NET_X - 0.40 |
| 目标台 | 右半台 | 左半台 |
| home_side | "left" | **"right"** |

### 关键修复（2026-05-28）

1. **发球验证方向** — `_serve_clears_net` 硬编码左臂方向，已改为双向支持
2. **ball_spin 惩罚** — weight=-0.3，温和抑制侧旋
3. **mdp 文件** — 从双臂版复制（支持 home_side 参数），默认值改为 "right"
4. **train.py/play.py** — 添加 `import test_isaac_rail_right.tasks`

### 训练命令

```bash
cd /home/glq/isaac_ws/test_isaac_rail_right
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Fairino3-PingPong-Rail-Right-Centerline-v0 \
  --num_envs 1024 --max_iterations 4000 --headless \
  --run_name right_<tag>
```

### 待解决
- y 轴角速度（侧旋）仍偏大，可能需要物理层面调整（球-拍摩擦/阻尼）
