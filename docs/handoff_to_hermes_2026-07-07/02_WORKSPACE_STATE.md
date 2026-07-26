# 工作空间当前状态 (2026-07-07)

## 概览

| 工作空间 | 路径 | 状态 | 训练运行数 | 最好模型 |
|---------|------|------|:---------:|---------|
| **rail 左臂** | `/home/glq/isaac_ws/test_isaac_rail/` | ✅ 完成 | 92 | model_1543.pt (97.1%) |
| **rail 右臂** | `/home/glq/isaac_ws/test_isaac_rail_right/` | ✅ 完成 | 40 | model_2225.pt (84%) |
| **双臂** | `/home/glq/isaac_ws/test_isaac_dual/` | 🔄 进行中 | 96+5 | merged_1543_2225 |
| **legacy** | `/home/glq/isaac_ws/test_isaac/` | 📦 历史参考 | 12 | model_2946.pt (40.7%) |

## test_isaac_rail（主项目 — 左臂 7-DOF）

```
路径: /home/glq/isaac_ws/test_isaac_rail/
Task ID: Fairino3-PingPong-Rail-Centerline-v0
机器人位置: (-0.97, 0, 0.76)
旋转: (0.707, 0, 0, 0.707) — 绕 Z +90°
初始关节: j1=-1.743, j2=-2.94, j3=0.0, j4=-1.847, j5=2.7, j6=1.054
动作空间: 7-DOF (rail_y + j1-j6)
Python 环境: env_isaaclab
```

### 训练历史
- v1: 方向错误（作废）
- v2: 正确方向，model_850 达 22.2% clean_right
- v5: 底座降桌高，model_448 达 97% clean_right
- v6: 扩展发球范围泛化，model_1543 达 97.1%
- v7b: 高度缩放奖励，model_1842 达 94.7%

### 关键配置文件
- `source/test_isaac_rail/test_isaac_rail/tasks/manager_based/fairino3_rail_pingpong/fairino3_rail_pingpong_env_cfg.py`
- `source/test_isaac_rail/test_isaac_rail/tasks/manager_based/fairino3_rail_pingpong/mdp/_shared.py`
- `source/test_isaac_rail/test_isaac_rail/tasks/manager_based/fairino3_rail_pingpong/mdp/rewards.py`

## test_isaac_rail_right（右臂 7-DOF）

```
路径: /home/glq/isaac_ws/test_isaac_rail_right/
Task ID: Fairino3-PingPong-Rail-Right-Centerline-v0
机器人位置: (2.07, 0, 0.76)
旋转: (0.707, 0, 0, -0.707) — 绕 Z -90°
home_side: "right"
发球方向: vx 正向
```

### 关键区别（vs 左臂）
- 机器人位置对称于 NET_X=0.55（距离相等，1.52m）
- 旋转方向相反（-0.707 vs 0.707）
- 发球从右侧发出

## test_isaac_dual（双臂对打）

```
路径: /home/glq/isaac_ws/test_isaac_dual/
Task ID: Fairino3-PingPong-Dual-Centerline-v0
动作空间: 14-DOF (7 左 + 7 右)
观测空间: 46-dim
```

### 架构特点
- **DualArmActor**: 两个独立 MLP (32→7) + 共享 14-dim distribution
- **工厂函数奖励**: `_make_arm_rewards()` 通过 home_side 生成镜像奖励
- **发球**: 仅左臂发球，`_serve_clears_net` 验证已移除
- **关键参数**: init_std=0.15, soft关节角 j2=-2.94/j5=2.7

### 已修复的关键 Bug
1. `actor_critic` → `actor` 属性名错误
2. `mlp.` 前缀未去除导致权重加载失败
3. 左臂 env_cfg 有 6 处方向性错误（RIGHT_MID_TARGET_X 等）

### 当前瓶颈
- 双臂 rally 率极低 (~0.04%)
- opponent_compatible 单臂天花板 ~1%
- 右臂在双臂环境中难以学习

## test_isaac（legacy 6-DOF）

```
路径: /home/glq/isaac_ws/test_isaac/
Git 仓库: ✅
训练运行: 12 个
```

### 历史意义
- 最早的实验基地，证明了 6-DOF 的物理限制
- v2i legal dense gate 在此首次实现
- 12 次训练运行提供丰富的对比数据

## .claude/ 和 CLAUDE.md

- **所有四个工作空间都没有 `.claude/` 目录**
- **所有四个工作空间都没有 `CLAUDE.md` 文件**
- 只有本 handoff 仓库有 `CLAUDE.md`
- 记忆系统位于 `~/.claude/projects/.../memory/`
