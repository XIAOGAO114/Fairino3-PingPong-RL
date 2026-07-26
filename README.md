# 🏓 Fairino3 双臂乒乓球对打 — 强化学习

> 哈尔滨工业大学 大创项目 | Isaac Lab 0.54.3 + Isaac Sim 4.5.0 + RSL RL PPO

## Demo

![双臂对打 Rally](videos/demo_dual_arm_rally.gif)

## 项目概述

使用 NVIDIA Isaac Lab 训练 **Fairino3 工业机械臂** 打乒乓球。最终目标是两个 7-DOF 机械臂（左臂+右臂）持续对打（rally）。

```
6-DOF 固定底座 → 7-DOF + 导轨 (左臂) → 7-DOF 右臂 → 双臂对打
```

| 阶段 | 任务 ID | 成果 |
|------|--------|------|
| 左臂 7-DOF | `Fairino3-PingPong-Rail-Centerline-v0` | **97%** clean_right, 0% illegal second hit |
| 右臂 7-DOF | `Fairino3-PingPong-Rail-Right-v0` | 84% 成功率，视觉完美 |
| 双臂对打 | `Fairino3-PingPong-Dual-Centerline-v0` | Rally 初步实现 |

## 环境要求

| 组件 | 版本 |
|------|------|
| Isaac Sim | **4.5.0** |
| Isaac Lab | **0.54.3** |
| Python | ≥ 3.10 |
| RSL RL | (Isaac Lab 内置) |
| GPU | NVIDIA (Isaac Sim 依赖) |

> ⚠️ 版本必须匹配。Isaac Lab 0.54.x 对应 Isaac Sim 4.5.0，其他版本组合未经测试。

## 快速开始

### 1. 安装 Isaac Lab

参考 [Isaac Lab 官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)，确保 Isaac Sim 4.5.0 + Isaac Lab 0.54.3 环境可用。

### 2. 安装本扩展

```bash
cd source/fairino3_pingpong
pip install -e .     # 或: python -m pip install -e .
```

### 3. 验证安装

```bash
# 列出已注册任务，应能看到 Fairino3 相关条目
python -c "import isaaclab_tasks; isaaclab_tasks.utils.import_packages('fairino3_pingpong.tasks')"
```

## 训练

### 单臂左臂（7-DOF + 导轨）

```bash
python scripts/train.py \
  --task Fairino3-PingPong-Rail-Centerline-v0 \
  --num_envs 512 --max_iterations 500 --headless \
  --run_name left_arm_exp1
```

### 单臂右臂（镜像）

```bash
python scripts/train.py \
  --task Fairino3-PingPong-Rail-Right-v0 \
  --num_envs 512 --max_iterations 500 --headless \
  --run_name right_arm_exp1
```

### 双臂对打

```bash
python scripts/train.py \
  --task Fairino3-PingPong-Dual-Centerline-v0 \
  --num_envs 512 --max_iterations 1000 --headless \
  --run_name dual_arm_exp1
```

> 建议先用 `--num_envs 16 --max_iterations 1` 做 smoke test 确认环境正常。

## 推理 & 可视化

```bash
# 单臂推理（GUI 模式，可观察击球效果）
python scripts/play.py \
  --task Fairino3-PingPong-Rail-Centerline-v0 \
  --num_envs 1 --real-time \
  --checkpoint checkpoints/left_v9_model_6034.pt

# 双臂对打推理
python scripts/play.py \
  --task Fairino3-PingPong-Dual-Centerline-v0 \
  --num_envs 1 --real-time \
  --checkpoint <path/to/dual_checkpoint.pt>
```

## 预训练模型

| 文件 | 用途 | 指标 |
|------|------|------|
| `checkpoints/left_v9_model_6034.pt` | 左臂 v9（最优） | 97% clean_right |
| `checkpoints/right_v9_model_5472.pt` | 右臂 v9（最优） | 84% 成功率 |
| `checkpoints/left_scratch_model_499.pt` | 左臂从头训练基线 | 基线参考 |

## 核心技术设计

### 动作空间

**7-DOF**: `rail_y` (Y轴导轨 scale=0.04) + `j1`–`j6` (旋转关节 scale=0.28)

增加 Y 轴 prismatic joint 是解决 6-DOF 方案 lateral freedom 不足的关键——机械臂可以横向移动覆盖更广击球区域。

### 状态机 (`mdp/_shared.py`)

击球检测基于物理事件而非时间步计数：

1. **`first_return_hit_event()`** — 检测有效击球：球先落左台 → 球接近(vx < -threshold) → 接触 + 速度反向
2. **`illegal_second_hit_event()`** — 检测二次触碰：follow-through 后的额外接触（基于速度增量 + 方向判断）
3. **`legal_post_hit_mask()`** — **核心门控**：`has_return_hit & ~has_illegal_second_hit`，阻止二次触碰后获得任何正向奖励
4. **`clean_over_net_event()`** — 球过网且高于网高
5. **`left_table_bounce_event()`** — 球落左台（有效击球前提）

### 奖励设计 (`mdp/rewards.py`)

- 击球前奖励：以 incoming phase 和左台弹跳为门控
- 击球后奖励：以 `legal_post_hit_mask()` 为门控，确保非法二次触碰切断所有正向 shaping
- 惩罚项：二次接触、击球后球拍追踪球、关节限位、球台碰撞、出界
- 终止成功：`right_table_bounce_reward` + `require_clean_over_net=True`

### 双臂架构 (`tasks/fairino3_dual/models/`)

- **DualArmActor**: 两个独立 MLP（左臂/右臂各自策略网络）+ 共享 Value 网络
- **镜像奖励**: `_make_arm_rewards()` 通过 `home_side` 参数生成左右对称奖励函数
- **opponent_compatible 奖励**: 回球适配对手训练分布，提升 rally 成功率

## 仓库结构

```
├── README.md                    # 本文件
├── source/fairino3_pingpong/    # Isaac Lab 扩展包（可 pip install -e .）
│   ├── config/extension.toml    # 扩展元数据
│   ├── setup.py
│   ├── fairino3_pingpong/
│   │   ├── assets/              # Fairino3 URDF + USD 模型
│   │   ├── tasks/fairino3_rail/ # 单臂训练配置（左臂/右臂）
│   │   └── tasks/fairino3_dual/ # 双臂对打配置
├── scripts/                     # 训练/评估脚本
│   ├── train.py                 # 训练入口
│   ├── play.py                  # 推理可视化
│   └── eval_rally.py            # Rally 评估
├── checkpoints/                 # 预训练模型权重 (.pt)
├── graph/                       # 评估图表和实验结果
├── videos/                      # 演示视频
└── docs/                        # 补充文档（项目演进、关键经验等）
    └── handoff_to_hermes_2026-07-07/
```

## License

MIT
