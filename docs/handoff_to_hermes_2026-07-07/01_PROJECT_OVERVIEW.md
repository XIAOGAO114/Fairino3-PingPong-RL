# 项目总览

## 我们在做什么

用 NVIDIA Isaac Lab 训练 Fairino3 工业机械臂打乒乓球。最终目标是两个机械臂（左臂+右臂）能持续对打（rally）。

## 项目演进

```
6-DOF 固定底座 (legacy) → 7-DOF + 导轨 (左臂) → 7-DOF 右臂 → 双臂对打
```

### 阶段 1: Legacy 6-DOF (test_isaac)
- **问题:** 6 个旋转关节，固定底座在球台中心线
- **根因:** 缺乏横向自由度（Y轴），导致击球后 follow-through 产生二次触碰（illegal second hit）
- **最好成绩:** 40.7% clean_right_per_hit, 36.4% true_second_done_rate
- **教训:** 机械问题不能纯靠 reward shaping 解决

### 阶段 2: Rail 7-DOF 左臂 (test_isaac_rail) ★ 主项目
- **改进:** 增加 Y 轴 prismatic joint (rail_y)，7 个自由度
- **成果:** `model_448.pt` 达到 97% clean_right, 0% illegal second hits
- **关键设计:** `legal_post_hit_mask()` — 门控所有 post-hit 奖励
- **训练位置:** x=-0.97, 面向 +X 方向

### 阶段 3: Rail 7-DOF 右臂 (test_isaac_rail_right)
- **镜像左臂配置:** x=2.07, 面向 -X 方向（180° 旋转）
- **成果:** `model_2225.pt` 视觉完美，84% 训练成功率
- **关键修复:** angular_damping=0.05 物理抑制球旋转

### 阶段 4: 双臂对打 (test_isaac_dual)
- **架构:** DualArmActor（两个独立 MLP + 共享 distribution）
- **工厂函数:** `_make_arm_rewards()` 通过 home_side 参数生成左右镜像奖励
- **主要创新:** opponent_compatible 奖励 — 回球适配对手训练分布
- **当前状态:** 单臂合并模型 rally ~0.04%，需要进一步工作

## 核心技术栈

| 层面 | 技术 |
|------|------|
| 仿真引擎 | NVIDIA Isaac Sim (PhysX) |
| RL 框架 | Isaac Lab ManagerBasedRLEnv |
| RL 算法 | RSL RL PPO (Actor-Critic) |
| 机器人 | Fairino3 工业臂 (URDF → USD) |
| 动作空间 | 7-DOF: rail_y (scale=0.04) + j1-j6 (scale=0.28) |

## 项目文件结构

```
/home/glq/isaac_ws/
├── test_isaac/              # legacy 6-DOF (历史参考)
├── test_isaac_rail/         # ★ 主项目：7-DOF 左臂
├── test_isaac_rail_right/   # 7-DOF 右臂（镜像）
└── test_isaac_dual/         # 双臂对打

每个项目结构相同:
project/
├── source/<pkg>/<pkg>/tasks/manager_based/<task>/
│   ├── <task>_env_cfg.py    # 主配置（场景、观测、动作、奖励、终止）
│   ├── mdp/_shared.py       # 状态机（击球检测、事件管理）
│   ├── mdp/rewards.py       # 奖励函数（~20项）
│   ├── mdp/terminations.py  # 终止条件
│   ├── mdp/observations.py  # 观测定义
│   ├── mdp/events.py        # 事件（发球等）
│   └── agents/rsl_rl_ppo_cfg.py  # PPO 超参数
├── scripts/rsl_rl/train.py  # 训练入口
├── scripts/rsl_rl/play.py   # 可视化推理
└── logs/rsl_rl/             # 训练输出和模型

本 handoff 仓库:
/home/glq/Desktop/project/pingpong_claudecode_handoff_2026-05-25/
├── HANDOFF.md               # 原始转接文档（2026-05-25）
├── CLAUDE.md                # Claude Code 项目指令
├── LEARNING_HANDOFF.md      # 学生学习计划（7步教学法）
├── artifacts/               # 源代码快照
│   ├── task_source/         # mdp/ + env_cfg 快照
│   ├── scripts/             # eval 脚本
│   └── reference/           # 评估 JSON + 实验记录
├── output/                  # 双臂实验总结 + 演示视频
└── handoff_to_hermes_2026-07-07/  # ← 本转接包
```
