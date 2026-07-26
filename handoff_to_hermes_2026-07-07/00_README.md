# Handoff to Hermes — 2026-07-07

## 这是什么

本文件夹是 Fairino3 乒乓球 RL 项目的完整转接包，从当前 Claude Code 会话转交给 Hermes（新会话）。

**转接日期:** 2026-07-07
**项目周期:** 2026-05-25 ~ 2026-07-07（约 6 周）
**学生:** 哈尔滨工业大学（HIT）智能机器人技术本科

## 快速导航

| 文件 | 内容 |
|------|------|
| `01_PROJECT_OVERVIEW.md` | 项目总览：是什么、做了什么、为什么 |
| `02_WORKSPACE_STATE.md` | 四个工作空间的当前状态 |
| `03_BEST_CHECKPOINTS.md` | 所有最佳模型 checkpoint 路径和指标 |
| `04_KEY_LEARNINGS.md` | 关键经验教训和设计原则 |
| `05_COMMANDS_REFERENCE.md` | 完整命令参考 |
| `06_CURRENT_ISSUES.md` | 当前未解决的问题和待办事项 |
| `memories/` | 所有项目记忆文件（19个）的副本 |
| `configs/` | 关键配置参数速查 |

## 最重要的 3 件事

1. **主项目是 `test_isaac_rail`（左臂 7-DOF + 导轨）**，已成功训练出 97% 成功率模型
2. **双臂对打 (`test_isaac_dual`) 已搭建完成**，但 rally 率极低 (~0.04%)，需要进一步工作
3. **学生学习模式已定义在 `LEARNING_HANDOFF.md`**，7 步教学流程必须严格遵守

## 环境

- **Python:** `/home/glq/.conda/envs/env_isaaclab/bin/python`
- **Conda env:** `env_isaaclab`
- **OS:** Ubuntu Linux 6.8.0-124-generic
- **GPU:** NVIDIA (Isaac Sim requires)
- **Isaac Lab:** NVIDIA Isaac Lab (RL training framework)
- **RL Algorithm:** RSL RL PPO
