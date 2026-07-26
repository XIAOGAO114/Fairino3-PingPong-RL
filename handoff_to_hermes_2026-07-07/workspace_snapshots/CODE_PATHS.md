# 工作空间代码快照 (2026-07-07)

本文件记录四个工作空间的关键源文件路径，方便 Hermes 快速定位代码。

## 注意
本 handoff 仓库的 `artifacts/task_source/` 是 2026-05-25 的快照，可能已过期。
实际代码在以下工作空间路径中。

---

## test_isaac_rail (左臂 7-DOF) — 主项目

### 任务配置
```
/home/glq/isaac_ws/test_isaac_rail/source/test_isaac_rail/test_isaac_rail/
├── assets/fairino3_v6_rail.py                    # URDF→USD 转换
└── tasks/manager_based/fairino3_rail_pingpong/
    ├── __init__.py                                 # Task 注册
    ├── fairino3_rail_pingpong_env_cfg.py           # ★ 主配置 (~500行)
    ├── agents/rsl_rl_ppo_cfg.py                    # ★ PPO 配置
    └── mdp/
        ├── _shared.py                              # ★ 状态机 (~400行)
        ├── rewards.py                              # ★ 奖励函数 (~600行)
        ├── terminations.py                         # ★ 终止条件
        ├── observations.py                         # ★ 观测定义
        └── events.py                               # 事件（发球）
```

### 入口脚本
```
/home/glq/isaac_ws/test_isaac_rail/scripts/rsl_rl/
├── train.py
└── play.py
```

### 训练输出
```
/home/glq/isaac_ws/test_isaac_rail/logs/rsl_rl/fairino3_rail_centerline_v1/
```

---

## test_isaac_rail_right (右臂 7-DOF)

### 任务配置
```
/home/glq/isaac_ws/test_isaac_rail_right/source/test_isaac_rail_right/test_isaac_rail_right/
├── assets/
└── tasks/manager_based/fairino3_rail_pingpong/
    ├── * (结构与左臂相同，home_side="right")
```

### 训练输出
```
/home/glq/isaac_ws/test_isaac_rail_right/logs/rsl_rl/fairino3_rail_right_centerline_v1/
```

---

## test_isaac_dual (双臂对打)

### 任务配置
```
/home/glq/isaac_ws/test_isaac_dual/source/test_isaac_dual/test_isaac_dual/
├── assets/
├── tasks/manager_based/fairino3_dual_pingpong/
│   ├── * (含 DualArmActor 模型 + _make_arm_rewards 工厂函数)
└── models/
    └── dual_arm_actor.py                           # ★ 双臂模型
```

### 训练输出
```
/home/glq/isaac_ws/test_isaac_dual/logs/rsl_rl/
├── fairino3_dual_centerline_v1/    (96 个子目录)
└── fairino3_rail_centerline_v1/    (5 个子目录)
```

---

## test_isaac (legacy 6-DOF)

### 任务配置
```
/home/glq/isaac_ws/test_isaac/source/test_isaac/test_isaac/tasks/manager_based/fairino3_pingpong/
```

### 训练输出
```
/home/glq/isaac_ws/test_isaac/logs/rsl_rl/   (12 个运行目录)
```

---

## 本 handoff 仓库快照

```
/home/glq/Desktop/project/pingpong_claudecode_handoff_2026-05-25/artifacts/
├── task_source/
│   ├── mdp/_shared.py          (2026-05-25 快照)
│   ├── mdp/rewards.py
│   ├── mdp/terminations.py
│   └── fairino3_pingpong_env_cfg.py
├── scripts/
│   └── eval_single_checkpoints.py
└── reference/
    ├── experiments.md
    └── evals/                  (评估 JSON)
```

## 当前未跟踪的改动

⚠️ 以下文件在早期版本中被修改过，当前状态未知：
- 左臂 env_cfg.py：2026-05-31 发现 6 处方向性 bug，需要验证是否已修复
- 双臂 train.py：添加了 `--unfreeze`, `--freeze-left`, `--right-std` 等自定义参数
- 双臂 events.py：双向镜像发球
- 双臂 _shared.py：严格交替 rally 计数器
