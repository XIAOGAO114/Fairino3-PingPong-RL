---
name: dual-arm-bugfixes
description: 双臂项目两个关键 bug：actor_critic→actor、load_left_actor 的 mlp. 前缀缺失
metadata: 
  node_type: memory
  type: project
  originSessionId: b4204dcd-3d52-4262-90d7-c1952cfbc118
---

## 双臂项目 Bug 修复（2026-05-29 发现）

### Bug 1: actor_critic 属性不存在

**文件:** `test_isaac_dual/scripts/rsl_rl/train.py`

RSL-RL PPO 中模型属性叫 `actor` 而非 `actor_critic`。原代码：
```python
actor = getattr(runner.alg, "actor_critic", None)  # 永远返回 None！
```
修复为：
```python
actor = getattr(runner.alg, "actor", None)
```
导致左右臂预训练权重从未加载，双臂都用随机初始化。

### Bug 2: load_left_actor/load_right_actor key 不匹配

**文件:** `test_isaac_dual/.../models/dual_arm_actor.py`

单臂 checkpoint 中 MLP 的 key 是 `mlp.0.weight`，但 `DualArmActor.left_actor` (MLP) 期望 `0.weight`。
原代码：
```python
left_state[k] = v  # 保留 "mlp." 前缀
```
修复为：
```python
left_state[k[4:]] = v  # 去掉 "mlp." 前缀
```

两个 bug 都已在 `2026-05-29_01-00-46_fixed3_h5_2225` 合并 checkpoint 中验证通过。
