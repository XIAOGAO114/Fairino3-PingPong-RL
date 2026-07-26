---
name: 2026-05-30-dual-arm-investigation
description: 2026-05-30 双臂右臂问题排查：多轮训练无效，最终定位为单臂训练发球参数与双臂环境不匹配
metadata: 
  node_type: memory
  type: project
  originSessionId: e1f40d3e-de12-44e8-b39d-84172d1c91f3
---

## 双臂右臂问题排查（2026-05-30）

### 问题
右臂在双臂环境中始终无法击球（right_first_return_hit ~0.0001-0.0003），左臂正常（~0.004）。跨越 400+ 轮持续无改善。

### 排查过程

| 实验 | 改动 | right_hit | 结论 |
|------|------|-----------|------|
| v1 | 基础 100 轮 | 0.0003 | 有问题 |
| v2 | 续训 300 轮 | 0.0003 | 不是时间问题 |
| v3 | 移除 table_collision 终止 | 0.0001 | 无关 |
| v4 | 移除 early_hit_penalty | 0.0010* | 短暂改善后回落 |
| v5 | std 冻结 0.3 | 0.0001 | 探索不足 |
| v6 | std 初始 0.5 | 0.0001 | 无关 |
| v7 | 续训 500 轮 | 0.0003 | 横盘 |
| v8 | 匹配发球参数 | 0.0003 | 无关 |

*峰值，后续回落

### 根因定位

1. **observation 验证通过** — 诊断脚本确认右臂 observation 数值正确，坐标变换正确
2. **动作值验证通过** — play 中打印的值合理
3. **发球参数不匹配验证不成立** — v8 匹配了右臂单臂训练参数（max_clearance_height=1.80），仍无改善
4. **根本原因**：右臂单臂模型 model_3872 训练时的发球分布与双臂环境不同。即使参数"匹配"，镜像发球的实现导致实际球轨迹分布仍有差异。解决办法是让右臂用双臂一致的扁平发球重训。

### 右臂扁平发球重训

- 从 model_3872 续训，更新 events.py：max_clearance_height=1.42，net_clearance=0.04
- 200 轮后 model_4071：first_return_hit=5.4%，right_table_bounce=83%
- 合并 checkpoint：merged_3438_4071（左 model_3438 + 右 model_4071）
- **待验证**：在双臂环境中 play 检验

### 关键修改文件

双臂项目：
- `events.py`：双向镜像发球，max_clearance_height=1.42，net_clearance=0.04
- `_shared.py`：严格交替 rally 计数器 + 上升沿触发 reward
- `terminations.py`：ball_resting_on_table（球停在桌上 45 步则终止）
- `env_cfg.py`：移除 table_collision 终止（只留罚分 -20），移除 early_hit_penalty，first_return_hit 20→20，落台奖励 250→40，高度惩罚 -12→-18，rally_exchange 共享 8.0
- `train.py`：添加 `--unfreeze` 标志

右臂项目：
- `events.py`：更新为扁平发球参数（max_clearance_height=1.42，net_clearance=0.04）

### 结论（用户确认）

**右臂能打了！** 扁平发球重训（model_4071）解决了根因。单臂发球参数必须与双臂环境匹配。

### 待办

1. 双臂微调：从 merged_3438_4071 起步，std 锁死，`--unfreeze` 只训 MLP
2. 预期难点：std 易涨、左臂可能退化、rally 行为需要两个臂同时学习
