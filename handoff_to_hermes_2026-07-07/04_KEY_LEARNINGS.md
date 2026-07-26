# 关键经验教训

## 设计原则

### 1. 精简奖励 + 机械自由度 > 训练规模
- 旧 6-DOF: 28 项 reward → 最好 40%
- 新 7-DOF: 21 项 reward → 449 iter 达 78.6%
- **Why:** 冗余奖励互相干扰，砍掉后 critic 信号更清晰
- **规则:** 权重 <1 的奖励项大概率是噪声，直接砍掉

### 2. 机械问题优先用机械解决
- 6-DOF 连击的根因是缺乏横向自由度，不是 reward shaping 不够
- 导轨 (rail_y) 从根本上解决了 illegal second hit 问题
- 不要过度依赖 reward shaping 弥补物理自由度不足

### 3. 改动前先自查再汇报
- 任何较大改动前必须：描述改什么、为什么、影响范围
- 自查后再提交用户审批
- **Why:** 避免滑轨方向搞反等低级错误，减少无用训练和返工

## 关键技术教训

### Reward 设计
- `legal_post_hit_mask()` 门控模式：所有 post-hit 正向奖励必须通过此 mask
- 密集奖励需要门控，否则非法行为也能获得正向信号
- 高度奖励 vs 成功率奖励的平衡很微妙（v7: 太严→75%，v7b: 适度→94.7%）

### 训练稳定性
- `entropy_coef=0.0015` 导致 action_std 持续上升（0.85→1.19），成功率横盘
- `init_std` 对关节极限率影响巨大（0.15 vs 0.7 → <1% vs 55-78% joint limit）
- 峰值后退化普遍存在，可能需要学习率衰减

### 双臂特定教训
- 单臂模型直接合并到双臂环境，表现大幅下降（97%→3%）
- observation 分布偏移是主因
- 右臂在双臂中从零训练困难（发球分布不匹配）
- opponent_compatible 单臂天花板 ~1%，真实 rally 效果必须在双臂合并后评估

### Bug 教训
- `actor_critic` vs `actor` 属性名：RSL-RL PPO 中模型属性叫 `actor` 不是 `actor_critic`
- `mlp.` 前缀：单臂 checkpoint key 带 `mlp.` 前缀，DualArmActor 的 MLP 不带
- 发球方向必须与 `_serve_clears_net` 的 vx 检查一致
- 训练启动后立即验证 log 中是否有 `Learning iteration` 行
- Isaac Lab 训练 fork 后父进程立即退出，需跟踪日志而非进程
- `--checkpoint` 参数只需文件名，配合 `--load_run` 指定目录

## 物理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `angular_damping` | 0.05 | 球旋转在飞行中自然衰减，所有项目已同步 |
| 球-拍恢复系数 | (URDF 默认) | 影响回球速度 |
| 仿真子步 | (Isaac Lab 默认) | 影响接触检测精度 |

## 用户偏好

- 关心视觉正确性 > 训练指标
- 如果 play 看起来不对，相信视觉观察
- 信任 `clean_right_per_hit`、`clean_over_net_per_hit`、`true_second_done_rate` 作为主要指标
- 不信任旧 `right`/`legal_right` 指标
