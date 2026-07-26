---
name: lesson-reward-simplicity
description: Clean reward design + mechanical DOF matters more than training scale
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8d173e72-1f62-47c7-aa8e-f0a8ec5fb2c8
---

精简奖励 + 机械自由度 > 训练规模。

**事实:** 旧 6-DOF 项目堆了 28 项 reward，数千 iter 最好 40%。滑轨 7-DOF 项目砍到 21 项 reward，底座降桌高 + j5 不超限，449 iter 达 78.6% right_table_bounce，second_paddle_contact=0。

**Why:** 冗余奖励互相干扰，特别是多个功能重叠的项（target_landing vs predicted_landing、ball_over_net vs net_direction/height/speed、centerline_x 在滑轨场景下无意义）。砍掉后 critic 信号更清晰。机械层面，滑轨给了关键的横向自由度，从根本上解决了连击问题。

**How to apply:** 以后设计 reward 时，优先检查是否有功能重叠的项直接砍掉。奖励权重 <1 的项大概率是噪声。机械问题优先用机械解决，不要纯靠 reward shaping 补。
