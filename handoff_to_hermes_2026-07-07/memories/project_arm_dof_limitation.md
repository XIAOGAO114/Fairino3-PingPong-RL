---
name: arm-dof-limitation
description: "6-DOF centerline arm lacks lateral freedom, root cause of double-hit problem"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2de03140-06d9-4131-82d0-23b8101f1508
---

单臂 Centerline 配置下，机械臂固定在球台中心线 (x=0.55 - table_half_length, y=0)，只有 6 个旋转关节。

**Why:** 球从右侧发来，拍面需要在拦截后立刻向右侧打出回球，然后迅速脱离球。但 6-DOF 固定基座的机械臂缺乏平行网方向的自由度（如基座横向平移或额外的 yaw 自由度），导致拍面击球后 natural follow-through 轨迹容易停留在球附近，产生二次触碰。

**How to apply:** 后续若要从根本上解决连击，应优先考虑：
- 机械层面：增加基座横向自由度
- 策略层面：训练专门的"击后快速后撤"动作（post_hit_retreat 惩罚已有但不够）
- 不要过度依赖 reward shaping 来弥补物理自由度不足
