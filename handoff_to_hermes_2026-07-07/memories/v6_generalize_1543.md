---
name: v6-generalize-1543
description: "model_1543.pt: v6 generalization training with expanded serve ranges, 97.1% clean_right"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0ad698e3-337a-45f5-9276-7a8aaf6fc94f
---

## model_1543.pt — v6 Generalization Training

**Path:** `/home/glq/isaac_ws/test_isaac_rail/logs/rsl_rl/fairino3_rail_centerline_v1/2026-05-26_11-34-16_v6_generalize_from1244_300/model_1543.pt`

**Eval (seed=42, 3500 steps):**
- clean_right_per_hit: 97.1%
- clean_over_net_per_hit: 100%
- true_second_done_rate: 0.0%
- trace 19/20 clean_right (1 robot_joint_limit)

**Config:** v5 height config (ideal_z=0.95, 3x penalty) + expanded serve ranges:
- pos: x=±0.20, y=±0.35, z=±0.08
- vel: vx=(-2.2,-0.7), vy=±0.45, vz=(0.05,0.55)
- max_clearance_height: TABLE_TOP_Z+0.80

**Issue:** Visual returns still look too high. Height rewards too weak vs success rewards.
**Next:** Bake height sensitivity into predicted_landing and right_table_bounce.
