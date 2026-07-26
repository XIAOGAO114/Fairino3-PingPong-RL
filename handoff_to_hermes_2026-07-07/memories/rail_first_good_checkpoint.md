---
name: rail-first-good-checkpoint
description: model_448.pt is the first solid rail 7-DOF checkpoint with 97% clean_right and 0% illegal second hits
metadata: 
  node_type: memory
  type: project
  originSessionId: 0ad698e3-337a-45f5-9276-7a8aaf6fc94f
---

## model_448.pt — Rail 7-DOF First Clean Checkpoint

**Path:** `/home/glq/isaac_ws/test_isaac_rail/logs/rsl_rl/fairino3_rail_centerline_v1/2026-05-26_01-35-19_cont_v1_400iter/model_448.pt`

**Eval results (seed=42, 3500 steps, single env):**
- clean_over_net_per_hit: **97.0%**
- clean_right_per_hit: **97.0%**
- true_second_done_rate: **0.0%**
- clean_right: 19/20 trace episodes
- 1 failure: episode 14 missed the ball entirely (ball_out_of_bounds)

**Compared to best 6-DOF:**
- 6-DOF v2h: clean_right_per_hit=40.7%, true_second_done_rate=36.4%
- Rail 448: clean_right_per_hit=97.0%, true_second_done_rate=0.0%

**Why:** Rail Y-axis prismatic joint gives the arm lateral freedom. legal_post_hit_mask gates all post-hit rewards.

**Issue:** Some returns go too high (lobs), bad for future two-arm rally. Next: add low_return_height_bonus to reward returns below 3x net height.
