---
name: v7b-height-scale-1842
description: "model_1842.pt: height-scaled success rewards (soft), 94.7% clean_right, visually improved height"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0ad698e3-337a-45f5-9276-7a8aaf6fc94f
---

## model_1842.pt — v7b Height-Scaled Rewards

**Path:** `/home/glq/isaac_ws/test_isaac_rail/logs/rsl_rl/fairino3_rail_centerline_v1/2026-05-26_12-15-48_v7b_soft_height_from1543_300/model_1842.pt`

**Eval (seed=42):**
- clean_right_per_hit: 94.7%
- trace 20/20 clean_right
- true_second: 0.0%
- mean_vx: 0.932, mean_x: 0.903

**Key change:** predicted_landing and right_table_bounce multiplied by `height_factor`:
- max_good_z = TABLE_TOP_Z + 0.50 (1.26m), min_factor=0.4
- Ball at 1.26m: full reward. Ball at 2.0m: 63% reward.

**Prior versions:**
- v7 (harsh): max_good_z=0.30, min_factor=0.2 — collapsed to 75% clean_right
- v5_cont (1244): no height scaling, ideal_z=0.95 — 97% clean_right but visually high
- v6 (1543): expanded serve ranges — 97.1%

**Config:** ideal_z=NET_HEIGHT+0.04 (0.95m), 3x penalty, expanded serve ranges, height_factor on predicted_landing + right_table_bounce
