---
name: dual-arm-project
description: "test_isaac_dual workspace — two Fairino3+rail arms, scaffold ready, rally logic TBD"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0ad698e3-337a-45f5-9276-7a8aaf6fc94f
---

## Dual-Arm Project

**Workspace:** `/home/glq/isaac_ws/test_isaac_dual/`
**Task ID:** `Fairino3-PingPong-Dual-Centerline-v0`

**Setup:**
- Left robot: `robot` (legacy name), at x=-0.97, facing +x, j1=-1.743
- Right robot: `right_robot`, at x=2.07, facing -x (rot 180°), j1=1.743
- Action: 14-DOF (7 left + 7 right)
- Observation: 46-dim (both joints, paddles, ball world-frame, ball-to-paddle vectors)
- Expanded serve ranges from v6

**Current state:**
- Scene loads, smoke test passes
- Rewards/terminations mostly left-side only (copied from single-arm)
- Right robot has basic joint_limit + table_collision terminations
- Rally logic (alternating hits, symmetric rewards for right side) NOT yet implemented

**Next steps:**
- Add right-side paddle contact detection
- Make rewards symmetric (work for both sides)
- Implement rally state machine (left hits → right hits → left hits → ...)
- Train from scratch
