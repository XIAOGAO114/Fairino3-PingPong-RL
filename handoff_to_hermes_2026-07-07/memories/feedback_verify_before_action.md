---
name: verify-before-action
description: 较大改动前先自查然后交给用户验证，获得允许后再执行
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2de03140-06d9-4131-82d0-23b8101f1508
---

对代码、配置、训练参数等做较大改动前，必须先自查一遍，然后把改动方案汇报给用户，等用户确认允许后再执行。

**Why:** 避免类似滑轨方向搞反的问题，减少无用训练和返工。

**How to apply:** 改动前先明确描述：要改什么、为什么、影响范围。自查后再提交用户审批。
