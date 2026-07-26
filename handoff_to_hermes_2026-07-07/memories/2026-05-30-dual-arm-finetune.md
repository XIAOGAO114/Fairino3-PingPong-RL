---
name: 2026-05-30-dual-arm-finetune
description: 双臂微调完整结论：v5右臂从零训练最优(left_bounce 5.09%峰值)，right_hit 0.017%在增长，需更长时间
metadata: 
  node_type: memory
  type: project
  originSessionId: dcab671d-69ba-44c2-a414-86c01203a2a1
---

## 双臂微调最终分析（2026-05-30）

### 五轮实验总结

| 版本 | 左臂 | 右臂 | std | 迭代 | left_bounce(end) | left_bounce(peak) | right_hit | right_collision |
|------|------|------|-----|------|:---:|:---:|:---:|:---:|
| v1 | 3438冻 | 4071训 | 0.4(错) | 100 | 0% | 0% | 0% | - |
| v3 | 1543冻 | 4071训 | 1.54冻 | 599 | 1.64% | 5.33% | 0% | -0.002 |
| v4 | 1543冻 | 4071训 | **0.92冻** | 100 | 2.30% | 4.89% | 0.002% | **-0.033** |
| **v5-100** | 1543冻 | **随机** | **0.92冻** | 100 | **3.68%** | 4.54% | 0.005% | **0.000** |
| **v5-600** | 1543冻 | **随机** | **0.92冻** | 599 | **2.92%** | **5.09%** | **0.017%** | **0.000** |

### 关键发现

1. **v5 最优**: 右臂从零训练 + 低 std(0.3)→action_std=0.92 + 左臂 model_1543 冻结
2. **右臂正在学习**: right_hit 600轮增长3.4倍 (0.005%→0.017%)，零碰撞
3. **左臂峰值 5.09%**: 创所有版本最高
4. **峰值后退化**: 普遍存在，中期峰值后回落，可能需学习率衰减
5. **model_4071 不适合双臂**: v3/v4 的 model_4071 碰撞多(v4 恶化14x)、hit率零

### 代码改动

- `train.py`: `--unfreeze` + `--freeze-left` + `--right-std X`
- `rsl_rl_ppo_cfg.py`: entropy_coef=0.0005, lr=1e-4

### 训练命令(v5)

```bash
# 初始训练
python train.py --task Fairino3-PingPong-Dual-Centerline-v0 \
  --num_envs 512 --max_iterations 100 --headless \
  --run_name v5_scratch_right \
  --pretrained_checkpoint <model_1543.pt> \
  --unfreeze --freeze-left --right-std 0.3

# 续训
python train.py --task Fairino3-PingPong-Dual-Centerline-v0 \
  --num_envs 512 --max_iterations 500 --headless \
  --run_name v5_cont_500 \
  --load_run <v5_run> --checkpoint model_99.pt \
  --resume --unfreeze --freeze-left --right-std 0.3
```

### Best checkpoint
`2026-05-30_18-15-26_v5_cont_500/model_598.pt`
