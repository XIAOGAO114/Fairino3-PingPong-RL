# 最佳模型 Checkpoint 汇总

## 单臂模型

### 左臂 — model_1543.pt (v6 Generalize) ★ 推荐

```
路径: /home/glq/isaac_ws/test_isaac_rail/logs/rsl_rl/fairino3_rail_centerline_v1/
      2026-05-26_11-34-16_v6_generalize_from1244_300/model_1543.pt
```
| 指标 | 值 |
|------|-----|
| clean_right_per_hit | **97.1%** |
| clean_over_net_per_hit | 100% |
| true_second_done_rate | 0.0% |
| trace clean_right | 19/20 |
| 训练位置 | x=-0.97, y=0, z=0.76 |

**配置:** v5 height (ideal_z=0.95, 3x penalty) + 扩展发球范围 + angular_damping=0.05
**问题:** 回球偏高（高度奖励太弱 vs 成功率奖励）

### 左臂 — model_1842.pt (v7b Height Scaled)

```
路径: /home/glq/isaac_ws/test_isaac_rail/logs/rsl_rl/fairino3_rail_centerline_v1/
      2026-05-26_12-15-48_v7b_soft_height_from1543_300/model_1842.pt
```
| 指标 | 值 |
|------|-----|
| clean_right_per_hit | **94.7%** |
| trace clean_right | 20/20 |
| true_second | 0.0% |

**改进:** height_factor 乘到 predicted_landing 和 right_table_bounce（max_good_z=1.26m, min_factor=0.4）

### 左臂 — model_448.pt (Rail First Clean)

```
路径: /home/glq/isaac_ws/test_isaac_rail/logs/rsl_rl/fairino3_rail_centerline_v1/
      2026-05-26_01-35-19_cont_v1_400iter/model_448.pt
```
第一个干净的 7-DOF checkpoint，97% clean_right，0% illegal second。
证明导轨从根本上解决了连击问题。

### 右臂 — model_2225.pt (v6 Angular Damping) ★ 推荐

```
路径: /home/glq/isaac_ws/test_isaac_rail_right/logs/rsl_rl/fairino3_rail_right_centerline_v1/
      2026-05-28_15-08-48_right_v6_angdamp_cont1_1525/model_2225.pt
```
| 指标 | 值 |
|------|-----|
| 训练成功率 | **84.4%** |
| 视觉表现 | **完美 — 所有回球过网落台** |
| 训练位置 | x=2.07, y=0, z=0.76 |

**注意:** 80% 训练指标被 left_table_bounce 检测噪声拉低。

### 右臂 — model_1325.pt (v4)

```
路径: /home/glq/isaac_ws/test_isaac_rail_right/logs/rsl_rl/fairino3_rail_right_centerline_v1/
      2026-05-28_02-25-56_right_v4_spin03_0528_0228/model_1325.pt
```
83% 成功率，无 angular_damping。

## 双臂合并模型

### merged_1543_2225（最终保底）

```
路径: /home/glq/isaac_ws/test_isaac_dual/logs/rsl_rl/fairino3_dual_centerline_v1/
      2026-05-30_20-28-16_merged_1543_2225_final/model_0.pt
```
左臂 model_1543 + 右臂 model_2225，冻结评估：

| 指标 | 值 |
|------|-----|
| left_bounce | 2-3.5% |
| left_hit | 0.4-0.5% |
| right_hit | 0.01-0.05% |
| rally_exchange | 0.01% |
| episode_length | 260-300 |

### merged_2541_2225

```
路径: /home/glq/isaac_ws/test_isaac_dual/logs/rsl_rl/fairino3_dual_centerline_v1/
      merged_2541_2225/model_0.pt
```
左臂 v7b 高度压低版 (84%) + 右臂 v6 (83%)

## Compat 训练模型

| 用途 | 臂 | 成功率 | Compat | 路径片段 |
|------|:--:|:---:|:---:|------|
| v2 peak | 左 | 73.2% | 25.9% | `.../2026-05-30_21-58-16_compat_v2_w80/model_2150.pt` |
| R-lean peak | 右 | 62.4% | **45.2%** | `.../2026-05-31_00-22-14_compat_peak_lean_v2/model_2675.pt` |
| 从头训练 | 左 | 85.7% | ~1% | `.../2026-05-31_17-30-40_left_from_scratch_2k/model_1999.pt` |

## Legacy 6-DOF 模型

| 模型 | 版本 | clean_right_per_hit | true_second_done |
|------|------|:---:|:---:|
| model_2797.pt | v2g baseline | 17.4% | 60.0% |
| model_2946.pt | v2h clean gate | 57.7% (trace) / 40.7% (batch) | 26.9% / 36.4% |

## Play 命令（可视化验证）

```bash
# 左臂单臂
cd /home/glq/isaac_ws/test_isaac_rail
python scripts/rsl_rl/play.py \
  --task Fairino3-PingPong-Rail-Centerline-v0 \
  --num_envs 1 --real-time \
  --checkpoint <左臂模型路径>

# 右臂单臂
cd /home/glq/isaac_ws/test_isaac_rail_right
python scripts/rsl_rl/play.py \
  --task Fairino3-PingPong-Rail-Right-Centerline-v0 \
  --num_envs 1 --real-time \
  --checkpoint <右臂模型路径>

# 双臂对打
cd /home/glq/isaac_ws/test_isaac_dual
python scripts/rsl_rl/play.py \
  --task Fairino3-PingPong-Dual-Centerline-v0 \
  --num_envs 1 --real-time \
  --checkpoint <合并模型路径>
```
