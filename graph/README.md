# Dual-Arm Ping-Pong RL — 图表文档

> 最后更新: 2026-07-13
> 生成: `conda run -n env_isaaclab python graph/generate_charts.py`

---

## 文件清单

```
graph/
├── README.md
├── generate_charts.py                 ← 一键生成所有图
├── architecture-diagram.html          ← 系统架构图 (浏览器打开)
│
├── eval_compat.py                     ← 单臂兼容率评估脚本
├── eval_rally.py                      ← 双臂 rally 评估脚本
│
├── eval_left_compat_v9.log           ← 左臂 v9: 17.0%
├── eval_right_compat_v9.log          ← 右臂 v9: 19.5%
├── eval_rally_6034_5472.log          ← 500步: 11.2%
├── eval_rally_6034_5472_2000steps.log ← 2000步: 35.3% ★
│
├── 01_left_compat_progression.png     ← 左臂 compat v1→v9
├── 02_right_compat_progression.png    ← 右臂 compat v2→v10
├── 03_left_vs_right_best_compat.png   ← 左右臂 v9 对比
├── 04_dual_rally_comparison.png       ← compat Before/After + rally
├── 05_loss_curves.png                 ← PPO 训练动态
├── 06_reward_breakdown.png            ← 17 项奖励分解
├── 07_compat_rate_summary.png         ← 版本终值柱状图
├── 08_baseline_success.png            ← 基线成功率
├── 09_eval_results.png                ← 评估三合一
├── 10_performance.png                 ← FPS + 耗时 + 回合长度
└── 11_summary_poster.png             ← 综合汇总 (放报告)
```

---

## 核心数据

| 指标 | Before | After (v9) | 提升 |
|------|:---:|:---:|:---:|
| 左臂 Compat Rate | 0.8% | **17.0%** | 21× |
| 右臂 Compat Rate | 0.4% | **19.5%** | 49× |
| Rally > 0 | ~0.01% | **35.3%** | — |
| Rally ≥ 2 | ~0% | **11.6%** | — |
| Rally Max | 0 | **4** | — |

## Rally 方程

```
rally>0 = L_compat × R_compat × chain_factor
35.3%  = 17.0%   × 19.5%    × 10.6
```

理论乘积 3.3%，实际 35.3%，链式因子 10.6×。Rally 链式效应将单臂兼容率的微弱改善放大了一个数量级。

## 使用说明

**重新生成图表**: 训练了新模型后更新日志路径即可
```bash
conda run -n env_isaaclab python graph/generate_charts.py
```

**跑评估获取新数据**:
```bash
# 单臂 compat
conda run -n env_isaaclab python graph/eval_compat.py left <model.pt>
conda run -n env_isaaclab python graph/eval_compat.py right <model.pt>

# 双臂 rally
conda run -n env_isaaclab python graph/eval_rally.py <merged_model.pt> 2000
```

**架构图**: 浏览器打开 `graph/architecture-diagram.html`
