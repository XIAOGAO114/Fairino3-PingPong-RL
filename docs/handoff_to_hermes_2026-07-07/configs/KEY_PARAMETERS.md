# 关键配置参数速查

## 球台尺寸（Isaac Lab 默认）

| 参数 | 值 |
|------|-----|
| TABLE_HALF_LENGTH | 1.37m |
| TABLE_HALF_WIDTH | 0.7625m |
| TABLE_TOP_Z | 0.76m |
| NET_X | 0.55m |
| NET_HEIGHT | 0.91m (TABLE_TOP_Z + 0.1525) |

## 机器人位置

| 项目 | 位置 (x, y, z) | 旋转 (w, x, y, z) |
|------|------|------|
| 左臂 (rail) | (-0.97, 0, 0.76) | (0.707, 0, 0, 0.707) |
| 右臂 (rail_right) | (2.07, 0, 0.76) | (0.707, 0, 0, -0.707) |
| 双臂左 | (-0.97, 0, 0.76) | (0.707, 0, 0, 0.707) |
| 双臂右 | (2.07, 0, 0.76) | (0.707, 0, 0, -0.707) |

## 关节角度

### 硬关节角（原始 URDF 限制附近，容易超限）
```
j1=-1.743, j2=-3.368, j3=0.0, j4=-1.847, j5=3.030, j6=1.054
```

### 软关节角（推荐，留有余量）
```
j1=-1.743, j2=-2.94, j3=0.0, j4=-1.847, j5=2.7, j6=1.054
```
- j2 从 -3.368 改为 -2.94（距下限 ~3.5 留 0.56 rad）
- j5 从 3.030 改为 2.7（远离上限）
- 关节极限率从 55-78% 降至 <1%

## 动作空间

| 自由度 | 类型 | scale | 范围 |
|--------|------|:---:|------|
| rail_y | prismatic | 0.04 | 滑轨 Y 轴位移 |
| j1 | revolute | 0.28 | 底座旋转 |
| j2 | revolute | 0.28 | 肩部 |
| j3 | revolute | 0.28 | 肘部 |
| j4 | revolute | 0.28 | 腕部1 |
| j5 | revolute | 0.28 | 腕部2 |
| j6 | revolute | 0.28 | 腕部3 |

## 发球参数（当前）

| 参数 | 左臂 | 右臂 |
|------|------|------|
| 位置 x 偏移 | ±0.20 | ±0.20 |
| 位置 y 偏移 | ±0.35 | ±0.35 |
| 位置 z 偏移 | ±0.08 | ±0.08 |
| vx 范围 | (-2.2, -0.7) | (0.7, 2.2) |
| vy 范围 | ±0.45 | ±0.45 |
| vz 范围 | (0.05, 0.55) | (0.05, 0.55) |
| max_clearance_height | TABLE_TOP_Z + 0.80 | TABLE_TOP_Z + 0.80 |

## PPO 超参数

| 参数 | 单臂 | 双臂 |
|------|------|------|
| init_std | 0.7 | **0.15** |
| hidden_dims | [512, 256, 128] | [512, 256, 128] |
| activation | elu | elu |
| entropy_coef | 0.0015 | **0.0005** |
| learning_rate | 1e-4 (默认) | **1e-4** |
| 左臂 std 初始 | - | 从 model_1543 加载 (1.0-2.5) |

## 高度控制参数（h5 最佳配置）

| 参数 | 值 |
|------|-----|
| `height_factor` max_good_z | TABLE_TOP_Z + 0.35 (1.11m) |
| `height_factor` min_factor | 0.2 |
| `high_return_height_penalty` weight | -18.0 |
| `high_return_height_penalty` max_height | TABLE_TOP_Z + 0.1525*3 (1.22m) |

## 物理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| angular_damping (球) | 0.05 | 所有项目已同步 |
| 球半径 | ~0.02m (标准乒乓球) | |
| 球质量 | (URDF 默认) | |
| 拍面尺寸 | (URDF 默认) | |

## 检测参数（illegal_second_hit）

| 参数 | v2j 严格值 | 说明 |
|------|:---:|------|
| window_steps | 60 | 检测窗口 |
| min_delta_speed | 0.03 | 最小速度变化 |
| grace_steps | 15 | 击球后宽限步数 |
| contact_distance | 0.10 | 接触距离阈值 |

## Task ID 对照

| 项目 | Task ID |
|------|------|
| 左臂单臂 | `Fairino3-PingPong-Rail-Centerline-v0` |
| 右臂单臂 | `Fairino3-PingPong-Rail-Right-Centerline-v0` |
| 双臂对打 | `Fairino3-PingPong-Dual-Centerline-v0` |
| Legacy 6-DOF | `Fairino3-PingPong-Centerline-v0` |

## opponent_compatible 配置（最终版本）

| 组件 | weight |
|------|:---:|
| opponent_compatible | 80 |
| net_direction | 10 |
| net_height | 3 |
| net_speed | 6 |
| predicted_landing | 30 |
| near_net_landing_speed | 12 |
| legal_return_separation | 12 |
| min_vy | -0.80 |
