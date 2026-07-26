# 命令参考

## Python 环境

```bash
Python: /home/glq/.conda/envs/env_isaaclab/bin/python
Conda: env_isaaclab
激活: conda activate env_isaaclab
```

## 所有命令从对应工作空间根目录运行

---

## test_isaac_rail (左臂 7-DOF)

### 冒烟测试（1 迭代，16 环境）
```bash
cd /home/glq/isaac_ws/test_isaac_rail
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Fairino3-PingPong-Rail-Centerline-v0 \
  --num_envs 16 --max_iterations 1 --headless \
  --run_name smoke_<tag>
```

### 从头训练
```bash
cd /home/glq/isaac_ws/test_isaac_rail
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Fairino3-PingPong-Rail-Centerline-v0 \
  --num_envs 512 --max_iterations 500 --headless \
  --run_name rail_<tag>
```

### 续训
```bash
cd /home/glq/isaac_ws/test_isaac_rail
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Fairino3-PingPong-Rail-Centerline-v0 \
  --num_envs 512 --max_iterations 500 --headless \
  --resume \
  --load_run <run_dir_name> \
  --checkpoint model_<N>.pt \
  --run_name continue_<tag>
```

### Play（可视化推理）
```bash
cd /home/glq/isaac_ws/test_isaac_rail
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/play.py \
  --task Fairino3-PingPong-Rail-Centerline-v0 \
  --num_envs 1 --real-time \
  --checkpoint <path/to/model.pt>
```

---

## test_isaac_rail_right (右臂 7-DOF)

### 训练
```bash
cd /home/glq/isaac_ws/test_isaac_rail_right
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Fairino3-PingPong-Rail-Right-Centerline-v0 \
  --num_envs 1024 --max_iterations 4000 --headless \
  --run_name right_<tag>
```

### Play
```bash
cd /home/glq/isaac_ws/test_isaac_rail_right
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/play.py \
  --task Fairino3-PingPong-Rail-Right-Centerline-v0 \
  --num_envs 1 --real-time \
  --checkpoint <path/to/model.pt>
```

---

## test_isaac_dual (双臂对打)

### 训练（从合并模型起步，冻结左臂）
```bash
cd /home/glq/isaac_ws/test_isaac_dual
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Fairino3-PingPong-Dual-Centerline-v0 \
  --num_envs 512 --max_iterations 100 --headless \
  --run_name v5_scratch_right \
  --pretrained_checkpoint model_1543.pt \
  --unfreeze --freeze-left --right-std 0.3
```

### 续训
```bash
cd /home/glq/isaac_ws/test_isaac_dual
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Fairino3-PingPong-Dual-Centerline-v0 \
  --num_envs 512 --max_iterations 500 --headless \
  --run_name v5_cont_500 \
  --load_run <v5_run> --checkpoint model_99.pt \
  --resume --unfreeze --freeze-left --right-std 0.3
```

### Play
```bash
cd /home/glq/isaac_ws/test_isaac_dual
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/play.py \
  --task Fairino3-PingPong-Dual-Centerline-v0 \
  --num_envs 1 --real-time \
  --checkpoint <path/to/model.pt>
```

---

## test_isaac (legacy 6-DOF)

### 训练
```bash
cd /home/glq/isaac_ws/test_isaac
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Fairino3-PingPong-Centerline-v0 \
  --num_envs 512 --max_iterations 500 --headless \
  --run_name centerline_<tag>
```

---

## 评估脚本

### 单环境 trace 评估
```bash
cd /home/glq/isaac_ws/test_isaac
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/eval_single_checkpoints.py \
  --task Fairino3-PingPong-Centerline-v0 \
  --num_envs 1 --steps 3500 --seed 42 --headless \
  --trace_episodes 20 \
  --json_out <output.json> \
  --checkpoints <checkpoint_path>
```

## 常用参数说明

| 参数 | 说明 |
|------|------|
| `--task` | Task ID，决定加载哪个配置 |
| `--num_envs` | 并行环境数（512/1024 适合训练，1/16 适合测试） |
| `--max_iterations` | 训练迭代次数 |
| `--headless` | 无头模式（不渲染） |
| `--real-time` | Play 时实时渲染 |
| `--resume` | 续训模式 |
| `--load_run` | 续训时指定 run 目录名 |
| `--checkpoint` | 模型文件名（只需文件名，配合 --load_run） |
| `--run_name` | 本次运行的名称 |
| `--pretrained_checkpoint` | 预训练模型路径（双臂用） |
| `--unfreeze` | 双臂：解冻模型进行训练 |
| `--freeze-left` | 双臂：冻结左臂 |
| `--right-std` | 双臂：右臂初始 std |
