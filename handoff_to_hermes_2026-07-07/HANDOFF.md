# Claude Code Handoff: Fairino3 Single-Arm Ping-Pong

Date: 2026-05-25
Source workspace: `/home/glq/Desktop/project/pingpong_new`
Isaac workspace: `/home/glq/isaac_ws/test_isaac`

## User Goal

Continue single-arm generalization training. The immediate goal is to reduce consecutive paddle hits and make visually correct returns: one valid hit, ball clearly crosses the net, then lands on the right side of the table.

The user has lost confidence in the current iteration because visual play still appears to use second hits to get the ball over the net.

## Current Diagnosis

The user's latest observation is likely correct.

The v2h change fixed only the final success reward/termination:

- `right_table_bounce_reward(... require_clean_over_net=True)`
- `terminations.right_table_bounce(... require_clean_over_net=True)`
- `allow_illegal_success=False`
- `illegal_reward_scale=0.0`

However, several dense post-hit shaping rewards still give positive reward after an illegal second paddle hit if the ball then goes right / crosses the net / predicts a right-table landing.

Likely still-positive after illegal second hit:

- `return_ball_velocity`
- `ball_over_net`
- `net_direction`
- `net_height`
- `net_speed`
- `centerline_x`
- `predicted_right_table_landing`
- `predicted_right_near_net_landing_speed`
- `ball_to_target`
- possibly `rally_return`

These mostly gate on `has_return_hit()` plus rightward velocity or net/landing conditions. They do not consistently gate on `~has_illegal_second_hit()`.

So v2h makes illegal/unclean right-table bounces not count as final success, but it does not fully cut off the dense reward path where a second hit helps the ball over the net.

## Best Current Checkpoint

Use v2h final only as a debugging baseline, not as a solved policy:

`/home/glq/isaac_ws/test_isaac/logs/rsl_rl/fairino3_centerline_shared_single_v1/2026-05-25_13-54-05_centerline_v2h_clean_success_gate_from2797_150/model_2946.pt`

Training run:

`/home/glq/isaac_ws/test_isaac/logs/rsl_rl/fairino3_centerline_shared_single_v1/2026-05-25_13-54-05_centerline_v2h_clean_success_gate_from2797_150`

Parent model:

`/home/glq/isaac_ws/test_isaac/logs/rsl_rl/fairino3_centerline_shared_single_v1/2026-05-25_13-17-11_centerline_v2g_legal_sep12_illegal010_from2648_150/model_2797.pt`

## Important Metrics So Far

v2g `model_2797.pt`, single-env trace, `seed=42`, first 20 episodes:

- `clean_right=4/20`
- `clean_right_per_hit=17.4%`
- `true_second_done_rate=60.0%`

v2h `model_2946.pt`, same trace:

- `clean_right=15/20`
- `clean_right_per_hit=57.7%`
- `true_second_done_rate=26.9%`

But visual play still looked bad to the user, and batch clean success remained around 40%.

v2h batch reference, `64 envs x 1800 steps`:

- `clean_over_net_per_hit=48.3%`
- `clean_right_per_hit=40.7%`
- `legal_right=100%`
- `true_second_done_rate=36.4%`

## Recommended Next Fix: v2i Legal Dense Gate

Implement a shared legal post-hit mask and apply it to all positive post-hit return-quality rewards.

Desired behavior:

- Before the first valid return hit: no return-quality reward.
- After the first valid return hit: positive shaping can be paid only while no illegal second hit has been observed.
- Once illegal second hit is detected: all positive return-quality shaping becomes zero for the rest of the episode.
- Keep the illegal second-hit penalty active.
- Keep final success requiring `clean_over_net` and no illegal second hit.

Possible helper:

```python
def legal_post_hit_mask(env, num_envs: int, robot_cfg=..., ball_cfg=..., contact_distance=0.10) -> torch.Tensor:
    illegal_second_hit_event(...)
    return has_return_hit(env, num_envs) & (~has_illegal_second_hit(env, num_envs))
```

Then replace gates such as:

```python
returning = (vel_x > 0.05) & has_return_hit(env, num_envs)
```

with:

```python
returning = (vel_x > 0.05) & legal_post_hit_mask(env, num_envs, ...)
```

Apply to at least:

- `return_ball_velocity`
- `ball_over_net`
- `ball_to_target`
- `predicted_right_table_landing`
- `predicted_right_near_net_landing_speed`
- `net_direction`
- `net_height`
- `net_speed`
- `centerline_x`
- `rally_return`

After patching, sync files into Isaac workspace, smoke test, then train a short v2i from `model_2946.pt`.

## Key Files

Workspace source files:

- `task_source/mdp/_shared.py`
- `task_source/mdp/rewards.py`
- `task_source/mdp/terminations.py`
- `task_source/fairino3_pingpong_env_cfg.py`
- `scripts/eval_single_checkpoints.py`
- `reference/experiments.md`

Isaac workspace target files:

- `/home/glq/isaac_ws/test_isaac/source/test_isaac/test_isaac/tasks/manager_based/fairino3_pingpong/mdp/_shared.py`
- `/home/glq/isaac_ws/test_isaac/source/test_isaac/test_isaac/tasks/manager_based/fairino3_pingpong/mdp/rewards.py`
- `/home/glq/isaac_ws/test_isaac/source/test_isaac/test_isaac/tasks/manager_based/fairino3_pingpong/mdp/terminations.py`
- `/home/glq/isaac_ws/test_isaac/source/test_isaac/test_isaac/tasks/manager_based/fairino3_pingpong/fairino3_pingpong_env_cfg.py`
- `/home/glq/isaac_ws/test_isaac/scripts/eval_single_checkpoints.py`

## Commands

Smoke train:

```bash
cd /home/glq/isaac_ws/test_isaac
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Fairino3-PingPong-Centerline-v0 \
  --num_envs 16 \
  --max_iterations 1 \
  --headless \
  --resume \
  --load_run 2026-05-25_13-54-05_centerline_v2h_clean_success_gate_from2797_150 \
  --checkpoint model_2946.pt \
  --run_name smoke_v2i_legal_dense_gate
```

Short training candidate:

```bash
cd /home/glq/isaac_ws/test_isaac
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Fairino3-PingPong-Centerline-v0 \
  --num_envs 512 \
  --max_iterations 100 \
  --headless \
  --resume \
  --load_run 2026-05-25_13-54-05_centerline_v2h_clean_success_gate_from2797_150 \
  --checkpoint model_2946.pt \
  --run_name centerline_v2i_legal_dense_gate_from2946_100
```

Single-env trace eval:

```bash
cd /home/glq/isaac_ws/test_isaac
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/eval_single_checkpoints.py \
  --task Fairino3-PingPong-Centerline-v0 \
  --num_envs 1 \
  --steps 3500 \
  --seed 42 \
  --headless \
  --trace_episodes 20 \
  --json_out /home/glq/Desktop/project/pingpong_new/reference/evals/eval_clean_gate_v2i_single_env_seed42_trace.json \
  --checkpoints <checkpoint>
```

Play current v2h:

```bash
cd /home/glq/isaac_ws/test_isaac
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/play.py \
  --task Fairino3-PingPong-Centerline-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint /home/glq/isaac_ws/test_isaac/logs/rsl_rl/fairino3_centerline_shared_single_v1/2026-05-25_13-54-05_centerline_v2h_clean_success_gate_from2797_150/model_2946.pt
```

## Process Notes

- The user cares more about visual correctness than old `right/legal_right`.
- Do not rely on old `right` alone.
- Primary metrics should be:
  - episode trace clean success count
  - `clean_over_net_per_hit`
  - `clean_right_per_hit`
  - `true_second_done_rate`
- If play looks wrong, trust the visual observation and inspect which rewards are still positive after illegal second hit.

