# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A handoff snapshot from a Fairino3 single-arm ping-pong RL training project using NVIDIA Isaac Lab.

**Primary workspace (active):** `/home/glq/isaac_ws/test_isaac_rail/` — 7-DOF rail-mounted Fairino3 arm with Y-axis prismatic joint.

**Legacy workspace:** `/home/glq/isaac_ws/test_isaac/` — 6-DOF centerline arm (fixed base, no lateral freedom). Kept for reference checkpoints.

This repo holds reference copies of the key source files and evaluation artifacts for context.

**Primary goal:** train a single-arm policy that returns a ping-pong ball cleanly (one hit, ball crosses net, lands on right table) without illegal second paddle contacts.

**Start here:** `HANDOFF.md` — it contains the current diagnosis, best checkpoint paths, and the v2i legal dense gate plan (now already implemented).

## Project Layout

| Purpose | Path |
|---------|------|
| Handoff workspace (this repo) | `/home/glq/Desktop/project/pingpong_claudecode_handoff_2026-05-25/` |
| **Active Isaac workspace (rail)** | `/home/glq/isaac_ws/test_isaac_rail/` |
| Legacy Isaac workspace (6-DOF) | `/home/glq/isaac_ws/test_isaac/` |
| Source files (snapshot) | `artifacts/task_source/` |

### Active Rail Project Structure

```
test_isaac_rail/
├── source/test_isaac_rail/test_isaac_rail/
│   ├── assets/fairino3_v6_rail/          # URDF + USD for 7-DOF rail robot
│   └── tasks/manager_based/fairino3_rail_pingpong/
│       ├── fairino3_rail_pingpong_env_cfg.py   # Main config
│       ├── mdp/_shared.py                       # State machines (shared with legacy)
│       ├── mdp/rewards.py                       # Reward functions
│       ├── mdp/terminations.py                  # Termination conditions
│       ├── mdp/observations.py                  # Observation terms
│       └── agents/rsl_rl_ppo_cfg.py             # PPO config
```

**Task ID:** `Fairino3-PingPong-Rail-Centerline-v0`
**Action space:** 7-DOF — `rail_y` (scale=0.04) + `j1`–`j6` (scale=0.28)
**Robot position:** `(-1.35, 0.0, 0.98)`
**Initial joint pose:** `rail_y=0.0, j1=-1.743, j2=-3.368, j3=0.0, j4=-1.847, j5=3.030, j6=1.054`

## Architecture

The project uses Isaac Lab's `ManagerBasedRLEnv` framework. The key extension points are in the `mdp/` package:

- **`_shared.py`** — Episode state machines. Each detector tracks per-env state (contact, hit, illegal second hit, left bounce, clean over-net) with rising-edge detection and episode-reset logic. The pattern: check for cached state → apply reset mask → compute event → cache result per step.
- **`rewards.py`** — All reward terms. Each is a function taking `env` + config params, returning a `(num_envs,)` tensor.
- **`terminations.py`** — Termination conditions, same signature pattern.
- **`fairino3_pingpong_env_cfg.py`** — Assembles scene, observations, actions, events, rewards, terminations into a `@configclass`. Contains two scene variants: `Fairino3PingPongEnvCfg` (corner robot) and `Fairino3PingPongCenterlineEnvCfg` (centerline robot, the active one).

### Core State Machine (in `_shared.py`)

1. **`first_return_hit_event()`** — Detects the first useful paddle hit. Requires: ball bounced on left table first, ball was incoming (vx < -threshold), contact + velocity reversal within a multi-step window (handles physics substep timing).
2. **`illegal_second_hit_event()`** — Detects a second paddle contact after the valid hit. Gated by grace steps (allow follow-through), window steps, delta-speed threshold, and rightward velocity requirement. Cached per step.
3. **`legal_post_hit_mask()`** — `has_return_hit & ~has_illegal_second_hit`. The key gate for all positive post-hit rewards. Already implemented (v2i).
4. **`clean_over_net_event()`** — Ball crossed net while above net height, within time window after hit, with no illegal second contact.
5. **`left_table_bounce_event()`** — Ball bounced on left half of table (prerequisite for a valid hit).

### Reward Design Principles

- Pre-hit rewards are gated by incoming phase and left-bounce prerequisites.
- Post-hit rewards should be gated by `legal_post_hit_mask()` so illegal second hits cut off all positive shaping.
- Penalties for: second contact, paddle-ball proximity after hit, paddle chasing ball, joint limits, table collision, out-of-bounds.
- Terminal success: `right_table_bounce_reward` with `require_clean_over_net=True`, `illegal_reward_scale=0.0`.

## Key Training Evolution (Legacy 6-DOF)

| Version | What changed | Best model |
|---------|-------------|------------|
| v2g | Baseline with legal gate on terminations only | `model_2797.pt` |
| v2h | `require_clean_over_net` on terminal reward/termination | `model_2946.pt` (visual still bad) |
| v2i | `legal_post_hit_mask` gates all 10 post-hit rewards | Code in place |
| v2j | Stricter illegal detection params (window 60, delta 0.03) | In progress |

The rail project addresses the root cause (6-DOF lateral limitation) with a 7th prismatic DOF.

## Commands

All run from the rail workspace. Python env: `/home/glq/.conda/envs/env_isaaclab/bin/python`

### Smoke test (1 iteration, 16 envs)
```bash
cd /home/glq/isaac_ws/test_isaac_rail
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Fairino3-PingPong-Rail-Centerline-v0 \
  --num_envs 16 --max_iterations 1 --headless \
  --run_name smoke_<tag>
```

### Training from scratch
```bash
cd /home/glq/isaac_ws/test_isaac_rail
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Fairino3-PingPong-Rail-Centerline-v0 \
  --num_envs 512 --max_iterations 500 --headless \
  --run_name rail_<tag>
```

### Play (visual inspection)
```bash
cd /home/glq/isaac_ws/test_isaac_rail
/home/glq/.conda/envs/env_isaaclab/bin/python scripts/rsl_rl/play.py \
  --task Fairino3-PingPong-Rail-Centerline-v0 \
  --num_envs 1 --real-time \
  --checkpoint <path/to/model.pt>
```

## Important Conventions

- **Always smoke test before full training.** A 1-iteration run catches config/syntax errors.
- **The user cares about visual correctness, not just metrics.** If play looks wrong, trust the visual observation over aggregate stats.
- **Primary metrics:** `clean_right_per_hit`, `clean_over_net_per_hit`, `true_second_done_rate`.
- **Episode state is stored on `env` as private attributes** (e.g., `env._fairino_return_hit_state`). These are per-env tensors, reset on episode boundaries via `episode_reset_mask()`.
- **Rail project is the main workspace now.** The legacy 6-DOF centerline project showed that reward shaping alone can't fully overcome the lack of lateral freedom. The rail adds a Y-axis prismatic joint (`rail_y`) to give the arm the missing degree of freedom.
