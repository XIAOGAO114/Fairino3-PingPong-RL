"""Inspect all chart data to annotate quality issues."""
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RAIL = "/home/glq/isaac_ws/test_isaac_rail/logs/rsl_rl/fairino3_rail_centerline_v1"
RIGHT = "/home/glq/isaac_ws/test_isaac_rail_right/logs/rsl_rl/fairino3_rail_right_centerline_v1"
DUAL = "/home/glq/isaac_ws/test_isaac_dual/logs/rsl_rl/fairino3_dual_centerline_v1"

def load(log_dir, tag):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return None, None
    events = ea.Scalars(tag)
    if len(events) < 1:
        return None, None
    return [e.step for e in events], [e.value for e in events]

# ===== LEFT ARM =====
print("=" * 70)
print("LEFT ARM COMPAT RUNS: opponent_compatible reward")
print("=" * 70)
left_runs = [
    ('v1 (w=80)',     f'{RAIL}/2026-05-30_21-35-06_compat_v1'),
    ('v2 (w=120)',    f'{RAIL}/2026-07-11_00-51-14_compat_v2_left'),
    ('v3 (lr-fix)',   f'{RAIL}/2026-07-11_03-53-47_compat_v3_left'),
    ('v4',            f'{RAIL}/2026-07-11_14-22-26_compat_v4_left'),
    ('v5',            f'{RAIL}/2026-07-11_14-54-29_compat_v5_left'),
    ('v6',            f'{RAIL}/2026-07-11_15-26-01_compat_v6_left'),
    ('v7',            f'{RAIL}/2026-07-11_16-03-49_compat_v7_left'),
    ('v8',            f'{RAIL}/2026-07-11_17-07-12_compat_v8_left'),
    ('v9 (+vz)',      f'{RAIL}/2026-07-12_05-51-26_compat_v9_left'),
]
for name, d in left_runs:
    s, v = load(d, 'Episode_Reward/opponent_compatible')
    s2, v2 = load(d, 'Train/mean_reward')
    n_c = len(v) if v else 0
    n_r = len(v2) if v2 else 0
    last5_c = np.mean(v[-5:]) if v and len(v) >= 5 else (v[-1] if v and len(v) > 0 else 0)
    first_c = v[0] if v and len(v) > 0 else 0
    max_c = max(v) if v else 0
    last5_r = np.mean(v2[-5:]) if v2 and len(v2) >= 5 else (v2[-1] if v2 and len(v2) > 0 else 0)
    print(f"  {name:>14s}: compat_pts={n_c:>4d}  start={first_c:>6.2f}  peak={max_c:>6.2f}  "
          f"last5={last5_c:>6.2f}  |  reward_pts={n_r:>4d}  last5={last5_r:>6.0f}")

# ===== RIGHT ARM =====
print()
print("=" * 70)
print("RIGHT ARM COMPAT RUNS: opponent_compatible reward")
print("=" * 70)
right_runs = [
    ('v2 (w=120)', f'{RIGHT}/2026-07-11_01-33-57_compat_v2_right_from2225'),
    ('v3',         f'{RIGHT}/2026-07-11_04-21-40_compat_v3_right'),
    ('v4',         f'{RIGHT}/2026-07-11_18-02-14_compat_v4_right'),
    ('v5',         f'{RIGHT}/2026-07-11_18-54-47_compat_v5_right'),
    ('v6',         f'{RIGHT}/2026-07-12_01-21-23_compat_v6_right'),
    ('v7',         f'{RIGHT}/2026-07-12_03-01-14_compat_v7_right'),
    ('v8',         f'{RIGHT}/2026-07-12_03-19-21_compat_v8_right'),
    ('v9 (+vz)',   f'{RIGHT}/2026-07-12_03-40-11_compat_v9_right'),
    ('v10',        f'{RIGHT}/2026-07-12_04-09-25_compat_v10_right'),
]
for name, d in right_runs:
    s, v = load(d, 'Episode_Reward/opponent_compatible')
    s2, v2 = load(d, 'Train/mean_reward')
    n_c = len(v) if v else 0
    n_r = len(v2) if v2 else 0
    last5_c = np.mean(v[-5:]) if v and len(v) >= 5 else (v[-1] if v and len(v) > 0 else 0)
    first_c = v[0] if v and len(v) > 0 else 0
    max_c = max(v) if v else 0
    last5_r = np.mean(v2[-5:]) if v2 and len(v2) >= 5 else (v2[-1] if v2 and len(v2) > 0 else 0)
    print(f"  {name:>14s}: compat_pts={n_c:>4d}  start={first_c:>6.2f}  peak={max_c:>6.2f}  "
          f"last5={last5_c:>6.2f}  |  reward_pts={n_r:>4d}  last5={last5_r:>6.0f}")

# ===== DUAL =====
print()
print("=" * 70)
print("DUAL ARM MERGED RUNS")
print("=" * 70)
dual_runs = [
    ('merged_1543_2225', f'{DUAL}/2026-05-30_20-28-16_merged_1543_2225_final'),
    ('merged_2541_2724', f'{DUAL}/2026-07-11_01-54-22_merged_v2_2541_2724'),
    ('merged_v2_matched', f'{DUAL}/2026-07-11_02-52-00_merged_v2_matched'),
    ('merged_5535_4474', f'{DUAL}/2026-07-12_01-41-46_merged_5535_4474'),
    ('merged_6034_5472', f'{DUAL}/2026-07-12_06-16-56_merged_6034_5472'),
]
for name, d in dual_runs:
    s, v = load(d, 'Episode_Reward/rally_exchange')
    s2, v2 = load(d, 'Train/mean_reward')
    print(f"  {name:>22s}: rally={v}  reward={v2}")

# ===== LEFT v9 REWARD DETAIL =====
print()
print("=" * 70)
print("LEFT ARM v9: ALL REWARD COMPONENTS")
print("=" * 70)
d = f'{RAIL}/2026-07-12_05-51-26_compat_v9_left'
tags_to_check = [
    'Episode_Reward/left_table_bounce',
    'Episode_Reward/left_paddle_to_ball',
    'Episode_Reward/left_paddle_to_intercept',
    'Episode_Reward/left_first_return_hit',
    'Episode_Reward/left_return_ball',
    'Episode_Reward/left_second_paddle_contact',
    'Episode_Reward/left_post_hit_paddle_clearance',
    'Episode_Reward/left_post_hit_paddle_retreat',
    'Episode_Reward/left_net_direction',
    'Episode_Reward/left_net_height',
    'Episode_Reward/left_net_speed',
    'Episode_Reward/left_high_return_height',
    'Episode_Reward/left_predicted_landing',
    'Episode_Reward/left_near_net_landing_speed',
    'Episode_Reward/left_legal_return_separation',
    'Episode_Reward/left_right_table_bounce',
    'Episode_Reward/left_post_return_idle',
    'Episode_Reward/opponent_compatible',
    'Episode_Reward/opponent_vz_quality',
    'Episode_Reward/left_joint_limit',
    'Episode_Reward/left_table_collision',
]
for tag in tags_to_check:
    s, v = load(d, tag)
    if v and len(v) > 0:
        n = len(v)
        pct = n // 5
        avg = np.mean(v[-max(1, pct):])
        print(f"  {tag:<55s}  n={n:>4d}  last20%_avg={avg:>8.2f}  range=[{min(v):.1f}, {max(v):.1f}]")
    else:
        print(f"  {tag:<55s}  NO DATA")

# ===== LEFT v1 DETAIL (check why it's empty) =====
print()
print("=" * 70)
print("LEFT ARM v1: ALL TAGS (debug)")
print("=" * 70)
d = f'{RAIL}/2026-05-30_21-35-06_compat_v1'
ea = EventAccumulator(d)
ea.Reload()
all_tags = sorted(ea.Tags().get('scalars', []))
print(f"  Total tags: {len(all_tags)}")
for t in all_tags:
    s, v = load(d, t)
    n = len(v) if v else 0
    if n > 0:
        print(f"  {t:<55s}  n={n:>4d}  last={v[-1]:.2f}")

# ===== BASELINE DETAIL =====
print()
print("=" * 70)
print("LEFT BASELINE v6 (97.1%)")
print("=" * 70)
d = f'{RAIL}/2026-05-26_11-34-16_v6_generalize_from1244_300'
for tag in ['Episode_Reward/left_right_table_bounce', 'Episode_Reward/left_table_bounce', 'Train/mean_reward']:
    s, v = load(d, tag)
    if v:
        print(f"  {tag:<55s}  n={len(v)}  last5={np.mean(v[-5:]):.2f}  range=[{min(v):.1f}, {max(v):.1f}]")

print()
print("=" * 70)
print("RIGHT BASELINE v6 (84%)")
print("=" * 70)
d = f'{RIGHT}/2026-05-28_15-08-48_right_v6_angdamp_cont1_1525'
for tag in ['Episode_Reward/right_left_table_bounce_reward', 'Train/mean_reward']:
    s, v = load(d, tag)
    if v:
        print(f"  {tag:<55s}  n={len(v)}  last5={np.mean(v[-5:]):.2f}  range=[{min(v):.1f}, {max(v):.1f}]")

# ===== LEFT v9 Loss =====
print()
print("=" * 70)
print("LEFT v9: LOSS CURVES")
print("=" * 70)
d = f'{RAIL}/2026-07-12_05-51-26_compat_v9_left'
for tag in ['Loss/value', 'Loss/surrogate', 'Loss/entropy', 'Loss/learning_rate']:
    s, v = load(d, tag)
    if v:
        print(f"  {tag:<30s}  n={len(v)}  range=[{min(v):.4f}, {max(v):.4f}]")
    else:
        print(f"  {tag:<30s}  NO DATA")
