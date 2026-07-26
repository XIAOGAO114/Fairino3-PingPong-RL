#!/usr/bin/env python3
"""Generate training charts from TensorBoard event files."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUTDIR = "/home/glq/Desktop/project/Hermes_大创乒乓球项目/graph"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RAIL_LOG = "/home/glq/isaac_ws/test_isaac_rail/logs/rsl_rl/fairino3_rail_centerline_v1"
RIGHT_LOG = "/home/glq/isaac_ws/test_isaac_rail_right/logs/rsl_rl/fairino3_rail_right_centerline_v1"
DUAL_LOG = "/home/glq/isaac_ws/test_isaac_dual/logs/rsl_rl/fairino3_dual_centerline_v1"


def load_scalar(log_dir, tag):
    """Load a scalar time series from tensorboard event files."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return None, None
    events = ea.Scalars(tag)
    if len(events) < 2:
        return None, None
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def load_smoothed(log_dir, tag, window=20):
    """Load and smooth scalar data with a moving average."""
    steps, values = load_scalar(log_dir, tag)
    if steps is None or len(steps) < window:
        return None, None
    smoothed = np.convolve(values, np.ones(window) / window, mode='valid')
    # Align steps: valid mode removes (window-1) points
    aligned_steps = steps[window - 1:]
    return aligned_steps, smoothed


# ============================================================
# Chart 1: Left Arm Compat Rate Progression (v1→v9)
# ============================================================
print("Chart 1: Left Arm Compat Rate Progression...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

left_compat_runs = [
    ('v1 w=80', f'{RAIL_LOG}/2026-05-30_21-35-06_compat_v1'),
    ('v2 w=120', f'{RAIL_LOG}/2026-07-11_00-51-14_compat_v2_left'),
    ('v3 lr-fix', f'{RAIL_LOG}/2026-07-11_03-53-47_compat_v3_left'),
    ('v4', f'{RAIL_LOG}/2026-07-11_14-22-26_compat_v4_left'),
    ('v5', f'{RAIL_LOG}/2026-07-11_14-54-29_compat_v5_left'),
    ('v6', f'{RAIL_LOG}/2026-07-11_15-26-01_compat_v6_left'),
    ('v7', f'{RAIL_LOG}/2026-07-11_16-03-49_compat_v7_left'),
    ('v8', f'{RAIL_LOG}/2026-07-11_17-07-12_compat_v8_left'),
    ('v9 +vz', f'{RAIL_LOG}/2026-07-12_05-51-26_compat_v9_left'),
]
n = len(left_compat_runs)
colors = plt.cm.viridis(np.linspace(0.1, 0.95, n))

ax = axes[0]
for (label, logdir), color in zip(left_compat_runs, colors):
    steps, values = load_smoothed(logdir, 'Episode_Reward/opponent_compatible')
    if steps is not None:
        ax.plot(steps, values, label=label, color=color, linewidth=1.3, alpha=0.85)
ax.set_xlabel('Iteration')
ax.set_ylabel('Opponent Compatible Reward')
ax.set_title('Left Arm: Compat Reward Progression')
ax.legend(fontsize=6.5, loc='lower right', ncol=2)
ax.grid(True, alpha=0.3)

ax = axes[1]
for (label, logdir), color in zip(left_compat_runs, colors):
    steps, values = load_smoothed(logdir, 'Train/mean_reward')
    if steps is not None:
        ax.plot(steps, values, label=label, color=color, linewidth=1.3, alpha=0.85)
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Episode Reward')
ax.set_title('Left Arm: Total Reward')
ax.legend(fontsize=6.5, loc='lower right', ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/01_left_compat_progression.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {OUTDIR}/01_left_compat_progression.png")


# ============================================================
# Chart 2: Right Arm Compat Rate Progression (v2→v10)
# ============================================================
print("Chart 2: Right Arm Compat Rate Progression...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

right_compat_runs = [
    ('v2 w=120', f'{RIGHT_LOG}/2026-07-11_01-33-57_compat_v2_right_from2225'),
    ('v3', f'{RIGHT_LOG}/2026-07-11_04-21-40_compat_v3_right'),
    ('v4', f'{RIGHT_LOG}/2026-07-11_18-02-14_compat_v4_right'),
    ('v5', f'{RIGHT_LOG}/2026-07-11_18-54-47_compat_v5_right'),
    ('v6', f'{RIGHT_LOG}/2026-07-12_01-21-23_compat_v6_right'),
    ('v7', f'{RIGHT_LOG}/2026-07-12_03-01-14_compat_v7_right'),
    ('v8', f'{RIGHT_LOG}/2026-07-12_03-19-21_compat_v8_right'),
    ('v9 +vz', f'{RIGHT_LOG}/2026-07-12_03-40-11_compat_v9_right'),
    ('v10', f'{RIGHT_LOG}/2026-07-12_04-09-25_compat_v10_right'),
]
n = len(right_compat_runs)
colors = plt.cm.plasma(np.linspace(0.1, 0.95, n))

ax = axes[0]
for (label, logdir), color in zip(right_compat_runs, colors):
    steps, values = load_smoothed(logdir, 'Episode_Reward/opponent_compatible')
    if steps is not None:
        ax.plot(steps, values, label=label, color=color, linewidth=1.3, alpha=0.85)
ax.set_xlabel('Iteration')
ax.set_ylabel('Opponent Compatible Reward')
ax.set_title('Right Arm: Compat Reward Progression')
ax.legend(fontsize=6.5, loc='lower right', ncol=2)
ax.grid(True, alpha=0.3)

ax = axes[1]
for (label, logdir), color in zip(right_compat_runs, colors):
    steps, values = load_smoothed(logdir, 'Train/mean_reward')
    if steps is not None:
        ax.plot(steps, values, label=label, color=color, linewidth=1.3, alpha=0.85)
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Episode Reward')
ax.set_title('Right Arm: Total Reward')
ax.legend(fontsize=6.5, loc='lower right', ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/02_right_compat_progression.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {OUTDIR}/02_right_compat_progression.png")


# ============================================================
# Chart 3: Left vs Right Best Compat Comparison (v9)
# ============================================================
print("Chart 3: Left vs Right Best Compat...")
fig, ax = plt.subplots(figsize=(10, 5))

best_left = f'{RAIL_LOG}/2026-07-12_05-51-26_compat_v9_left'
best_right = f'{RIGHT_LOG}/2026-07-12_03-40-11_compat_v9_right'

for logdir, label, color, ls in [
    (best_left, 'Left Arm v9 compat', '#2196F3', '-'),
    (best_right, 'Right Arm v9 compat', '#FF5722', '-'),
]:
    steps, values = load_smoothed(logdir, 'Episode_Reward/opponent_compatible')
    if steps is not None:
        ax.plot(steps, values, label=label, color=color, linewidth=2, linestyle=ls)

for logdir, label, color in [
    (best_left, 'Left vz_quality', '#64B5F6'),
    (best_right, 'Right vz_quality', '#FF8A65'),
]:
    steps, values = load_smoothed(logdir, 'Episode_Reward/opponent_vz_quality')
    if steps is not None and len(steps) > 1:
        ax.plot(steps, values, label=label, color=color, linewidth=1, linestyle='--', alpha=0.7)

ax.set_xlabel('Iteration')
ax.set_ylabel('Reward')
ax.set_title('Left vs Right Arm: Best Compat Reward (v9 with vz quality)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/03_left_vs_right_best_compat.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {OUTDIR}/03_left_vs_right_best_compat.png")


# ============================================================
# Chart 4: Compat & Rally from Real Eval Runs
# ============================================================
print("Chart 4: Compat & Rally Eval Results...")

# Use eval log data (same parse functions as chart 9)
import re

def parse_compat_log_pct(path):
    try:
        with open(path) as f:
            for line in f:
                if '% compatible' in line:
                    return float(line.strip().split()[-2].replace('%', ''))
    except: pass
    return 0

def parse_rally_log_pct(path):
    try:
        with open(path) as f:
            content = f.read()
            m_gt0 = re.search(r'>0 rate: ([\d.]+)%', content)
            m_ge2 = re.search(r'>=2 rate: ([\d.]+)%', content)
            if m_gt0: return float(m_gt0.group(1)), float(m_ge2.group(1))
    except: pass
    return 0, 0

logd = '/home/glq/Desktop/project/Hermes_大创乒乓球项目/graph'
left_pct = parse_compat_log_pct(f'{logd}/eval_left_compat_v9.log')
right_pct = parse_compat_log_pct(f'{logd}/eval_right_compat_v9.log')
# Prefer 2000-step log if available, fall back to 500-step
rally_gt0, rally_ge2 = parse_rally_log_pct(f'{logd}/eval_rally_6034_5472_2000steps.log')
if rally_gt0 == 0:
    rally_gt0, rally_ge2 = parse_rally_log_pct(f'{logd}/eval_rally_6034_5472.log')
rally_eps = 1452  # from 2000-step eval
theory = left_pct * right_pct / 100
chain_factor = rally_gt0 / theory if theory > 0 else 1

# Build comparison data: before compat vs after compat
# Historical: left ~0.8%, right ~0.4% (from memory/handoff docs)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Subplot 1: Compat Rate Before/After
ax = axes[0]
x = np.arange(2)
width = 0.35
before = [0.8, 0.4]  # pre-compat (v1 era)
after = [left_pct, right_pct]
bars1 = ax.bar(x - width/2, before, width, label='Before Compat Training', color='#BDBDBD', edgecolor='white')
bars2 = ax.bar(x + width/2, after, width, label='After Compat Training (v9)', color=['#2196F3', '#FF5722'], edgecolor='white')
for bar, val in zip(bars1, before):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', fontsize=9)
for bar, val in zip(bars2, after):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(['Left Arm', 'Right Arm'])
ax.set_ylabel('Compatible Rate (%)')
ax.set_title('Compat Rate: Before vs After')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
improvement_l = left_pct / 0.8
improvement_r = right_pct / 0.4
ax.text(0.5, 0.95, f'Left: {improvement_l:.0f}x  |  Right: {improvement_r:.0f}x', 
        transform=ax.transAxes, ha='center', fontsize=9, style='italic')

# Subplot 2: Rally Rate
ax = axes[1]
rally_cats = ['Rally > 0', 'Rally ≥ 2']
rally_vals = [rally_gt0, rally_ge2]
bars = ax.bar(rally_cats, rally_vals, color=['#4CAF50', '#8BC34A'], edgecolor='white', linewidth=1, width=0.4)
for bar, val in zip(bars, rally_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', 
            ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Rate (%)')
ax.set_title(f'Rally Rate: merged_6034_5472\n(256 envs x 500 steps, 680 episodes)')
ax.set_ylim(0, max(rally_vals) * 1.5)
ax.grid(True, alpha=0.3, axis='y')

# Subplot 3: Rally Equation
ax = axes[2]
theory = left_pct * right_pct / 100
eq_cats = ['Theory\n(L% × R%)', 'Actual\nRally>0', 'Rally≥2']
eq_vals = [theory, rally_gt0, rally_ge2]
eq_colors = ['#9E9E9E', '#4CAF50', '#FF9800']
bars = ax.bar(eq_cats, eq_vals, color=eq_colors, edgecolor='white', linewidth=1, width=0.4)
for bar, val in zip(bars, eq_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.1f}%', 
            ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Rate (%)')
ax.set_title('Rally Equation: Theory vs Reality')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/04_dual_rally_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {OUTDIR}/04_dual_rally_comparison.png")
print(f"  Data: left={left_pct:.1f}% right={right_pct:.1f}% rally>0={rally_gt0:.1f}% rally≥2={rally_ge2:.1f}% theory={theory:.1f}%")


# ============================================================
# Chart 5: Training Dynamics - Loss Curves (left arm v9)
# ============================================================
print("Chart 5: Loss Curves...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
logdir = best_left

# Value loss
ax = axes[0, 0]
steps, values = load_smoothed(logdir, 'Loss/value')
if steps is not None:
    ax.plot(steps, values, color='#2196F3', linewidth=1)
ax.set_xlabel('Iteration'); ax.set_ylabel('Value Loss')
ax.set_title('Value Loss'); ax.grid(True, alpha=0.3)

# Surrogate loss
ax = axes[0, 1]
steps, values = load_smoothed(logdir, 'Loss/surrogate')
if steps is not None:
    ax.plot(steps, values, color='#FF5722', linewidth=1)
ax.set_xlabel('Iteration'); ax.set_ylabel('Surrogate Loss')
ax.set_title('Surrogate Loss (Policy)'); ax.grid(True, alpha=0.3)

# Entropy
ax = axes[1, 0]
steps, values = load_smoothed(logdir, 'Loss/entropy')
if steps is not None:
    ax.plot(steps, values, color='#4CAF50', linewidth=1)
ax.set_xlabel('Iteration'); ax.set_ylabel('Entropy')
ax.set_title('Policy Entropy (Exploration)'); ax.grid(True, alpha=0.3)

# Learning rate
ax = axes[1, 1]
steps, values = load_scalar(logdir, 'Loss/learning_rate')
if steps is not None:
    ax.plot(steps, values, color='#9C27B0', linewidth=1)
ax.set_xlabel('Iteration'); ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule'); ax.grid(True, alpha=0.3)

plt.suptitle('Training Dynamics: Left Arm compat_v9', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/05_loss_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {OUTDIR}/05_loss_curves.png")


# ============================================================
# Chart 6: Reward Component Breakdown (left arm v9)
# ============================================================
print("Chart 6: Reward Breakdown...")
fig, ax = plt.subplots(figsize=(13, 6))

reward_tags = [
    # Single-arm compat runs use UNPREFIXED tag names (no left_/right_ prefix)
    ('Episode_Reward/left_table_bounce',       'table_bounce'),
    ('Episode_Reward/paddle_to_ball',          'paddle_to_ball'),
    ('Episode_Reward/paddle_to_intercept',     'paddle_to_intercept'),
    ('Episode_Reward/first_return_hit',        'first_hit'),
    ('Episode_Reward/return_ball',             'return_ball'),
    ('Episode_Reward/second_paddle_contact',   'second_contact'),
    ('Episode_Reward/net_direction',           'net_direction'),
    ('Episode_Reward/net_height',              'net_height'),
    ('Episode_Reward/net_speed',               'net_speed'),
    ('Episode_Reward/predicted_landing',       'pred_landing'),
    ('Episode_Reward/near_net_landing_speed',  'landing_speed'),
    ('Episode_Reward/right_table_bounce',      'right_table_bounce'),
    ('Episode_Reward/opponent_compatible',     'opponent_compat'),
    ('Episode_Reward/opponent_vz_quality',     'opponent_vz_qual'),
    ('Episode_Reward/action_rate',             'action_rate'),
    ('Episode_Reward/ball_out_penalty',        'ball_out_pen'),
    ('Episode_Reward/joint_vel',               'joint_vel'),
]

colors_rb = plt.cm.tab20(np.linspace(0, 0.95, len(reward_tags)))
means = []
names = []

for (tag, name), color in zip(reward_tags, colors_rb):
    steps, values = load_scalar(logdir, tag)
    if steps is not None and len(values) > 0:
        n_last = max(1, len(values) // 5)
        avg = float(np.mean(values[-n_last:]))
        means.append(avg)
        names.append(name)
        ax.bar(name, avg, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.text(name, avg, f'{avg:.1f}', ha='center', va='bottom', fontsize=7)

ax.set_ylabel('Mean Reward (last 20% of training)')
ax.set_title('Reward Component Breakdown: Left Arm compat_v9')
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/06_reward_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {OUTDIR}/06_reward_breakdown.png")


# ============================================================
# Chart 7: Compat Rate Summary Bar Chart (All Versions)
# ============================================================
print("Chart 7: Compat Rate Summary...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left arm
left_versions = [r[0] for r in left_compat_runs]
left_rates = []
for _, logdir in left_compat_runs:
    steps, values = load_scalar(logdir, 'Episode_Reward/opponent_compatible')
    if steps is not None and len(values) > 0:
        left_rates.append(float(np.mean(values[-20:])))
    else:
        left_rates.append(0)

ax = axes[0]
bars = ax.bar(left_versions, left_rates,
              color=plt.cm.Blues(np.linspace(0.3, 0.9, len(left_versions))))
for bar, val in zip(bars, left_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'{val:.1f}', ha='center', va='bottom', fontsize=7)
ax.set_ylabel('Mean Compat Reward (last 20 iters)')
ax.set_title('Left Arm: Compat Reward by Version')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.grid(True, alpha=0.3, axis='y')

# Right arm
right_versions = [r[0] for r in right_compat_runs]
right_rates = []
for _, logdir in right_compat_runs:
    steps, values = load_scalar(logdir, 'Episode_Reward/opponent_compatible')
    if steps is not None and len(values) > 0:
        right_rates.append(float(np.mean(values[-20:])))
    else:
        right_rates.append(0)

ax = axes[1]
bars = ax.bar(right_versions, right_rates,
              color=plt.cm.Oranges(np.linspace(0.3, 0.9, len(right_versions))))
for bar, val in zip(bars, right_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'{val:.1f}', ha='center', va='bottom', fontsize=7)
ax.set_ylabel('Mean Compat Reward (last 20 iters)')
ax.set_title('Right Arm: Compat Reward by Version')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/07_compat_rate_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {OUTDIR}/07_compat_rate_summary.png")


# ============================================================
# Chart 8: Baseline Single-Arm Success Rate
# ============================================================
print("Chart 8: Baseline Success Rates...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

left_base = f'{RAIL_LOG}/2026-05-26_11-34-16_v6_generalize_from1244_300'
right_base = f'{RIGHT_LOG}/2026-05-28_15-08-48_right_v6_angdamp_cont1_1525'

ax = axes[0]
for tag, name, color in [
    # Single-arm runs use unprefixed tag names
    ('Episode_Reward/right_table_bounce', 'right_table_bounce (success)', '#4CAF50'),
    ('Episode_Reward/left_table_bounce', 'left_table_bounce (prereq)', '#2196F3'),
    ('Train/mean_reward', 'mean_reward', '#FF5722'),
]:
    steps, values = load_smoothed(left_base, tag)
    if steps is not None:
        ax.plot(steps, values, label=name, color=color, linewidth=1.5, alpha=0.85)
ax.set_xlabel('Iteration'); ax.set_ylabel('Reward')
ax.set_title('Left Arm Baseline (v6 generalize, 97.1%)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
for tag, name, color in [
    ('Episode_Reward/right_table_bounce', 'right_table_bounce (success)', '#4CAF50'),
    ('Episode_Reward/left_table_bounce', 'left_table_bounce (prereq)', '#2196F3'),
    ('Train/mean_reward', 'mean_reward', '#FF5722'),
]:
    steps, values = load_smoothed(right_base, tag)
    if steps is not None:
        ax.plot(steps, values, label=name, color=color, linewidth=1.5, alpha=0.85)
ax.set_xlabel('Iteration'); ax.set_ylabel('Reward')
ax.set_title('Right Arm Baseline (v6 angdamp, 84%)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/08_baseline_success.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {OUTDIR}/08_baseline_success.png")


# ============================================================
# Chart 9: Real Eval Results — Compat & Rally Rates
# ============================================================
print("Chart 9: Real Eval Results...")

# Parse eval log files
def parse_compat_log(path):
    """Extract compat rate from eval log."""
    try:
        with open(path) as f:
            for line in f:
                if '% compatible' in line:
                    # "left: 3015/17774 = 17.0% compatible"
                    parts = line.strip().split()
                    arm = parts[0].rstrip(':')
                    compat_hits = int(parts[1].split('/')[0])
                    total_crosses = int(parts[1].split('/')[1])
                    pct = float(parts[3].replace('%', ''))
                    return arm, compat_hits, total_crosses, pct
    except:
        pass
    return None, 0, 0, 0

def parse_rally_log(path):
    """Extract rally stats from eval log."""
    try:
        with open(path) as f:
            content = f.read()
            import re
            m_total = re.search(r'\((\d+) episodes\)', content)
            m_mean = re.search(r'Mean: ([\d.]+)', content)
            m_gt0 = re.search(r'>0 rate: ([\d.]+)%', content)
            m_ge2 = re.search(r'>=2 rate: ([\d.]+)%', content)
            if m_total:
                return int(m_total.group(1)), float(m_mean.group(1)), float(m_gt0.group(1)), float(m_ge2.group(1))
    except:
        pass
    return 0, 0, 0, 0

log_dir = '/home/glq/Desktop/project/Hermes_大创乒乓球项目/graph'
left_arm, left_hits, left_total, left_pct = parse_compat_log(f'{log_dir}/eval_left_compat_v9.log')
right_arm, right_hits, right_total, right_pct = parse_compat_log(f'{log_dir}/eval_right_compat_v9.log')
rally_eps, rally_mean, rally_gt0, rally_ge2 = parse_rally_log(f'{log_dir}/eval_rally_6034_5472_2000steps.log')
if rally_eps == 0:
    rally_eps, rally_mean, rally_gt0, rally_ge2 = parse_rally_log(f'{log_dir}/eval_rally_6034_5472.log')

# Also known historical values from memory
hist_left = 0.8   # v1-v2 era
hist_right = 0.4  # v1-v2 era

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Subplot 1: Compat Rate
ax = axes[0]
categories = ['Left Arm v9\n(model_6034)', 'Right Arm v9\n(model_5472)']
compat_pcts = [left_pct, right_pct]
bars = ax.bar(categories, compat_pcts, color=['#2196F3', '#FF5722'], edgecolor='white', linewidth=1)
for bar, pct, hits, total in zip(bars, compat_pcts, [left_hits, right_hits], [left_total, right_total]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{pct:.1f}%\n({hits}/{total})', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Compatible Rate (%)')
ax.set_title('Single-Arm Compat Rate\n(net-crossings in opponent serve zone)')
ax.set_ylim(0, max(compat_pcts) * 1.3)
ax.grid(True, alpha=0.3, axis='y')

# Subplot 2: Rally Rate
ax = axes[1]
rally_cats = ['Rally > 0', 'Rally ≥ 2']
rally_pcts = [rally_gt0, rally_ge2]
bars = ax.bar(rally_cats, rally_pcts, color=['#4CAF50', '#8BC34A'], edgecolor='white', linewidth=1)
for bar, pct in zip(bars, rally_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Rate (%)')
ax.set_title(f'Dual-Arm Rally Rate\nmerged_6034_5472 ({rally_eps} episodes)')
ax.set_ylim(0, max(rally_pcts) * 1.5)
ax.grid(True, alpha=0.3, axis='y')

# Subplot 3: Rally Equation Validation
ax = axes[2]
# Theory: rally_gt0 ≈ left_compat × right_compat
theory = left_pct / 100 * right_pct / 100 * 100
actual = rally_gt0
eq_cats = ['Predicted\n(L×R compat)', 'Actual\n(rally>0)', 'Rally≥2']
eq_vals = [theory, actual, rally_ge2]
eq_colors = ['#9E9E9E', '#4CAF50', '#FF9800']
bars = ax.bar(eq_cats, eq_vals, color=eq_colors, edgecolor='white', linewidth=1)
for bar, val, cat in zip(bars, eq_vals, eq_cats):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Rate (%)')
ax.set_title('Rally Equation Validation')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/09_eval_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {OUTDIR}/09_eval_results.png")

print(f"\n  Eval summary: left={left_pct:.1f}% right={right_pct:.1f}% "
      f"rally>0={rally_gt0:.1f}% rally>=2={rally_ge2:.1f}% "
      f"theory={theory:.1f}%")


# ============================================================
# Chart 10: Performance Metrics — FPS & Episode Length
# ============================================================
print("Chart 10: Performance Metrics...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

left_v9_dir = f'{RAIL_LOG}/2026-07-12_05-51-26_compat_v9_left'

# FPS
ax = axes[0]
s_fps, v_fps = load_smoothed(left_v9_dir, 'Perf/total_fps', window=20)
if s_fps is not None:
    ax.plot(s_fps, v_fps, color='#2196F3', linewidth=1.5)
ax.set_xlabel('Iteration'); ax.set_ylabel('FPS')
ax.set_title(f'Training Throughput\n(512 envs, headless, RTX 4060)')
ax.grid(True, alpha=0.3)

# Collection time
ax = axes[1]
s_col, v_col = load_smoothed(left_v9_dir, 'Perf/collection_time', window=20)
s_lrn, v_lrn = load_smoothed(left_v9_dir, 'Perf/learning_time', window=20)
if s_col is not None:
    ax.plot(s_col, v_col, label='Collection', color='#FF5722', linewidth=1.2)
if s_lrn is not None:
    ax.plot(s_lrn, v_lrn, label='Learning', color='#4CAF50', linewidth=1.2)
ax.set_xlabel('Iteration'); ax.set_ylabel('Time (s)')
ax.set_title('Time per Iteration')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Episode length
ax = axes[2]
s_ep, v_ep = load_smoothed(left_v9_dir, 'Train/mean_episode_length', window=20)
if s_ep is not None:
    ax.plot(s_ep, v_ep, color='#9C27B0', linewidth=1.5)
ax.set_xlabel('Iteration'); ax.set_ylabel('Steps')
ax.set_title('Mean Episode Length (max 500)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/10_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {OUTDIR}/10_performance.png")


# ============================================================
# Chart 11: Summary Poster Figure
# ============================================================
print("Chart 11: Summary Poster...")
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Title
fig.suptitle('Dual-Arm Ping-Pong RL: Opponent-Compatible Reward Training', fontsize=16, fontweight='bold', y=0.98)

# (0,0): Compat Rate Before/After
ax = fig.add_subplot(gs[0, 0])
x = np.arange(2); w = 0.35
before = [0.8, 0.4]; after = [left_pct, right_pct]
ax.bar(x - w/2, before, w, label='Pre-Compat', color='#BDBDBD')
ax.bar(x + w/2, after, w, label='Post-Compat (v9)', color=['#2196F3', '#FF5722'])
for i, (b, a) in enumerate(zip(before, after)):
    ax.text(i - w/2, b + 0.3, f'{b}%', ha='center', fontsize=9)
    ax.text(i + w/2, a + 0.3, f'{a}%', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(['Left Arm', 'Right Arm']); ax.set_ylabel('%')
ax.set_title('Compat Rate Improvement'); ax.legend(fontsize=7)

# (0,1): Rally Rate
ax = fig.add_subplot(gs[0, 1])
bars = ax.bar(['Rally>0', 'Rally≥2'], [rally_gt0, rally_ge2], color=['#4CAF50', '#8BC34A'], width=0.4)
for b, v in zip(bars, [rally_gt0, rally_ge2]):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Rally Rate (merged_6034_5472)')

# (0,2): Key Numbers
ax = fig.add_subplot(gs[0, 2])
ax.axis('off')
improvement_l = left_pct / 0.8; improvement_r = right_pct / 0.4
summary_text = (
    f"Left Compat:  0.8% → 17.0%  ({improvement_l:.0f}x)\n"
    f"Right Compat: 0.4% → 19.5%  ({improvement_r:.0f}x)\n"
    f"Rally>0:      11.2%\n"
    f"Chain Factor: {chain_factor:.1f}x\n"
    f"PPO iters:    500/run\n"
    f"Env:          Isaac Lab + Fairino3"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace')

# (1,0): Left Compat Progression (simplified)
ax = fig.add_subplot(gs[1, 0])
for (label, logdir), color in zip(left_compat_runs, plt.cm.Blues(np.linspace(0.3, 0.9, len(left_compat_runs)))):
    steps, values = load_smoothed(logdir, 'Episode_Reward/opponent_compatible')
    if steps is not None:
        alpha = 1.0 if 'v9' in label else 0.5
        lw = 2 if 'v9' in label else 0.8
        ax.plot(steps, values, linewidth=lw, alpha=alpha, color=color)
ax.set_xlabel('Iteration'); ax.set_ylabel('Compat Reward')
ax.set_title('Left Arm Compat Reward Progression')

# (1,1): Right Compat Progression (simplified)
ax = fig.add_subplot(gs[1, 1])
for (label, logdir), color in zip(right_compat_runs, plt.cm.Oranges(np.linspace(0.3, 0.9, len(right_compat_runs)))):
    steps, values = load_smoothed(logdir, 'Episode_Reward/opponent_compatible')
    if steps is not None:
        alpha = 1.0 if 'v9' in label else 0.5
        lw = 2 if 'v9' in label else 0.8
        ax.plot(steps, values, linewidth=lw, alpha=alpha, color=color)
ax.set_xlabel('Iteration'); ax.set_ylabel('Compat Reward')
ax.set_title('Right Arm Compat Reward Progression')

# (1,2): Rally Equation
ax = fig.add_subplot(gs[1, 2])
eq_cats = ['Theory\nL%×R%', f'Theory\n×{chain_factor:.1f}', 'Actual\nRally>0']
eq_vals = [theory, theory * chain_factor, rally_gt0]
eq_colors = ['#9E9E9E', '#FF9800', '#4CAF50']
bars = ax.bar(eq_cats, eq_vals, color=eq_colors, width=0.5)
for b, v in zip(bars, eq_vals):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, f'{v:.1f}%', ha='center', fontsize=10)
ax.set_title('Rally Equation Validation')

# (2,0): Loss
ax = fig.add_subplot(gs[2, 0])
s, v = load_smoothed(left_v9_dir, 'Loss/value')
if s is not None: ax.plot(s, v, color='#2196F3', linewidth=1)
ax.set_xlabel('Iter'); ax.set_ylabel('Value Loss'); ax.set_title('PPO Value Loss')

# (2,1): Reward Breakdown
ax = fig.add_subplot(gs[2, 1])
reward_tags_poster = [
    ('Episode_Reward/paddle_to_ball', 'paddle'),
    ('Episode_Reward/first_return_hit', 'hit'),
    ('Episode_Reward/return_ball', 'return'),
    ('Episode_Reward/net_direction', 'net_dir'),
    ('Episode_Reward/net_height', 'net_h'),
    ('Episode_Reward/predicted_landing', 'landing'),
    ('Episode_Reward/right_table_bounce', 'r_table'),
    ('Episode_Reward/opponent_compatible', 'compat'),
    ('Episode_Reward/opponent_vz_quality', 'vz'),
]
vals_p = []
names_p = []
for tag, name in reward_tags_poster:
    _, v = load_scalar(left_v9_dir, tag)
    if v is not None and len(v) > 0:
        vals_p.append(float(np.mean(v[-100:])))
        names_p.append(name)
ax.barh(names_p, vals_p, color=plt.cm.tab20(np.linspace(0, 0.9, len(names_p))))
ax.set_xlabel('Mean Reward'); ax.set_title('Reward Component Breakdown')

# (2,2): Architecture note
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')
arch_text = (
    "System Architecture:\n"
    "• Isaac Lab 5.1 + Isaac Sim 5.1\n"
    "• PPO (RSL-RL backend)\n"
    "• 2× Fairino3 7-DOF (rail)\n"
    "• DualArmActor: 2× MLP(32→7)\n"
    "• 14-DOF action, 46-dim obs\n"
    "• 512 parallel envs\n"
    "• RTX 4060 8GB, ~12000 FPS"
)
ax.text(0.05, 0.95, arch_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace')

plt.savefig(f'{OUTDIR}/11_summary_poster.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  -> {OUTDIR}/11_summary_poster.png")


# ============================================================
# Summary
# ============================================================
print("\nDone! Generated charts:")
total = 0
for f in sorted(os.listdir(OUTDIR)):
    if f.endswith('.png'):
        size_kb = os.path.getsize(os.path.join(OUTDIR, f)) / 1024
        print(f"  {f}  ({size_kb:.0f} KB)")
        total += size_kb
print(f"  Total: {total:.0f} KB")
