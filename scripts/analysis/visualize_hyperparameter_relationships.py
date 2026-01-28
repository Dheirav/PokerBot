#!/usr/bin/env python3
"""
Generate visualizations for proven hyperparameter relationships.

This script creates publication-quality plots showing the empirical relationships
between population size, matchups, hands, and mutation sigma based on tournament data.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Tournament data: (name, win_rate)
configs = [
    # Batch 1 winners
    ('p40_m8_h375_s0.1_g50', 78.7),
    ('p40_m6_h500_s0.15_g200', 73.1),
    ('p40_m8_h375_s0.1_g200', 72.2),
    ('p20_m6_h500_s0.15_g200', 71.3),
    ('p12_m6_h500_s0.15_g200', 68.5),
    
    # Batch 2 winners
    ('p12_m8_h500_s0.08_g200', 81.2),
    ('p12_m6_h750_s0.08_g50', 67.4),
    ('p12_m6_h375_s0.1_g50', 63.9),
    ('p12_m6_h750_s0.1_g50', 62.5),
    ('p12_m6_h750_s0.1_g200', 61.1),
]

# Parse configs
parsed = []
for name, wr in configs:
    parts = name.replace('_g50', '').replace('_g200', '').split('_')
    pop = int(parts[0][1:])
    matchups = int(parts[1][1:])
    hands = int(parts[2][1:])
    sigma = float(parts[3][1:])
    parsed.append((pop, matchups, hands, sigma, wr, name))

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'p12': '#e74c3c', 'p20': '#f39c12', 'p40': '#27ae60'}

# Create output directory
output_dir = Path('tournament_reports/hyperparameter_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating hyperparameter relationship visualizations...")
print("=" * 80)

# ============================================================================
# Figure 1: Population vs Matchups
# ============================================================================
print("\n1. Creating Population vs Matchups plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Relationship 1: Population Size ↔ Matchups per Agent', 
             fontsize=16, fontweight='bold', y=1.02)

# Left plot: Absolute values
for pop_size in [12, 20, 40]:
    data = [(p, m, wr) for p, m, h, s, wr, n in parsed if p == pop_size]
    if data:
        pops, matchups, wrs = zip(*data)
        color = f'p{pop_size}'
        ax1.scatter(matchups, wrs, s=200, alpha=0.7, 
                   color=colors.get(color, '#3498db'), 
                   label=f'pop={pop_size}', edgecolors='black', linewidth=1.5)

ax1.set_xlabel('Matchups per Agent', fontsize=12, fontweight='bold')
ax1.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Win Rate vs Matchups (by Population)', fontsize=13)
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(55, 85)

# Right plot: Ratio-based
for pop_size in [12, 20, 40]:
    data = [(p, m/p, wr) for p, m, h, s, wr, n in parsed if p == pop_size]
    if data:
        pops, ratios, wrs = zip(*data)
        color = f'p{pop_size}'
        ax2.scatter(ratios, wrs, s=200, alpha=0.7, 
                   color=colors.get(color, '#3498db'),
                   label=f'pop={pop_size}', edgecolors='black', linewidth=1.5)

# Add trend zones
ax2.axvspan(0.15, 0.25, alpha=0.2, color='green', label='Large Pop Zone (15-25%)')
ax2.axvspan(0.50, 0.67, alpha=0.2, color='orange', label='Small Pop Zone (50-67%)')

ax2.set_xlabel('Matchups / Population Ratio', fontsize=12, fontweight='bold')
ax2.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Win Rate vs Matchup Ratio', fontsize=13)
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(55, 85)

plt.tight_layout()
plt.savefig(output_dir / 'relationship_1_population_vs_matchups.png', 
            dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_dir / 'relationship_1_population_vs_matchups.png'}")

# ============================================================================
# Figure 2: Matchups vs Hands
# ============================================================================
print("\n2. Creating Matchups vs Hands plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Relationship 2: Matchups ↔ Hands per Matchup', 
             fontsize=16, fontweight='bold', y=1.02)

# Left plot: Matchups vs Hands (bubble size = win rate)
matchups_data = {}
for p, m, h, s, wr, n in parsed:
    key = (m, h)
    if key not in matchups_data:
        matchups_data[key] = []
    matchups_data[key].append(wr)

for (m, h), wrs in matchups_data.items():
    avg_wr = np.mean(wrs)
    size = 100 + avg_wr * 10  # Scale bubble size by win rate
    color_val = avg_wr / 100  # Normalize for colormap
    ax1.scatter(m, h, s=size, alpha=0.6, c=[color_val], cmap='RdYlGn', 
               vmin=0.5, vmax=0.85, edgecolors='black', linewidth=1.5)
    ax1.text(m, h, f'{avg_wr:.0f}%', ha='center', va='center', 
            fontsize=9, fontweight='bold')

ax1.set_xlabel('Matchups per Agent', fontsize=12, fontweight='bold')
ax1.set_ylabel('Hands per Matchup', fontsize=12, fontweight='bold')
ax1.set_title('Performance Map (bubble size = win rate)', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.set_xticks([6, 8])
ax1.set_yticks([375, 500, 750])

# Right plot: Total evaluations vs Win rate
total_evals_data = []
for p, m, h, s, wr, n in parsed:
    total_evals = m * h
    total_evals_data.append((total_evals, wr, f'm{m}+h{h}'))

total_evals_data.sort(key=lambda x: x[0])
evals, wrs, labels = zip(*total_evals_data)

scatter = ax2.scatter(evals, wrs, s=150, alpha=0.7, c=wrs, cmap='RdYlGn', 
                     vmin=55, vmax=85, edgecolors='black', linewidth=1.5)

# Highlight best performers
best_configs = [(3000, 78.7, 'm8+h375'), (4000, 81.2, 'm8+h500')]
for ev, wr, lbl in best_configs:
    ax2.scatter(ev, wr, s=400, alpha=0.3, color='gold', edgecolors='gold', 
               linewidth=3, marker='*', zorder=10)

# Add optimal zone
ax2.axvspan(3000, 4500, alpha=0.15, color='green', label='Optimal Zone')

ax2.set_xlabel('Total Evaluations (matchups × hands)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Win Rate vs Total Evaluations', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(55, 85)

plt.colorbar(scatter, ax=ax2, label='Win Rate (%)')
plt.tight_layout()
plt.savefig(output_dir / 'relationship_2_matchups_vs_hands.png', 
            dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_dir / 'relationship_2_matchups_vs_hands.png'}")

# ============================================================================
# Figure 3: Population vs Sigma
# ============================================================================
print("\n3. Creating Population vs Sigma plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Relationship 3: Population Size ↔ Mutation Sigma', 
             fontsize=16, fontweight='bold', y=1.02)

# Left plot: Scatter with colors by population
for pop_size in [12, 20, 40]:
    data = [(s, wr) for p, m, h, s, wr, n in parsed if p == pop_size]
    if data:
        sigmas, wrs = zip(*data)
        color = f'p{pop_size}'
        ax1.scatter(sigmas, wrs, s=200, alpha=0.7, 
                   color=colors.get(color, '#3498db'),
                   label=f'pop={pop_size}', edgecolors='black', linewidth=1.5)

# Add empirical formula line
pop_range = np.linspace(12, 60, 100)
sigma_formula = 0.5 / np.sqrt(pop_range)
ax1_twin = ax1.twiny()
ax1_twin.plot(sigma_formula, pop_range, 'k--', alpha=0.3, linewidth=2, 
             label='σ ≈ 0.5/√pop')
ax1_twin.set_xlabel('Theoretical Sigma (from formula)', fontsize=10, style='italic')
ax1_twin.set_xlim(0.04, 0.16)

ax1.set_xlabel('Mutation Sigma', fontsize=12, fontweight='bold')
ax1.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Win Rate vs Sigma (by Population)', fontsize=13)
ax1.legend(fontsize=11, loc='lower left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(55, 85)
ax1.set_xlim(0.06, 0.16)

# Right plot: Heat map showing population vs sigma
pop_values = [12, 20, 40]
sigma_values = [0.08, 0.10, 0.12, 0.15]

# Create matrix
matrix = np.zeros((len(pop_values), len(sigma_values)))
for i, pop in enumerate(pop_values):
    for j, sig in enumerate(sigma_values):
        matches = [wr for p, m, h, s, wr, n in parsed if p == pop and abs(s - sig) < 0.01]
        matrix[i, j] = np.mean(matches) if matches else np.nan

im = ax2.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=60, vmax=82)
ax2.set_xticks(range(len(sigma_values)))
ax2.set_yticks(range(len(pop_values)))
ax2.set_xticklabels([f'{s:.2f}' for s in sigma_values])
ax2.set_yticklabels([f'{p}' for p in pop_values])
ax2.set_xlabel('Mutation Sigma', fontsize=12, fontweight='bold')
ax2.set_ylabel('Population Size', fontsize=12, fontweight='bold')
ax2.set_title('Performance Heat Map', fontsize=13)

# Add text annotations
for i in range(len(pop_values)):
    for j in range(len(sigma_values)):
        if not np.isnan(matrix[i, j]):
            text = ax2.text(j, i, f'{matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", 
                          fontsize=10, fontweight='bold')

# Add formula overlay
ax2.text(0.95, 0.05, 'Formula: σ ≈ 0.5/√pop', 
        transform=ax2.transAxes, fontsize=11, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
        ha='right', va='bottom')

plt.colorbar(im, ax=ax2, label='Win Rate (%)')
plt.tight_layout()
plt.savefig(output_dir / 'relationship_3_population_vs_sigma.png', 
            dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_dir / 'relationship_3_population_vs_sigma.png'}")

# ============================================================================
# Figure 4: Comprehensive Overview
# ============================================================================
print("\n4. Creating comprehensive overview plot...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Hyperparameter Relationships - Comprehensive Overview', 
             fontsize=18, fontweight='bold', y=0.98)

# Top left: Win rate distribution by population
ax1 = fig.add_subplot(gs[0, 0])
pop_groups = {}
for p, m, h, s, wr, n in parsed:
    if p not in pop_groups:
        pop_groups[p] = []
    pop_groups[p].append(wr)

positions = []
data_to_plot = []
labels = []
for pop in sorted(pop_groups.keys()):
    positions.append(pop)
    data_to_plot.append(pop_groups[pop])
    labels.append(f'p{pop}')

bp = ax1.boxplot(data_to_plot, positions=positions, widths=3, patch_artist=True,
                 showmeans=True, meanline=True)
for patch, pop in zip(bp['boxes'], sorted(pop_groups.keys())):
    color = colors.get(f'p{pop}', '#3498db')
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_xlabel('Population Size', fontsize=11, fontweight='bold')
ax1.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
ax1.set_title('Win Rate Distribution by Population', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(55, 85)

# Top middle: Matchups distribution
ax2 = fig.add_subplot(gs[0, 1])
matchup_groups = {}
for p, m, h, s, wr, n in parsed:
    if m not in matchup_groups:
        matchup_groups[m] = []
    matchup_groups[m].append(wr)

positions = list(sorted(matchup_groups.keys()))
data_to_plot = [matchup_groups[m] for m in positions]

bp = ax2.boxplot(data_to_plot, positions=positions, widths=0.5, patch_artist=True,
                 showmeans=True, meanline=True)
for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.7)

ax2.set_xlabel('Matchups per Agent', fontsize=11, fontweight='bold')
ax2.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
ax2.set_title('Win Rate Distribution by Matchups', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(55, 85)

# Top right: Sigma distribution
ax3 = fig.add_subplot(gs[0, 2])
sigma_groups = {}
for p, m, h, s, wr, n in parsed:
    if s not in sigma_groups:
        sigma_groups[s] = []
    sigma_groups[s].append(wr)

positions = list(sorted(sigma_groups.keys()))
data_to_plot = [sigma_groups[s] for s in positions]

bp = ax3.boxplot(data_to_plot, positions=positions, widths=0.01, patch_artist=True,
                 showmeans=True, meanline=True)
for patch in bp['boxes']:
    patch.set_facecolor('#9b59b6')
    patch.set_alpha(0.7)

ax3.set_xlabel('Mutation Sigma', fontsize=11, fontweight='bold')
ax3.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
ax3.set_title('Win Rate Distribution by Sigma', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(55, 85)

# Middle row: 3D-style visualization (population, matchups, win rate)
ax4 = fig.add_subplot(gs[1, :])
for p, m, h, s, wr, n in parsed:
    color = colors.get(f'p{p}', '#3498db')
    size = (wr - 50) * 10  # Scale by win rate
    ax4.scatter(p, m, s=size, alpha=0.6, color=color, edgecolors='black', linewidth=1)
    
# Add champion marker
champion = [(p, m, wr) for p, m, h, s, wr, n in parsed if wr > 78][0]
ax4.scatter(champion[0], champion[1], s=500, alpha=0.3, color='gold', 
           edgecolors='gold', linewidth=3, marker='*', zorder=10, label='Champion')

ax4.set_xlabel('Population Size', fontsize=12, fontweight='bold')
ax4.set_ylabel('Matchups per Agent', fontsize=12, fontweight='bold')
ax4.set_title('Population vs Matchups (bubble size = win rate)', fontsize=13)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

# Bottom left: Best configs ranking
ax5 = fig.add_subplot(gs[2, :])
top_configs = sorted(parsed, key=lambda x: x[4], reverse=True)[:8]
names = [f"{n.split('_')[0]}\n{n.split('_')[1]}\n{n.split('_')[2]}" for p, m, h, s, wr, n in top_configs]
wrs = [wr for p, m, h, s, wr, n in top_configs]
colors_list = [colors.get(f'p{p}', '#3498db') for p, m, h, s, wr, n in top_configs]

bars = ax5.barh(range(len(names)), wrs, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.set_yticks(range(len(names)))
ax5.set_yticklabels(names, fontsize=9)
ax5.set_xlabel('Win Rate (%)', fontsize=12, fontweight='bold')
ax5.set_title('Top 8 Configurations', fontsize=13)
ax5.grid(True, alpha=0.3, axis='x')
ax5.set_xlim(55, 85)

# Add value labels
for i, (bar, wr) in enumerate(zip(bars, wrs)):
    ax5.text(wr + 0.5, i, f'{wr:.1f}%', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'comprehensive_overview.png', 
            dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_dir / 'comprehensive_overview.png'}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("✓ All visualizations generated successfully!")
print(f"\nOutput location: {output_dir.absolute()}")
print("\nGenerated files:")
print("  1. relationship_1_population_vs_matchups.png")
print("  2. relationship_2_matchups_vs_hands.png")
print("  3. relationship_3_population_vs_sigma.png")
print("  4. comprehensive_overview.png")
print("\nThese visualizations have been added to your tournament reports.")
print("=" * 80)
