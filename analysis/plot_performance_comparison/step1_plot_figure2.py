# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
import os

# Set random seed for reproducibility (must match original)
np.random.seed(42)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "legend.frameon": True,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": False,
    "legend.fontsize": 7,
    "legend.borderpad": 0.4,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.4,
})

# ========== Data loading ==========
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load SPAN 10-run data
span_df = pd.read_csv(os.path.join(script_dir, 'span_10runs.csv'))
span_runs = span_df[['AUC', 'AUPR', 'ACC', 'F1', 'Precision', 'Recall']].values

# Load system means
means_df = pd.read_csv(os.path.join(script_dir, 'system_means.csv'))
system_means = {}
for _, row in means_df.iterrows():
    system_means[row['System']] = row[['AUC', 'AUPR', 'ACC', 'F1', 'Precision', 'Recall']].values.astype(float)

# ========== Prepare data ==========
metric_names = ['AUC', 'AUPR', 'ACC', 'F1', 'Precision', 'Recall']
panel_a_data = [span_runs[:, i] for i in range(6)]

# ========== Plotting ==========
fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.5), dpi=300)

# ========== Panel a: SPAN 10-run boxplot + jittered scatter ==========
ax1 = axes[0]

n_runs = 10
bp = ax1.boxplot(panel_a_data, positions=range(1, 7), widths=0.5, patch_artist=True,
                 showmeans=True,
                 meanprops=dict(marker='D', markerfacecolor='#E74C3C',
                               markeredgecolor='black', markersize=7))

# Beautify box patches
colors = ['#D6EAF8', '#D6EAF8', '#FADBD8', '#FADBD8', '#D6EAF8', '#D6EAF8']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('#2C3E50')
    patch.set_linewidth(1.2)
for median in bp['medians']:
    median.set_color('#2C3E50')
    median.set_linewidth(2)

# Jittered scatter points (10 runs, different colors)
run_colors = plt.cm.tab10(np.linspace(0, 1, n_runs))
for i, data in enumerate(panel_a_data):
    x_jittered = np.random.normal(i+1, 0.06, n_runs)
    for j, (x, y) in enumerate(zip(x_jittered, data)):
        ax1.scatter(x, y, s=40, color=run_colors[j], edgecolors='white',
                   linewidth=0.5, alpha=0.85, zorder=3)

# Run legend (all 10 runs, upper right)
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=run_colors[i], markersize=6,
                              label=f'Run {i+1}') for i in range(n_runs)]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=7,
          frameon=True, ncol=5, title='Independent runs', title_fontsize=7, handletextpad=0.2, columnspacing=0.2)

# Statistical annotations + p-value (3 lines, below lowest point)
for i, data in enumerate(panel_a_data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    cv = std / mean * 100
    # Get lower whisker endpoint
    lower_whisker_y = bp['whiskers'][2*i+1].get_ydata()[1]
    # Get lowest scatter point
    min_scatter_y = np.min(data)
    # Take the lower of the two, then subtract 0.004
    lowest_y = min(lower_whisker_y, min_scatter_y) - 0.004   
    ax1.text(i+1, lowest_y, f'{mean:.4f} \u00b1 {std:.4f}\nCV={cv:.2f}%',
                ha='center', va='top', fontsize=7, color='#555555')

ax1.set_xticks(range(1, 7))
ax1.set_xticklabels(metric_names, fontsize=7)
ax1.set_ylabel('Performance', fontsize=7)
ax1.set_ylim(0.895, 1.005)
ax1.set_title('Robustness analysis of SPAN across 10 independent runs',
              fontsize=7, fontweight='bold', loc='left')

ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))
ax1.tick_params(axis="y", labelsize=7, direction="in")
ax1.tick_params(direction="in", width=1.0, length=4)

ax1.grid(True, axis="y", which="major", color="#E0E0E0", linewidth=0.5, linestyle="-")
ax1.grid(True, axis="y", which="minor", color="#F0F0F0", linewidth=0.3, linestyle="-")
# Vertical grid lines (x-axis direction, preserved from original)
ax1.grid(True, axis="x", which="major", color="#E0E0E0", linewidth=0.5, linestyle="-")
ax1.set_axisbelow(True)

# Remove top and right spines (matching Figure 3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_linewidth(1.0)
ax1.spines["bottom"].set_linewidth(1.0)

# Panel label: 8pt bold upright lowercase "a" at top-left
ax1.text(-0.02, 1.02, 'a', transform=ax1.transAxes,
         fontsize=8, fontweight='bold', va='bottom', ha='right')

# ========== Panel b: Multi-system comparison ==========
ax2 = axes[1]

x_pos = np.arange(1, 7)
dodge = 0.15

# System styles — Okabe-Ito colorblind-safe palette
system_styles = {
    'SPAN':      {'color': '#D55E00', 'marker': 'o', 'size': 9,  'zorder': 5},
    'TriFusion': {'color': '#CC79A7', 'marker': 's', 'size': 70, 'zorder': 4},
    'MDformer':  {'color': '#95A5A6', 'marker': '^', 'size': 70, 'zorder': 4},
    'AMHMDA':    {'color': '#0072B2', 'marker': 'D', 'size': 60, 'zorder': 4},
    'HGCLAMIR':  {'color': '#E69F00', 'marker': 'p', 'size': 60, 'zorder': 4},
}

# SPAN: mean + 95% CI (based on 10-run data)
span_means = [np.mean(d) for d in panel_a_data]
span_ci = [1.96 * np.std(d, ddof=1) / np.sqrt(len(d)) for d in panel_a_data]

# Other baseline systems (from system_means.csv)
offsets = {'TriFusion': 0, 'MDformer': 1, 'AMHMDA': 2, 'HGCLAMIR': 3}

# Collect legend handles and labels
legend_handles = []
legend_labels = []

# Plot SPAN with error bars
span_handle = ax2.errorbar(x_pos - 0.5*dodge, span_means, yerr=span_ci, fmt='o', markersize=7,
            color='#D55E00', capsize=4, capthick=2, elinewidth=1.5,
            zorder=5)
legend_handles.insert(0, span_handle)
legend_labels.insert(0, 'SPAN (mean \u00b1 95% CI, n=10)')

# Plot baseline systems
for name, offset in offsets.items():
    style = system_styles[name]
    means = system_means[name]
    handle = ax2.scatter(x_pos + offset*dodge, means, marker=style['marker'],
               s=style['size'], color=style['color'],
               zorder=style['zorder'])
    legend_handles.append(handle)
    legend_labels.append(f'{name} (mean)')

# Legend (SPAN first)
ax2.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(1.0, 0.95),
          fontsize=7, frameon=True, fancybox=False,
          framealpha=0.9, edgecolor='#CCCCCC',
          labelspacing=0.8, handlelength=1.2, handletextpad=0.4, borderpad=0.4)

# Improvement percentage annotation (SPAN vs TriFusion) — placed to the RIGHT of all data points
trifusion_means = system_means['TriFusion']
for i in [0, 2, 3]:
    improvement = (span_means[i] - trifusion_means[i]) / trifusion_means[i] * 100
    ax2.annotate(f'+{improvement:.1f}%',
                xy=(x_pos[i] + 1*dodge - 0.12, span_means[i]),
                ha='left', va='center', fontsize=7, color='#D55E00', fontweight='bold')

ax2.set_xticks(range(1, 7))
ax2.set_xticklabels(metric_names, fontsize=7)
ax2.set_ylabel('Performance', fontsize=7)
ax2.set_ylim(0.85, 1.005)
ax2.set_title('Comparative performance against state-of-the-art methods',
              fontsize=7, fontweight='bold', loc='left')

# Y-axis ticks: major every 0.02, minor every 0.01 
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
ax2.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))
ax2.tick_params(axis="y", labelsize=7, direction="in")
ax2.tick_params(direction="in", width=1.0, length=4)

# Grid: major + minor 
# Horizontal grid lines (y-axis direction)
ax2.grid(True, axis="y", which="major", color="#E0E0E0", linewidth=0.5, linestyle="-")
ax2.grid(True, axis="y", which="minor", color="#F0F0F0", linewidth=0.3, linestyle="-")
# Vertical grid lines (x-axis direction, preserved from original)
ax2.grid(True, axis="x", which="major", color="#E0E0E0", linewidth=0.5, linestyle="-")
ax2.set_axisbelow(True)

# Remove top and right spines
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_linewidth(1.0)
ax2.spines["bottom"].set_linewidth(1.0)

# Panel label: 8pt bold upright lowercase "b" at top-left
ax2.text(-0.02, 1.02, 'b', transform=ax2.transAxes,
         fontsize=8, fontweight='bold', va='bottom', ha='right')

plt.tight_layout()
plt.subplots_adjust(hspace=0.2)

# Ensure tick direction is inward after tight_layout
for ax in axes:
    ax.tick_params(direction="in", which="both")

# Save in two formats:
# - PDF: vector format for final artwork submission 
# - PNG: high-resolution raster for embedding in Word manuscript
pdf_path = os.path.join(script_dir, 'figure2_performance.pdf')
png_path = os.path.join(script_dir, 'figure2_performance.png')

fig.savefig(pdf_path, bbox_inches='tight',
            facecolor='white', edgecolor='none',
            dpi=300)
fig.savefig(png_path, bbox_inches='tight',
            facecolor='white', edgecolor='none',
            dpi=600)
plt.close(fig)
print("Figure 2 saved to:")
print(f"  PDF (vector): {pdf_path}")
print(f"  PNG (600 DPI): {png_path}")
