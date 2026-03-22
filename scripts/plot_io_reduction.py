#!/usr/bin/env python3
"""Generate I/O reduction figures: fig_io_vs_l and fig_io_theory."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re

plt.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'

FONTSIZE_LABEL = 24
FONTSIZE_TICK  = 22
FONTSIZE_LEGEND = 22
LINEWIDTH = 2.5
MARKERSIZE = 10

# ---- Parsing helpers ----

def parse_sections(filepath):
    """Parse result file into sections keyed by header line.
    Returns dict: section_header -> list of (L, QPS, Recall, MeanIOs, FilterSkips) tuples.
    """
    sections = {}
    current = None
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('=== '):
                current = line.strip('= ')
                sections[current] = []
                continue
            if current is None:
                continue
            if line.startswith('---') or line.strip() == '' or line.startswith('['):
                if line.startswith('['):
                    current = None
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    L = int(parts[0])
                    QPS = float(parts[1])
                    recall = float(parts[2])
                    mean_ios = float(parts[3])
                    fskips = float(parts[4])
                    sections[current].append((L, QPS, recall, mean_ios, fskips))
                except ValueError:
                    continue
    return sections


def parse_report_sections(filepath):
    """Parse result file into sections keyed by [REPORT] description.
    Returns dict: report_desc -> list of (L, QPS, Recall, MeanIOs, FilterSkips).
    """
    sections = {}
    current_report = None
    current_table = None
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r'\[REPORT\]\s+(.*)', line)
            if m:
                current_report = m.group(1)
                continue
            if line.startswith('=== '):
                current_table = line.strip('= ')
                if current_report:
                    key = current_report
                    sections[key] = []
                continue
            if current_report is None or current_table is None:
                continue
            if line.startswith('---') or line.strip() == '':
                continue
            if line.startswith('['):
                current_table = None
                current_report = None
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    L = int(parts[0])
                    QPS = float(parts[1])
                    recall = float(parts[2])
                    mean_ios = float(parts[3])
                    fskips = float(parts[4])
                    key = current_report if current_report else current_table
                    if key not in sections:
                        sections[key] = []
                    sections[key].append((L, QPS, recall, mean_ios, fskips))
                except ValueError:
                    continue
    return sections


# ---- Load main data (sel=10%) ----
main_file = '/home/node33/PipeANN/data/filter/results/fig_filter_bigann100M_main.txt'
main_data = parse_report_sections(main_file)

# Extract 1T data for I/O vs L plot
pipe_1t_key = 'Baseline(post-filter) sel=10% T=1 bigann100M'
gate_1t_key = 'FilterAware(mode=8) sel=10% T=1 bigann100M'

pipe_1t = main_data[pipe_1t_key]
gate_1t = main_data[gate_1t_key]

pipe_L = [r[0] for r in pipe_1t]
pipe_ios = [r[3] for r in pipe_1t]
gate_L = [r[0] for r in gate_1t]
gate_ios = [r[3] for r in gate_1t]

# ========== fig_io_vs_l: X=L, Y=Mean I/Os per query ==========
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(pipe_L, pipe_ios,
        color=(0.404, 0.553, 0.706), marker='^', markersize=MARKERSIZE,
        linewidth=LINEWIDTH, linestyle='--', label='PipeANN', zorder=2)
ax.plot(gate_L, gate_ios,
        color=(0.922, 0.514, 0.478), marker='s', markersize=MARKERSIZE,
        linewidth=LINEWIDTH, linestyle='-', label='GateANN', zorder=3)

ax.set_xlabel('Search list size $L$', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Mean IOs per query', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.legend(fontsize=FONTSIZE_LEGEND, loc='upper left',
          frameon=False)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

out_base = '/home/node33/GateANN/figures/fig_io_vs_l'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close(fig)

# ========== fig_io_theory: measured I/O reduction vs expected 1/s ==========
# Load selectivity sweep data
sel_file = '/home/node33/PipeANN/data/filter/results/fig_filter_bigann100M_sel.txt'
sel_data = parse_report_sections(sel_file)

selectivities = [0.05, 0.10, 0.20]
sel_labels = ['5%', '10%', '20%']

# For the 10% sel, use main_file (1T data); for 5% and 20%, use sel_file (1T data)
# We pick L=200 for the comparison point
target_L = 200

def get_ios_at_L(data_list, L_target):
    """Get MeanIOs for the row closest to L_target."""
    for row in data_list:
        if row[0] == L_target:
            return row[3]
    return None

pipe_ios_by_sel = []
gate_ios_by_sel = []

# sel=5% T=1
pipe_5_key = 'Baseline(post-filter) sel=5% T=1 bigann100M'
gate_5_key = 'FilterAware(mode=8) sel=5% T=1 bigann100M'
pipe_ios_by_sel.append(get_ios_at_L(sel_data[pipe_5_key], target_L))
gate_ios_by_sel.append(get_ios_at_L(sel_data[gate_5_key], target_L))

# sel=10% T=1 (from main_file)
pipe_ios_by_sel.append(get_ios_at_L(pipe_1t, target_L))
gate_ios_by_sel.append(get_ios_at_L(gate_1t, target_L))

# sel=20% T=1
pipe_20_key = 'Baseline(post-filter) sel=20% T=1 bigann100M'
gate_20_key = 'FilterAware(mode=8) sel=20% T=1 bigann100M'
pipe_ios_by_sel.append(get_ios_at_L(sel_data[pipe_20_key], target_L))
gate_ios_by_sel.append(get_ios_at_L(sel_data[gate_20_key], target_L))

# Measured reduction ratio = PipeANN_IOs / GateANN_IOs  (how many times fewer IOs)
measured_ratio = [p / g for g, p in zip(gate_ios_by_sel, pipe_ios_by_sel)]
# Expected ratio = 1/selectivity
expected_ratio = [1.0 / s for s in selectivities]

# Colors: one per selectivity (matching sel_qps/sel_speedup)
bar_colors = [(0.922, 0.514, 0.478),   # 5% red
              (0.404, 0.553, 0.706),   # 10% blue
              (0.396, 0.761, 0.647)]    # 20% green
gray_color = (0.827, 0.827, 0.827)

x = np.arange(len(selectivities))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4))

# Plot measured bars (one color per selectivity) and theoretical bars (gray)
bars1 = ax.bar(x - width/2, measured_ratio, width,
               color=bar_colors, edgecolor='black', linewidth=0.5,
               label='Measured')
bars2 = ax.bar(x + width/2, expected_ratio, width,
               color=gray_color, edgecolor='black', linewidth=0.5,
               label='Expected (1/$s$)')

# Add value annotations on measured bars
for i in range(len(selectivities)):
    ax.text(x[i] - width/2, measured_ratio[i] + 0.5,
            f'{measured_ratio[i]:.1f}$\\times$', ha='center', va='bottom',
            fontsize=FONTSIZE_TICK)

ax.set_xlabel('Selectivity', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('IO reduction ratio', fontsize=FONTSIZE_LABEL)
ax.set_xticks(x)
ax.set_xticklabels(sel_labels)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.legend(fontsize=FONTSIZE_LEGEND, loc='upper right',
          frameon=False)
ax.grid(True, linestyle='--', alpha=0.4, axis='y')
ax.set_ylim(0, 40)

plt.tight_layout()

out_base = '/home/node33/GateANN/figures/fig_io_theory'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close(fig)
