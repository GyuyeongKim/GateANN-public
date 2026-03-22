#!/usr/bin/env python3
"""Generate selectivity figures: fig_sel_qps and fig_sel_speedup."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re

# Font: Nimbus Sans
plt.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'

FONTSIZE_LABEL = 20
FONTSIZE_TICK  = 18
FONTSIZE_LEGEND = 18
LINEWIDTH = 2.5
MARKERSIZE = 8

# Colors per selectivity level (spectrum: lighter = lower selectivity)
PIPE_COLORS = {
    '5%':  (0.702, 0.776, 0.853),    # light blue
    '10%': (0.404, 0.553, 0.706),    # medium blue
    '20%': (0.263, 0.361, 0.459),    # dark blue
}
GATE_COLORS = {
    '5%':  (0.953, 0.710, 0.686),    # light coral
    '10%': (0.922, 0.514, 0.478),    # medium red
    '20%': (0.600, 0.333, 0.310),    # dark red
}
# Linestyles per selectivity
PIPE_LINESTYLES = {
    '5%':  (0, (5, 2)),         # dashed
    '10%': (0, (5, 2, 1, 2)),   # dash-dot
    '20%': (0, (1, 2)),         # dotted
}
GATE_LINESTYLES = {
    '5%':  '-',                 # solid
    '10%': (0, (3, 1, 1, 1)),   # dash-dot-dot
    '20%': (0, (5, 1)),         # long dash
}

# ---- Parsing helper ----

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
                    sections[current_report] = []
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
                    sections[current_report].append((L, QPS, recall, mean_ios, fskips))
                except ValueError:
                    continue
    return sections


def pareto_filter(recall, qps):
    """Remove dominated points where another has both higher recall and higher QPS."""
    pairs = list(zip(recall, qps))
    result = [(r, q) for r, q in pairs
              if not any(r2 > r and q2 > q for r2, q2 in pairs)]
    if result:
        return [p[0] for p in result], [p[1] for p in result]
    return recall, qps


# ---- Load selectivity sweep data (T=32) ----
sel_file = '/home/node33/PipeANN/data/filter/results/fig_filter_bigann100M_sel.txt'
sel_data = parse_report_sections(sel_file)

# Also load main file for sel=10% T=32
main_file = '/home/node33/PipeANN/data/filter/results/fig_filter_bigann100M_main.txt'
main_data = parse_report_sections(main_file)

# Keys for 32T data
sel_configs = [
    ('5%',  0.05,
     'Baseline(post-filter) sel=5% T=32 bigann100M',
     'FilterAware(mode=8) sel=5% T=32 bigann100M',
     sel_data),
    ('10%', 0.10,
     'Baseline(post-filter) sel=10% T=32 bigann100M',
     'FilterAware(mode=8) sel=10% T=32 bigann100M',
     main_data),
    ('20%', 0.20,
     'Baseline(post-filter) sel=20% T=32 bigann100M',
     'FilterAware(mode=8) sel=20% T=32 bigann100M',
     sel_data),
]

# ========== fig_sel_qps: X=Recall, Y=QPS (log), 32T ==========
fig, ax = plt.subplots(figsize=(8, 4.5))

for i, (sel_label, sel_val, pipe_key, gate_key, data_src) in enumerate(sel_configs):
    pipe_rows = data_src[pipe_key]
    gate_rows = data_src[gate_key]

    p_recall = [r[2] for r in pipe_rows]
    p_qps = [r[1] for r in pipe_rows]
    g_recall = [r[2] for r in gate_rows]
    g_qps = [r[1] for r in gate_rows]

    # Filter for recall >= 0.25
    p_r, p_q = zip(*[(r, q) for r, q in zip(p_recall, p_qps) if r >= 0.25])
    g_r, g_q = zip(*[(r, q) for r, q in zip(g_recall, g_qps) if r >= 0.25])

    # Remove dominated points (avoids V-shaped dips from outlier L values)
    p_r, p_q = pareto_filter(list(p_r), list(p_q))
    g_r, g_q = pareto_filter(list(g_r), list(g_q))

    ax.plot(p_r, p_q,
            color=PIPE_COLORS[sel_label], marker='^', markersize=MARKERSIZE,
            linewidth=LINEWIDTH, linestyle=PIPE_LINESTYLES[sel_label],
            label=f'PipeANN {sel_label}', zorder=2)
    ax.plot(g_r, g_q,
            color=GATE_COLORS[sel_label], marker='s', markersize=MARKERSIZE,
            linewidth=LINEWIDTH, linestyle=GATE_LINESTYLES[sel_label],
            label=f'GateANN {sel_label}', zorder=3)

ax.set_yscale('log')
ax.set_xlabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Throughput (QPS)', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.legend(bbox_to_anchor=(0.5, 1.35), loc='upper center', ncol=3,
          frameon=False, fontsize=FONTSIZE_LEGEND)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(left=0.8, right=1.01)
ax.set_ylim(top=1.5e5)

plt.tight_layout()

out_base = '/home/node33/GateANN/figures/fig_sel_qps'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close(fig)

# ========== fig_sel_speedup: bar chart at ~90% recall ==========

def get_qps_at_recall(rows, target_recall):
    """Find the QPS at the L value that gives recall closest to target."""
    best = None
    best_diff = float('inf')
    for row in rows:
        diff = abs(row[2] - target_recall)
        if diff < best_diff:
            best_diff = diff
            best = row
    return best[1] if best else None

target_recall = 0.90

sel_labels_bar = []
speedups = []

for sel_label, sel_val, pipe_key, gate_key, data_src in sel_configs:
    pipe_rows = data_src[pipe_key]
    gate_rows = data_src[gate_key]

    pipe_qps = get_qps_at_recall(pipe_rows, target_recall)
    gate_qps = get_qps_at_recall(gate_rows, target_recall)
    speedup = gate_qps / pipe_qps if pipe_qps else 0

    sel_labels_bar.append(sel_label)
    speedups.append(speedup)

x = np.arange(len(sel_labels_bar))

# Colored bars per selectivity
bar_colors = [(0.922, 0.514, 0.478), (0.404, 0.553, 0.706), (0.396, 0.761, 0.647)]

fig, ax = plt.subplots(figsize=(8, 3.5))

bars = ax.bar(x, speedups, width=0.5, color=bar_colors, edgecolor='black',
              linewidth=0.5)

# Add value labels on top of bars (using multiplication sign)
for bar, sp in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f'{sp:.1f}$\\times$', ha='center', va='bottom',
            fontsize=FONTSIZE_LEGEND)

ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
ax.set_xlabel('Selectivity', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Speedup ($\\times$)', fontsize=FONTSIZE_LABEL)
ax.set_xticks(x)
ax.set_xticklabels(sel_labels_bar)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.grid(True, linestyle='--', alpha=0.4, axis='y')
ax.set_ylim(0, max(speedups) * 1.15)

plt.tight_layout()

out_base = '/home/node33/GateANN/figures/fig_sel_speedup'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close(fig)
