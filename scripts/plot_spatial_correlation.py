#!/usr/bin/env python3
"""Generate Figure: Spatial label correlation effect on BigANN-100M.
Style matches plot_selectivity.py (Figure 13)."""
import re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULT_FILE = "/home/node33/PipeANN/data/filter/results/fig_spatial_correlation.txt"
FIG_DIR = "/home/node33/PipeANN/figures"

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

# Color scheme: blue shades for PipeANN, red shades for GateANN
PIPE_COLORS = {
    '0.0': (0.702, 0.776, 0.853),   # light blue
    '0.5': (0.404, 0.553, 0.706),   # medium blue
    '1.0': (0.263, 0.361, 0.459),   # dark blue
}
GATE_COLORS = {
    '0.0': (0.953, 0.710, 0.686),   # light coral
    '0.5': (0.922, 0.514, 0.478),   # medium red
    '1.0': (0.600, 0.333, 0.310),   # dark red
}
PIPE_LINESTYLES = {
    '0.0': (0, (5, 2)),
    '0.5': (0, (5, 2, 1, 2)),
    '1.0': (0, (1, 2)),
}
GATE_LINESTYLES = {
    '0.0': '-',
    '0.5': (0, (3, 1, 1, 1)),
    '1.0': (0, (5, 1)),
}

# Parse results
data = {}
current_key = None
with open(RESULT_FILE) as f:
    for line in f:
        m = re.match(r'\[REPORT\]\s+(\S+)\s+(alpha=[\d.]+)\s+T=(\d+)', line)
        if m:
            system = m.group(1).split('(')[0]
            alpha_str = m.group(2)
            threads = int(m.group(3))
            current_key = (system, alpha_str, threads)
            data[current_key] = {'L': [], 'QPS': [], 'Recall': []}
            continue
        if current_key:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    L = int(parts[0])
                    qps = float(parts[1])
                    recall = float(parts[2])
                    data[current_key]['L'].append(L)
                    data[current_key]['QPS'].append(qps)
                    data[current_key]['Recall'].append(recall)
                except ValueError:
                    pass

# --- Throughput vs Recall (T=32) ---
fig, ax = plt.subplots(figsize=(8, 4.5))

for alpha_val in ['0.0', '0.5', '1.0']:
    alpha_key = f'alpha={alpha_val}'

    # PipeANN
    pipe_key = ('PipeANN', alpha_key, 32)
    if pipe_key in data:
        d = data[pipe_key]
        r_filt = [(r, q) for r, q in zip(d['Recall'], d['QPS']) if r >= 0.7]
        if r_filt:
            r, q = zip(*r_filt)
            ax.plot(r, q, color=PIPE_COLORS[alpha_val], marker='^',
                    markersize=MARKERSIZE, linewidth=LINEWIDTH,
                    linestyle=PIPE_LINESTYLES[alpha_val],
                    label=f'PipeANN $\\alpha$={alpha_val}', zorder=2)

    # GateANN
    gate_key = ('GateANN', alpha_key, 32)
    if gate_key in data:
        d = data[gate_key]
        r_filt = [(r, q) for r, q in zip(d['Recall'], d['QPS']) if r >= 0.7]
        if r_filt:
            r, q = zip(*r_filt)
            ax.plot(r, q, color=GATE_COLORS[alpha_val], marker='s',
                    markersize=MARKERSIZE, linewidth=LINEWIDTH,
                    linestyle=GATE_LINESTYLES[alpha_val],
                    label=f'GateANN $\\alpha$={alpha_val}', zorder=3)

ax.set_yscale('log')
ax.set_xlabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Throughput (QPS)', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.legend(bbox_to_anchor=(0.5, 1.35), loc='upper center', ncol=3,
          frameon=False, fontsize=FONTSIZE_LEGEND)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(left=0.7, right=1.02)
ax.set_ylim(top=1.5e5)

plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig_spatial_tput')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out + '.eps', bbox_inches='tight')
print(f"Saved: {out}.png/.eps")
plt.close()

# --- Summary ---
print("\n=== SPATIAL CORRELATION SUMMARY ===")
for alpha_val in ['0.0', '0.5', '1.0']:
    alpha_key = f'alpha={alpha_val}'
    print(f"\nalpha={alpha_val}:")
    for sys_name in ['PipeANN', 'GateANN']:
        key = (sys_name, alpha_key, 32)
        if key not in data:
            continue
        d = data[key]
        for target in [0.90, 0.95]:
            for i, r in enumerate(d['Recall']):
                if r >= target:
                    print(f"  {sys_name}: recall>={target:.0%} -> QPS={d['QPS'][i]:.0f} (L={d['L'][i]})")
                    break
        max_r = max(d['Recall'])
        max_i = d['Recall'].index(max_r)
        print(f"  {sys_name}: max recall={max_r:.4f}, QPS={d['QPS'][max_i]:.0f}")
