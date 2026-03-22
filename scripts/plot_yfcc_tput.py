#!/usr/bin/env python3
"""Generate YFCC-10M throughput figure matching figure 7 style."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Use Nimbus Sans (metric-compatible Helvetica clone)
plt.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'

# ---- Data: PipeANN 32T (combined base + supplement + supplement2 + largeL + plateau) ----
# Sorted by recall (ascending)
pipe_recall = [0.0963, 0.1202, 0.1388, 0.1541, 0.1674, 0.1793, 0.1901,
               0.2094, 0.2262, 0.2546, 0.2782, 0.2989, 0.3251, 0.3601,
               0.4106, 0.4738,
               0.4962, 0.5147, 0.5307, 0.5567, 0.5772, 0.6017, 0.6318,
               0.6723, 0.6994, 0.7195, 0.7484, 0.7592, 0.7770,
               0.8074, 0.8276]
pipe_qps    = [10318, 9467, 8639, 7931, 7327, 6803, 6345,
               5605, 5008, 4113, 3478, 3012, 2503, 1951,
               1351, 834,
               697, 600, 527, 425, 354, 284, 214,
               143, 107, 86, 61, 54, 43,
               29, 22]

# ---- Data: GateANN 32T (combined base + supplement2 + largeL + plateau) ----
# Sorted by recall (ascending)
gate_recall = [0.0760, 0.1003, 0.1201, 0.1367, 0.1515, 0.1645, 0.1762,
               0.1974, 0.2154, 0.2456, 0.2705, 0.2918, 0.3186, 0.3544,
               0.4050, 0.4684,
               0.4905, 0.5090, 0.5250, 0.5509, 0.5711, 0.5955,
               0.6254, 0.6658, 0.6925, 0.7126, 0.7413, 0.7524, 0.7700,
               0.8004, 0.8205, 0.8472,
               0.8777, 0.8958, 0.9129]
gate_qps    = [107849, 146324, 125162, 109027, 86963, 101127, 86889,
               86032, 78109, 64379, 55138, 48258, 40468, 32598,
               23215, 15241,
               11276, 10850, 9692, 7415, 6610, 5315,
               4141, 2674, 2018, 1512, 995, 826, 608,
               327, 207, 106,
               45, 26, 14]

# ---- Filter: only show recall >= 0.25 for cleaner plot ----
def filt(recall, qps, min_r=0.25):
    r, q = zip(*[(r, q) for r, q in zip(recall, qps) if r >= min_r])
    return list(r), list(q)

pipe_recall, pipe_qps = filt(pipe_recall, pipe_qps)
gate_recall, gate_qps = filt(gate_recall, gate_qps)

# ---- Style matching figure 7 ----
FONTSIZE_LABEL = 20
FONTSIZE_TICK  = 18
FONTSIZE_LEGEND = 18
LINEWIDTH = 2.5
MARKERSIZE = 10

fig, ax = plt.subplots(figsize=(8, 3.5))

ax.plot(pipe_recall, pipe_qps,
        color=(0.404, 0.553, 0.706), marker='^', markersize=MARKERSIZE,
        linewidth=LINEWIDTH, label='PipeANN', zorder=2)
ax.plot(gate_recall, gate_qps,
        color=(0.922, 0.514, 0.478), marker='s', markersize=MARKERSIZE,
        linewidth=LINEWIDTH, label='GateANN', zorder=3)

ax.set_yscale('log')
ax.set_xlabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Throughput (QPS)', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.legend(fontsize=FONTSIZE_LEGEND, loc='upper right',
          frameon=False)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(left=0.5, right=1.0)

plt.tight_layout()

# Save PNG + EPS
out_base = '/home/node33/GateANN/figures/fig_yfcc_tput'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
