#!/usr/bin/env python3
"""Generate BigANN-100M thread scaling figure (L=200)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'

# ---- Style ----
FONTSIZE_LABEL = 18
FONTSIZE_TICK  = 16
FONTSIZE_LEGEND = 16
LINEWIDTH = 2
MARKERSIZE = 10

# ======================================================================
# Data: QPS at L=200 for BigANN-100M (sel=10%)
# Extracted from fig_filter_bigann100M_main.txt (T=1,4,8,16,32)
# and fig_filter_bigann100M_diskann.txt (T=1,32)
# ======================================================================

threads = [1, 4, 8, 16, 32]

# DiskANN (mode=0): only T=1 and T=32 available in data files
diskann_threads = [1, 32]
diskann_qps     = [101, 1837]  # L=200

# PipeANN (mode=2): T=1,4,8,16,32
pipe_qps = [678, 1895, 2090, 2100, 2098]  # L=200

# GateANN (mode=8): T=1,4,8,16,32
gate_qps = [1595, 5857, 9897, 16907, 20494]  # L=200

# ======================================================================
# Figure: Thread Scaling
# ======================================================================
fig, ax = plt.subplots(figsize=(7.38, 3.3))

ax.plot(diskann_threads, diskann_qps,
        color=(0.396, 0.761, 0.647), marker='^', markersize=6,
        linewidth=LINEWIDTH, linestyle='-', label='DiskANN', zorder=2)
ax.plot(threads, pipe_qps,
        color=(0.404, 0.553, 0.706), marker='o', markersize=6,
        linewidth=LINEWIDTH, linestyle='--', label='PipeANN', zorder=3)
ax.plot(threads, gate_qps,
        color=(0.922, 0.514, 0.478), marker='s', markersize=6,
        linewidth=LINEWIDTH, linestyle='-', label='GateANN', zorder=4)

ax.set_yscale('log')
ax.set_xlabel('Number of threads', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Throughput (QPS)', fontsize=FONTSIZE_LABEL)
ax.set_xticks(threads)
ax.set_xticklabels([str(t) for t in threads])
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.legend(fontsize=FONTSIZE_LEGEND, loc='upper center',
          bbox_to_anchor=(0.5, 1.28), ncol=3,
          frameon=False)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
out_base = '/Users/gykim/workspace/GateANN/figures/fig_thread_scaling'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close()
