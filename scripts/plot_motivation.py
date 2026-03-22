#!/usr/bin/env python3
"""Generate motivation figures: thread scaling (DiskANN+PipeANN only) and
recall collapse (post-filter vs naive pre-filter)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'

# ======================================================================
# Figure 1: Motivation Thread Scaling (DiskANN + PipeANN only)
# Data: QPS at L=200 for BigANN-100M (sel=10%)
# From fig_filter_bigann100M_main.txt and fig_filter_bigann100M_diskann.txt
# ======================================================================

# ---- Style for thread scaling ----
FONTSIZE_LABEL_TS = 22
FONTSIZE_TICK_TS  = 19
FONTSIZE_LEGEND_TS = 20
LINEWIDTH_TS = 2.5
MARKERSIZE_TS = 10

threads = [1, 4, 8, 16, 32]

# DiskANN (mode=0): only T=1 and T=32 available
diskann_threads = [1, 32]
diskann_qps     = [101, 1837]  # L=200

# PipeANN (mode=2): T=1,4,8,16,32
pipe_qps = [678, 1895, 2090, 2100, 2098]  # L=200

fig, ax = plt.subplots(figsize=(5.18, 4.0))

ax.plot(diskann_threads, diskann_qps,
        color=(0.396, 0.761, 0.647), marker='^', markersize=MARKERSIZE_TS,
        linewidth=LINEWIDTH_TS, linestyle='-', label='DiskANN', zorder=2)
ax.plot(threads, pipe_qps,
        color=(0.404, 0.553, 0.706), marker='o', markersize=MARKERSIZE_TS,
        linewidth=LINEWIDTH_TS, linestyle='--', label='PipeANN', zorder=3)

ax.set_xlabel('Threads', fontsize=FONTSIZE_LABEL_TS)
ax.set_ylabel('Throughput (QPS)', fontsize=FONTSIZE_LABEL_TS)
ax.set_xticks(threads)
ax.set_xticklabels([str(t) for t in threads])
ax.tick_params(labelsize=FONTSIZE_TICK_TS)
ax.set_yscale('log')
ax.legend(fontsize=FONTSIZE_LEGEND_TS, loc='upper center',
          bbox_to_anchor=(0.5, 1.32), ncol=2,
          frameon=False)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
out_base = '/home/node33/GateANN/figures/fig_motivation_thread_scaling'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close()

# ======================================================================
# Figure 2: Motivation — Post-filter vs Naive Pre-filter (Recall vs QPS)
# BigANN-100M sel=10%, T=32
# Post-filter data from main results; pre-filter from naive_prefilter_raw.txt
# ======================================================================

import re, os

FONTSIZE_LABEL_C = 22
FONTSIZE_TICK_C  = 19
FONTSIZE_LEGEND_C = 20
LINEWIDTH_C = 2.5
MARKERSIZE_C = 10

RESULTS_DIR = '/home/node33/PipeANN/data/filter/results'

def parse_table(filepath, report_key):
    """Parse L / QPS / Recall rows from a [REPORT] section."""
    rows = []
    found_report = False
    in_table = False
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if re.match(r'\[REPORT\]\s+' + re.escape(report_key), s):
                found_report = True
                in_table = False
                continue
            if found_report and s.startswith('=== '):
                in_table = True
                continue
            if not in_table:
                continue
            if s.startswith('---') or s.startswith('L ') or s == '':
                continue
            if s.startswith('['):
                break
            parts = s.split()
            if len(parts) >= 3:
                try:
                    rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    pass
    return rows

# Load PipeANN post-filter T=32
main_file = os.path.join(RESULTS_DIR, 'fig_filter_bigann100M_main.txt')
post_rows = parse_table(main_file, 'Baseline(post-filter) sel=10% T=32 bigann100M')

# Load naive pre-filter T=32 (from Part D-3)
# File uses "### NaivePrefilter T=32 ###" format, not [REPORT]
naive_file = os.path.join(RESULTS_DIR, 'naive_prefilter_raw.txt')
naive_rows = []
if os.path.exists(naive_file):
    with open(naive_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    naive_rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    pass
else:
    print("WARNING: naive_prefilter_raw.txt not yet generated (Part D-3)")

print(f"Post-filter points: {len(post_rows)}, Naive pre-filter points: {len(naive_rows)}")

fig, ax = plt.subplots(figsize=(5.14, 4.2))

if post_rows:
    recall_post = [r[2] for r in post_rows]
    qps_post    = [r[1] for r in post_rows]
    ax.plot(recall_post, qps_post,
            color=(0.404, 0.553, 0.706), marker='o', markersize=MARKERSIZE_C,
            linewidth=LINEWIDTH_C, linestyle='-', label='Post-filter', zorder=3)

if naive_rows:
    recall_naive = [r[2] for r in naive_rows]
    qps_naive    = [r[1] for r in naive_rows]
    ax.plot(recall_naive, qps_naive,
            color=(0.922, 0.514, 0.478), marker='s', markersize=MARKERSIZE_C,
            linewidth=LINEWIDTH_C, linestyle='--', label='Naive pre-filter', zorder=2)

ax.set_xlabel('Recall@10', fontsize=FONTSIZE_LABEL_C)
ax.set_ylabel('Throughput (QPS)', fontsize=FONTSIZE_LABEL_C)
ax.set_yscale('log')
ax.tick_params(labelsize=FONTSIZE_TICK_C)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width, 1.922 / 4.2])
ax.legend(fontsize=FONTSIZE_LEGEND_C, loc='upper center',
          bbox_to_anchor=(0.5, 1.555), ncol=1,
          frameon=False)
out_base = '/home/node33/GateANN/figures/fig_motivation_collapse'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close()
