#!/usr/bin/env python3
"""Generate Zipf workload figures: fig_zipf_lat and fig_zipf_tput."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re

plt.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'

FONTSIZE_LABEL = 22
FONTSIZE_TICK  = 20
FONTSIZE_LEGEND = 20
LINEWIDTH = 2.5
MARKERSIZE = 10

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


# ---- Load zipf data ----
zipf_file = '/home/node33/PipeANN/data/filter/results/fig_filter_bigann100M_zipf.txt'
zipf_data = parse_report_sections(zipf_file)

# Keys (PipeANN and GateANN only -- no DiskANN in these plots)
pipe_1t_key = 'Baseline(post-filter) zipf T=1 bigann100M'
gate_1t_key = 'FilterAware(mode=8) zipf T=1 bigann100M'

pipe_32t_key = 'Baseline(post-filter) zipf T=32 bigann100M'
gate_32t_key = 'FilterAware(mode=8) zipf T=32 bigann100M'

# Extract data
pipe_1t = zipf_data[pipe_1t_key]
gate_1t = zipf_data[gate_1t_key]

pipe_32t = zipf_data[pipe_32t_key]
gate_32t = zipf_data[gate_32t_key]

def extract_recall_qps(rows):
    recall = [r[2] for r in rows]
    qps = [r[1] for r in rows]
    return recall, qps

def filt(recall, vals, min_r=0.7):
    r, v = zip(*[(r, v) for r, v in zip(recall, vals) if r >= min_r])
    return list(r), list(v)

# ========== fig_zipf_lat: X=Latency (ms, log), Y=Recall@10, 1T ==========

p_recall, p_qps = extract_recall_qps(pipe_1t)
g_recall, g_qps = extract_recall_qps(gate_1t)

p_lat = [1000.0 / q for q in p_qps]
g_lat = [1000.0 / q for q in g_qps]

p_recall, p_lat = filt(p_recall, p_lat)
g_recall, g_lat = filt(g_recall, g_lat)

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(p_lat, p_recall,
        color=(0.404, 0.553, 0.706), marker='^', markersize=MARKERSIZE,
        linewidth=LINEWIDTH, linestyle='--', label='PipeANN', zorder=2)
ax.plot(g_lat, g_recall,
        color=(0.922, 0.514, 0.478), marker='s', markersize=MARKERSIZE,
        linewidth=LINEWIDTH, linestyle='-', label='GateANN', zorder=3)

ax.set_xscale('log')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xlabel('Latency (ms)', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_ylim(bottom=0.7, top=1.005)
ax.legend(fontsize=FONTSIZE_LEGEND, loc='lower right', frameon=False)

plt.tight_layout()

out_base = '/home/node33/GateANN/figures/fig_zipf_lat'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close(fig)

# ========== fig_zipf_tput: X=Recall, Y=QPS (log), 32T ==========

p_recall, p_qps = extract_recall_qps(pipe_32t)
g_recall, g_qps = extract_recall_qps(gate_32t)

p_recall, p_qps = filt(p_recall, p_qps)
g_recall, g_qps = filt(g_recall, g_qps)

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(p_recall, p_qps,
        color=(0.404, 0.553, 0.706), marker='^', markersize=MARKERSIZE,
        linewidth=LINEWIDTH, linestyle='--', label='PipeANN', zorder=2)
ax.plot(g_recall, g_qps,
        color=(0.922, 0.514, 0.478), marker='s', markersize=MARKERSIZE,
        linewidth=LINEWIDTH, linestyle='-', label='GateANN', zorder=3)

ax.set_yscale('log')
ax.set_xlabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Throughput (QPS)', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(left=0.7, right=1.005)

plt.tight_layout()

out_base = '/home/node33/GateANN/figures/fig_zipf_tput'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close(fig)
