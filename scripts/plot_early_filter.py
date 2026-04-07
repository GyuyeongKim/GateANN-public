#!/usr/bin/env python3
"""Generate Early Filter Check comparison figures.

BigANN-100M (sel=10%):
  fig_early_filter_lat:  Recall vs Latency (1T) — PipeANN, EarlyFilter, GateANN
  fig_early_filter_tput: Recall vs QPS (32T)    — PipeANN, EarlyFilter, GateANN
"""

import re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

COLOR_PIPEANN  = (0.404, 0.553, 0.706)  # blue
COLOR_EARLY    = (0.945, 0.784, 0.533)   # orange
COLOR_GATEANN  = (0.922, 0.514, 0.478)   # red

RESULTS_DIR = '/Users/gykim/workspace/PipeANN/data/filter/results'
FIG_DIR = '/Users/gykim/workspace/PipeANN/figures'

# ---- Parsing helper ----
def parse_table(filepath, report_key, table_prefix='=== '):
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
            if not found_report:
                continue
            if s.startswith('=== '):
                in_table = True
                continue
            if s.startswith('---') or s.startswith('L ') or s == '':
                continue
            if s.startswith('[REPORT]'):
                break
            parts = s.split()
            if len(parts) >= 3:
                try:
                    rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
                    in_table = True  # data found, we're in the table
                except ValueError:
                    pass
    return rows

def filt(recall, vals, min_r=0.0):
    r, v = zip(*[(r, v) for r, v in zip(recall, vals) if r >= min_r])
    return list(r), list(v)

# ---- Load data ----

# PipeANN (mode=2) from main results
main_file = os.path.join(RESULTS_DIR, 'fig_filter_bigann100M_main.txt')
pipe_1t = parse_table(main_file, 'Baseline(post-filter) sel=10% T=1 bigann100M')
pipe_32t = parse_table(main_file, 'Baseline(post-filter) sel=10% T=32 bigann100M')

# GateANN (mode=8) from main results
gate_1t = parse_table(main_file, 'FilterAware(mode=8) sel=10% T=1 bigann100M')
gate_32t = parse_table(main_file, 'FilterAware(mode=8) sel=10% T=32 bigann100M')

# Early Filter (mode=9)
early_file = os.path.join(RESULTS_DIR, 'fig_early_filter.txt')
early_1t = parse_table(early_file, 'EarlyFilter(mode=9) sel=10% T=1 bigann100M')
early_32t = parse_table(early_file, 'EarlyFilter(mode=9) sel=10% T=32 bigann100M')

print(f"Data points: PipeANN 1T={len(pipe_1t)} 32T={len(pipe_32t)}")
print(f"             GateANN 1T={len(gate_1t)} 32T={len(gate_32t)}")
print(f"             Early   1T={len(early_1t)} 32T={len(early_32t)}")

# ---- (a) Recall vs Latency (1T) ----
fig, ax = plt.subplots(figsize=(6, 4))

for rows, color, marker, ls, label, zorder in [
    (pipe_1t,  COLOR_PIPEANN, 'o', '--', 'PipeANN (Post)',      2),
    (early_1t, COLOR_EARLY,   'D', '-',  'PipeANN (Early)',  3),
    (gate_1t,  COLOR_GATEANN, 's', '-',  'GateANN (Pre)',      4),
]:
    if not rows:
        continue
    recall = [r[2] for r in rows]
    lat = [1000.0 / r[1] for r in rows]  # ms
    recall, lat = filt(recall, lat, 0.7)
    ax.plot(lat, recall, color=color, marker=marker, markersize=MARKERSIZE,
            linewidth=LINEWIDTH, linestyle=ls, label=label, zorder=zorder)

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
out = os.path.join(FIG_DIR, 'fig_early_filter_lat')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out + '.eps', bbox_inches='tight')
print(f"Saved: {out}.png/.eps")
plt.close()

# ---- (b) Recall vs QPS (32T) ----
fig, ax = plt.subplots(figsize=(6, 4))

# Post and Early overlap (≈identical QPS) — plot both with distinct markers
for rows, color, marker, ms, ls, label, zorder in [
    (pipe_32t,  COLOR_PIPEANN, 'o', MARKERSIZE+2, '--', 'PipeANN (Post)',  2),
    (early_32t, COLOR_EARLY,   'D', MARKERSIZE,   '-',  'PipeANN (Early)', 3),
    (gate_32t,  COLOR_GATEANN, 's', MARKERSIZE,   '-',  'GateANN (Pre)',   4),
]:
    if not rows:
        continue
    recall = [r[2] for r in rows]
    qps = [r[1] for r in rows]
    recall, qps = filt(recall, qps, 0.7)
    ax.plot(recall, qps, color=color, marker=marker, markersize=ms,
            linewidth=LINEWIDTH, linestyle=ls, label=label, zorder=zorder)

ax.set_yscale('log')
ax.set_xlabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Throughput (QPS)', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(left=0.7, right=1.02)

plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig_early_filter_tput')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out + '.eps', bbox_inches='tight')
print(f"Saved: {out}.png/.eps")
plt.close()

# ---- Summary ----
print("\n=== EARLY FILTER SUMMARY ===")
for label, rows_1t, rows_32t in [
    ('PipeANN', pipe_1t, pipe_32t),
    ('EarlyFilter', early_1t, early_32t),
    ('GateANN', gate_1t, gate_32t),
]:
    if not rows_32t:
        continue
    for target in [0.90, 0.95]:
        for L, qps, r in rows_32t:
            if r >= target:
                print(f"  {label} T=32: recall>={target:.0%} -> QPS={qps:.0f} (L={L})")
                break
