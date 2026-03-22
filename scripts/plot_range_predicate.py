#!/usr/bin/env python3
"""Generate Figure: Range predicate experiment on BigANN-100M (norm bins).
(a) Recall vs Latency (T=1), (b) Recall vs QPS (T=32)."""
import re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULT_FILE = "/home/node33/PipeANN/data/filter/results/fig_range_predicate.txt"
FIG_DIR = "/home/node33/PipeANN/figures"

plt.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'

# Parse
data = {}
current_key = None
with open(RESULT_FILE) as f:
    for line in f:
        m = re.match(r'\[REPORT\]\s+(\S+)\s+range\s+T=(\d+)', line)
        if m:
            system = m.group(1).split('(')[0]
            threads = int(m.group(2))
            current_key = (system, threads)
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

print(f"Data loaded: {[(k, len(v['L'])) for k, v in data.items()]}")

FONTSIZE_LABEL = 22
FONTSIZE_TICK = 20
FONTSIZE_LEGEND = 20
LINEWIDTH = 2.5
MARKERSIZE = 10

colors = {'DiskANN': (0.396, 0.761, 0.647), 'PipeANN': (0.404, 0.553, 0.706), 'GateANN': (0.922, 0.514, 0.478)}
markers = {'DiskANN': '^', 'PipeANN': 'o', 'GateANN': 's'}

# --- (a) Latency (T=1) ---
fig, ax = plt.subplots(figsize=(6, 4))
for sys_name in ['DiskANN', 'PipeANN', 'GateANN']:
    key = (sys_name, 1)
    if key not in data:
        continue
    d = data[key]
    lat = [1000.0 / q for q in d['QPS']]
    pairs = [(r, l) for r, l in zip(d['Recall'], lat) if r >= 0.7]
    if not pairs:
        continue
    r, l = zip(*pairs)
    ax.plot(l, r, color=colors[sys_name], marker=markers[sys_name],
            markersize=MARKERSIZE, linewidth=LINEWIDTH, label=sys_name, zorder=3)

ax.set_xscale('log')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xlabel('Latency (ms)', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.legend(fontsize=FONTSIZE_LEGEND, loc='lower right', frameon=False)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_ylim(bottom=0.7, top=1.005)
plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig_range_lat')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out + '.eps', bbox_inches='tight')
print(f"Saved: {out}.png/.eps")
plt.close()

# --- (b) Throughput (T=32) ---
fig, ax = plt.subplots(figsize=(6, 4))
for sys_name in ['DiskANN', 'PipeANN', 'GateANN']:
    key = (sys_name, 32)
    if key not in data:
        continue
    d = data[key]
    pairs = [(r, q) for r, q in zip(d['Recall'], d['QPS']) if r >= 0.7]
    if not pairs:
        continue
    r, q = zip(*pairs)
    ax.plot(r, q, color=colors[sys_name], marker=markers[sys_name],
            markersize=MARKERSIZE, linewidth=LINEWIDTH, label=sys_name, zorder=3)

ax.set_yscale('log')
ax.set_xlabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Throughput (QPS)', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(left=0.7, right=1.02)
plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig_range_tput')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out + '.eps', bbox_inches='tight')
print(f"Saved: {out}.png/.eps")
plt.close()

# Summary
print("\n=== RANGE PREDICATE SUMMARY ===")
for T in [1, 32]:
    print(f"\n--- T={T} ---")
    for sys_name in ['DiskANN', 'PipeANN', 'GateANN']:
        key = (sys_name, T)
        if key not in data:
            continue
        d = data[key]
        for target in [0.85, 0.89]:
            for i, r in enumerate(d['Recall']):
                if r >= target:
                    print(f"  {sys_name}: recall>={target:.0%} -> QPS={d['QPS'][i]:.0f} (L={d['L'][i]})")
                    break
