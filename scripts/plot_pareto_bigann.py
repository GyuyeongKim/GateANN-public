#!/usr/bin/env python3
"""Generate BigANN-100M Pareto figures: latency (1T) and throughput (32T)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Font
plt.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'

# ---- Style ----
FONTSIZE_LABEL = 16
FONTSIZE_TICK  = 14
FONTSIZE_LEGEND = 14
LINEWIDTH = 2.5
MARKERSIZE = 10

# ======================================================================
# Data from fig_filter_bigann100M_main.txt and fig_filter_bigann100M_diskann.txt
# ======================================================================

# --- DiskANN (mode=0), T=1 ---
diskann_1t_recall = [0.1003, 0.1504, 0.1995, 0.2500, 0.2964, 0.3416, 0.3841,
                     0.4587, 0.5236, 0.6301, 0.7106, 0.7717, 0.8377, 0.8831,
                     0.9041, 0.9405, 0.9611, 0.9734, 0.9812, 0.9896,
                     0.9959, 0.9984, 0.9995, 0.9997]
diskann_1t_qps    = [361, 368, 353, 343, 322, 313, 297,
                     279, 243, 204, 175, 151, 129, 109,
                     101, 82, 70, 61, 56, 46,
                     30, 22, 15, 11]

# --- DiskANN (mode=0), T=32 ---
diskann_32t_recall = [0.1003, 0.1504, 0.1995, 0.2500, 0.2964, 0.3416, 0.3841,
                      0.4587, 0.5236, 0.6301, 0.7106, 0.7717, 0.8377, 0.8831,
                      0.9041, 0.9405, 0.9611, 0.9734, 0.9812, 0.9896,
                     0.9959, 0.9984, 0.9995, 0.9997]
diskann_32t_qps    = [8655, 8602, 7885, 7765, 6895, 6057, 6002,
                      5367, 4773, 3987, 3315, 2915, 2428, 2024,
                      1837, 1527, 1307, 1124, 991, 794,
                     550, 391, 262, 198]

# --- PipeANN (mode=2), T=1 ---
pipe_1t_recall = [0.1373, 0.1867, 0.2324, 0.2751, 0.3162, 0.3542, 0.3907,
                  0.4590, 0.5210, 0.6271, 0.7087, 0.7698, 0.8369, 0.8825,
                  0.9037, 0.9402, 0.9609, 0.9732, 0.9810, 0.9896,
                     0.9958, 0.9984, 0.9995, 0.9997]
pipe_1t_qps    = [1770, 1749, 1700, 1650, 1587, 1524, 1461,
                  1337, 1268, 1133, 1028, 935, 821, 732,
                  678, 579, 508, 447, 407, 334,
                     196, 149, 97, 69]

# --- PipeANN (mode=2), T=32 ---
pipe_32t_recall = [0.1353, 0.1856, 0.2314, 0.2738, 0.3144, 0.3527, 0.3887,
                   0.4577, 0.5197, 0.6265, 0.7085, 0.7700, 0.8371, 0.8826,
                   0.9038, 0.9403, 0.9609, 0.9732, 0.9810, 0.9896,
                     0.9958, 0.9984, 0.9995, 0.9997]
pipe_32t_qps    = [15010, 15625, 13411, 11818, 10494, 9480, 8600,
                   7216, 6260, 4912, 4009, 3404, 2759, 2324,
                   2098, 1695, 1420, 1221, 1071, 860,
                     612, 430, 287, 215]

# --- GateANN (mode=8), T=1 ---
gate_1t_recall = [0.1061, 0.1524, 0.1985, 0.2428, 0.2839, 0.3237, 0.3623,
                     0.4319, 0.4935, 0.5951, 0.6726, 0.7320, 0.7965, 0.8412,
                     0.8633, 0.9030, 0.9266, 0.9420, 0.9532, 0.9669, 0.9802,
                     0.9885, 0.9940, 0.9963]
gate_1t_qps    = [3152, 4391, 3965, 3575, 3464, 3165, 3055,
                     3182, 3556, 2986, 2621, 2325, 1965, 1721,
                     1595, 1304, 1097, 991, 874, 710, 336,
                     346, 230, 170]

# --- GateANN (mode=8), T=32 ---
gate_32t_recall = [0.1054, 0.1519, 0.1983, 0.2424, 0.2832, 0.3231, 0.3615,
                     0.4316, 0.4927, 0.5941, 0.6720, 0.7314, 0.7959, 0.8411,
                     0.8639, 0.9036, 0.9272, 0.9424, 0.9535, 0.9672, 0.9803,
                     0.9885, 0.9940, 0.9963]
gate_32t_qps    = [16334, 89938, 76230, 73659, 68347, 60671, 56583,
                     50556, 44432, 35301, 30383, 26810, 18388, 20354,
                     20494, 16017, 13949, 11730, 10501, 8518, 3557,
                     2759, 2393, 1516]

# ======================================================================
# Figure 1: Latency Pareto (1 thread) -- FLIPPED: x=Latency, y=Recall
# ======================================================================
# Latency = 1000 / QPS (ms)
diskann_1t_lat = [1000.0 / q for q in diskann_1t_qps]
pipe_1t_lat    = [1000.0 / q for q in pipe_1t_qps]
gate_1t_lat    = [1000.0 / q for q in gate_1t_qps]

MARKERSIZE_PARETO = 6

fig, ax = plt.subplots(figsize=(4.1, 2.94))

ax.plot(diskann_1t_lat, diskann_1t_recall,
        color=(0.396, 0.761, 0.647), marker='^', markersize=MARKERSIZE_PARETO,
        linewidth=LINEWIDTH, linestyle='-', label='DiskANN', zorder=2)
ax.plot(pipe_1t_lat, pipe_1t_recall,
        color=(0.404, 0.553, 0.706), marker='o', markersize=MARKERSIZE_PARETO,
        linewidth=LINEWIDTH, linestyle='--', label='PipeANN', zorder=3)
ax.plot(gate_1t_lat, gate_1t_recall,
        color=(0.922, 0.514, 0.478), marker='s', markersize=MARKERSIZE_PARETO,
        linewidth=LINEWIDTH, linestyle='-', label='GateANN', zorder=4)

ax.set_xscale('log')
ax.set_xlabel('Latency (ms)', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_ylim(bottom=0.7, top=1.005)
plt.tight_layout()
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width, 1.52 / 2.94])
ax.legend(fontsize=FONTSIZE_LEGEND, loc='upper center',
          bbox_to_anchor=(0.5, 1.53), ncol=2, frameon=False)
out_base = '/home/node33/GateANN/figures/fig_pareto_bigann_lat'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close()

# ======================================================================
# Figure 2: Throughput Pareto (32 threads, log scale)
# ======================================================================
fig, ax = plt.subplots(figsize=(4.1, 3.1))

ax.plot(diskann_32t_recall, diskann_32t_qps,
        color=(0.396, 0.761, 0.647), marker='^', markersize=MARKERSIZE_PARETO,
        linewidth=LINEWIDTH, linestyle='-', label='DiskANN', zorder=2)
ax.plot(pipe_32t_recall, pipe_32t_qps,
        color=(0.404, 0.553, 0.706), marker='o', markersize=MARKERSIZE_PARETO,
        linewidth=LINEWIDTH, linestyle='--', label='PipeANN', zorder=3)
ax.plot(gate_32t_recall, gate_32t_qps,
        color=(0.922, 0.514, 0.478), marker='s', markersize=MARKERSIZE_PARETO,
        linewidth=LINEWIDTH, linestyle='-', label='GateANN', zorder=4)

ax.set_yscale('log')
ax.set_xlabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Throughput (QPS)', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(left=0.7, right=1.005)

plt.tight_layout()
out_base = '/home/node33/GateANN/figures/fig_pareto_bigann_tput'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close()
