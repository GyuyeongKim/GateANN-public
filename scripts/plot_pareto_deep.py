#!/usr/bin/env python3
"""Generate DEEP-100M Pareto figures: latency (1T) and throughput (32T)."""

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
# Data from fig_filter_deep100M_main.txt, fig_filter_deep100M_diskann.txt,
# and fig_filter_deep100M_diskann_32T.txt
# ======================================================================

# --- DiskANN (mode=0), T=1 (from fig_filter_deep100M_diskann.txt) ---
diskann_1t_recall = [0.0990, 0.1478, 0.1959, 0.2433, 0.2872, 0.3281, 0.3650,
                      0.4282, 0.4838, 0.5774, 0.6487, 0.7073, 0.7721, 0.8199,
                      0.8444, 0.8895, 0.9183, 0.9383, 0.9527, 0.9695, 0.9849,
                      0.9927, 0.9966, 0.9979]
diskann_1t_qps    = [358, 358, 341, 330, 316, 303, 278,
                   253, 236, 209, 165, 148, 124, 107,
                   98, 80, 68, 60, 53, 43, 29,
                   21, 14, 11]

# --- DiskANN (mode=0), T=32 (from fig_filter_deep100M_diskann_32T.txt) ---
diskann_32t_recall = [0.0990, 0.1478, 0.1959, 0.2433, 0.2872, 0.3281, 0.3650,
                       0.4282, 0.4838, 0.5774, 0.6487, 0.7073, 0.7721, 0.8199,
                       0.8444, 0.8895, 0.9183, 0.9383, 0.9527, 0.9695, 0.9849,
                       0.9927, 0.9966, 0.9979]
diskann_32t_qps    = [7395, 8264, 7347, 6966, 6571, 6181, 5714,
                    4781, 4562, 3799, 3129, 2703, 2343, 2013,
                    1799, 1463, 1262, 1101, 960, 774, 540,
                    382, 258, 195]

# --- PipeANN (mode=2), T=1 ---
pipe_1t_recall = [0.1298, 0.1802, 0.2227, 0.2625, 0.2979, 0.3335, 0.3667,
                   0.4283, 0.4843, 0.5771, 0.6486, 0.7068, 0.7717, 0.8198,
                   0.8442, 0.8896, 0.9183, 0.9385, 0.9528, 0.9697, 0.9850,
                   0.9928, 0.9966, 0.9979]
pipe_1t_qps    = [1096, 1379, 1387, 1494, 1406, 1376, 1343,
                1255, 1207, 1088, 983, 901, 782, 649,
                665, 568, 494, 435, 322, 315, 172,
                135, 80, 65]

# --- PipeANN (mode=2), T=32 ---
pipe_32t_recall = [0.1292, 0.1798, 0.2223, 0.2610, 0.2979, 0.3328, 0.3656,
                    0.4268, 0.4820, 0.5757, 0.6483, 0.7072, 0.7724, 0.8201,
                    0.8445, 0.8898, 0.9184, 0.9386, 0.9529, 0.9697, 0.9850,
                    0.9928, 0.9966, 0.9979]
pipe_32t_qps    = [15310, 15162, 13014, 11441, 10156, 9166, 8337,
                 7077, 6106, 4803, 3957, 3355, 2728, 2302,
                 2083, 1685, 1411, 1217, 1067, 857, 594,
                 417, 278, 209]

# --- GateANN (mode=8), T=1 ---
gate_1t_recall = [0.0988, 0.1436, 0.1855, 0.2267, 0.2651, 0.3022, 0.3357,
                   0.3985, 0.4544, 0.5465, 0.6177, 0.6742, 0.7358, 0.7823,
                   0.8074, 0.8527, 0.8838, 0.9054, 0.9208, 0.9411, 0.9627,
                   0.9768, 0.9867, 0.9910]
gate_1t_qps    = [2731, 4301, 3903, 3663, 3420, 3248, 3111,
                2730, 3071, 2720, 2399, 2148, 1821, 1600,
                1490, 1202, 1050, 924, 827, 620, 378,
                344, 238, 168]

# --- GateANN (mode=8), T=32 ---
gate_32t_recall = [0.0986, 0.1431, 0.1850, 0.2260, 0.2645, 0.3018, 0.3353,
                    0.3981, 0.4537, 0.5456, 0.6175, 0.6746, 0.7360, 0.7827,
                    0.8084, 0.8534, 0.8845, 0.9059, 0.9212, 0.9414, 0.9629,
                    0.9770, 0.9868, 0.9911]
gate_32t_qps    = [24233, 84580, 82225, 74653, 69444, 68691, 67326,
                 53715, 49191, 41489, 37690, 32668, 26631, 22588,
                 20201, 16558, 13766, 11949, 10446, 8324, 5613,
                 3965, 2642, 1947]

# ======================================================================
# Figure 1: Latency Pareto (1 thread) -- FLIPPED: x=Latency, y=Recall
# ======================================================================
diskann_1t_lat = [1000.0 / q for q in diskann_1t_qps]
pipe_1t_lat    = [1000.0 / q for q in pipe_1t_qps]
gate_1t_lat    = [1000.0 / q for q in gate_1t_qps]

fig, ax = plt.subplots(figsize=(4.1, 3.1))

ax.plot(diskann_1t_lat, diskann_1t_recall,
        color=(0.396, 0.761, 0.647), marker='^', markersize=6,
        linewidth=LINEWIDTH, linestyle='-', label='DiskANN', zorder=2)
ax.plot(pipe_1t_lat, pipe_1t_recall,
        color=(0.404, 0.553, 0.706), marker='o', markersize=6,
        linewidth=LINEWIDTH, linestyle='--', label='PipeANN', zorder=3)
ax.plot(gate_1t_lat, gate_1t_recall,
        color=(0.922, 0.514, 0.478), marker='s', markersize=6,
        linewidth=LINEWIDTH, linestyle='-', label='GateANN', zorder=4)

ax.set_xscale('log')
ax.set_xlabel('Latency (ms)', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_ylim(bottom=0.7, top=1.005)

plt.tight_layout()
out_base = '/Users/gykim/workspace/GateANN/figures/fig_pareto_deep_lat'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close()

# ======================================================================
# Figure 2: Throughput Pareto (32 threads, log scale)
# ======================================================================
fig, ax = plt.subplots(figsize=(4.1, 3.1))

ax.plot(diskann_32t_recall, diskann_32t_qps,
        color=(0.396, 0.761, 0.647), marker='^', markersize=6,
        linewidth=LINEWIDTH, linestyle='-', label='DiskANN', zorder=2)
ax.plot(pipe_32t_recall, pipe_32t_qps,
        color=(0.404, 0.553, 0.706), marker='o', markersize=6,
        linewidth=LINEWIDTH, linestyle='--', label='PipeANN', zorder=3)
ax.plot(gate_32t_recall, gate_32t_qps,
        color=(0.922, 0.514, 0.478), marker='s', markersize=6,
        linewidth=LINEWIDTH, linestyle='-', label='GateANN', zorder=4)

ax.set_yscale('log')
ax.set_xlabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Throughput (QPS)', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(left=0.7, right=1.005)

plt.tight_layout()
out_base = '/Users/gykim/workspace/GateANN/figures/fig_pareto_deep_tput'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close()
