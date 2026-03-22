#!/usr/bin/env python3
"""Generate F-DiskANN comparison figures.

BigANN-100M (2 figures):
  fig_fdiskann_lat: DiskANN, PipeANN, F-DiskANN, FilterANN (1T latency)
  fig_fdiskann_tput: DiskANN, PipeANN, FilterANN (32T throughput)
DEEP-100M (2 figures):
  fig_fdiskann_deep_lat: DiskANN, F-DiskANN, GateANN (1T latency)
  fig_fdiskann_deep_tput: DiskANN, F-DiskANN, GateANN (32T throughput)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---- Colors (from EPS setrgbcolor) ----
COLOR_DISKANN   = (0.396, 0.761, 0.647)  # green
COLOR_PIPEANN   = (0.404, 0.553, 0.706)  # blue
COLOR_FDISKANN  = (0.945, 0.784, 0.533)  # orange
COLOR_FILTERANN = (0.922, 0.514, 0.478)  # red

# ---- BigANN style (DejaVu Sans, from EPS) ----
BIGANN_FONTSIZE_LABEL  = 16
BIGANN_FONTSIZE_TICK   = 11.2
BIGANN_FONTSIZE_LEGEND = 14
BIGANN_LINEWIDTH       = 2.5
BIGANN_MARKERSIZE      = 4

# ---- DEEP style (NimbusSans, from EPS) ----
DEEP_FONTSIZE_LABEL  = 18
DEEP_FONTSIZE_TICK   = 12.6
DEEP_FONTSIZE_LEGEND = 16
DEEP_LINEWIDTH       = 2.5
DEEP_MARKERSIZE      = 10

# ======================================================================
# BigANN-100M Data
# ======================================================================

# --- DiskANN (mode=0), T=1 ---
diskann_1t_recall = [0.1003, 0.1504, 0.1995, 0.2500, 0.2964, 0.3416, 0.3841,
                     0.4587, 0.5236, 0.6301, 0.7106, 0.7717, 0.8377, 0.8831,
                     0.9041, 0.9405, 0.9611, 0.9734, 0.9812, 0.9896]
diskann_1t_qps    = [361, 368, 353, 343, 322, 313, 297,
                     279, 243, 204, 175, 151, 129, 109,
                     101, 82, 70, 61, 56, 46]

# --- DiskANN (mode=0), T=32 ---
diskann_32t_recall = [0.1003, 0.1504, 0.1995, 0.2500, 0.2964, 0.3416, 0.3841,
                      0.4587, 0.5236, 0.6301, 0.7106, 0.7717, 0.8377, 0.8831,
                      0.9041, 0.9405, 0.9611, 0.9734, 0.9812, 0.9896]
diskann_32t_qps    = [8655, 8602, 7885, 7765, 6895, 6057, 6002,
                      5367, 4773, 3987, 3315, 2915, 2428, 2024,
                      1837, 1527, 1307, 1124, 991, 794]

# --- PipeANN (Baseline post-filter, mode=2), T=1 ---
pipe_1t_recall = [0.1373, 0.1867, 0.2324, 0.2751, 0.3162, 0.3542, 0.3907,
                  0.4590, 0.5210, 0.6271, 0.7087, 0.7698, 0.8369, 0.8825,
                  0.9037, 0.9402, 0.9609, 0.9732, 0.9810, 0.9896]
pipe_1t_qps    = [1770, 1749, 1700, 1650, 1587, 1524, 1461,
                  1337, 1268, 1133, 1028, 935, 821, 732,
                  678, 579, 508, 447, 407, 334]

# --- PipeANN (Baseline post-filter, mode=2), T=32 ---
pipe_32t_recall = [0.1353, 0.1856, 0.2314, 0.2738, 0.3144, 0.3527, 0.3887,
                   0.4577, 0.5197, 0.6265, 0.7085, 0.7700, 0.8371, 0.8826,
                   0.9038, 0.9403, 0.9609, 0.9732, 0.9810, 0.9896]
pipe_32t_qps    = [15010, 15625, 13411, 11818, 10494, 9480, 8600,
                   7216, 6260, 4912, 4009, 3404, 2759, 2324,
                   2098, 1695, 1420, 1221, 1071, 860]

# --- F-DiskANN (v2), T=1 --- first 8 L values (L=10..50)
fdiskann_v2_1t_recall = [0.28, 0.40, 0.48, 0.56, 0.65, 0.70, 0.76, 0.90]
fdiskann_v2_1t_qps    = [1054.52, 1100.20, 1079.95, 1057.29, 1029.66, 991.99, 950.68, 876.08]

# --- FilterANN (mode=8), T=1 --- main (20) + highL extra (4) = 24 pts
filterann_1t_recall = [0.1061, 0.1524, 0.1985, 0.2428, 0.2839, 0.3237, 0.3623,
                       0.4319, 0.4935, 0.5951, 0.6726, 0.7320, 0.7965, 0.8412,
                       0.8633, 0.9030, 0.9266, 0.9420, 0.9532, 0.9669,
                       0.9802, 0.9885, 0.9940, 0.9963]
filterann_1t_qps    = [3152, 4391, 3965, 3575, 3464, 3165, 3055,
                       3182, 3556, 2986, 2621, 2325, 1965, 1721,
                       1595, 1304, 1097, 991, 874, 710,
                       514, 361, 245, 177]

# --- FilterANN (mode=8), T=32 --- main (20) + highL extra (4) = 24 pts
filterann_32t_recall = [0.1054, 0.1519, 0.1983, 0.2424, 0.2832, 0.3231, 0.3615,
                        0.4316, 0.4927, 0.5941, 0.6720, 0.7314, 0.7959, 0.8411,
                        0.8639, 0.9036, 0.9272, 0.9424, 0.9535, 0.9672,
                        0.9804, 0.9886, 0.9940, 0.9963]
filterann_32t_qps    = [16334, 89938, 76230, 73659, 68347, 60671, 56583,
                        50556, 44432, 35301, 30383, 26810, 18388, 20354,
                        20494, 16017, 13949, 11730, 10501, 8518,
                        6110, 4299, 2861, 2078]

# ======================================================================
# Figure 1: BigANN Latency Pareto (1 thread)
#   X = Latency (ms), log scale; Y = Recall@10, linear
#   4 series: DiskANN, PipeANN, F-DiskANN, FilterANN
# ======================================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

diskann_1t_lat     = [1000.0 / q for q in diskann_1t_qps]
pipe_1t_lat        = [1000.0 / q for q in pipe_1t_qps]
fdiskann_v2_1t_lat = [1000.0 / q for q in fdiskann_v2_1t_qps]
filterann_1t_lat   = [1000.0 / q for q in filterann_1t_qps]

fig, ax = plt.subplots(figsize=(4.02, 3.08))

ax.plot(diskann_1t_lat, diskann_1t_recall,
        color=COLOR_DISKANN, marker='^', markersize=BIGANN_MARKERSIZE,
        linewidth=BIGANN_LINEWIDTH, label='DiskANN', zorder=2)
ax.plot(pipe_1t_lat, pipe_1t_recall,
        color=COLOR_PIPEANN, marker='o', markersize=BIGANN_MARKERSIZE,
        linewidth=BIGANN_LINEWIDTH, label='PipeANN', zorder=3)
ax.plot(fdiskann_v2_1t_lat, fdiskann_v2_1t_recall,
        color=COLOR_FDISKANN, marker='v', markersize=BIGANN_MARKERSIZE,
        linewidth=BIGANN_LINEWIDTH, label='F-DiskANN', zorder=4)
ax.plot(filterann_1t_lat, filterann_1t_recall,
        color=COLOR_FILTERANN, marker='s', markersize=BIGANN_MARKERSIZE,
        linewidth=BIGANN_LINEWIDTH, label='FilterANN', zorder=5)

ax.set_xscale('log')
ax.set_xlabel('Latency (ms)', fontsize=BIGANN_FONTSIZE_LABEL)
ax.set_ylabel('Recall@10', fontsize=BIGANN_FONTSIZE_LABEL)
ax.tick_params(labelsize=BIGANN_FONTSIZE_TICK)
ax.legend(fontsize=BIGANN_FONTSIZE_LEGEND, loc='upper left',
          framealpha=0.9, edgecolor='lightgray')
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_ylim(bottom=0.5)

plt.tight_layout()
out_base = '/home/node33/GateANN/figures/fig_fdiskann_lat'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close()

# ======================================================================
# Figure 2: BigANN Throughput Pareto (32 threads, log scale)
#   X = Recall@10 (linear); Y = Throughput QPS (log scale)
#   3 series: DiskANN, PipeANN, FilterANN (no F-DiskANN)
# ======================================================================
fig, ax = plt.subplots(figsize=(3.91, 3.07))

ax.plot(diskann_32t_recall, diskann_32t_qps,
        color=COLOR_DISKANN, marker='^', markersize=BIGANN_MARKERSIZE,
        linewidth=BIGANN_LINEWIDTH, label='DiskANN', zorder=2)
ax.plot(pipe_32t_recall, pipe_32t_qps,
        color=COLOR_PIPEANN, marker='o', markersize=BIGANN_MARKERSIZE,
        linewidth=BIGANN_LINEWIDTH, label='PipeANN', zorder=3)
ax.plot(filterann_32t_recall, filterann_32t_qps,
        color=COLOR_FILTERANN, marker='s', markersize=BIGANN_MARKERSIZE,
        linewidth=BIGANN_LINEWIDTH, label='FilterANN', zorder=5)

ax.set_yscale('log')
ax.set_xlabel('Recall@10', fontsize=BIGANN_FONTSIZE_LABEL)
ax.set_ylabel('Throughput (QPS)', fontsize=BIGANN_FONTSIZE_LABEL)
ax.tick_params(labelsize=BIGANN_FONTSIZE_TICK)
ax.legend(fontsize=BIGANN_FONTSIZE_LEGEND, loc='upper right',
          framealpha=0.9, edgecolor='lightgray')
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(left=0.5)

plt.tight_layout()
out_base = '/home/node33/GateANN/figures/fig_fdiskann_tput'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close()

# ======================================================================
# DEEP-100M Data
# ======================================================================

# --- DiskANN (mode=0), T=1 ---
deep_diskann_1t_recall = [0.0990, 0.1478, 0.1959, 0.2433, 0.2872, 0.3281, 0.3650,
                          0.4282, 0.4838, 0.5774, 0.6487, 0.7073, 0.7721, 0.8199,
                          0.8444, 0.8895, 0.9183, 0.9383, 0.9527, 0.9695]
deep_diskann_1t_qps    = [358, 358, 341, 330, 316, 303, 278,
                          253, 236, 209, 165, 148, 124, 107,
                          98, 80, 68, 60, 53, 43]

# --- DiskANN (mode=0), T=32 ---
deep_diskann_32t_recall = [0.0990, 0.1478, 0.1959, 0.2433, 0.2872, 0.3281, 0.3650,
                           0.4282, 0.4838, 0.5774, 0.6487, 0.7073, 0.7721, 0.8199,
                           0.8444, 0.8895, 0.9183, 0.9383, 0.9527, 0.9695]
deep_diskann_32t_qps    = [7395, 8264, 7347, 6966, 6571, 6181, 5714,
                           4781, 4562, 3799, 3129, 2703, 2343, 2013,
                           1799, 1463, 1262, 1101, 960, 774]

# --- F-DiskANN (DEEP), T=1 --- 24 L values
deep_fdiskann_1t_recall = [0.29, 0.40, 0.48, 0.57, 0.64, 0.70, 0.77,
                           0.87, 0.97, 1.12, 1.27, 1.38, 1.50, 1.59,
                           1.63, 1.73, 1.80, 1.84, 1.88, 1.91,
                           1.95, 1.97, 1.98, 1.98]
deep_fdiskann_1t_qps    = [855.76, 993.71, 993.44, 967.00, 941.21, 913.32, 884.34,
                           822.57, 764.40, 671.22, 595.88, 535.12, 463.55, 406.95,
                           375.25, 316.55, 273.40, 240.53, 214.51, 174.82,
                           126.44, 91.09, 61.19, 45.65]

# --- F-DiskANN (DEEP), T=32 --- 24 L values
deep_fdiskann_32t_recall = [0.29, 0.39, 0.48, 0.56, 0.64, 0.71, 0.76,
                            0.87, 0.97, 1.13, 1.27, 1.38, 1.50, 1.59,
                            1.63, 1.73, 1.80, 1.85, 1.88, 1.91,
                            1.95, 1.97, 1.98, 1.98]
deep_fdiskann_32t_qps    = [8105.96, 7715.69, 7403.32, 7143.14, 6737.99, 6338.52, 6017.51,
                            5201.36, 4846.10, 4004.01, 3411.39, 2968.23, 2474.03, 2095.59,
                            1890.61, 1579.55, 1327.91, 1160.63, 1014.98, 787.15,
                            598.10, 423.82, 284.59, 214.15]

# --- GateANN (mode=8, DEEP), T=1 ---
deep_gate_1t_recall = [0.0988, 0.1436, 0.1855, 0.2267, 0.2651, 0.3022, 0.3357,
                       0.3985, 0.4544, 0.5465, 0.6177, 0.6742, 0.7358, 0.7823,
                       0.8074, 0.8527, 0.8838, 0.9054, 0.9208, 0.9411]
deep_gate_1t_qps    = [2731, 4301, 3903, 3663, 3420, 3248, 3111,
                       2730, 3071, 2720, 2399, 2148, 1821, 1600,
                       1490, 1202, 1050, 924, 827, 620]

# --- GateANN (mode=8, DEEP), T=32 ---
deep_gate_32t_recall = [0.0986, 0.1431, 0.1850, 0.2260, 0.2645, 0.3018, 0.3353,
                        0.3981, 0.4537, 0.5456, 0.6175, 0.6746, 0.7360, 0.7827,
                        0.8084, 0.8534, 0.8845, 0.9059, 0.9212, 0.9414]
deep_gate_32t_qps    = [24233, 84580, 82225, 74653, 69444, 68691, 67326,
                        53715, 49191, 41489, 37690, 32668, 26631, 22588,
                        20201, 16558, 13766, 11949, 10446, 8324]

# ======================================================================
# Figure 3: DEEP Latency Pareto (1 thread)
#   X = Latency (ms), log scale; Y = Recall@10, linear
#   3 series: DiskANN, F-DiskANN, GateANN
#   NimbusSans font, larger markers, wider figure
# ======================================================================
plt.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'

deep_diskann_1t_lat  = [1000.0 / q for q in deep_diskann_1t_qps]
deep_fdiskann_1t_lat = [1000.0 / q for q in deep_fdiskann_1t_qps]
deep_gate_1t_lat     = [1000.0 / q for q in deep_gate_1t_qps]

fig, ax = plt.subplots(figsize=(7.77, 3.37))

ax.plot(deep_diskann_1t_lat, deep_diskann_1t_recall,
        color=COLOR_DISKANN, marker='^', markersize=DEEP_MARKERSIZE,
        linewidth=DEEP_LINEWIDTH, label='DiskANN', zorder=2)
ax.plot(deep_fdiskann_1t_lat, deep_fdiskann_1t_recall,
        color=COLOR_FDISKANN, marker='v', markersize=DEEP_MARKERSIZE,
        linewidth=DEEP_LINEWIDTH, label='F-DiskANN', zorder=3)
ax.plot(deep_gate_1t_lat, deep_gate_1t_recall,
        color=COLOR_FILTERANN, marker='s', markersize=DEEP_MARKERSIZE,
        linewidth=DEEP_LINEWIDTH, label='GateANN', zorder=4)

ax.set_xscale('log')
ax.set_xlabel('Latency (ms)', fontsize=DEEP_FONTSIZE_LABEL)
ax.set_ylabel('Recall@10', fontsize=DEEP_FONTSIZE_LABEL)
ax.tick_params(labelsize=DEEP_FONTSIZE_TICK)
ax.legend(fontsize=DEEP_FONTSIZE_LEGEND, loc='upper left',
          framealpha=0.9, edgecolor='lightgray')
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_ylim(bottom=0.5)

plt.tight_layout()
out_base = '/home/node33/GateANN/figures/fig_fdiskann_deep_lat'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close()

# ======================================================================
# Figure 4: DEEP Throughput Pareto (32 threads, log scale)
#   X = Recall@10 (linear); Y = Throughput QPS (log scale)
#   3 series: DiskANN, F-DiskANN, GateANN
# ======================================================================
fig, ax = plt.subplots(figsize=(7.89, 3.38))

ax.plot(deep_diskann_32t_recall, deep_diskann_32t_qps,
        color=COLOR_DISKANN, marker='^', markersize=DEEP_MARKERSIZE,
        linewidth=DEEP_LINEWIDTH, label='DiskANN', zorder=2)
ax.plot(deep_fdiskann_32t_recall, deep_fdiskann_32t_qps,
        color=COLOR_FDISKANN, marker='v', markersize=DEEP_MARKERSIZE,
        linewidth=DEEP_LINEWIDTH, label='F-DiskANN', zorder=3)
ax.plot(deep_gate_32t_recall, deep_gate_32t_qps,
        color=COLOR_FILTERANN, marker='s', markersize=DEEP_MARKERSIZE,
        linewidth=DEEP_LINEWIDTH, label='GateANN', zorder=4)

ax.set_yscale('log')
ax.set_xlabel('Recall@10', fontsize=DEEP_FONTSIZE_LABEL)
ax.set_ylabel('Throughput (QPS)', fontsize=DEEP_FONTSIZE_LABEL)
ax.tick_params(labelsize=DEEP_FONTSIZE_TICK)
ax.legend(fontsize=DEEP_FONTSIZE_LEGEND, loc='upper right',
          framealpha=0.9, edgecolor='lightgray')
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(left=0.5)

plt.tight_layout()
out_base = '/home/node33/GateANN/figures/fig_fdiskann_deep_tput'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close()
