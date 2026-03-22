#!/usr/bin/env python3
"""Generate BW (pipeline depth) sweep figures: QPS vs W and Recall vs L.

fig_bw_sweep_qps:    QPS at L=200 vs pipeline depth W (1T dashed, 32T solid)
fig_bw_sweep_recall: Recall@10 vs search list size L for W=4,8,16,32
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'

FONTSIZE_LABEL  = 18
FONTSIZE_TICK   = 16
FONTSIZE_LEGEND = 16
LINEWIDTH       = 2.5
MARKERSIZE      = 10

# ======================================================================
# Data from fig_bw_sweep_fixed.txt
# BW values: 1, 2, 4, 8, 16, 32
# L values:  10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 120, 150, 200, 300, 500
# ======================================================================

L_values = [10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 120, 150, 200, 300, 500]
BW_values = [1, 2, 4, 8, 16, 32]

# --- T=1 data: QPS for each (BW, L) ---
qps_1t = {
    1:  [2425, 3186, 2703, 3110, 2820, 2530, 2327, 1940, 1672, 1298, 1068, 905, 718, 558, 375, 225],
    2:  [3169, 4337, 3795, 3377, 3305, 3813, 3509, 3014, 2647, 2128, 1777, 1501, 1259, 979, 671, 408],
    4:  [3255, 4368, 3939, 3529, 3238, 3150, 3435, 3854, 3451, 2856, 2475, 2180, 1825, 1434, 1006, 617],
    8:  [3278, 4657, 4139, 3822, 3767, 4694, 4356, 3862, 3493, 2945, 2539, 2212, 1842, 1501, 1082, 644],
    16: [3220, 4593, 4077, 3977, 4914, 4604, 4306, 3842, 2788, 1873, 2230, 2211, 1901, 1534, 1089, 691],
    32: [3211, 4567, 4120, 3727, 3402, 3210, 4345, 3879, 3517, 2929, 1707, 1473, 1841, 1488, 1068, 680],
}

# --- T=32 data: QPS for each (BW, L) ---
qps_32t = {
    1:  [15614, 66797, 51499, 51668, 47920, 42928, 35725, 32259, 28891, 21487, 18642, 18043, 15239, 11721, 8053, 4998],
    2:  [17430, 81929, 72342, 51523, 57414, 52197, 48108, 43723, 38184, 31681, 26187, 22869, 18926, 16363, 11672, 7210],
    4:  [16913, 85001, 79381, 66606, 65262, 56774, 48732, 49036, 42523, 34017, 31093, 27792, 21863, 19588, 13979, 8521],
    8:  [16989, 85371, 78312, 75602, 80901, 76734, 72220, 57426, 53038, 40997, 37200, 33204, 26364, 20508, 13970, 8434],
    16: [20046, 98171, 97107, 81745, 76278, 70459, 63658, 61511, 52456, 41827, 36926, 29491, 23496, 18665, 13261, 8421],
    32: [20121, 102776, 103225, 93053, 79071, 68337, 72611, 61386, 54707, 43928, 37673, 33713, 27294, 20934, 13993, 8554],
}

# --- Recall data for each BW ---
recall = {
    1:  [0.1122, 0.1599, 0.2055, 0.2496, 0.2892, 0.3273, 0.3639, 0.4313, 0.4911, 0.5916, 0.6694, 0.7292, 0.7944, 0.8617, 0.9262, 0.9667],
    2:  [0.1081, 0.1557, 0.2026, 0.2469, 0.2883, 0.3268, 0.3643, 0.4322, 0.4930, 0.5929, 0.6706, 0.7300, 0.7949, 0.8622, 0.9263, 0.9668],
    4:  [0.1061, 0.1523, 0.1985, 0.2428, 0.2837, 0.3238, 0.3625, 0.4324, 0.4934, 0.5948, 0.6721, 0.7314, 0.7960, 0.8628, 0.9264, 0.9668],
    8:  [0.1061, 0.1524, 0.1986, 0.2429, 0.2839, 0.3240, 0.3625, 0.4320, 0.4934, 0.5951, 0.6723, 0.7318, 0.7964, 0.8633, 0.9267, 0.9669],
    16: [0.1061, 0.1524, 0.1986, 0.2429, 0.2841, 0.3240, 0.3624, 0.4321, 0.4934, 0.5943, 0.6721, 0.7319, 0.7964, 0.8633, 0.9267, 0.9669],
    32: [0.1061, 0.1524, 0.1986, 0.2429, 0.2838, 0.3236, 0.3625, 0.4321, 0.4934, 0.5952, 0.6719, 0.7314, 0.7964, 0.8632, 0.9267, 0.9669],
}

# ======================================================================
# Figure 1: QPS vs W (pipeline depth) at L=200
#   X = Pipeline depth W (log2 scale)
#   Y = QPS at L=200
#   2 series: 1 thread (dashed), 32 threads (solid)
#   B&W linestyles
# ======================================================================

# Index for L=300 in L_values (~93% recall, closest to 90%)
l_idx = L_values.index(300)

qps_1t_by_bw  = [qps_1t[bw][l_idx] for bw in BW_values]
qps_32t_by_bw = [qps_32t[bw][l_idx] for bw in BW_values]

fig, ax = plt.subplots(figsize=(3.90, 3.08))

ax.plot(BW_values, qps_1t_by_bw,
        color=(0.404, 0.553, 0.706), marker='o', markersize=6,
        linewidth=LINEWIDTH, linestyle='--',
        label='1 thread', zorder=2)
ax.plot(BW_values, qps_32t_by_bw,
        color=(0.922, 0.514, 0.478), marker='s', markersize=6,
        linewidth=LINEWIDTH, linestyle='-',
        label='32 threads', zorder=3)

ax.set_xscale('log', base=2)
ax.set_xticks(BW_values)
ax.set_xticklabels([str(w) for w in BW_values])
# Italic W in x-label, italic L=200 in y-label
ax.set_xlabel('Pipeline depth $W$', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('QPS at $L$=300', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.legend(fontsize=FONTSIZE_LEGEND, loc='center right',
          frameon=False)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
out_base = '/home/node33/GateANN/figures/fig_bw_sweep_qps'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close()

# ======================================================================
# Figure 2: Recall vs L for W=4, 8, 16, 32 (4 series only)
#   X = Search list size L
#   Y = Recall@10
#   B&W: different linestyles for different W values
# ======================================================================

BW_COLORS = {
    4:  (0.404, 0.553, 0.706),
    8:  (0.922, 0.514, 0.478),
    16: (0.396, 0.761, 0.647),
    32: (0.945, 0.784, 0.533),
}
BW_LINESTYLES = {
    4:  '-',
    8:  '--',
    16: '-.',
    32: ':',
}
BW_MARKERS = {
    4:  'o',
    8:  's',
    16: '^',
    32: 'D',
}

fig, ax = plt.subplots(figsize=(4.02, 3.07))

for bw in [4, 8, 16, 32]:
    ax.plot(L_values, recall[bw],
            color=BW_COLORS[bw], marker=BW_MARKERS[bw], markersize=6,
            linewidth=LINEWIDTH, linestyle=BW_LINESTYLES[bw],
            label=f'$W$={bw}', zorder=2)

# Italic L in x-label
ax.set_xlabel('Search list size $L$', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.legend(fontsize=FONTSIZE_LEGEND, loc='lower right', ncol=1,
          frameon=False)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
out_base = '/home/node33/GateANN/figures/fig_bw_sweep_recall'
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out_base + '.eps', bbox_inches='tight')
print(f"Saved: {out_base}.png and {out_base}.eps")
plt.close()
