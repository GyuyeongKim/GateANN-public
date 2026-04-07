#!/usr/bin/env python3
"""Generate R_max Pareto curves (figure 14b). x-axis starts at 0.7."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'

# PipeANN 32T, sel=10%, BigANN-100M (from main results)
pipe_recall = [0.1353, 0.1856, 0.2314, 0.2738, 0.3144, 0.3527, 0.3887,
                0.4577, 0.5197, 0.6265, 0.7085, 0.7700, 0.8371, 0.8826,
                0.9038, 0.9403, 0.9609, 0.9732, 0.9810, 0.9896, 0.9958,
                0.9984, 0.9995, 0.9997, 0.9998, 0.9999,
                0.9999, 0.9999, 0.9999, 0.9999]
pipe_qps    = [15010, 15625, 13411, 11818, 10494, 9480, 8600,
             7216, 6260, 4912, 4009, 3404, 2759, 2324,
             2098, 1695, 1420, 1221, 1071, 860, 612,
             430, 287, 215, 140, 84,
             72, 61, 54, 43]

# GateANN 32T data for each R_max
data = {
    8: {
        'recall': [0.0905, 0.1362, 0.1782, 0.2172, 0.2557, 0.2895, 0.3221,
                  0.3805, 0.4314, 0.5117, 0.5752, 0.6235, 0.6766, 0.7156,
                  0.7334, 0.7746, 0.8037, 0.8259, 0.8434, 0.8697, 0.9032,
                  0.9318, 0.9548, 0.9671, 0.9797, 0.9895,
                  0.9916, 0.9934, 0.9946, 0.9960],
        'qps':    [18403, 73328, 73670, 70706, 71357, 62951, 60723,
                  51574, 44777, 34553, 28760, 25918, 21292, 18450,
                  16721, 14086, 12233, 10734, 9461, 7804, 4366,
                  3469, 2378, 1831, 1298, 828,
                  703, 606, 531, 422],
        'dram': '3GB',
    },
    16: {
        'recall': [0.1001, 0.1478, 0.1924, 0.2355, 0.2750, 0.3133, 0.3487,
                  0.4131, 0.4690, 0.5603, 0.6284, 0.6828, 0.7431, 0.7860,
                  0.8081, 0.8471, 0.8728, 0.8915, 0.9070, 0.9268, 0.9488,
                  0.9658, 0.9795, 0.9860, 0.9917, 0.9961],
        'qps':    [19983, 90764, 84221, 77367, 71724, 66408, 64564,
                  54460, 49358, 41864, 35425, 30276, 24856, 21600,
                  19576, 16014, 13437, 11593, 10245, 8290, 3876,
                  3225, 2389, 1798, 977, 834],
        'dram': '6GB',
    },
    32: {
        'recall': [0.1054, 0.1519, 0.1981, 0.2425, 0.2832, 0.3232, 0.3616,
                  0.4314, 0.4925, 0.5941, 0.6720, 0.7318, 0.7967, 0.8417,
                  0.8639, 0.9034, 0.9272, 0.9424, 0.9535, 0.9673, 0.9803,
                  0.9885, 0.9940, 0.9963],
        'qps':    [16301, 87304, 79964, 69844, 60835, 61859, 57647,
                  51172, 45489, 35832, 31316, 31766, 27473, 23111,
                  20896, 16606, 14146, 12107, 10559, 8479, 3557,
                  2759, 2393, 1516],
        'dram': '12GB',
    },
    48: {
        'recall': [0.1051, 0.1518, 0.1983, 0.2438, 0.2856, 0.3264, 0.3656,
                  0.4372, 0.5016, 0.6074, 0.6885, 0.7510, 0.8181, 0.8642,
                  0.8860, 0.9240, 0.9463, 0.9603, 0.9697, 0.9804, 0.9901,
                  0.9948, 0.9976, 0.9986],
        'qps':    [16465, 80591, 79779, 72742, 66601, 57050, 56652,
                  48690, 43015, 32138, 28258, 29306, 25742, 22345,
                  20441, 16482, 14246, 12186, 10631, 8543, 3192,
                  2905, 1899, 1576],
        'dram': '18GB',
    },
    64: {
        'recall': [0.1039, 0.1510, 0.1976, 0.2447, 0.2873, 0.3280, 0.3666,
                  0.4402, 0.5058, 0.6140, 0.6966, 0.7593, 0.8278, 0.8741,
                  0.8956, 0.9330, 0.9542, 0.9674, 0.9759, 0.9854, 0.9934,
                  0.9969, 0.9988, 0.9994],
        'qps':    [16024, 71221, 67160, 58095, 54632, 50918, 48027,
                  38054, 36912, 32501, 28561, 24598, 21138, 18436,
                  18529, 15255, 12551, 11136, 10264, 8135, 2414,
                  2503, 1519, 1450],
        'dram': '24GB',
    },
}

# Filter to recall >= 0.7
def filt(recall, qps, min_r=0.7):
    r, q = zip(*[(r, q) for r, q in zip(recall, qps) if r >= min_r])
    return list(r), list(q)

FONTSIZE_LABEL = 20
FONTSIZE_TICK  = 18
FONTSIZE_LEGEND = 18
LINEWIDTH = 2.5
MARKERSIZE = 10

fig, ax = plt.subplots(figsize=(8, 4.5))

# PipeANN baseline (dashed, blue)
pr, pq = filt(pipe_recall, pipe_qps)
ax.plot(pr, pq, color=(0.404, 0.553, 0.706), marker='o', markersize=MARKERSIZE,
        linewidth=LINEWIDTH, linestyle='--', label='PipeANN', zorder=2)

# GateANN R_max values: distinct colors and markers
rmax_colors = [
    (0.922, 0.514, 0.478),
    (0.945, 0.784, 0.533),
    (0.922, 0.514, 0.478),
    (0.624, 0.796, 0.910),
    (0.396, 0.761, 0.647),
]
rmax_markers = ['s', 'D', '^', 'v', 'P']

for idx, (rval, d) in enumerate(data.items()):
    dr, dq = filt(d['recall'], d['qps'])
    ax.plot(dr, dq, color=rmax_colors[idx], marker=rmax_markers[idx], markersize=MARKERSIZE,
            linewidth=LINEWIDTH, linestyle='-',
            label=f'$R_{{\\max}}$={rval} ({d["dram"]})', zorder=3)

ax.set_yscale('log')
ax.set_xlabel('Recall@10', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Throughput (QPS)', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.legend(fontsize=FONTSIZE_LEGEND, ncol=3,
          frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, 1.35))
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(left=0.8, right=1.02)

plt.subplots_adjust(top=0.75)

out = '/Users/gykim/workspace/GateANN/figures/fig_nbrs_pareto'
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out + '.eps', bbox_inches='tight')
print(f"Saved: {out}.png and {out}.eps")
