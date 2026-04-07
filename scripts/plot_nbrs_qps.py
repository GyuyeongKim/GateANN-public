#!/usr/bin/env python3
"""Generate R_max DRAM-QPS tradeoff plot (figure 14a).
X: DRAM overhead (GB), Y: QPS at 90% recall."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'

# GateANN 32T, sel=10%, BigANN-100M — same data as plot_nbrs_pareto.py
rmax_vals = [8, 16, 32, 48, 64]
dram_gb   = [3, 6, 12, 18, 24]

data = {
    8: {
        'recall': [0.0905, 0.1362, 0.1782, 0.2172, 0.2557, 0.2895, 0.3221,
                   0.3805, 0.4314, 0.5117, 0.5752, 0.6235, 0.6766, 0.7156,
                   0.7334, 0.7746, 0.8037, 0.8259, 0.8434, 0.8697,
                   0.9032, 0.9318, 0.9548, 0.9671],
        'qps':    [18403, 73328, 73670, 70706, 71357, 62951, 60723,
                   51574, 44777, 34553, 28760, 25918, 21292, 18450,
                   16721, 14086, 12233, 10734, 9461, 7804,
                   4366, 3469, 2378, 1831],
    },
    16: {
        'recall': [0.1001, 0.1478, 0.1924, 0.2355, 0.2750, 0.3133, 0.3487,
                   0.4131, 0.4690, 0.5603, 0.6284, 0.6828, 0.7431, 0.7860,
                   0.8081, 0.8471, 0.8728, 0.8915, 0.9070, 0.9268,
                   0.9488, 0.9658, 0.9795, 0.9860],
        'qps':    [19983, 90764, 84221, 77367, 71724, 66408, 64564,
                   54460, 49358, 41864, 35425, 30276, 24856, 21600,
                   19576, 16014, 13437, 11593, 10245, 8290,
                   3876, 3225, 2389, 1798],
    },
    32: {
        'recall': [0.1054, 0.1519, 0.1981, 0.2425, 0.2832, 0.3232, 0.3616,
                   0.4314, 0.4925, 0.5941, 0.6720, 0.7318, 0.7967, 0.8417,
                   0.8639, 0.9034, 0.9272, 0.9424, 0.9535, 0.9673,
                   0.9803, 0.9885, 0.9940, 0.9963],
        'qps':    [16301, 87304, 79964, 69844, 60835, 61859, 57647,
                   51172, 45489, 35832, 31316, 31766, 27473, 23111,
                   20896, 16606, 14146, 12107, 10559, 8479,
                   3557, 2759, 2393, 1516],
    },
    48: {
        'recall': [0.1051, 0.1518, 0.1983, 0.2438, 0.2856, 0.3264, 0.3656,
                   0.4372, 0.5016, 0.6074, 0.6885, 0.7510, 0.8181, 0.8642,
                   0.8860, 0.9240, 0.9463, 0.9603, 0.9697, 0.9804,
                   0.9901, 0.9948, 0.9976, 0.9986],
        'qps':    [16465, 80591, 79779, 72742, 66601, 57050, 56652,
                   48690, 43015, 32138, 28258, 29306, 25742, 22345,
                   20441, 16482, 14246, 12186, 10631, 8543,
                   3192, 2905, 1899, 1576],
    },
    64: {
        'recall': [0.1039, 0.1510, 0.1976, 0.2447, 0.2873, 0.3280, 0.3666,
                   0.4402, 0.5058, 0.6140, 0.6966, 0.7593, 0.8278, 0.8741,
                   0.8956, 0.9330, 0.9542, 0.9674, 0.9759, 0.9854,
                   0.9934, 0.9969, 0.9988, 0.9994],
        'qps':    [16024, 71221, 67160, 58095, 54632, 50918, 48027,
                   38054, 36912, 32501, 28561, 24598, 21138, 18436,
                   18529, 15255, 12551, 11136, 10264, 8135,
                   2414, 2503, 1519, 1450],
    },
}

# PipeANN baseline (32T, sel=10%, BigANN-100M)
pipe_recall = [0.1353, 0.1856, 0.2314, 0.2738, 0.3144, 0.3527, 0.3887,
               0.4577, 0.5197, 0.6265, 0.7085, 0.7700, 0.8371, 0.8826,
               0.9038, 0.9403, 0.9609, 0.9732, 0.9810, 0.9896,
               0.9958, 0.9984, 0.9995, 0.9997]
pipe_qps    = [15010, 15625, 13411, 11818, 10494, 9480, 8600,
               7216, 6260, 4912, 4009, 3404, 2759, 2324,
               2098, 1695, 1420, 1221, 1071, 860,
               612, 430, 287, 215]

TARGET = 0.90

def interp_qps(recall, qps, target):
    """Interpolate QPS at a target recall."""
    r = np.array(recall)
    q = np.array(qps)
    order = np.argsort(r)
    r, q = r[order], q[order]
    if target < r[0] or target > r[-1]:
        return None
    return float(np.interp(target, r, q))

FONTSIZE_LABEL = 20
FONTSIZE_TICK  = 18
FONTSIZE_LEGEND = 18
FONTSIZE_ANNOT = 16
LINEWIDTH = 2.5
MARKERSIZE = 10

fig, ax = plt.subplots(figsize=(8, 3.5))

# GateANN at 90% recall
xs, ys = [], []
for rval, gb in zip(rmax_vals, dram_gb):
    d = data[rval]
    q = interp_qps(d['recall'], d['qps'], TARGET)
    if q is not None:
        xs.append(gb)
        ys.append(q)
        print(f"  R_max={rval} ({gb}GB): {q:.0f} QPS at {TARGET:.0%} recall")

# PipeANN at 90% recall (DRAM overhead = 0)
pipe_qps_90 = interp_qps(pipe_recall, pipe_qps, TARGET)
print(f"  PipeANN: {pipe_qps_90:.0f} QPS at {TARGET:.0%} recall")
ax.plot([0], [pipe_qps_90], color=(0.404, 0.553, 0.706), marker='o',
        markersize=MARKERSIZE, linewidth=0, label='PipeANN', zorder=3)

# GateANN at 90% recall
ax.plot(xs, ys, color=(0.922, 0.514, 0.478), marker='o',
        markersize=MARKERSIZE, linewidth=LINEWIDTH,
        label='GateANN', zorder=3)

# Annotate R_max on each GateANN point (above line, avoid overlap)
annot_offsets = {3: (0, 18), 6: (0, 18), 12: (0, 18), 18: (0, 18), 24: (0, -18)}
for gb, q, rval in zip(xs, ys, rmax_vals):
    ox, oy = annot_offsets[gb]
    ax.annotate(f'$R_{{\\max}}$={rval}', (gb, q),
                textcoords='offset points', xytext=(ox, oy),
                fontsize=FONTSIZE_ANNOT, ha='center', va='center')

ax.set_xlabel('DRAM Overhead (GB)', fontsize=FONTSIZE_LABEL)
ax.set_ylabel('Throughput (QPS)', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False,
          loc='center right', bbox_to_anchor=(0.98, 0.35))
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xticks([0] + dram_gb)
ax.set_ylim(bottom=0, top=23000)

# "Better" arrow: toward upper-left (lower DRAM, higher QPS)
from matplotlib.patches import FancyArrowPatch
arr_tail = (17.5, 3500)
arr_head = (12, 9500)
arrow_patch = FancyArrowPatch(
    arr_tail, arr_head,
    arrowstyle='simple,head_length=1.4,head_width=2.4,tail_width=1.1',
    mutation_scale=20,
    fc='none', ec=(0.90, 0.35, 0.30, 0.65),
    linewidth=1.8, zorder=1)
ax.add_patch(arrow_patch)
# Compute text rotation from display coordinates
fig.canvas.draw()
p1 = ax.transData.transform(arr_tail)
p2 = ax.transData.transform(arr_head)
ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
if ang > 90:
    ang -= 180
mid_x = (arr_tail[0] + arr_head[0]) / 2
mid_y = (arr_tail[1] + arr_head[1]) / 2
ax.text(mid_x, mid_y, 'Better', fontsize=19, ha='center', va='center',
        color=(0.90, 0.35, 0.30, 0.75), fontweight='bold', fontstyle='italic',
        rotation=ang, rotation_mode='anchor', zorder=2)

plt.tight_layout()

out = '/Users/gykim/workspace/GateANN/figures/fig_nbrs_qps'
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out + '.eps', bbox_inches='tight')
print(f"Saved: {out}.png and {out}.eps")
