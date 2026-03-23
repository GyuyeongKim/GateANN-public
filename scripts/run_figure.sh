#!/bin/bash
set -e

# Convenience wrapper: run experiment + plot for a single figure.
#
# Usage: ./scripts/run_figure.sh <figure_number>
# Example: ./scripts/run_figure.sh 5    # runs fig05 experiment + plot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FIG="$1"

if [ -z "$FIG" ]; then
    echo "Usage: $0 <figure_number>"
    echo ""
    echo "Available figures:"
    echo "  1   — Motivation (thread scaling + naive pre-filter)"
    echo "  5   — Pareto curves (BigANN-100M + DEEP-100M)"
    echo "  6   — Thread scaling"
    echo "  7   — I/O reduction"
    echo "  8   — BigANN-1B"
    echo "  9   — YFCC-10M multi-label"
    echo "  10  — Vamana comparison"
    echo "  11  — Filtered-DiskANN comparison"
    echo "  12  — Selectivity sweep"
    echo "  13  — R_max sweep"
    echo "  14  — Zipf distribution"
    echo "  15  — Spatial correlation"
    echo "  16  — Range predicates"
    echo "  17  — Pipeline depth (BW sweep)"
    echo "  18  — Ablation (early filter)"
    echo ""
    echo "Tables:"
    echo "  t4  — SSD speed impact"
    echo "  t5  — Time breakdown"
    exit 1
fi

# Map figure number to experiment script
case "$FIG" in
    1)  EXP="fig01_motivation.sh" ;;
    5)  EXP="fig05_pareto_main.sh" ;;
    6)  EXP="fig06_thread_scaling.sh" ;;
    7)  EXP="fig07_io_reduction.sh" ;;
    8)  EXP="fig08_billion.sh" ;;
    9)  EXP="fig09_yfcc.sh" ;;
    10) EXP="fig10_vamana.sh" ;;
    11) EXP="fig11_fdiskann.sh" ;;
    12) EXP="fig12_selectivity.sh" ;;
    13) EXP="fig13_nbrs_sweep.sh" ;;
    14) EXP="fig14_zipf.sh" ;;
    15) EXP="fig15_spatial.sh" ;;
    16) EXP="fig16_range.sh" ;;
    17) EXP="fig17_bw_sweep.sh" ;;
    18) EXP="fig18_ablation.sh" ;;
    t4|T4) EXP="tab04_ssd_impact.sh" ;;
    t5|T5) EXP="tab05_breakdown.sh" ;;
    *)  echo "Unknown figure: $FIG"; exit 1 ;;
esac

echo "=== Running: $EXP ==="
bash "$SCRIPT_DIR/$EXP"

# Generate plots (skip for tables — they print to stdout)
case "$FIG" in
    t4|T4|t5|T5) echo "(Table output — no plot script needed)" ;;
    *)
        echo ""
        echo "=== Generating figures ==="
        cd "$REPO_ROOT"
        python3 scripts/generate_all_figures.py
        echo "Figures saved to: figures/"
        ;;
esac
