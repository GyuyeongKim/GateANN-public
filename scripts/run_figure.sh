#!/bin/bash
set -e

# Convenience wrapper: run experiment + plot for a single figure.
#
# Usage: ./scripts/run_figure.sh <figure_number>
# Example: ./scripts/run_figure.sh 4    # runs fig04 experiment + plot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FIG="$1"

if [ -z "$FIG" ]; then
    echo "Usage: $0 <figure_number>"
    echo ""
    echo "Available figures:"
    echo "  1   — Motivation (thread scaling + naive pre-filter)"
    echo "  4   — Pareto curves (BigANN-100M + DEEP-100M)"
    echo "  5   — Thread scaling"
    echo "  6   — I/O reduction"
    echo "  7   — BigANN-1B"
    echo "  8   — YFCC-10M multi-label"
    echo "  9   — Vamana comparison"
    echo "  10  — Filtered-DiskANN comparison"
    echo "  11  — Selectivity sweep"
    echo "  12  — R_max sweep"
    echo "  13  — Zipf distribution"
    echo "  14  — Spatial correlation"
    echo "  15  — Range predicates"
    echo "  16  — Pipeline depth (BW sweep)"
    echo "  17  — Ablation (early filter)"
    echo ""
    echo "Tables:"
    echo "  t3  — SSD speed impact"
    echo "  t4  — Time breakdown"
    exit 1
fi

# Map figure number to experiment script
case "$FIG" in
    1)  EXP="fig01_motivation.sh" ;;
    4)  EXP="fig04_pareto_main.sh" ;;
    5)  EXP="fig05_thread_scaling.sh" ;;
    6)  EXP="fig06_io_reduction.sh" ;;
    7)  EXP="fig07_billion.sh" ;;
    8)  EXP="fig08_yfcc.sh" ;;
    9)  EXP="fig09_vamana.sh" ;;
    10) EXP="fig10_fdiskann.sh" ;;
    11) EXP="fig11_selectivity.sh" ;;
    12) EXP="fig12_nbrs_sweep.sh" ;;
    13) EXP="fig13_zipf.sh" ;;
    14) EXP="fig14_spatial.sh" ;;
    15) EXP="fig15_range.sh" ;;
    16) EXP="fig16_bw_sweep.sh" ;;
    17) EXP="fig17_ablation.sh" ;;
    t3|T3) EXP="tab03_ssd_impact.sh" ;;
    t4|T4) EXP="tab04_breakdown.sh" ;;
    *)  echo "Unknown figure: $FIG"; exit 1 ;;
esac

echo "=== Running: $EXP ==="
bash "$SCRIPT_DIR/$EXP"

# Generate plots (skip for tables — they print to stdout)
case "$FIG" in
    t3|T3|t4|T4) echo "(Table output — no plot script needed)" ;;
    *)
        echo ""
        echo "=== Generating figures ==="
        cd "$REPO_ROOT"
        python3 scripts/generate_all_figures.py
        echo "Figures saved to: figures/"
        ;;
esac
