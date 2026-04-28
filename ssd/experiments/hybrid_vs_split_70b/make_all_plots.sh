#!/bin/bash
# Run after both split + hybrid finish — produces all breakdown graphs.
set -u
cd "$(dirname "$0")/../.."   # ssd repo root

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
RESULTS="${1:-$PWD/experiments/hybrid_vs_split_70b/results}"

# Per-config plots (uses bench/plot_mesa_breakdown.py — produces *.csv too,
# which the 3-way script consumes). Also runs timeline Gantt for one step.
for d in "$RESULTS/split_k5_dfo2_exit40" "$RESULTS/hybrid_k5_K1_3_K2_2_exit40"; do
    [ -d "$d" ] || continue
    echo "=== plot_mesa_breakdown $d ==="
    "$PY" bench/plot_mesa_breakdown.py "$d"
    echo "=== plot_mesa_timeline $d ==="
    "$PY" bench/plot_mesa_timeline.py "$d" --step 100 --warmup 5 || true
done

# Pre-hybrid baseline (A) — pre-hybrid BOTH-AWQ run (target + draft both
# quantized). Matches the new experiment default. The dense-draft baseline
# at tmp/final_exp2_quant_70b/mesa_k5_f4_dfo2_exit40 is kept for legacy.
BASELINE=/home/chokwans99/PSD/ssd/experiments/quant_awq/70b/draft_awq/mesa_k5_f4_dfo2_exit40
if [ -d "$BASELINE" ] && ls "$BASELINE"/mesa_profile_*.json >/dev/null 2>&1; then
    echo "=== plot_mesa_breakdown (baseline A) $BASELINE ==="
    "$PY" bench/plot_mesa_breakdown.py "$BASELINE"
fi

# Compare table (text)
echo "=== compare_breakdown.py (table) ==="
"$PY" experiments/hybrid_vs_split_70b/compare_breakdown.py "$RESULTS" \
    > "$RESULTS/compare_table.txt" 2>&1
cat "$RESULTS/compare_table.txt" | head -80

# 3-way figures
echo "=== plot_3way.py ==="
"$PY" experiments/hybrid_vs_split_70b/plot_3way.py "$RESULTS" "$BASELINE"

# Top-line summary (uses bench/extract_sweep_metrics.py)
echo "=== top-line metrics ==="
"$PY" bench/extract_sweep_metrics.py "$RESULTS" | tee "$RESULTS/SUMMARY.md"

echo ""
echo "===================================================="
echo "  Generated artifacts in $RESULTS/:"
echo "===================================================="
ls -lh "$RESULTS"/*.png "$RESULTS"/*.csv "$RESULTS"/*.md "$RESULTS"/*.txt 2>/dev/null
echo ""
echo "Per-run artifacts:"
for d in "$RESULTS"/split_k5_dfo2_exit40 "$RESULTS"/hybrid_k5_K1_3_K2_2_exit40; do
    [ -d "$d" ] || continue
    echo "  $(basename $d)/"
    ls -lh "$d"/*.png "$d"/*.csv 2>/dev/null | sed 's/^/    /'
done
