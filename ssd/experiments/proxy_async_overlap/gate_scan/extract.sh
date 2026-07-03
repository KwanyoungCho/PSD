#!/usr/bin/env bash
# Extract the metric table from gate_scan cell logs.
DIR="$(dirname "$0")"
printf "%-11s %8s %8s %7s %7s %7s %7s %6s %6s %8s %8s\n" \
  cell TPS tok/st accept cache p1_hit p2_hit L_p1 L_p2 T_target T_draft
for label in A_base C_sd A_jit A_pod A_jit_pod E8_deep16 E9_deep16; do
  log="${DIR}/${label}/run.log"
  [ -f "$log" ] || { echo "${label}: MISSING"; continue; }
  tps=$(grep "Final Decode Throughput" "$log" | tail -1 | grep -oE "[0-9.]+" | head -1)
  tok=$(grep "Avg Tokens per step (incl recovery)" "$log" | tail -1 | grep -oE "[0-9.]+$")
  acc=$(grep "Avg Fraction of Speculated" "$log" | tail -1 | grep -oE "0\.[0-9]+" | head -1)
  ch=$(grep "Avg Cache Hits" "$log" | tail -1 | grep -oE "[0-9.]+$")
  p1=$(grep "Phase 1 (draft) Hit Rate" "$log" | tail -1 | grep -oE "[0-9.]+$")
  p2=$(grep "Phase 2 (proxy) Hit Rate" "$log" | tail -1 | grep -oE "[0-9.]+$")
  l1=$(grep "Phase 1 Accepted Len" "$log" | tail -1 | grep -oE "[0-9.]+$")
  l2=$(grep "Phase 2 Accepted Len" "$log" | tail -1 | grep -oE "[0-9.]+$")
  tt=$(grep "Avg target time per full step" "$log" | tail -1 | grep -oE "[0-9.]+$")
  td=$(grep "Avg draft step time" "$log" | tail -1 | grep -oE "[0-9.]+$")
  printf "%-11s %8s %8s %7s %7s %7s %7s %6s %6s %8s %8s\n" \
    "$label" "${tps:-—}" "${tok:-—}" "${acc:-—}" "${ch:-—}" "${p1:-—}" "${p2:-—}" \
    "${l1:-—}" "${l2:-—}" "${tt:-—}" "${td:-—}"
done
