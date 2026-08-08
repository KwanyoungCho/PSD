#!/usr/bin/env bash
# Threshold-calibration adoption gate (docs: threshold_calibration_p1p2_*/ANALYSIS.md).
#
# Paired on/on-only comparison of calibrated expansion floors and verify caps
# against the adopted champion configuration.  Relative A/B (eslab18 OK);
# absolute adoption numbers still require the clean box.
#
# Arms (all P1 on + P2 on, K1=9 K2=4, G: P1 18 / P2 8):
#   A_base : champion  — proxy .01 conf .03 p1 0/0,      M1 14 M2 8
#   B_safe : calibrated safe     — conf .01 p1 .001/.01, M1 14 M2 8
#   C_bal  : calibrated balanced — conf .01 p1 .01/.01,  M1 14 M2 8
#   D_balM : C_bal + target-row cut — M1 12 M2 7
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="${ROOT}/experiments/proxy_async_overlap/tree_sweep/run_p1_p2_tree_formal_20260807.sh"
OUT_ROOT="${OUT_ROOT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/threshold_ab_20260808}"
mkdir -p "${OUT_ROOT}"

run_arm () {
  local arm="$1" seed="$2" conf="$3" p1s="$4" p1c="$5" m1="$6" m2="$7"
  echo "[$(date -Is)] ARM ${arm} seed=${seed} conf=${conf} p1=${p1s}/${p1c} M=${m1}/${m2}"
  OUT="${OUT_ROOT}/${arm}" ARMS="both" SEEDS="${seed}" \
  RUN_NS="${AB_NS:-10}" RUN_OUTLEN="${AB_OUTLEN:-384}" \
  TREE_PROXY_THRESHOLD=0.01 TREE_CONF_THRESHOLD="${conf}" \
  P1_START_THRESHOLD="${p1s}" P1_CONF_THRESHOLD="${p1c}" \
  P1_TREE_VERIFY_NODES="${m1}" P2_TREE_VERIFY_NODES="${m2}" \
  RESUME="${RESUME:-1}" GPU_SET="${GPU_SET:-0,1,2,3,4}" \
    bash "${BASE}"
  sleep 15
}

case_of () {
  case "$1" in
    A_base) echo "0.03 0    0    14 8" ;;
    B_safe) echo "0.01 0.001 0.01 14 8" ;;
    C_bal)  echo "0.01 0.01 0.01 14 8" ;;
    D_balM) echo "0.01 0.01 0.01 12 7" ;;
  esac
}

arm_order_of () {
  case "$1" in
    42)   echo "A_base B_safe C_bal D_balM" ;;
    123)  echo "C_bal D_balM A_base B_safe" ;;
    2024) echo "B_safe A_base D_balM C_bal" ;;
    *)    echo "A_base B_safe C_bal D_balM" ;;
  esac
}

IFS=',' read -r -a seeds <<<"${AB_SEEDS:-42,123,2024}"
for seed in "${seeds[@]}"; do
  for arm in $(arm_order_of "${seed}"); do
    # shellcheck disable=SC2046
    run_arm "${arm}" "${seed}" $(case_of "${arm}")
  done
done

echo "[$(date -Is)] THRESHOLD_AB_DONE"
for arm in A_base B_safe C_bal D_balM; do
  for d in "${OUT_ROOT}/${arm}"/*_both_s*; do
    [ -d "${d}" ] || continue
    t=$(grep -m1 "Final Decode Throughput" "${d}/run.log" 2>/dev/null | grep -o "[0-9.]*" | tail -1)
    tk=$(grep -m1 "Tokens per step (incl" "${d}/metrics.txt" 2>/dev/null | grep -o "[0-9.]*$")
    a1=$(grep -m1 "Phase 1 Accepted Len" "${d}/metrics.txt" 2>/dev/null | grep -o "[0-9.]*$")
    a2=$(grep -m1 "Phase 2 Accepted Len" "${d}/metrics.txt" 2>/dev/null | grep -o "[0-9.]*$")
    echo "${arm} $(basename "${d}") tps=${t} tok/step=${tk} p1al=${a1} p2al=${a2}"
  done
done
