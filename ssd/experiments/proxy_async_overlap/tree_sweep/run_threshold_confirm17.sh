#!/usr/bin/env bash
# D_balM 채택 확정 게이트 — eslab17 클린박스 절대치 (threshold_ab_20260808 후속).
# 전제: GPU 0-4 유휴. D(보정 threshold + M1 12/M2 7) vs A(champion) paired,
# 3-seed 순서회전, full-scale ns=20 outlen=384.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="${ROOT}/experiments/proxy_async_overlap/tree_sweep/run_p1_p2_tree_formal_20260807.sh"
OUT_ROOT="${OUT_ROOT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/threshold_confirm17}"
mkdir -p "${OUT_ROOT}"

gmax=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3,4 | sort -n | tail -1)
if [ "${gmax}" -ge 1000 ]; then
  echo "GPU 0-4 not clean (max used ${gmax}MiB) — 17 클린박스 전제 위반" >&2
  exit 2
fi

run_arm () {
  local arm="$1" seed="$2" conf="$3" p1s="$4" p1c="$5" m1="$6" m2="$7"
  echo "[$(date -Is)] ARM ${arm} seed=${seed}"
  OUT="${OUT_ROOT}/${arm}" ARMS="both" SEEDS="${seed}" \
  RUN_NS="${CONFIRM_NS:-20}" RUN_OUTLEN="${CONFIRM_OUTLEN:-384}" \
  TREE_PROXY_THRESHOLD=0.01 TREE_CONF_THRESHOLD="${conf}" \
  P1_START_THRESHOLD="${p1s}" P1_CONF_THRESHOLD="${p1c}" \
  P1_TREE_VERIFY_NODES="${m1}" P2_TREE_VERIFY_NODES="${m2}" \
  RESUME="${RESUME:-1}" GPU_SET="${GPU_SET:-0,1,2,3,4}" \
    bash "${BASE}"
  sleep 15
}

for seed in 42 123 2024; do
  case "${seed}" in
    123) order="D_balM A_base" ;;
    *)   order="A_base D_balM" ;;
  esac
  for arm in ${order}; do
    if [ "${arm}" = "A_base" ]; then
      run_arm A_base "${seed}" 0.03 0 0 14 8
    else
      run_arm D_balM "${seed}" 0.01 0.01 0.01 12 7
    fi
  done
done

echo "[$(date -Is)] CONFIRM17_DONE"
for arm in A_base D_balM; do
  for d in "${OUT_ROOT}/${arm}"/*_both_s*; do
    [ -d "${d}" ] || continue
    t=$(grep -m1 "Final Decode Throughput" "${d}/run.log" 2>/dev/null | grep -o "[0-9.]*" | tail -1)
    echo "${arm} $(basename "${d}") tps=${t}"
  done
done
