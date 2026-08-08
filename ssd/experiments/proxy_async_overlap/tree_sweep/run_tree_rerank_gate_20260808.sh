#!/usr/bin/env bash
# Narrow, evidence-driven gate for hit-time dynamic-tree reranking.
#
# Candidate caps come from post-hoc analysis of real cache-hit walks, not a
# broad parameter sweep:
#   P1 generated=18: verify 12 or 14 (18 is the generation-matched baseline)
#   P2 generated=8 : verify 7       (8 is the generation-matched baseline)
#
# Usage:
#   STAGE=screen bash experiments/proxy_async_overlap/tree_sweep/run_tree_rerank_gate_20260808.sh
#   STAGE=final  P1_WINNER=14 P2_WINNER=7 \
#     bash experiments/proxy_async_overlap/tree_sweep/run_tree_rerank_gate_20260808.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="${ROOT}/experiments/proxy_async_overlap/tree_sweep/run_p1_p2_tree_formal_20260807.sh"
OUT_ROOT="${OUT_ROOT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/tree_rerank_gate_20260808}"
STAGE="${STAGE:-screen}"
mkdir -p "${OUT_ROOT}"

run_case () {
  local name="$1" arms="$2" seeds="$3" run_ns="$4" outlen="$5"
  local p1_verify="$6" p2_verify="$7"
  echo "[$(date -Is)] CASE ${name}: arms=${arms} seeds=${seeds} P1=18/${p1_verify} P2=8/${p2_verify}"
  OUT="${OUT_ROOT}/${name}" ARMS="${arms}" SEEDS="${seeds}" \
  RUN_NS="${run_ns}" RUN_OUTLEN="${outlen}" \
  P1_TREE_MAX_NODES=18 P1_TREE_VERIFY_NODES="${p1_verify}" \
  P2_TREE_MAX_NODES=8 P2_TREE_VERIFY_NODES="${p2_verify}" \
  RESUME="${RESUME:-0}" GPU_SET="${GPU_SET:-0,1,2,3,4}" \
    bash "${BASE}"
}

case "${STAGE}" in
  screen)
    # Eight prompts/case are sufficient only to reject obviously harmful
    # caps.  The winner must pass the paired final gate below.
    run_case p1_v18 p1_tree 42 "${SCREEN_NS:-2}" "${SCREEN_OUTLEN:-192}" 18 8
    run_case p1_v14 p1_tree 42 "${SCREEN_NS:-2}" "${SCREEN_OUTLEN:-192}" 14 8
    run_case p1_v12 p1_tree 42 "${SCREEN_NS:-2}" "${SCREEN_OUTLEN:-192}" 12 8
    run_case p2_v8  p2_tree 42 "${SCREEN_NS:-2}" "${SCREEN_OUTLEN:-192}" 18 8
    run_case p2_v7  p2_tree 42 "${SCREEN_NS:-2}" "${SCREEN_OUTLEN:-192}" 18 7
    ;;
  final)
    : "${P1_WINNER:?set P1_WINNER to the screened P1 verification cap}"
    : "${P2_WINNER:?set P2_WINNER to the screened P2 verification cap}"
    # Baseline and selected rerank are separate directories but use the same
    # three seeds.  Compare paired per-seed deltas, not pooled single runs.
    run_case both_baseline both "${FINAL_SEEDS:-42,123,2024}" \
      "${FINAL_NS:-5}" "${FINAL_OUTLEN:-256}" 18 8
    run_case both_rerank both "${FINAL_SEEDS:-42,123,2024}" \
      "${FINAL_NS:-5}" "${FINAL_OUTLEN:-256}" \
      "${P1_WINNER}" "${P2_WINNER}"
    ;;
  *)
    echo "STAGE must be screen or final; got ${STAGE}" >&2
    exit 2
    ;;
esac

echo "[$(date -Is)] RERANK_${STAGE}_DONE"
