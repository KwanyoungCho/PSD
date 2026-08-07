#!/usr/bin/env bash
# Sparse-to-fine DUET P1/P2 tree sweep wrapper.
#
# A spec is:
#   label:K1:K2:N1:N2:C:P1_SCALE:P2_WIDTH:arms
# where arms is one of chain,p1_tree,p2_tree,both (or a comma list).
# Specs are separated by semicolons.  Root counts stay fixed by the formal
# runner (P1 roots/context=2, P2 R=W=10) unless explicitly overridden.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FORMAL="${ROOT}/experiments/proxy_async_overlap/tree_sweep/run_p1_p2_tree_formal_20260807.sh"
OUT="${OUT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/p1_p2_sparse_sweep_20260807}"
SERVER_LABEL="${SERVER_LABEL:-$(hostname -s)}"
SEEDS="${SEEDS:-42}"
RUN_NS="${RUN_NS:-8}"
RUN_OUTLEN="${RUN_OUTLEN:-192}"

if [[ -z "${SWEEP_SPECS:-}" ]]; then
  echo "SWEEP_SPECS is required; example:" >&2
  echo "  p1k7:7:4:14:8:3:1.0:10:chain,p1_tree;"\
"p2k5:9:5:18:10:3:1.0:10:chain,p2_tree" >&2
  exit 2
fi

mkdir -p "${OUT}"
IFS=';' read -r -a specs <<<"${SWEEP_SPECS}"
for spec in "${specs[@]}"; do
  IFS=':' read -r label k1 k2 n1 n2 c p1_scale p2_width arms <<<"${spec}"
  if [[ -z "${label}" || -z "${k1}" || -z "${k2}" \
      || -z "${n1}" || -z "${n2}" || -z "${c}" \
      || -z "${p1_scale}" || -z "${p2_width}" || -z "${arms}" ]]; then
    echo "invalid sweep spec: ${spec}" >&2
    exit 2
  fi
  echo "[$(date -Is)] SWEEP label=${label} K1=${k1} K2=${k2} "\
"N1=${n1} N2=${n2} C=${c} P1scale=${p1_scale} P2W=${p2_width} "\
"arms=${arms}"
  OUT="${OUT}/${label}" SERVER_LABEL="${SERVER_LABEL}_${label}" \
  SEEDS="${SEEDS}" ARMS="${arms}" RUN_NS="${RUN_NS}" \
  RUN_OUTLEN="${RUN_OUTLEN}" K1="${k1}" K2="${k2}" \
  P1_TREE_MAX_NODES="${n1}" P2_TREE_MAX_NODES="${n2}" TREE_C="${c}" \
  P1_TREE_FORWARD_SCALE="${p1_scale}" P2_WIDTH="${p2_width}" \
    bash "${FORMAL}" || exit $?
done

echo "[$(date -Is)] SPARSE_SWEEP_DONE server=${SERVER_LABEL}" \
  | tee "${OUT}/DONE_${SERVER_LABEL}"
