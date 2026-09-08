#!/usr/bin/env bash
# Current-tree P2 confidence-threshold A/B on the balanced medium subset.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${HERE}"

run_arm() {
  local seed=$1
  local conf=$2
  local label=$3
  local port=$4
  DATA=${HERE}/postfull_screening_subset.jsonl \
    OUTLEN=256 SEED=${seed} K1=8 K2=5 EXIT_LAYER=49 \
    N1=14 M1=12 N2=10 M2=10 ROOT_COUNT=10 \
    P2_PROXY=0.01 P2_CONF=${conf} PORT=${port} \
    bash ./run_screen_arm.sh "threshold_s${seed}_c${label}"
}

run_arm 1   0.02 002 18971
run_arm 1   0.03 003 18972
run_arm 42  0.02 002 18973
run_arm 42  0.03 003 18974
run_arm 123 0.02 002 18975
run_arm 123 0.03 003 18976

echo "[$(date -Is)] post-full threshold multiseed sweep complete"
