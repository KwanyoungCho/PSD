#!/usr/bin/env bash
# Medium-size repeated-seed check after the one-seed full N2=12 regression.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${HERE}"

run_arm() {
  local seed=$1
  local n2=$2
  local port=$3
  DATA=${HERE}/postfull_screening_subset.jsonl \
    OUTLEN=256 SEED=${seed} K1=8 K2=5 EXIT_LAYER=49 \
    N1=14 M1=12 N2=${n2} M2=10 ROOT_COUNT=10 PORT=${port} \
    bash ./run_screen_arm.sh "multiseed_s${seed}_n${n2}m10"
}

# Seed 42 already has N2=10 and N2=12 arms; only fill the midpoint.
run_arm 42 11 18961

# Two independent seeds complete the balanced 3 x 3 comparison.
run_arm 1 10 18962
run_arm 1 11 18963
run_arm 1 12 18964
run_arm 123 10 18965
run_arm 123 11 18966
run_arm 123 12 18967

echo "[$(date -Is)] post-full multiseed sweep complete"
