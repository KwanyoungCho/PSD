#!/usr/bin/env bash
# MESA K1=K2=7 sweep — exit_layer=52, dfo × pfo grid at PROFILE_MESA=0.
#
# Cells (12):
#   dfo ∈ {2, 3, 4, 5} × pfo ∈ {1, 2, 3}
# pfo is derived from async_fan_out − dfo, so per-cell we pass:
#   --f $(dfo + pfo)  --mesa_draft_fan_out $dfo
#
# Other params match 20260512_ours_label_perf paper baseline except:
#   --mesa_exit_layer 52 (was 56)
#   K1=K2=7  →  --k 14  --mesa_phase1_k 7  --mesa_phase2_k 7
# SSD_PROFILE_MESA=0 (cold path, paper-grade TPS).
# SSD_FORCE_SPLIT_K1K2=1 (MESA split mode).

set -uo pipefail

ROOT="/home/chokwans99/PSD/ssd"
SWEEP_DIR="${ROOT}/experiments/proxy_async_overlap/mesa_k1k2_7_exit52_dfo_pfo_sweep"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT_BASE=12800
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
# MESA cold path: PROFILE_MESA=0, SSD_FORCE_SPLIT_K1K2=1.
export SSD_PROFILE_MESA=0
export SSD_FORCE_SPLIT_K1K2=1

run_one() {
  local DFO=$1 PFO=$2
  local F=$((DFO + PFO))
  local OUTDIR="${SWEEP_DIR}/dfo${DFO}_pfo${PFO}"
  mkdir -p "${OUTDIR}"

  export SSD_DIST_PORT="$((SSD_DIST_PORT_BASE + DFO * 10 + PFO))"

  echo "[$(date -Is)] === START dfo=${DFO} pfo=${PFO} (f=${F}, port ${SSD_DIST_PORT}) ===" | tee "${OUTDIR}/run.log"
  pkill -9 -f "bench.py" 2>/dev/null || true
  sleep 8

  "${PY}" -O bench/bench.py \
    --llama --size 8 \
    --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b \
    --quant_awq \
    --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4 \
    --quant_group_size 128 \
    --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 50 \
    --input_len 512 --output_len 512 --all --max_model_len 2048 \
    --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b \
    --quant_awq_draft \
    --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1 \
    --async --spec --k 14 --f "${F}" \
    --mesa --mesa_exit_layer 52 --mesa_phase1_k 7 --mesa_phase2_k 7 \
    --mesa_draft_fan_out "${DFO}" --mesa_policy b \
    >> "${OUTDIR}/run.log" 2>&1
  local status=$?

  if [[ ${status} -ne 0 ]]; then
    echo "[$(date -Is)] === FAILED dfo=${DFO} pfo=${PFO} status=${status} ===" | tee -a "${OUTDIR}/run.log"
  else
    echo "[$(date -Is)] === END dfo=${DFO} pfo=${PFO} ===" | tee -a "${OUTDIR}/run.log"
  fi

  {
    echo "dfo=${DFO} pfo=${PFO} f=${F} status=${status}"
    grep -E "Final Decode Throughput|Avg target time|Avg Tokens per step|Avg Fraction of Speculated|Avg Cache Hits|Phase 1.*hit|Phase 2.*hit|accepted_lens_phase|p1_|p2_" "${OUTDIR}/run.log" || true
  } > "${OUTDIR}/headline.txt"

  return 0
}

echo "[$(date -Is)] === SWEEP START (12 runs: dfo {2,3,4,5} × pfo {1,2,3}) ==="
for DFO in 2 3 4 5; do
  for PFO in 1 2 3; do
    run_one "${DFO}" "${PFO}"
  done
done
echo "[$(date -Is)] === SWEEP COMPLETE ==="

echo ""
echo "=== Headline summary ==="
for DFO in 2 3 4 5; do
  for PFO in 1 2 3; do
    OUTDIR="${SWEEP_DIR}/dfo${DFO}_pfo${PFO}"
    echo "--- dfo=${DFO} pfo=${PFO} ---"
    cat "${OUTDIR}/headline.txt" 2>/dev/null || echo "(no headline)"
  done
done
