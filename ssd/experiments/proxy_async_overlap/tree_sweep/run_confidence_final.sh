#!/usr/bin/env bash
# Final quality/speed gate.  Four structural arms, three seeds, rotated order.
# Parameters are frozen within an arm; this is the single confirmatory run,
# not an exploratory sweep.
set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
OUT="${ROOT}/experiments/proxy_async_overlap/tree_sweep/confidence_final"
mkdir -p "${OUT}"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
export SSD_DUET_EXIT_REPLICA=1 SSD_ASYNC_PROXY_SEND=1 SSD_PROXY_STREAM=0

BASE=(--llama --size 8
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128 --b 1 --temp 0.7
  --input_len 512 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --gpus 5 --async --spec --duet
  --duet_exit_layer 56 --f 3 --duet_k1 9 --duet_k2 4
  --duet_p1_fanout 2
  --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1
  --duet_p2_budget 10 --numseqs 2 --output_len 256)

cleanup_jobs () {
  local child
  for child in $(jobs -pr); do
    kill "${child}" 2>/dev/null || true
  done
}
trap cleanup_jobs EXIT

run_one () {
  local cycle="$1" seed="$2" arm="$3" port="$4"
  local log="${OUT}/c${cycle}_s${seed}_${arm}.log"
  shift 4
  if [[ "${RESUME:-0}" == "1" ]] && grep -q "EXIT:0" "${log}" 2>/dev/null; then
    echo "[$(date -Is)] resume-skip cycle=${cycle} seed=${seed} arm=${arm}"
    return 0
  fi
  echo "[$(date -Is)] cycle=${cycle} seed=${seed} arm=${arm}"
  SSD_DIST_PORT="${port}" SSD_TREE_EXEC=1 SSD_TREE_ARENA=1 \
    SSD_PROFILE=0 SSD_PROFILE_DUET=0 \
    timeout 15m "${PY}" -O bench/bench.py "${BASE[@]}" --seed "${seed}" \
      "$@" >"${log}" 2>&1
  local rc=$?
  echo "EXIT:${rc}" >>"${log}"
  grep -E "Final Decode Throughput|Avg Tokens per step|Hit Rate|Accepted Len|Avg target time|Avg target verify|Avg draft step|p2exec stats" \
    "${log}" || true
  return "${rc}"
}

run_arm () {
  local cycle="$1" seed="$2" arm="$3" port="$4"
  case "${arm}" in
    chain)
      run_one "${cycle}" "${seed}" "${arm}" "${port}" \
        --duet_tree_policy off ;;
    legacy_r6)
      run_one "${cycle}" "${seed}" "${arm}" "${port}" \
        --duet_tree_policy level --duet_tree_root_count 6 \
        --duet_tree_nv 8 --duet_tree_beta 0.5 ;;
    confidence_nv6)
      run_one "${cycle}" "${seed}" "${arm}" "${port}" \
        --duet_tree_policy confidence --duet_tree_nv 6 ;;
    confidence_nv8)
      run_one "${cycle}" "${seed}" "${arm}" "${port}" \
        --duet_tree_policy confidence --duet_tree_nv 8 ;;
    *) return 2 ;;
  esac
}

seeds=(42 123 2024)
orders=(
  "chain legacy_r6 confidence_nv6 confidence_nv8"
  "legacy_r6 confidence_nv6 confidence_nv8 chain"
  "confidence_nv6 confidence_nv8 chain legacy_r6"
)
port=15600
for cycle in 0 1 2; do
  for arm in ${orders[$cycle]}; do
    port=$((port + 1))
    run_arm "${cycle}" "${seeds[$cycle]}" "${arm}" "${port}"
  done
done

"${PY}" - "${OUT}" <<'PY'
import json, pathlib, re, statistics, sys

out = pathlib.Path(sys.argv[1])
patterns = {
    "tps": r"Final Decode Throughput: ([0-9.]+)",
    "tok_step": r"Avg Tokens per step \(incl recovery\): ([0-9.]+)",
    "target_ms": r"Avg target time per full step \(ms\): ([0-9.]+)",
    "verify_ms": r"Avg target verify time \(ms\): ([0-9.]+)",
    "p1_hit": r"Avg Phase 1 \(draft\) Hit Rate: ([0-9.]+)",
    "p2_hit": r"Avg Phase 2 \(proxy\) Hit Rate: ([0-9.]+)",
    "p1_al": r"Avg Phase 1 Accepted Len: ([0-9.]+)",
    "p2_al": r"Avg Phase 2 Accepted Len: ([0-9.]+)",
    "draft_ms": r"Avg draft step time \(ms\): ([0-9.]+)",
}
rows = {}
for path in sorted(out.glob("c*_*.log")):
    text = path.read_text(errors="replace")
    if "EXIT:0" not in text:
        continue
    arm = path.stem.split("_", 2)[2]
    row = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            row[key] = float(m.group(1))
    rows.setdefault(arm, []).append(row)
summary = {}
for arm, vals in rows.items():
    summary[arm] = {"n": len(vals)}
    for key in patterns:
        xs = [v[key] for v in vals if key in v]
        if xs:
            summary[arm][key] = round(statistics.mean(xs), 4)
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo CONFIDENCE_FINAL_DONE
