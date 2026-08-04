#!/usr/bin/env bash
set -u
ROOT=/home/chokwans99/PSD/ssd
PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
cd $ROOT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
run_prof () {
  local mode="$1" dir="$2" port="$3" extra=""
  rm -rf "$dir"; mkdir -p "$dir"
  if [ "$mode" != "off" ]; then extra="--duet_tree_nv 8 --duet_tree_beta 0.5 --duet_tree_root_count 6"; fi
  SSD_PROFILE_DUET=1 SSD_PROFILE_DIR="$dir" SSD_DIST_PORT=$port $PY -O bench/bench.py --llama --size 8 \
    --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b \
    --quant_awq --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4 \
    --quant_group_size 128 --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 4 \
    --input_len 512 --output_len 192 --all --max_model_len 2048 \
    --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b \
    --quant_awq_draft --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1 \
    --async --spec --duet --duet_exit_layer 56 --f 3 \
    --duet_k1 9 --duet_k2 4 --duet_p1_fanout 2 \
    --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1 \
    --duet_p2_budget 10 \
    --duet_tree_policy "$mode" $extra \
    > "$dir/run.log" 2>&1
  echo "EXIT:$?" >> "$dir/run.log"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "chokwans99" ] && kill -9 $p 2>/dev/null
  done
  sleep 8
}
run_prof off   $ROOT/experiments/proxy_async_overlap/tree_sweep/timeline_v2/prof_chain 14030
run_prof level $ROOT/experiments/proxy_async_overlap/tree_sweep/timeline_v2/prof_treeR6 14031
echo PROF_PAIR18_DONE
