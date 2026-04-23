#!/bin/bash
# Full bench matrix: pre-quant (dense) vs INT4 across targets.
# Each experiment: numseqs=10, output_len=256, all 4 datasets = 10240 tok total.
# MESA configs: mid exit (2L/3, default) and early exit (L/2).
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12298
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
OUT=/home/chokwans99/PSD/ssd/tmp/full_matrix

L2_7B=/data2/chokwans99/models/layerskip-llama2-7B
L3_8B=/data2/chokwans99/models/layerskip-llama3-8B
CL_34B=$(ls -d /data2/chokwans99/models/models--facebook--layerskip-codellama-34B/snapshots/*/ | head -1)
CL_34B="${CL_34B%/}"
DRAFT_L2=/data2/chokwans99/models/TinyLlama-1.1B-Chat-v1.0
DRAFT_L3=/data2/chokwans99/models/Llama-3.2-1B-Instruct

COMMON_BASE="--llama --size 8 --b 1 --temp 0.6 --output_len 256 --max_model_len 2048 --all --numseqs 10"

run() {
    local tag="$1"; shift
    local gpus="$1"; shift
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    echo ""
    echo "[$(date +%H:%M:%S)] === $tag === (GPUs=$gpus)"
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true
    sleep 2
    local t0=$SECONDS
    CUDA_VISIBLE_DEVICES=$gpus "$PY" -O bench/bench.py "$@" > "$subdir/run.log" 2>&1
    local rc=$?
    local wall=$((SECONDS - t0))
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    local acc=$(grep "Avg Fraction" "$subdir/run.log" | grep -oP '[\d.]+' | tail -1)
    local tok_step=$(grep "Avg Tokens per step (incl recovery)" "$subdir/run.log" | head -1 | grep -oP '[\d.]+' | tail -1)
    local ch=$(grep "Avg Cache Hits" "$subdir/run.log" | grep -oP '[\d.]+' | tail -1)
    local traceback=$(grep -c "Traceback" "$subdir/run.log")
    local err=""
    if [ $rc -ne 0 ] || [ -z "$tp" ]; then
        err=$(grep -E "ValueError|RuntimeError|QuantizedLinearNotImplementedError|OutOfMemory" "$subdir/run.log" | head -1 | cut -c1-80)
    fi
    echo "  rc=$rc wall=${wall}s tp=${tp:-FAIL} acc=${acc:-n/a} tok/step=${tok_step:-n/a} cache_hit=${ch:-n/a} ${err:+err=$err}"
}

# ================================================================
# Target A: Llama-3-8B (bf16 native, TP=2 + draft = 3 GPUs)
# 32 layers. mid=21 (2L/3), early=16 (L/2)
# ================================================================
echo ""
echo "################################################################"
echo "# Target A: layerskip-llama3-8B (bf16 native, TP=2)"
echo "################################################################"

T=$L3_8B
DRAFT=$DRAFT_L3
TP=2
GPUS_AR=0,1           # tp=2
GPUS_SPEC=0,1,2       # tp=2 + draft=1
MID=21
EARLY=16

C_AR="$COMMON_BASE --model_path $T --gpus $TP"
C_SPEC="$COMMON_BASE --model_path $T --gpus 3 --draft_path $DRAFT --async --spec --k 7 --flh 5 4 4 3 3 2 2 1"
C_MESA="$COMMON_BASE --model_path $T --gpus 3 --draft_path $DRAFT --async --spec --k 5 --f 4 --mesa --mesa_draft_fan_out 2"

run "A_01_ar_dense"            "$GPUS_AR"   $C_AR
run "A_02_ar_int4"             "$GPUS_AR"   $C_AR --quant_int4
run "A_03_spec_dense"          "$GPUS_SPEC" $C_SPEC
run "A_04_spec_int4"           "$GPUS_SPEC" $C_SPEC --quant_int4
run "A_05_mesa_mid_dense"      "$GPUS_SPEC" $C_MESA --mesa_exit_layer $MID
run "A_06_mesa_mid_int4"       "$GPUS_SPEC" $C_MESA --mesa_exit_layer $MID --quant_int4
run "A_07_mesa_early_dense"    "$GPUS_SPEC" $C_MESA --mesa_exit_layer $EARLY
run "A_08_mesa_early_int4"     "$GPUS_SPEC" $C_MESA --mesa_exit_layer $EARLY --quant_int4

# ================================================================
# Target B: Llama-2-7B (fp16 original → bf16 runtime override, TP=2)
# 32 layers. mid=21, early=16
# ================================================================
echo ""
echo "################################################################"
echo "# Target B: layerskip-llama2-7B (fp16 → bf16 runtime override, TP=2)"
echo "################################################################"

T=$L2_7B
DRAFT=$DRAFT_L2
TP=2
GPUS_AR=0,1
GPUS_SPEC=0,1,2
MID=21
EARLY=16

C_AR="$COMMON_BASE --model_path $T --gpus $TP"
C_SPEC="$COMMON_BASE --model_path $T --gpus 3 --draft_path $DRAFT --async --spec --k 7 --flh 5 4 4 3 3 2 2 1"
C_MESA="$COMMON_BASE --model_path $T --gpus 3 --draft_path $DRAFT --async --spec --k 5 --f 4 --mesa --mesa_draft_fan_out 2"

run "B_01_ar_dense"            "$GPUS_AR"   $C_AR
run "B_02_ar_int4"             "$GPUS_AR"   $C_AR --quant_int4 --quant_force_bf16_runtime
run "B_03_spec_dense"          "$GPUS_SPEC" $C_SPEC
run "B_04_spec_int4"           "$GPUS_SPEC" $C_SPEC --quant_int4 --quant_force_bf16_runtime
run "B_05_mesa_mid_dense"      "$GPUS_SPEC" $C_MESA --mesa_exit_layer $MID
run "B_06_mesa_mid_int4"       "$GPUS_SPEC" $C_MESA --mesa_exit_layer $MID --quant_int4 --quant_force_bf16_runtime
run "B_07_mesa_early_dense"    "$GPUS_SPEC" $C_MESA --mesa_exit_layer $EARLY
run "B_08_mesa_early_int4"     "$GPUS_SPEC" $C_MESA --mesa_exit_layer $EARLY --quant_int4 --quant_force_bf16_runtime

# ================================================================
# Target C: CodeLlama-34B (fp16 original → bf16 runtime override, TP=4 + draft = 5 GPUs)
# 48 layers. mid=32, early=24
# ================================================================
echo ""
echo "################################################################"
echo "# Target C: layerskip-codellama-34B (fp16 → bf16 runtime override, TP=4)"
echo "################################################################"

T=$CL_34B
DRAFT=$DRAFT_L2
TP=4
GPUS_AR=0,1,2,3
GPUS_SPEC=0,1,2,3,4
MID=32
EARLY=24

C_AR="$COMMON_BASE --model_path $T --gpus $TP"
C_SPEC="$COMMON_BASE --model_path $T --gpus 5 --draft_path $DRAFT --async --spec --k 7 --flh 5 4 4 3 3 2 2 1"
C_MESA="$COMMON_BASE --model_path $T --gpus 5 --draft_path $DRAFT --async --spec --k 5 --f 4 --mesa --mesa_draft_fan_out 2"

run "C_01_ar_dense"            "$GPUS_AR"   $C_AR
run "C_02_ar_int4"             "$GPUS_AR"   $C_AR --quant_int4 --quant_force_bf16_runtime
run "C_03_spec_dense"          "$GPUS_SPEC" $C_SPEC
run "C_04_spec_int4"           "$GPUS_SPEC" $C_SPEC --quant_int4 --quant_force_bf16_runtime
run "C_05_mesa_mid_dense"      "$GPUS_SPEC" $C_MESA --mesa_exit_layer $MID
run "C_06_mesa_mid_int4"       "$GPUS_SPEC" $C_MESA --mesa_exit_layer $MID --quant_int4 --quant_force_bf16_runtime
run "C_07_mesa_early_dense"    "$GPUS_SPEC" $C_MESA --mesa_exit_layer $EARLY
run "C_08_mesa_early_int4"     "$GPUS_SPEC" $C_MESA --mesa_exit_layer $EARLY --quant_int4 --quant_force_bf16_runtime

echo ""
echo "################################################################"
echo "# SUMMARY"
echo "################################################################"
printf "%-30s %-10s %-8s %-10s %-12s\n" "tag" "TP" "accept" "tok/step" "cache_hit"
for t in $(ls -d $OUT/[A-C]_*/ 2>/dev/null | sort); do
    tag=$(basename "$t")
    log="$t/run.log"
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    acc=$(grep "Avg Fraction" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    tok_step=$(grep "Avg Tokens per step (incl recovery)" "$log" 2>/dev/null | head -1 | grep -oP '[\d.]+' | tail -1)
    ch=$(grep "Avg Cache Hits" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    printf "%-30s %-10s %-8s %-10s %-12s\n" "$tag" "${tp:-FAIL}" "${acc:-n/a}" "${tok_step:-n/a}" "${ch:-n/a}"
done
