#!/bin/bash
# 70B Policy B with stats instrumentation.
# Q1: when cache miss, does h's argmax match actual reject position?
# Q2: in Policy B picks, h vs r̂ dominance.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12399
export SSD_FORCE_SPLIT_K1K2=1
export SSD_MESA_POLICY_B_STATS=1

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES explicitly"; exit 1
fi
echo "[gpu-set] $CUDA_VISIBLE_DEVICES"
echo "[host-load] $(uptime | awk -F'load average:' '{print $2}')"

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
OUT="$PWD/experiments/policy_b_smoke/out_70b_stats"
mkdir -p "$OUT"

# n=10, output=64, temp=0.6, seed=42, exit=57 — 같은 조건의 70B
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10 --seed 42"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

echo "[$(date +%H:%M:%S)] === 70B Policy B stats ==="
fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
"$PY" -O bench/bench.py $COMMON $QUANT \
    --k 16 --f 3 --mesa --mesa_exit_layer 57 \
    --mesa_phase1_k 8 --mesa_phase2_k 8 \
    --mesa_draft_fan_out 2 --mesa_policy b \
    >"$OUT/run.log" 2>&1
rc=$?
tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
err=$(grep -oP 'Traceback|RuntimeError|FAIL|AssertionError' "$OUT/run.log" | head -2)
echo "  rc=$rc TP=${tp:-?} ${err:+ERR=$err}"

echo ""
echo "===== Q1 analysis: h calibration on miss steps ====="
"$PY" -c "
import re
log = open('$OUT/run.log').read()

# Parse [POLICY-B-STATS] step lines (target side, sets h vector for that step)
# and [POLICY-B-STATS pos-match] lines (target side, post-verify, has accept_len + cache_hit).
# Both fire per verify step in the same order on rank 0.

step_h = []   # h vectors per step
for m in re.finditer(r'POLICY-B-STATS\] step K=(\d+) h=\[([^\]]+)\]', log):
    K = int(m.group(1))
    h = [float(x.strip().strip(\"'\")) for x in m.group(2).split(',')]
    step_h.append((K, h))

step_match = []   # (accept_len, cache_hit, phase_src, match_string) per step
for m in re.finditer(r'POLICY-B-STATS pos-match\] b=\d+ K=(\d+) accept_len=(\d+) cache_hit=(\d+) phase_src=(\d+) match_per_pos=([01]+)', log):
    K = int(m.group(1)); al = int(m.group(2)); ch = int(m.group(3))
    ps = int(m.group(4)); mstr = m.group(5)
    step_match.append((K, al, ch, ps, mstr))

print(f'h vectors logged: {len(step_h)}')
print(f'pos-match logged: {len(step_match)}')

# Pair by index — both fire in same order on rank 0
n = min(len(step_h), len(step_match))
print(f'Paired steps: {n}')
print()

# Q1: For miss steps, is actual reject position aligned with h's prediction?
# actual_reject_pos = accept_len (first position that was rejected; if accept_len==K, all accepted → reject at all-accept slot K)
# h: vector of length K+1, h[i] = P(first reject at position i), i ∈ [0, K]
# h's argmax = predicted most-likely reject position
miss_matches = []  # list of (actual_rej, argmax_h, top3_h_positions, h_vec)
hit_matches = []
import statistics
for i in range(n):
    Kh, hvec = step_h[i]
    Km, al, ch, ps, mstr = step_match[i]
    # actual reject = al (if al < K, that's where target rejected; if al == K, all accepted)
    actual_rej = al
    # argmax h
    arg_max = hvec.index(max(hvec))
    # top-3 of h
    sorted_idx = sorted(range(len(hvec)), key=lambda j: -hvec[j])
    top3 = sorted_idx[:3]
    if ch == 0:   # miss
        miss_matches.append((actual_rej, arg_max, top3, hvec))
    else:
        hit_matches.append((actual_rej, arg_max, top3, hvec))

def report(group_name, matches):
    if not matches:
        print(f'{group_name}: no data')
        return
    n_g = len(matches)
    correct1 = sum(1 for ar, am, t3, _ in matches if ar == am)
    correct3 = sum(1 for ar, am, t3, _ in matches if ar in t3)
    # 'within-1': actual reject within ±1 of argmax(h)
    within1 = sum(1 for ar, am, t3, _ in matches if abs(ar - am) <= 1)
    print(f'{group_name} (n={n_g}):')
    print(f'  actual_reject == argmax(h):     {correct1:>4} / {n_g} = {100*correct1/n_g:.1f}%')
    print(f'  actual_reject in top-3 of h:    {correct3:>4} / {n_g} = {100*correct3/n_g:.1f}%')
    print(f'  actual_reject within ±1 of argmax: {within1:>4} / {n_g} = {100*within1/n_g:.1f}%')
    # accept_len histogram
    al_hist = [0]*(matches[0][3].__len__())
    for ar, _, _, _ in matches:
        if ar < len(al_hist):
            al_hist[ar] += 1
    print(f'  actual_reject distribution: {al_hist}')
    # argmax h histogram
    am_hist = [0]*(matches[0][3].__len__())
    for _, am, _, _ in matches:
        am_hist[am] += 1
    print(f'  argmax(h) distribution:     {am_hist}')

report('=== MISS steps', miss_matches)
print()
report('=== HIT steps', hit_matches)

# Q2: also re-print the dominance summary
print()
print('=== Q2: h vs r̂ dominance (full sample) ===')
all_h = []
all_r = []
all_pos = []
for line in re.findall(r'POLICY-B-STATS\] step .* picks\(pos,h,r,h\*r\)\[0\.\.11\]=(.+?)$', log, re.MULTILINE):
    picks = re.findall(r'\((\d+),([\d.]+),([\d.]+),([\d.]+)\)', line)
    for pos, h, r, score in picks:
        all_pos.append(int(pos)); all_h.append(float(h)); all_r.append(float(r))
if all_h:
    print(f'  picks logged: {len(all_h)}')
    print(f'  h    median={statistics.median(all_h):.3f}  mean={statistics.mean(all_h):.3f}  stdev={statistics.stdev(all_h):.3f}')
    print(f'  r    median={statistics.median(all_r):.3f}  mean={statistics.mean(all_r):.3f}  stdev={statistics.stdev(all_r):.3f}')
    print(f'  ratio h/r: {statistics.mean(all_h)/statistics.mean(all_r):.2f}x')
    from collections import Counter
    pos_count = Counter(all_pos)
    print(f'  position distribution of picks: {dict(sorted(pos_count.items()))}')
"
