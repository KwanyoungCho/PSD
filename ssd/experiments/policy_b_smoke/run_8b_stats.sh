#!/bin/bash
# Tiny smoke (Policy B) with stats instrumentation. Generates per-step
# logs of: (1) h vs r̂ dominance per chosen pick; (2) per-position match
# rate (greedy view) for each verify step.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12399
export SSD_FORCE_SPLIT_K1K2=1
export SSD_MESA_POLICY_B_STATS=1     # ★ enable instrumentation

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES explicitly"; exit 1
fi
echo "[gpu-set] $CUDA_VISIBLE_DEVICES"

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/models/layerskip-llama3-8B
DRAFT=/data2/chokwans99/models/Llama-3.2-1B-Instruct
OUT="$PWD/experiments/policy_b_smoke/out_8b_stats"
mkdir -p "$OUT"

# n=3 sequences, output=20 → ~60 verify steps per run. Manageable log.
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 20 --max_model_len 1024 --random --numseqs 3 --seed 42"

echo "[$(date +%H:%M:%S)] === 8B Policy B stats ==="
fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
"$PY" -O bench/bench.py $COMMON \
    --k 16 --f 3 --mesa --mesa_phase1_k 8 --mesa_phase2_k 8 \
    --mesa_draft_fan_out 2 --mesa_policy b \
    >"$OUT/run.log" 2>&1
rc=$?
echo "  rc=$rc"

echo ""
echo "===== POLICY-B-STATS samples ====="
echo "--- first 5 [POLICY-B-STATS] (h vs r̂) lines ---"
grep "POLICY-B-STATS] step" "$OUT/run.log" | head -5
echo ""
echo "--- first 10 [POLICY-B-STATS pos-match] lines ---"
grep "POLICY-B-STATS pos-match" "$OUT/run.log" | head -10

echo ""
echo "===== aggregate analysis ====="
"$PY" -c "
import re
log = open('$OUT/run.log').read()

# Q1: per-position match analysis
lines = re.findall(r'POLICY-B-STATS pos-match.*accept_len=(\d+) cache_hit=(\d+) phase_src=(\d+) match_per_pos=([01]+)', log)
print(f'Total verify steps logged: {len(lines)}')
print()
# Group by (cache_hit, phase_src)
from collections import Counter, defaultdict
group_counts = Counter()
group_match_pos = defaultdict(lambda: [0]*16)   # match count per position
group_total = Counter()
zero_accept_steps = 0
zero_accept_match_after = defaultdict(int)
for al, ch, ps, m in lines:
    al = int(al); ch = int(ch); ps = int(ps)
    key = ('hit' if ch else 'miss', f'ps{ps}')
    group_counts[key] += 1
    for i, c in enumerate(m):
        group_match_pos[key][i] += int(c)
    group_total[key] += 1
    # zero accept analysis
    if al == 0:
        zero_accept_steps += 1
        for i, c in enumerate(m):
            if int(c):
                zero_accept_match_after[i] += 1
print(f'=== group counts ===')
for k, v in group_counts.items():
    print(f'  {k}: {v}')
print()
print(f'=== per-position match rate (within group) ===')
for k, n in group_total.items():
    rates = [group_match_pos[k][i]/n if n else 0 for i in range(16)]
    rstr = ' '.join(f'{r:.2f}' for r in rates[:9])
    print(f'  {k} (n={n}) pos[0..8]: {rstr}')
print()
print(f'=== zero-accept analysis ({zero_accept_steps} steps with accept_len=0) ===')
print(f'  if any later position matched (probabilistic but non-zero):')
for i in range(16):
    if zero_accept_match_after[i]:
        print(f'    pos {i}: matched in {zero_accept_match_after[i]} of {zero_accept_steps} ({100*zero_accept_match_after[i]/zero_accept_steps:.1f}%)')

# Q2: h vs r̂ dominance
print()
print(f'=== Q2: h vs r̂ dominance in chosen picks ===')
hr_lines = re.findall(r'POLICY-B-STATS\] step .* picks\(pos,h,r,h\*r\)\[0\.\.11\]=(.+?)$', log, re.MULTILINE)
print(f'Steps with picks logged: {len(hr_lines)}')
import statistics
all_h = []
all_r = []
all_pos = []
for line in hr_lines[:30]:   # first 30 steps
    picks = re.findall(r'\((\d+),([\d.]+),([\d.]+),([\d.]+)\)', line)
    for pos, h, r, score in picks:
        all_pos.append(int(pos)); all_h.append(float(h)); all_r.append(float(r))
if all_h:
    print(f'  h    median={statistics.median(all_h):.3f}  mean={statistics.mean(all_h):.3f}  stdev={statistics.stdev(all_h):.3f}')
    print(f'  r    median={statistics.median(all_r):.3f}  mean={statistics.mean(all_r):.3f}  stdev={statistics.stdev(all_r):.3f}')
    print(f'  ratio h/r: {statistics.mean(all_h)/statistics.mean(all_r):.2f}x')
    pos_count = Counter(all_pos)
    print(f'  position distribution of picks: {dict(sorted(pos_count.items()))}')"
