# AWQ W4A16 reproduction of `final_exp2` — TEMPLATE (numbers to be filled in)

Same six configurations as `tmp/final_exp2/REPORT.md`. Everything identical
except the target model's linear projections are now AWQ-calibrated W4A16
(Marlin), packed by our offline calibration + import pipeline.

## Setup

- **Target**: `facebook/layerskip-codellama-34B` — calibrated output at
  `/data2/chokwans99/awq_calibrated/codellama34b/` (AWQ, α=0.5, 128 C4
  samples, seq_len 512, group_size 128, zero-point True, w_bit 4).
  SSD-native artifact at
  `/data2/chokwans99/awq_artifacts/codellama34b_awq_tp4.rank{0..3}.awq.pt`.
- **Draft**: `TinyLlama-1.1B-Chat-v1.0` (TP=1, dense — unchanged).
- **Prompts**: 200 (humaneval/alpaca/gsm8k/ultrafeedback × 50),
  `output_len=256`, `temp=0.6`, B=1, `max_model_len=2048`.

### Configs

Identical to `final_exp2`:

| # | Name | Flags | MQ_LEN |
|:--:|------|-------|:------:|
| 1 | `ar` | no spec (TP=4 target only) | — |
| 2 | `baseline_k7_uniform` | `--k 7 --f 3` | 24 |
| 3 | `baseline_k7_geo` | `--k 7 --flh 5 4 4 3 3 2 2 1` | 24 |
| 4 | `mesa_k5_f4_dfo2_exit24` | `--k 5 --f 4 --mesa --exit 24 --dfo 2` | 24 |
| 5 | `mesa_k5_f4_dfo2_exit28` | exit=28 | 24 |
| 6 | `mesa_k5_f4_dfo2_exit32` | exit=32 | 24 |

The only command-line delta vs `final_exp2/run_all.sh` is the appended
`--quant_awq --quant_awq_artifact /data2/.../codellama34b_awq_tp4`.

## Results (dense fp16 baseline from `final_exp2` vs this AWQ run)

### Throughput

| Config | dense TP (tok/s) | **AWQ TP (tok/s)** | **Speedup** |
|--------|-----------:|-----------:|-----------:|
| AR (TP=4)              | 28.26 | TBD | TBD |
| Baseline K=7 uniform   | 67.99 | TBD | TBD |
| Baseline K=7 geo       | 68.45 | TBD | TBD |
| MESA K=5 exit=24       | 60.73 | TBD | TBD |
| MESA K=5 exit=28       | 58.48 | TBD | TBD |
| MESA K=5 exit=32       | 58.22 | TBD | TBD |

### Per-spec metrics

| Config | dense Accept / CacheHit / Tok/Step | AWQ Accept / CacheHit / Tok/Step |
|--------|:---:|:---:|
| Baseline K=7 uniform | 0.44 / 0.66 / 4.06 | TBD |
| Baseline K=7 geo     | 0.44 / 0.66 / 4.07 | TBD |
| MESA K=5 exit=24     | 0.52 / 0.83 / 3.62 | TBD |
| MESA K=5 exit=28     | 0.52 / 0.85 / 3.60 | TBD |
| MESA K=5 exit=32     | 0.52 / 0.87 / 3.58 | TBD |

### Per-phase breakdown

See `compare_breakdown_dense_vs_quant.png` for side-by-side stacked bars
(target / draft, dense | AWQ per config).

Expected pattern (Marlin kernel analysis, confirmed separately in
`sandbox/awq_spike/08_perf_bench.py`):
- Target: `graph_pre + graph_post` (or `verify_replay` for baselines) should
  shrink — most target forward time is in dense projections and these are
  now W4A16. Attention kernels / norms / lm_head remain unchanged.
- Draft: unchanged — draft is still dense TinyLlama.

## Artifacts

```
tmp/final_exp2_quant/
├── REPORT.md                               — 이 문서
├── run_all_quant.sh                        — 전체 실행 스크립트 (AWQ 플래그 추가)
├── SUMMARY.txt                             — 메트릭 표
├── compare_throughput.png                  — dense vs AWQ 처리량 pair-bar
├── compare_breakdown_dense_vs_quant.png    — per-phase dense|AWQ stacked bar
├── ar/run.log
├── baseline_k7_uniform/ (+ mesa_profile_*.json, plot 세트)
├── baseline_k7_geo/
├── mesa_k5_f4_dfo2_exit24/
├── mesa_k5_f4_dfo2_exit28/
└── mesa_k5_f4_dfo2_exit32/
```

## Reproduce

```bash
# 0. AWQ-calibrate the 34B target (once per model, ~15 min on 7 GPUs)
bash -c 'cd /home/chokwans99/PSD/ssd && \
    CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 python scripts/awq_calibrate.py \
        --model /data2/.../layerskip-codellama-34B \
        --out   /data2/chokwans99/awq_calibrated/codellama34b \
        --n-samples 128 --seq-len 512 --alpha 0.5 --dtype float16'

# 1. pack into TP=4 SSD-native artifact (once per TP size)
python scripts/awq_import.py --mode autoawq \
    --model /data2/chokwans99/awq_calibrated/codellama34b \
    --out   /data2/chokwans99/awq_artifacts/codellama34b_awq_tp4 \
    --tp 4 --dtype float16

# 2. run all 6 configs
bash tmp/final_exp2_quant/run_all_quant.sh

# 3. regenerate compare plots
python bench/plot_compare_dense_vs_quant.py tmp/final_exp2_quant

# 4. per-config MESA breakdown plots
for d in tmp/final_exp2_quant/*/; do
    python bench/plot_mesa_breakdown.py "$d"
done
```
