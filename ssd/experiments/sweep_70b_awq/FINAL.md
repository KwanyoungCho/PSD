# 70B AWQ MESA / Async-SSD Parameter Sweep — Final Report

**Stack**: layerskip-llama2-70B (AWQ TP=4) + TinyLlama-1.1B (AWQ TP=1)  
**Hardware**: 5× RTX 3090 (4 target TP + 1 draft)  
**Workload**: B=1, temp=0.6, max_model_len=2048, random prompts  
**Reference**: AR baseline = 32.87 tok/s (from `experiments/quant_awq/70b/ar/`)

Sweep ran 51 configs across stages 0→5 (orchestrate.py adaptive driver).
All raw data in `state.json`; per-config logs in `stage{0,1A,1B,2,3,4,5}/{config}/run.log`.

## Stage 5: Confirmation (numseqs=200, output_len=256)

Final calibrated operating points, measured with the largest workload.

| config | TP (tok/s) | draft_ms | verify_ms | accept | cache_hit |
|---|---:|---:|---:|---:|---:|
| async k=7 f=8 | 76.78 | 43.95 | 45.31 | 0.44 | 0.79 |
| mesa k=6 f=3 dfo=2 exit=53 a | 72.25 | 48.87 | 46.83 | 0.48 | 0.83 |
| AR (target only) | 32.87 | — | — | — | — |

**Verdict**: Async (k=7, f=8) wins by **76.78 vs 72.25 tok/s** (+6.3% async vs MESA). Both ~2.2-2.3× over AR.

## Async SSD Calibration Table

All async runs across all stages, sorted by TP.

| workload | k | f | TP | draft_ms | verify_ms | accept | CH | T/S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NS=128/OL=256 | 7 | 8 | 79.41 | 43.96 | 45.53 | 0.47 | 0.80 | 4.26 |
| NS=128/OL=256 | 6 | 8 | 77.59 | 36.94 | 43.21 | 0.48 | 0.81 | 3.90 |
| NS=128/OL=256 | 6 | 7 | 77.10 | 36.92 | 43.10 | 0.48 | 0.79 | 3.88 |
| NS=200/OL=256 | 7 | 8 | 76.78 | 43.95 | 45.31 | 0.44 | 0.79 | 4.10 |
| NS=128/OL=256 | 7 | 7 | 75.75 | 43.61 | 45.10 | 0.43 | 0.78 | 4.03 |
| NS=128/OL=256 | 6 | 9 | 75.60 | 37.34 | 43.27 | 0.47 | 0.81 | 3.80 |
| NS=128/OL=256 | 5 | 9 | 75.54 | 30.49 | 41.79 | 0.53 | 0.83 | 3.64 |
| NS=128/OL=256 | 5 | 8 | 74.30 | 27.45 | 41.79 | 0.52 | 0.81 | 3.59 |
| NS=128/OL=256 | 5 | 7 | 73.32 | 27.24 | 41.94 | 0.51 | 0.80 | 3.56 |
| NS=64/OL=128 | 6 | 8 | 66.47 | 37.26 | 44.13 | 0.44 | 0.76 | 3.62 |
| NS=128/OL=256 | 7 | 9 | 66.16 | 51.56 | 44.96 | 0.44 | 0.81 | 4.06 |
| NS=64/OL=128 | 7 | 8 | 65.10 | 44.20 | 45.77 | 0.39 | 0.75 | 3.70 |
| NS=64/OL=128 | 7 | 3 | 64.97 | — | — | 0.40 | 0.62 | — |
| NS=64/OL=128 | 7 | 6 | 64.69 | 39.13 | 45.95 | 0.39 | 0.72 | — |
| NS=64/OL=128 | 5 | 3 | 64.37 | — | — | 0.47 | 0.64 | — |
| NS=64/OL=128 | 7 | 2 | 63.56 | — | — | 0.39 | 0.55 | — |
| NS=64/OL=128 | 7 | 4 | 63.43 | — | — | 0.39 | 0.66 | — |
| NS=64/OL=128 | 3 | 3 | 60.84 | — | — | 0.62 | 0.71 | — |
| NS=16/OL=64 | 5 | 3 | 57.18 | — | — | 0.45 | 0.61 | — |
| NS=64/OL=128 | 8 | 8 | 54.26 | 59.25 | 46.43 | 0.37 | 0.75 | 3.92 |

## MESA Calibration Table

All MESA runs across all stages, sorted by TP.

| workload | k | f | dfo | exit | pol | TP | draft_ms | verify_ms | accept | CH | T/S |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| NS=128/OL=256 | 6 | 3 | 2 | 53 | a | 72.81 | 48.37 | 45.85 | 0.48 | 0.81 | 3.88 |
| NS=200/OL=256 | 6 | 3 | 2 | 53 | a | 72.25 | 48.87 | 46.83 | 0.48 | 0.83 | 3.91 |
| NS=128/OL=256 | 5 | 3 | 2 | 53 | a | 71.99 | 44.48 | 44.21 | 0.53 | 0.82 | 3.65 |
| NS=128/OL=256 | 5 | 4 | 2 | 53 | a | 71.06 | 45.97 | 43.92 | 0.52 | 0.84 | 3.62 |
| NS=128/OL=256 | 4 | 4 | 2 | 53 | a | 69.28 | 41.62 | 42.59 | 0.58 | 0.86 | 3.33 |
| NS=128/OL=256 | 4 | 5 | 2 | 53 | a | 67.84 | 42.81 | 42.94 | 0.57 | 0.87 | 3.29 |
| NS=128/OL=256 | 4 | 3 | 2 | 53 | a | 67.74 | 40.71 | 42.98 | 0.58 | 0.83 | 3.30 |
| NS=128/OL=256 | 6 | 4 | 2 | 53 | a | 67.58 | 52.22 | 45.82 | 0.48 | 0.84 | 3.87 |
| NS=128/OL=256 | 5 | 5 | 2 | 53 | a | 64.33 | 50.51 | 44.30 | 0.53 | 0.86 | 3.63 |
| NS=64/OL=128 | 5 | 4 | 2 | 53 | a | 64.19 | 45.36 | 43.81 | 0.48 | 0.82 | 3.39 |
| NS=128/OL=256 | 6 | 5 | 2 | 53 | a | 63.03 | 56.04 | 45.83 | 0.49 | 0.86 | 3.93 |
| NS=64/OL=128 | 5 | 4 | 2 | 53 | b | 62.08 | 46.98 | 45.97 | 0.48 | 0.83 | 3.41 |
| NS=64/OL=128 | 5 | 4 | 2 | 40 | b | 61.19 | 42.86 | 48.22 | 0.51 | 0.77 | 3.55 |
| NS=64/OL=128 | 5 | 4 | 2 | 46 | a | 61.04 | 44.03 | 46.22 | 0.48 | 0.80 | 3.38 |
| NS=64/OL=128 | 5 | 8 | 4 | 46 | a | 60.99 | 48.32 | 45.75 | 0.49 | 0.86 | 3.47 |
| NS=64/OL=128 | 5 | 8 | 4 | 46 | a | 60.99 | 47.44 | 45.06 | 0.48 | 0.86 | 3.41 |
| NS=64/OL=128 | 5 | 4 | 2 | 46 | b | 60.91 | 45.69 | 48.02 | 0.50 | 0.80 | 3.49 |
| NS=64/OL=128 | 5 | 4 | 2 | 46 | a | 60.20 | 45.60 | 46.47 | 0.48 | 0.80 | 3.38 |
| NS=64/OL=128 | 5 | 6 | 3 | 46 | a | 60.12 | 47.43 | 45.03 | 0.47 | 0.83 | 3.35 |
| NS=64/OL=128 | 5 | 4 | 2 | 46 | b | 60.02 | 45.09 | 47.88 | 0.48 | 0.81 | 3.42 |
| NS=64/OL=128 | 5 | 4 | 1 | 46 | a | 59.66 | 48.82 | 46.49 | 0.48 | 0.78 | 3.41 |
| NS=64/OL=128 | 5 | 8 | 4 | 40 | a | 59.65 | 47.77 | 47.44 | 0.48 | 0.83 | 3.39 |
| NS=64/OL=128 | 5 | 4 | 2 | 40 | a | 59.42 | 41.73 | 46.70 | 0.47 | 0.76 | 3.34 |
| NS=64/OL=128 | 5 | 8 | 4 | 46 | b | 59.28 | 49.95 | 47.96 | 0.49 | 0.87 | 3.47 |
| NS=64/OL=128 | 5 | 6 | 2 | 46 | a | 59.09 | 48.63 | 46.32 | 0.47 | 0.82 | 3.37 |
| NS=64/OL=128 | 5 | 8 | 4 | 53 | a | 58.86 | 50.34 | 44.86 | 0.49 | 0.88 | 3.47 |
| NS=64/OL=128 | 5 | 6 | 3 | 46 | b | 58.46 | 49.69 | 47.70 | 0.48 | 0.84 | 3.39 |
| NS=64/OL=128 | 5 | 8 | 2 | 46 | a | 55.97 | 51.31 | 44.23 | 0.48 | 0.83 | 3.40 |
| NS=64/OL=128 | 3 | 6 | 2 | 46 | a | 55.96 | 35.25 | 42.38 | 0.59 | 0.85 | 2.77 |
| NS=64/OL=128 | 7 | 6 | 2 | 46 | a | 54.11 | 61.27 | 50.59 | 0.41 | 0.80 | 3.84 |
| NS=16/OL=64 | 5 | 4 | 2 | 46 | a | 53.09 | — | — | 0.44 | 0.77 | — |

## Policy A vs B Comparison (Stage 2 + Stage 3, exit=46/40/53)

| config (NS=64/OL=128) | Policy A | Policy B | ΔTP | Δ% |
|---|---:|---:|---:|---:|
| k=5 f=4 dfo=2 exit=53 | 64.19 | 62.08 | -2.11 | -3.3% |
| k=5 f=4 dfo=2 exit=40 | 59.42 | 61.19 | +1.77 | +3.0% |
| k=5 f=4 dfo=2 exit=46 | 61.04 | 60.91 | -0.13 | -0.2% |
| k=5 f=4 dfo=2 exit=46 | 61.04 | 60.02 | -1.02 | -1.7% |
| k=5 f=8 dfo=4 exit=46 | 60.99 | 59.28 | -1.71 | -2.8% |
| k=5 f=6 dfo=3 exit=46 | 60.12 | 58.46 | -1.66 | -2.8% |

Policy B wins in some configs at smaller f (f=4, dfo=2), Policy A wins at larger f (f=8). **Effect is small (~±1 tok/s). Policy A remains the more robust default.**


## Decision Gate Answers

Per the explicit decision gate at the end of MESA-PARAMETER-SWEEP-PLAN.md:

### 1. Best async-SSD parameter set

`k=7, f=8` — TP **76.78** tok/s (NS=200/OL=256)

Coarse Stage 1A had picked k=6/f=8 (TP=66.47 at NS=64). Fine Stage 4 at NS=128 surfaced k=7/f=8 (TP=79.41) which held up at NS=200 (TP=76.78). Higher k saturates draft step time (k=7 f=9 dropped 13 tok/s vs k=7 f=8). Lower k has less budget but draft is faster — sweet spot is k=7/f=8.

### 2. Best MESA parameter set

`k=6, f=3, dfo=2, exit_layer=53, policy=a` — TP **72.25** tok/s (NS=200/OL=256)

Stage 1B coarse picked k=5/f=8/dfo=4 (TP=60.99 at NS=64). Stage 3 exit-layer sweep surfaced exit=53 as best. Stage 4 fine settled on k=6/f=3/dfo=2 (TP=72.81 at NS=128). Stage 5 confirmed at NS=200: TP=72.25. **Smaller total budget (f=3) plus deeper exit (53/80 = 66%) beats the larger budget MESA configs** — the deeper proxy gives better correction quality, and small f keeps draft step bounded.

### 3. Is draft still the dominant bottleneck?

**Mostly balanced now, but draft slightly dominates MESA**:

- Best async (k=7/f=8 NS=200): draft=44.0ms, verify=45.3ms — draft ≈ verify (~equal)

- Best MESA (k=6/f=3 NS=200): draft=48.9ms, verify=46.8ms — **draft still 2-3 ms over verify** (2-pass overhead)


In Stage 1A coarse (NS=64) draft and verify were both ~33-46 ms across configs; in fine sweep the draft path (esp for MESA) remains the slightly slower side. AWQ on the draft already removed the worst draft-bound asymmetry.

### 4. Gap large enough to justify Phase 1 / Phase 2 redesign?

**No, not at this stack.** Async beats MESA by **6.3%** (76.78 vs 72.25 tok/s). The MESA 2-pass structural cost (~2-3 ms extra draft) is small in absolute terms but larger than the cache-hit gain (MESA CH=0.83 vs async CH=0.79 — only +0.04). Token efficiency is genuinely improved (MESA accept ≈ async, plus better CH), but it doesn't compound to throughput because Phase 2 replay cost isn't free even at this scale.

A shorter-Phase-1 redesign would need to recover several ms per step *and* preserve the proxy's correction quality. At a 4.5 tok/s gap, the engineering cost of new layout/graph variants is unlikely to pay off on this stack. **Recommendation: defer redesign**, keep async (k=7, f=8) as the production default.

### 5. Is Policy B robust enough to be the default?

**No, Policy A remains the more robust default.** From Stage 2/3:

- Policy B helps at small budget (f=4, dfo=2): exit=40 +1.8, exit=46 -0.9, exit=53 -2.1

- Policy B hurts at larger budget (f=8, dfo=4 → -1.7) and at f=6/dfo=3 → -1.7

- No config showed Policy B improving by more than ~+1.8 tok/s


Policy B's joint `ĥ_i × r̂_i(v)` ranking concentrates budget on a few high-confidence (position, token) pairs. On 70B this matches the residual correction signal in some regimes but starves alternative branches — net wash. Keep Policy A as default; expose `--mesa_policy b` for users who want to experiment.


## Bottom Line

| Mode | Best Config | TP (NS=200) | vs AR | vs Async |
|---|---|---:|---:|---:|
| AR | (target only, TP=4) | 32.87 | 1.00× | 0.43× |
| Async SSD | k=7, f=8 | **76.78** | **2.34×** | **1.00×** |
| MESA | k=6, f=3, dfo=2, exit=53, policy=A | 72.25 | 2.20× | 0.94× |

**Production default**: Async SSD with k=7, f=8 on this stack.
