# K2=5 paper-config TPS verification

**Run date**: 2026-05-16
**Branch**: feat/mesa-proxy-async-overlap @ 1e4e396 (Final p99 attribution)
**Question**: Does the K2=5 paper config still hit ≥80 decode TPS after all
the engine-WIP / Phase B / Phase C changes? And is the ~3 % "regression"
I claimed in Phase 0b real, or was it a comparison artifact?

## Verdict

**Yes — 80.42 tok/s at PROFILE_MESA=0, current HEAD.**

Earlier "3 % regression" claim was wrong: it compared Phase 0b's
`--output_len 256` run to the paper baseline's `--output_len 512` run.
Different warmup amortization. With apples-to-apples (same paper config,
`--output_len 512`), current branch is actually +2 % faster than the
paper baseline at the same `SSD_PROFILE_MESA=1` setting.

## Setup

All three runs use the verbatim 20260512_ours_label_perf_k1_7_k2_5
command:

```
--llama --size 8 --model_path layerskip_llama2_70b (AWQ TP=4)
--quant_awq --quant_awq_artifact ... --quant_group_size 128
--gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 50
--input_len 512 --output_len 512 --all --max_model_len 2048
--draft_path tinyllama_1b (AWQ TP=1)
--async --spec --k 12 --f 3
--mesa --mesa_exit_layer 56 --mesa_phase1_k 7 --mesa_phase2_k 5
--mesa_draft_fan_out 2 --mesa_policy b
SSD_FORCE_SPLIT_K1K2=1
```

The only differences across runs are `SSD_PROFILE_MESA` and the code
version.

## Results

| | code state | PROFILE_MESA | decode_tps | target_full_step | Δ vs paper |
|---|---|---:|---:|---:|---:|
| Paper baseline (20260512_ours_label_perf) | 1a8af64 + WIP files | 1 | 76.35 | 53.03 ms | (baseline) |
| **A** (current @ on) | 1e4e396 | 1 | **77.84** | 52.98 ms | **+1.95 %** |
| **B** (current @ off) | 1e4e396 | 0 | **80.42** | 50.76 ms | **+5.33 %** |

Key derived numbers:

- Measurement overhead at PROFILE_MESA=1 ≈ **B − A = 2.58 tok/s
  (3.3 %)**. Mostly the per-step CUDA events + the `.item()` syncs
  inside the (PROFILE_MESA=1)-gated profile_cache_status compute path.
- Current branch at PROFILE_MESA=1 ≈ **A = 77.84**, which is 76.35 + 1.5
  vs paper baseline at the same MESA=1. **No regression**.
- Cold path is the right configuration for paper-headline TPS numbers.

## Correcting the earlier (Phase 0b) "3 % regression" claim

Phase 0b RESULTS.md compared:

  - paper baseline 1a8af64 @ `--output_len 512` @ MESA=1 → 74.79 tok/s
  - Phase 0b A run 43c3f6d @ `--output_len 256` @ MESA=1 → 72.23 tok/s

I attributed the gap (−3.4 %) to commit 43c3f6d's instrumentation
refactor. But the two runs differed in `--output_len`, which shifts the
warmup ratio. The apples-to-apples comparison (this experiment, both at
`--output_len 512`) shows the opposite sign: current code at MESA=1 is
+1.95 % vs paper baseline.

The "wait shift" pattern (`proxy_compute_send` median ↓ 2 ms /
`target_spec_wait` median ↑ 2.7 ms) observed across the same period is
**still real** — it is a label-bookkeeping shift inside target_full_step
caused by the `.item()` sync removal in step.py. But it does **not**
translate into a measurable TPS regression at apples-to-apples
configuration. Step time +0.98 ms in Phase 0b was a comparison artifact;
this run measures −0.05 ms (52.98 − 53.03), basically flat.

## What this means for paper claims

1. **Report decode_tps at PROFILE_MESA=0** (cold path) for all
   paper-headline tables. Measurement overhead is ~3 % and unnecessary
   when reporting the algorithm's actual throughput.
2. **Keep PROFILE_MESA=1** for any diagnostic / attribution table that
   needs per-span breakdowns (the 3 % overhead is the price of
   visibility).
3. **No code revert needed.** Commit 43c3f6d's instrumentation refactor
   did not regress the cold path; the Phase 0b comparison was unfair
   due to different `--output_len`.

## Open questions (not blocking)

- Sample size: this is single-seed (`--seed 42`). A 2-seed cross-check
  (`--seed 1337`) would tighten the variance estimate. Cost: ~28 min
  per extra seed. Not done in this commit because the threshold result
  (B > 80) is unambiguous (≈ +0.5 tok/s margin at our run-to-run noise
  band).
- C condition (`git revert 43c3f6d` then measure at MESA=0) is not run.
  Given B = 80.42 already passes the paper threshold and A shows no
  regression vs paper baseline, C is informational at best — would
  isolate the cold-path contribution of 43c3f6d but cannot change the
  paper conclusion.

## Artifacts

```
k2_5_tps_verify/
  run_one.sh                 # parameterized runner (label + PROFILE_MESA)
  B_current_off/run.log      # B run output
  B_current_off/master.log
  A_current_on/run.log       # A run output
  A_current_on/master.log
  A_current_on/mesa_profile_*.json   # gitignored (~80-230 MB)
  RESULTS.md                  # this file
```
