# Full-distribution replay: shared-analysis verification

Source under review: https://chatgpt.com/s/t_6aa29a78c4488191a8443e4d147cc79e

Protocol fixed before evaluation: 4 datasets (alpaca, c4, gsm, humaneval),
seeds 42 and 123, first 32 prompts each, maximum 256 generated tokens,
T=1, B=1, only-proxy chain, exit layer 56, configured K1=8/K2=4,
P2 budget 15. Calibrated AWQ target and TinyLlama paths match the preceding
entropy experiment. All observed verification chains must have actual K=4.

Capture full float32 p_T, p_D, p_E56, p_E79 and actual sampled draft tokens
at per-prompt verification steps 0,8,16,... . NPZ files are uncompressed and
ignored by git; no top-k or precision truncation. Metadata includes every
prompt's total step count so the sampling schedule can be audited.

Replay fixed distributions, without modifying verification or generation.
Train: prompt ids 1..16; held-out: 17..32 in each dataset, same split for both
seeds. Fit global temperatures and optional subtraction strengths using
training prompts only. Dataset/prompt clusters, not individual token positions,
are the independent units for bootstrap intervals. These are conditional
candidate-coverage experiments, not throughput or full P1/P2 validation.

Primary candidate rule reproduces production top-M=17 renormalization,
sampled-token exclusion, global top-15 allocation. Also evaluate M=64,
full-vocabulary normalization, exact hazards, oracle residuals, equal three
candidates per position, and an idealized fixed DS reference set (three q
alternatives at each of four rejection positions; no bonus candidates).

Calibration grid: tau=[.65,.8,.9,1,1.1,1.25,1.5],
lambda=[0,.25,.5,.75,1]. Tau changes the proxy used for prediction only;
the target sampler remains T=1. Include fixed-h and calibrated-h controls.

An additional oracle control replaces the real proxy by a temperature-scaled
true target with the same per-context entropy. It preserves target rankings;
it is a diagnostic construction, not a usable method or a causal decomposition.

During the review, after initial partial runs, an additional oracle control was
added: retain the exact proxy probability multiset and reassign it to the true
target token ranking. This preserves entropy and the complete probability shape.
It is a post-hoc mechanism control and is not used to tune any deployable policy.

After inspecting the coarse grid, the same global KL objective is additionally
optimized as a continuous convex function of beta=1/tau, using training prompts
only. This checks grid discretization; the original grid and its selections are
retained separately. Held-out values do not enter the Newton optimization.

```bash
source ssd/env.sh
ssd/.venv/bin/python ssd/experiments/proxy_source_ablation/probe_replay_20260910/run.py
CUDA_VISIBLE_DEVICES=7 OPENBLAS_NUM_THREADS=1 ssd/.venv/bin/python results/residial_dist/shared_review/replay.py
CUDA_VISIBLE_DEVICES=7 OPENBLAS_NUM_THREADS=1 ssd/.venv/bin/python results/residial_dist/shared_review/calibrate_exact.py
OPENBLAS_NUM_THREADS=1 ssd/.venv/bin/python results/residial_dist/shared_review/checks.py
OPENBLAS_NUM_THREADS=1 ssd/.venv/bin/python results/residial_dist/shared_review/aggregate.py
ssd/.venv/bin/python results/residial_dist/shared_review/make_report.py
ssd/.venv/bin/python results/residial_dist/shared_review/audit.py
```

The smoke run uses two built-in example prompts (`--example`) for 48 tokens;
it is excluded from the campaign and all research tables.
