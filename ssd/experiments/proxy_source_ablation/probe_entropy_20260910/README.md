# Full-vocabulary proxy/target entropy measurement

Measures the distribution of `H(pE) - H(pT)` at every verification context
and all 80 proxy layers. Same model and only-proxy chain configuration as
`probe_ds_seed_20260909`; this probe records scalar observations instead
of running the candidate-policy factorial. It consumes no random numbers.

```bash
cd /home/chokwans99/PSD
source ssd/env.sh
ssd/.venv/bin/python ssd/experiments/proxy_source_ablation/probe_entropy_20260910/run.py
OPENBLAS_NUM_THREADS=1 ssd/.venv/bin/python results/residial_dist/entropy/aggregate.py
```

Defaults: GPUs 2–6, four datasets, seeds 42/123/7, 32 prompts per dataset,
256 generated tokens, temperature 1, B=1, calibrated 70B AWQ + TinyLlama.
Check GPU availability before running. `run.py --help` exposes smaller
smoke runs and alternate output paths. Completed runs are skipped;
incomplete outputs require a fresh output directory.

`out/<dataset>_seed<seed>.json` is the manifest and the matching directory
contains chunked per-context NPZ observations. `*.complete.json` appears
only after generation, tap, finite-value, and prompt/step checks succeed.
The last partial chunk is flushed before engine teardown.

Raw arrays include full-vocabulary entropy, maximum token probability,
proxy probability at target's top token, top-token agreement, TVD,
sequence/step/position, bonus-row flag, true recovery hazard, prefix reach
probability, and target–draft overlap. No full logits or token text are
stored in the NPZ files. Entropy uses natural logs (nats), before any
candidate exclusion or truncation. The manifest records layer indices,
temperature, counters, chunk paths, and probability-normalization error.

The report is `results/residial_dist/entropy/REPORT.md`. Row weighting,
rejection weighting, root-only controls, prompt-level uncertainty, and
the final-layer numerical floor are reported separately. These are
quality measurements; throughput with probes enabled is not a benchmark.
