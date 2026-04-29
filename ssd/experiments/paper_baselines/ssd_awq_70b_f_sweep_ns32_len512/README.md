# SSD AWQ 70B Fan-Out Sweep

Purpose: find the largest useful uniform fan-out before target-side idle is mostly removed.

Fixed settings:

- Target: `layerskip_llama2_70b` AWQ, TP=4.
- Draft: `tinyllama_1b` AWQ, TP=1.
- GPUs: `0,1,2,3,4`.
- Decode: `--async --spec --k 7`.
- Sampling: `--temp 0.7 --seed 42`.
- Dataset: `--all --numseqs 32`, so 128 prompts total.
- Generation length: `--output_len 512`.
- Profiling: `SSD_PROFILE_MESA=1`.

Current runs:

| config | status | run dir | key files |
|---|---|---|---|
| `k7_f6` | completed | `20260429_k7_f6_temp07_seed42_ns32_all/` | `summary_metrics.csv`, `timeline_cache_hit.png`, `timeline_cache_miss.png`, `run.log` |
| `k7_f7` | running | `20260429_k7_f7_temp07_seed42_ns32_all/` | `run.log` |

Aggregate outputs:

- `summary_index.csv`: completed runs only.
- `summary_dashboard.png`: generated from `summary_index.csv`.
- `runs_manifest.csv`: completed and running run list.

Useful commands:

```bash
tail -f experiments/paper_baselines/ssd_awq_70b_f_sweep_ns32_len512/20260429_k7_f7_temp07_seed42_ns32_all/run.log
python bench/plot_ssd_summary.py experiments/paper_baselines/ssd_awq_70b_f_sweep_ns32_len512/summary_index.csv
```

Timeline convention:

- `timeline_cache_hit.png` / `timeline_cache_miss.png`: selected hit/miss handshake is drawn at the tail of the window.
- `timeline_cache_*_full_handshake.png`: older view with the selected handshake at the front.
- `timeline_cache_*_post_response.png`: post-response critical path only.

Run command template:

```bash
bash experiments/paper_baselines/ssd_awq_70b_f_sweep_ns32/run_one.sh <f>
```
