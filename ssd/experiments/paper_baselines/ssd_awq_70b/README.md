# SSD AWQ 70B Paper Baselines

This directory stores paper-facing SSD baseline runs in a stable layout.

Per-run files:

- `run.log`: raw benchmark log.
- `summary_metrics.csv`: one-row table for paper/report aggregation.
- `summary_metrics.md`: same row in Markdown.
- `timeline_cache_hit.png`: one steady-state cache-hit step timeline.
- `timeline_cache_miss.png`: one steady-state cache-miss step timeline.
- `mesa_profile_*.json`: raw profiling events for regenerating plots.
- `legacy_plots/`: older exploratory plots kept out of the main view.

Aggregate table:

- `summary_index.csv`: appended by `bench/summarize_ssd_run.py`.
- `summary_dashboard.png`: regenerated from `summary_index.csv` after each
  appended run.

Recommended postprocess command:

```bash
python bench/summarize_ssd_run.py <run_dir> --k <speculate_k> \
  --append-index experiments/paper_baselines/ssd_awq_70b/summary_index.csv
```

Regenerate only the dashboard:

```bash
python bench/plot_ssd_summary.py experiments/paper_baselines/ssd_awq_70b/summary_index.csv
```
