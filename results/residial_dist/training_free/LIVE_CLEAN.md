# Isolated runtime comparison

4096 actually emitted tokens / generate wall time; includes prefill, excludes model loading.

| Seed | Arm | Output TPS | Engine decode TPS | Cache hit (%) | Output tokens |
|---|---|---:|---:|---:|---:|
| 903 | matched_proxy | 80.174 | 83.899 | 80.636 | 4096 |
| 904 | matched_proxy | 89.133 | 93.484 | 83.160 | 4096 |
| 903 | optimized_proxy | 83.957 | 87.830 | 82.810 | 4096 |
| 904 | optimized_proxy | 82.169 | 85.890 | 81.978 | 4096 |

Pooled output-throughput change: -1.615%.

Two seeds, same 16 prompts, reverse arm order; no concurrent GPU2 analysis. Limited runtime screen, not a large-sample speedup claim or full Mirror-SD benchmark.
