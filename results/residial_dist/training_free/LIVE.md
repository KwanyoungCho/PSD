# No-probe runtime screen

T=0.7, same 16 prompts × two seeds; each arm uses the same prompt file and output limit 256.

| Arm | Decode TPS, seed 801 | Decode TPS, seed 802 | Pooled decode TPS | Pooled cache hit |
|---|---:|---:|---:|---:|
| legacy | 83.211 | 86.533 | 84.841 | 80.861% |
| matched_proxy | 87.824 | 83.810 | 85.768 | 80.853% |
| optimized_proxy | 91.671 | 88.910 | 90.272 | 85.544% |
| selected | 85.875 | 87.546 | 86.703 | 82.130% |
| original_residual | 88.568 | 88.601 | 88.585 | 77.478% |

Exploratory pilot: some generation intervals overlapped GPU2 analysis. The original residual control was added afterward. Two seeds on the same 16 prompts; first four arms used reverse order. Use LIVE_CLEAN.md for the isolated allocation comparison. Do not bootstrap individual steps as independent throughput replications or call this a full Mirror-SD benchmark.
