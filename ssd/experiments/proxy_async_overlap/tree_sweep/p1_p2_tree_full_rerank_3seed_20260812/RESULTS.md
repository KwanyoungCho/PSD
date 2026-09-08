# DUET tree with fused P1 rerank: full Spec-Bench, three seeds

- Full Spec-Bench: 480 questions / 560 turns per seed
- Seeds: 1, 42, 123; output 1,024; profiler off
- MT-Bench two turns are merged before prompt-level averaging
- Tree policy is unchanged: K1/K2=8/4, N1/M1=14/12, N2/M2=8/8
- `SSD_P1_RERANK_PRECOMPUTE=1` and GPU target topology enabled

## Completion

- seed 1: 560/560
- seed 42: 560/560
- seed 123: 560/560

## Overall (prompt-level)

| Seed | Questions (turns) | Decode TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step ms | Target verify ms | Outside verify ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 480 (560) | 64.541 | 4.449 | 0.776 | 0.610 | 4.348 | 0.167 | 2.201 | 69.852 | 63.234 | 6.619 |
| 42 | 480 (560) | 65.136 | 4.490 | 0.790 | 0.616 | 4.367 | 0.174 | 2.204 | 69.803 | 63.251 | 6.553 |
| 123 | 480 (560) | 66.582 | 4.547 | 0.787 | 0.621 | 4.412 | 0.166 | 2.225 | 69.287 | 62.958 | 6.330 |
| Mean ± SD | 480 (560) | 65.420 ± 1.050 | 4.496 ± 0.049 | 0.785 ± 0.007 | 0.616 ± 0.006 | 4.376 ± 0.033 | 0.169 ± 0.004 | 2.210 ± 0.013 | 69.648 ± 0.313 | 63.147 ± 0.164 | 6.500 ± 0.151 |

## Decode TPS by subtask

| Seed | mt_bench | translation | summarization | qa | math_reasoning | rag | Overall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 63.295 | 68.431 | 54.583 | 67.872 | 64.982 | 68.082 | 64.541 |
| 42 | 64.572 | 67.642 | 55.354 | 72.321 | 63.165 | 67.762 | 65.136 |
| 123 | 65.592 | 73.137 | 54.382 | 71.307 | 67.241 | 67.834 | 66.582 |
| Mean ± SD | 64.486 ± 1.151 | 69.737 ± 2.971 | 54.773 ± 0.513 | 70.500 ± 2.332 | 65.129 ± 2.042 | 67.893 ± 0.168 | 65.420 ± 1.050 |

## Accepted length by subtask

| Seed | mt_bench | translation | summarization | qa | math_reasoning | rag | Overall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.343 | 4.725 | 3.768 | 4.667 | 4.462 | 4.731 | 4.449 |
| 42 | 4.446 | 4.643 | 3.836 | 4.978 | 4.327 | 4.712 | 4.490 |
| 123 | 4.487 | 4.973 | 3.726 | 4.862 | 4.562 | 4.674 | 4.547 |
| Mean ± SD | 4.425 ± 0.074 | 4.780 ± 0.172 | 3.777 ± 0.056 | 4.836 ± 0.158 | 4.450 ± 0.118 | 4.706 ± 0.029 | 4.496 ± 0.049 |

## Cache hit rate by subtask

| Seed | mt_bench | translation | summarization | qa | math_reasoning | rag | Overall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.820 | 0.796 | 0.644 | 0.817 | 0.828 | 0.753 | 0.776 |
| 42 | 0.827 | 0.781 | 0.672 | 0.850 | 0.842 | 0.771 | 0.790 |
| 123 | 0.825 | 0.805 | 0.646 | 0.838 | 0.860 | 0.748 | 0.787 |
| Mean ± SD | 0.824 ± 0.003 | 0.794 ± 0.012 | 0.654 ± 0.016 | 0.835 ± 0.017 | 0.844 ± 0.016 | 0.757 ± 0.012 | 0.785 ± 0.007 |

## P1 conditional AL by subtask

| Seed | mt_bench | translation | summarization | qa | math_reasoning | rag | Overall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.102 | 4.528 | 4.049 | 4.329 | 4.273 | 4.790 | 4.348 |
| 42 | 4.199 | 4.475 | 4.235 | 4.630 | 4.073 | 4.573 | 4.367 |
| 123 | 4.224 | 4.777 | 3.970 | 4.519 | 4.266 | 4.637 | 4.412 |
| Mean ± SD | 4.175 ± 0.065 | 4.593 ± 0.161 | 4.085 ± 0.136 | 4.492 ± 0.152 | 4.204 ± 0.114 | 4.666 ± 0.111 | 4.376 ± 0.033 |

## P2 conditional AL by subtask

| Seed | mt_bench | translation | summarization | qa | math_reasoning | rag | Overall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.164 | 2.148 | 1.934 | 2.344 | 2.187 | 2.379 | 2.201 |
| 42 | 2.148 | 2.173 | 1.923 | 2.436 | 2.075 | 2.410 | 2.204 |
| 123 | 2.142 | 2.174 | 2.067 | 2.347 | 2.204 | 2.394 | 2.225 |
| Mean ± SD | 2.151 ± 0.011 | 2.165 ± 0.015 | 1.974 ± 0.080 | 2.375 ± 0.052 | 2.155 ± 0.071 | 2.394 ± 0.016 | 2.210 ± 0.013 |

## Integrity

- Complete validated seeds: 3/3
- Each completed seed must match all 560 dataset UIDs in exact order
- Each row is checked for the fixed tree configuration and metrics
- TPS is decode-only and aggregated at prompt/question level

Machine-readable table: `/home/eslab/chokwans99/PSD/ssd/experiments/proxy_async_overlap/tree_sweep/p1_p2_tree_full_rerank_3seed_20260812/overall.csv`
