# Seed 123 confidence 0.01 vs 0.02

> Profiler-off, fixed paired questions. TPS and AL are question-level; latencies are verification-step weighted.

| Arm | Q (turns) | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multiseed_s123_n10m10 | 60 (70) | 58.237 | 3.950 | 0.736 | 0.559 | 4.541 | 0.177 | 3.419 | 69.667 | 61.575 |
| threshold_s123_c002 | 60 (70) | 60.693 | 4.118 | 0.710 | 0.555 | 4.859 | 0.155 | 3.254 | 70.005 | 61.541 |

## Paired change from reference

| Arm | mean ΔTPS | TPS wins | mean ΔAL | AL wins |
|---|---:|---:|---:|---:|
| threshold_s123_c002 | 2.456 | 29/60 | 0.169 | 29/60 |

## Per-subtask

### multiseed_s123_n10m10

| Subtask | Q (turns) | TPS | AL | Hit | P1 AL | P2 AL |
|---|---:|---:|---:|---:|---:|---:|
| mt_bench | 10 (20) | 54.742 | 3.698 | 0.756 | 4.150 | 3.302 |
| translation | 10 (10) | 60.320 | 4.082 | 0.715 | 4.658 | 3.301 |
| summarization | 10 (10) | 60.710 | 4.168 | 0.674 | 5.118 | 3.741 |
| qa | 10 (10) | 59.998 | 4.030 | 0.777 | 4.664 | 3.308 |
| math_reasoning | 10 (10) | 57.924 | 3.894 | 0.808 | 4.425 | 3.269 |
| rag | 10 (10) | 55.728 | 3.824 | 0.689 | 4.286 | 3.658 |

### threshold_s123_c002

| Subtask | Q (turns) | TPS | AL | Hit | P1 AL | P2 AL |
|---|---:|---:|---:|---:|---:|---:|
| mt_bench | 10 (20) | 51.785 | 3.509 | 0.707 | 3.994 | 3.228 |
| translation | 10 (10) | 62.877 | 4.231 | 0.745 | 4.860 | 3.034 |
| summarization | 10 (10) | 54.209 | 3.724 | 0.674 | 4.471 | 3.329 |
| qa | 10 (10) | 69.051 | 4.660 | 0.788 | 5.252 | 3.658 |
| math_reasoning | 10 (10) | 61.364 | 4.138 | 0.663 | 5.466 | 3.195 |
| rag | 10 (10) | 64.870 | 4.448 | 0.681 | 5.230 | 3.050 |
