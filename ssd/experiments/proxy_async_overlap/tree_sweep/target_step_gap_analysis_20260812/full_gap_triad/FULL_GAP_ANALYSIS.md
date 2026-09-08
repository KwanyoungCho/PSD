# Chain vs tree full target-step decomposition

Latency-only diagnostic on the same tiny7 prompts, seed42, output 256. Warm-up spans are excluded using each raw JSONL's measured step count. AL and hit rate are not used for the causal conclusions.
The target posts its fused receive before the draft has finished the response. Therefore `target_recv_fused` is a blocking readiness wait, not pure metadata transport. The following Q receive is the closest separately measured payload-transfer span.

## Raw overall latency

| Metric | Chain | P2 tree only | P1+P2 tree |
|---|---:|---:|---:|
| Full target step | 67.833 ± 0.271 | 69.263 ± 0.632 | 72.437 ± 0.821 |
| Target verify | 61.061 ± 0.069 | 61.593 ± 0.181 | 62.411 ± 0.046 |
| Outside verify | 6.772 ± 0.203 | 7.670 ± 0.476 | 10.026 ± 0.780 |

## Status-specific critical path

### P1 hit

| Segment | Chain | P2 tree only | P1+P2 tree |
|---|---:|---:|---:|
| Full step | 64.890 ± 0.233 | 65.344 ± 0.501 | 71.379 ± 0.914 |
| Draft/spec wait | 2.476 ± 0.149 | 2.819 ± 0.310 | 4.739 ± 0.855 |
| Response→verify gap | 1.267 ± 0.020 | 1.288 ± 0.040 | 3.313 ± 0.012 |
| Target verify | 60.723 ± 0.060 | 60.809 ± 0.185 | 62.885 ± 0.059 |
| Post-verify | 0.425 ± 0.011 | 0.428 ± 0.008 | 0.441 ± 0.002 |

### P2 hit

| Segment | Chain | P2 tree only | P1+P2 tree |
|---|---:|---:|---:|
| Full step | 62.904 ± 0.156 | 68.118 ± 0.513 | 67.472 ± 0.651 |
| Draft/spec wait | 2.476 ± 0.123 | 3.417 ± 0.422 | 3.483 ± 0.571 |
| Response→verify gap | 1.271 ± 0.015 | 2.803 ± 0.053 | 2.845 ± 0.016 |
| Target verify | 58.740 ± 0.044 | 61.412 ± 0.076 | 60.693 ± 0.064 |
| Post-verify | 0.417 ± 0.008 | 0.487 ± 0.008 | 0.451 ± 0.003 |

### Miss

| Segment | Chain | P2 tree only | P1+P2 tree |
|---|---:|---:|---:|
| Full step | 71.758 ± 0.351 | 72.456 ± 0.599 | 72.285 ± 0.678 |
| Draft/spec wait | 12.207 ± 0.222 | 12.395 ± 0.467 | 12.525 ± 0.675 |
| Response→verify gap | 1.035 ± 0.017 | 1.105 ± 0.040 | 1.130 ± 0.009 |
| Target verify | 58.166 ± 0.102 | 58.575 ± 0.176 | 58.281 ± 0.040 |
| Post-verify | 0.350 ± 0.014 | 0.381 ± 0.004 | 0.349 ± 0.003 |

## Draft response decomposition

### P1 hit

| Segment | Chain | P2 tree only | P1+P2 tree |
|---|---:|---:|---:|
| Draft request receive/KV restore envelope | 0.363 ± 0.015 | 0.466 ± 0.023 | 0.760 ± 0.092 |
| Tree KV restore | — | 0.487 ± 0.072 | 0.537 ± 0.114 |
| Cache response compute | 0.904 ± 0.094 | 0.785 ± 0.172 | 2.472 ± 0.651 |
| Tree rerank total | — | — | 1.628 ± 0.428 |
| Generated-tree pack/validate | — | — | 0.445 ± 0.118 |
| Subtree selection | — | — | 0.366 ± 0.097 |
| GPU compaction | — | — | 0.332 ± 0.082 |
| Served-tree pack/validate | — | — | 0.390 ± 0.106 |
| Parent-q gather | — | — | 0.101 ± 0.027 |
| Response wire pack | 0.073 ± 0.007 | 0.090 ± 0.016 | 0.072 ± 0.021 |
| Fused metadata send | 0.111 ± 0.010 | 0.106 ± 0.022 | 0.109 ± 0.025 |
| Q/parent-q send | 0.086 ± 0.008 | 0.082 ± 0.013 | 0.073 ± 0.018 |
| Full send envelope | 0.242 ± 0.023 | 0.230 ± 0.042 | 0.238 ± 0.058 |
| Draft recv→send end | 1.643 ± 0.146 | 1.619 ± 0.259 | 3.586 ± 0.840 |

### P2 hit

| Segment | Chain | P2 tree only | P1+P2 tree |
|---|---:|---:|---:|
| Draft request receive/KV restore envelope | 0.361 ± 0.018 | 0.475 ± 0.024 | 0.686 ± 0.083 |
| Tree KV restore | — | 0.503 ± 0.061 | 0.461 ± 0.098 |
| Cache response compute | 0.918 ± 0.070 | 1.291 ± 0.283 | 1.258 ± 0.338 |
| Tree rerank total | — | 0.402 ± 0.075 | 0.376 ± 0.101 |
| Generated-tree pack/validate | — | — | — |
| Subtree selection | — | — | — |
| GPU compaction | — | — | — |
| Served-tree pack/validate | — | 0.363 ± 0.068 | 0.338 ± 0.091 |
| Parent-q gather | — | 0.126 ± 0.027 | 0.123 ± 0.035 |
| Response wire pack | 0.074 ± 0.005 | 0.075 ± 0.014 | 0.072 ± 0.021 |
| Fused metadata send | 0.112 ± 0.009 | 0.112 ± 0.025 | 0.106 ± 0.024 |
| Q/parent-q send | 0.087 ± 0.007 | 0.078 ± 0.013 | 0.074 ± 0.020 |
| Full send envelope | 0.245 ± 0.020 | 0.248 ± 0.049 | 0.235 ± 0.061 |
| Draft recv→send end | 1.659 ± 0.120 | 2.133 ± 0.374 | 2.296 ± 0.521 |

## Target receive and pre-verify decomposition

### P1 hit

| Segment | Chain | P2 tree only | P1+P2 tree |
|---|---:|---:|---:|
| Target response receive wait | 1.184 ± 0.130 | 1.310 ± 0.292 | 3.330 ± 0.856 |
| Blocking fused receive (includes draft readiness) | 1.004 ± 0.120 | 1.059 ± 0.291 | 3.008 ± 0.847 |
| Tree valid/phase scalar read | 0.000 ± 0.000 | 0.044 ± 0.002 | 0.075 ± 0.004 |
| Q/parent-q receive | 0.148 ± 0.010 | 0.126 ± 0.007 | 0.129 ± 0.006 |
| Response→verify gap | 1.267 ± 0.020 | 1.288 ± 0.040 | 3.313 ± 0.012 |
| Wire list/parse/validate | — | — | 0.679 ± 0.014 |
| Topology total prepare | — | — | 1.112 ± 0.004 |
| Topology CPU pack | — | — | 0.751 ± 0.009 |
| Topology H2D copies | — | — | 0.238 ± 0.004 |
| Parent-q select | — | — | 0.139 ± 0.001 |

### P2 hit

| Segment | Chain | P2 tree only | P1+P2 tree |
|---|---:|---:|---:|
| Target response receive wait | 1.206 ± 0.108 | 1.961 ± 0.407 | 2.077 ± 0.553 |
| Blocking fused receive (includes draft readiness) | 1.026 ± 0.098 | 1.647 ± 0.404 | 1.759 ± 0.546 |
| Tree valid/phase scalar read | 0.000 ± 0.000 | 0.073 ± 0.002 | 0.075 ± 0.003 |
| Q/parent-q receive | 0.148 ± 0.009 | 0.125 ± 0.007 | 0.127 ± 0.007 |
| Response→verify gap | 1.271 ± 0.015 | 2.803 ± 0.053 | 2.845 ± 0.016 |
| Wire list/parse/validate | — | 0.494 ± 0.011 | 0.497 ± 0.009 |
| Topology total prepare | — | 0.857 ± 0.010 | 0.860 ± 0.011 |
| Topology CPU pack | — | 0.504 ± 0.012 | 0.505 ± 0.013 |
| Topology H2D copies | — | 0.230 ± 0.002 | 0.231 ± 0.001 |
| Parent-q select | — | 0.137 ± 0.004 | 0.134 ± 0.002 |

## Target verify decomposition

### P1 hit

| Segment | Chain | P2 tree only | P1+P2 tree |
|---|---:|---:|---:|
| Verify total | 60.723 ± 0.060 | 60.809 ± 0.185 | 62.885 ± 0.059 |
| Verify setup | 0.361 ± 0.005 | 0.367 ± 0.006 | 1.107 ± 0.019 |
| Tree metadata/depth | — | — | 0.203 ± 0.004 |
| Tree mask prepare | — | — | 0.444 ± 0.007 |
| Input copy | — | — | 0.108 ± 0.002 |
| Attention buffers | — | — | 0.167 ± 0.002 |
| Target graph pre | 39.440 ± 0.086 | 39.534 ± 0.076 | 41.100 ± 0.039 |
| Exit/proxy side | 0.944 ± 0.000 | 0.969 ± 0.000 | 1.372 ± 0.004 |
| Target graph post | 15.725 ± 0.001 | 15.729 ± 0.001 | 16.309 ± 0.002 |
| Final logits | 0.247 ± 0.003 | 0.243 ± 0.003 | 0.253 ± 0.000 |
| Acceptance prep | 0.120 ± 0.007 | 0.119 ± 0.005 | 0.124 ± 0.011 |
| Sample/accept envelope | 3.861 ± 0.106 | 3.836 ± 0.099 | 3.926 ± 0.028 |
| Chain/tree accept core | 3.791 ± 0.105 | 3.768 ± 0.097 | 3.154 ± 0.026 |
| Tree KV commit | — | — | 0.974 ± 0.004 |

### P2 hit

| Segment | Chain | P2 tree only | P1+P2 tree |
|---|---:|---:|---:|
| Verify total | 58.740 ± 0.044 | 61.412 ± 0.076 | 60.693 ± 0.064 |
| Verify setup | 0.357 ± 0.005 | 1.004 ± 0.018 | 1.025 ± 0.004 |
| Tree metadata/depth | — | 0.178 ± 0.003 | 0.180 ± 0.003 |
| Tree mask prepare | — | 0.394 ± 0.009 | 0.400 ± 0.001 |
| Input copy | — | 0.105 ± 0.003 | 0.105 ± 0.001 |
| Attention buffers | — | 0.159 ± 0.002 | 0.164 ± 0.001 |
| Target graph pre | 38.078 ± 0.054 | 40.227 ± 0.066 | 40.147 ± 0.029 |
| Exit/proxy side | 0.940 ± 0.001 | 1.213 ± 0.004 | 1.320 ± 0.009 |
| Target graph post | 15.236 ± 0.001 | 15.891 ± 0.002 | 15.887 ± 0.001 |
| Final logits | 0.212 ± 0.000 | 0.247 ± 0.000 | 0.248 ± 0.000 |
| Acceptance prep | 0.117 ± 0.008 | 0.131 ± 0.010 | 0.122 ± 0.003 |
| Sample/accept envelope | 3.782 ± 0.079 | 3.842 ± 0.029 | 3.201 ± 0.034 |
| Chain/tree accept core | 3.712 ± 0.074 | 3.087 ± 0.051 | 2.434 ± 0.030 |
| Tree KV commit | — | 0.956 ± 0.036 | 1.006 ± 0.008 |

## Direct deltas against chain

| Status/comparison | Full step | Draft/spec wait | Pre-verify | Verify |
|---|---:|---:|---:|---:|
| P1: full tree − chain | 6.488 ± 1.112 | 2.263 ± 0.968 | 2.047 ± 0.025 | 2.162 ± 0.117 |
| P2: P2 tree − chain | 5.214 ± 0.669 | 0.940 ± 0.544 | 1.532 ± 0.045 | 2.672 ± 0.117 |
| P2: full tree − chain | 4.568 ± 0.789 | 1.007 ± 0.671 | 1.574 ± 0.005 | 1.953 ± 0.108 |
| Miss: full tree − chain | 0.528 ± 0.980 | 0.318 ± 0.866 | 0.095 ± 0.023 | 0.115 ± 0.135 |

Machine-readable: `overall_runs.csv`, `conditional_runs.csv`.
