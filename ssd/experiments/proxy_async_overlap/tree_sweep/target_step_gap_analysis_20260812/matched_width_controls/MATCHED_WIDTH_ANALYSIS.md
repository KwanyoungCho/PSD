# Matched-width tree latency controls

These latency-only controls keep the number of target verification rows equal between chain and tree, then increase only the tree node count. They are not AL or hit-rate evaluations.

- P1: chain K1=8 (9 rows), tree M1=8 (9 rows), tree M1=12 (13 rows).
- P2: chain K2=4 (5 rows), tree M2=4 (5 rows), tree M2=8 (9 rows).

## Critical path

### P1 hit

| Segment | Chain, 8 nodes / 9 rows | Tree, 8 nodes / 9 rows | Tree, 12 nodes / 13 rows |
|---|---:|---:|---:|
| Full step | 64.890 ± 0.233 | 67.363 ± 0.706 | 70.110 ± 0.047 |
| Draft/spec wait | 2.476 ± 0.149 | 3.417 ± 0.681 | 2.993 ± 0.089 |
| Pre-verify | 1.267 ± 0.020 | 2.921 ± 0.029 | 3.329 ± 0.042 |
| Target verify | 60.723 ± 0.060 | 60.585 ± 0.049 | 63.326 ± 0.000 |

### P2 hit

| Segment | Chain, 4 nodes / 5 rows | Tree, 4 nodes / 5 rows | Tree, 8 nodes / 9 rows |
|---|---:|---:|---:|
| Full step | 62.904 ± 0.156 | 63.744 ± 0.103 | 68.118 ± 0.513 |
| Draft/spec wait | 2.476 ± 0.123 | 3.057 ± 0.115 | 3.417 ± 0.422 |
| Pre-verify | 1.271 ± 0.015 | 2.438 ± 0.032 | 2.803 ± 0.053 |
| Target verify | 58.740 ± 0.044 | 57.738 ± 0.003 | 61.412 ± 0.076 |

## Pre-verify fixed machinery

### P1 hit

| Segment | Chain, 8 nodes / 9 rows | Tree, 8 nodes / 9 rows | Tree, 12 nodes / 13 rows |
|---|---:|---:|---:|
| Wire parse/validate | — | 0.537 ± 0.004 | 0.678 ± 0.007 |
| Topology CPU pack | — | 0.540 ± 0.005 | 0.752 ± 0.008 |
| Topology H2D | — | 0.232 ± 0.000 | 0.241 ± 0.004 |
| Parent-q select | — | 0.139 ± 0.001 | 0.141 ± 0.001 |

### P2 hit

| Segment | Chain, 4 nodes / 5 rows | Tree, 4 nodes / 5 rows | Tree, 8 nodes / 9 rows |
|---|---:|---:|---:|
| Wire parse/validate | — | 0.362 ± 0.008 | 0.494 ± 0.011 |
| Topology CPU pack | — | 0.325 ± 0.006 | 0.504 ± 0.012 |
| Topology H2D | — | 0.226 ± 0.007 | 0.230 ± 0.002 |
| Parent-q select | — | 0.130 ± 0.004 | 0.137 ± 0.004 |

## Verify internals

### P1 hit

| Segment | Chain, 8 nodes / 9 rows | Tree, 8 nodes / 9 rows | Tree, 12 nodes / 13 rows |
|---|---:|---:|---:|
| Verify setup | 0.361 ± 0.005 | 1.026 ± 0.014 | 1.122 ± 0.020 |
| Graph pre | 39.440 ± 0.086 | 40.038 ± 0.001 | 41.222 ± 0.022 |
| Exit/proxy side | 0.944 ± 0.000 | 1.280 ± 0.001 | 1.375 ± 0.008 |
| Graph post | 15.725 ± 0.001 | 15.910 ± 0.001 | 16.351 ± 0.000 |
| Acceptance envelope | 3.861 ± 0.106 | 3.163 ± 0.051 | 4.184 ± 0.001 |

### P2 hit

| Segment | Chain, 4 nodes / 5 rows | Tree, 4 nodes / 5 rows | Tree, 8 nodes / 9 rows |
|---|---:|---:|---:|
| Verify setup | 0.357 ± 0.005 | 0.975 ± 0.013 | 1.004 ± 0.018 |
| Graph pre | 38.078 ± 0.054 | 38.451 ± 0.080 | 40.227 ± 0.066 |
| Exit/proxy side | 0.940 ± 0.001 | 1.230 ± 0.007 | 1.213 ± 0.004 |
| Graph post | 15.236 ± 0.001 | 15.279 ± 0.002 | 15.891 ± 0.002 |
| Acceptance envelope | 3.782 ± 0.079 | 2.620 ± 0.105 | 3.842 ± 0.029 |

## Causal split

| Comparison | Full step | Draft wait | Pre-verify | Verify |
|---|---:|---:|---:|---:|
| P1 fixed tree machinery: tree8 − chain8 | +2.472 ms | +0.941 ms | +1.655 ms | -0.137 ms |
| P1 four extra nodes: tree12 − tree8 | +2.747 ms | -0.424 ms | +0.408 ms | +2.740 ms |
| P2 fixed tree machinery: tree4 − chain4 | +0.839 ms | +0.581 ms | +1.167 ms | -1.002 ms |
| P2 four extra nodes: tree8 − tree4 | +4.374 ms | +0.359 ms | +0.365 ms | +3.673 ms |

`tree8 − chain8` / `tree4 − chain4` estimates fixed tree protocol, metadata, custom-attention setup, and tree-walk cost at equal target row count. `tree12 − tree8` / `tree8 − tree4` estimates the marginal cost of four additional verification nodes, but generation paths and accepted outputs can still change; use status-conditional spans, not overall TPS, for this diagnostic.

Machine-readable: `conditional_runs.csv`.
