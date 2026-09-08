# K1=K2=8 phase-difficulty summary

Primary AL includes the one correction/recovery token. Confidence 
intervals resample questions, so long generations do not become 
independent pseudo-replicates.

| Arm | Source | Events | Questions | Event-pooled AL | Question-mean AL [95% CI] | P(AL=1) | P(AL≥3) | P(AL≥5) | Mean valid-k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | Miss (fresh JIT) | 648 | 30 | 3.150 | 3.652 [3.100, 4.295] | 0.343 | 0.452 | 0.235 | 8.00 |
| chain | P1 | 1993 | 30 | 4.611 | 4.891 [4.373, 5.443] | 0.224 | 0.622 | 0.432 | 8.00 |
| chain | P2 | 652 | 30 | 3.538 | 4.314 [3.701, 5.000] | 0.287 | 0.500 | 0.285 | 8.00 |
| tree | Miss (fresh JIT) | 466 | 30 | 3.584 | 3.288 [2.835, 3.721] | 0.352 | 0.519 | 0.305 | 8.00 |
| tree | P1 | 1598 | 29 | 5.533 | 5.563 [5.083, 6.042] | 0.093 | 0.756 | 0.549 | 12.00 |
| tree | P2 | 471 | 29 | 3.798 | 4.411 [3.757, 5.127] | 0.180 | 0.665 | 0.274 | 11.94 |

## Within-question phase contrasts

Negative `P2 - P1` means P2 has lower accepted length. The CI and 
sign fraction use questions as the independent unit.

| Arm | Contrast | Questions | Mean difference [95% CI] | Fraction below zero |
|---|---|---:|---:|---:|
| chain | P2 - P1 | 30 | -0.576 [-1.040, -0.132] | 0.733 |
| chain | Miss (fresh JIT) - P1 | 30 | -1.239 [-1.868, -0.654] | 0.767 |
| chain | P2 - Miss (fresh JIT) | 30 | 0.662 [0.094, 1.320] | 0.333 |
| tree | P2 - P1 | 29 | -1.151 [-1.777, -0.515] | 0.793 |
| tree | Miss (fresh JIT) - P1 | 29 | -2.196 [-2.814, -1.613] | 0.931 |
| tree | P2 - Miss (fresh JIT) | 29 | 1.044 [0.222, 1.958] | 0.345 |
