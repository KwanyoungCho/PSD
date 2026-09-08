# K1=K2=8 phase-difficulty summary

Primary AL includes the one correction/recovery token. Confidence 
intervals resample questions, so long generations do not become 
independent pseudo-replicates.

| Arm | Source | Events | Questions | Event-pooled AL | Question-mean AL [95% CI] | P(AL=1) | P(AL≥3) | P(AL≥5) | Mean valid-k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | Miss (fresh JIT) | 648 | 30 | 3.150 | 3.652 [3.100, 4.295] | 0.343 | 0.452 | 0.235 | 8.00 |
| chain | P1 | 1993 | 30 | 4.611 | 4.891 [4.373, 5.443] | 0.224 | 0.622 | 0.432 | 8.00 |
| chain | P2 | 652 | 30 | 3.538 | 4.314 [3.701, 5.000] | 0.287 | 0.500 | 0.285 | 8.00 |
| tree | Miss (fresh JIT) | 588 | 30 | 3.289 | 3.705 [3.163, 4.295] | 0.340 | 0.461 | 0.259 | 8.00 |
| tree | P1 | 1747 | 29 | 5.216 | 5.451 [4.927, 5.986] | 0.133 | 0.711 | 0.516 | 12.00 |
| tree | P2 | 570 | 27 | 3.763 | 4.060 [3.688, 4.548] | 0.225 | 0.630 | 0.298 | 11.71 |

## Within-question phase contrasts

Negative `P2 - P1` means P2 has lower accepted length. The CI and 
sign fraction use questions as the independent unit.

| Arm | Contrast | Questions | Mean difference [95% CI] | Fraction below zero |
|---|---|---:|---:|---:|
| chain | P2 - P1 | 30 | -0.576 [-1.045, -0.130] | 0.733 |
| chain | Miss (fresh JIT) - P1 | 30 | -1.239 [-1.881, -0.637] | 0.767 |
| chain | P2 - Miss (fresh JIT) | 30 | 0.662 [0.087, 1.315] | 0.333 |
| tree | P2 - P1 | 27 | -1.241 [-1.706, -0.814] | 0.815 |
| tree | Miss (fresh JIT) - P1 | 29 | -1.653 [-2.276, -1.035] | 0.897 |
| tree | P2 - Miss (fresh JIT) | 27 | 0.165 [-0.464, 0.760] | 0.444 |
