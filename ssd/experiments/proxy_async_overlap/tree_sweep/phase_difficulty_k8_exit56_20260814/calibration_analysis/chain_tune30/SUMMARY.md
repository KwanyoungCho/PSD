# K1=K2=8 phase-difficulty summary

Primary AL includes the one correction/recovery token. Confidence 
intervals resample questions, so long generations do not become 
independent pseudo-replicates.

| Arm | Source | Events | Questions | Event-pooled AL | Question-mean AL [95% CI] | P(AL=1) | P(AL≥3) | P(AL≥5) | Mean valid-k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | Miss (fresh JIT) | 648 | 30 | 3.150 | 3.652 [3.100, 4.295] | 0.343 | 0.452 | 0.235 | 8.00 |
| chain | P1 | 1993 | 30 | 4.611 | 4.891 [4.373, 5.443] | 0.224 | 0.622 | 0.432 | 8.00 |
| chain | P2 | 652 | 30 | 3.538 | 4.314 [3.701, 5.000] | 0.287 | 0.500 | 0.285 | 8.00 |

## Within-question phase contrasts

Negative `P2 - P1` means P2 has lower accepted length. The CI and 
sign fraction use questions as the independent unit.

| Arm | Contrast | Questions | Mean difference [95% CI] | Fraction below zero |
|---|---|---:|---:|---:|
| chain | P2 - P1 | 30 | -0.576 [-1.055, -0.122] | 0.733 |
| chain | Miss (fresh JIT) - P1 | 30 | -1.239 [-1.878, -0.666] | 0.767 |
| chain | P2 - Miss (fresh JIT) | 30 | 0.662 [0.089, 1.339] | 0.333 |
