# K1=K2=8 phase-difficulty summary

Primary AL includes the one correction/recovery token. Confidence 
intervals resample questions, so long generations do not become 
independent pseudo-replicates.

| Arm | Source | Events | Questions | Event-pooled AL | Question-mean AL [95% CI] | P(AL=1) | P(AL≥3) | P(AL≥5) | Mean valid-k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | Miss (fresh JIT) | 1894 | 120 | 3.311 | 3.450 [3.199, 3.709] | 0.328 | 0.484 | 0.251 | 8.00 |
| chain | P1 | 7166 | 116 | 4.696 | 4.740 [4.518, 4.980] | 0.219 | 0.633 | 0.444 | 8.00 |
| chain | P2 | 2186 | 110 | 3.611 | 3.954 [3.701, 4.222] | 0.290 | 0.514 | 0.296 | 8.00 |

## Within-question phase contrasts

Negative `P2 - P1` means P2 has lower accepted length. The CI and 
sign fraction use questions as the independent unit.

| Arm | Contrast | Questions | Mean difference [95% CI] | Fraction below zero |
|---|---|---:|---:|---:|
| chain | P2 - P1 | 110 | -0.729 [-0.965, -0.493] | 0.727 |
| chain | Miss (fresh JIT) - P1 | 116 | -1.206 [-1.490, -0.924] | 0.819 |
| chain | P2 - Miss (fresh JIT) | 110 | 0.482 [0.208, 0.775] | 0.336 |
