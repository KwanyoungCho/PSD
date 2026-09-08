# K1=K2=8 phase-difficulty summary

Primary AL includes the one correction/recovery token. Confidence 
intervals resample questions, so long generations do not become 
independent pseudo-replicates.

| Arm | Source | Events | Questions | Event-pooled AL | Question-mean AL [95% CI] | P(AL=1) | P(AL≥3) | P(AL≥5) | Mean valid-k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | Miss (fresh JIT) | 1855 | 120 | 3.302 | 3.381 [3.120, 3.653] | 0.330 | 0.480 | 0.250 | 8.00 |
| chain | P1 | 7190 | 116 | 4.766 | 4.806 [4.569, 5.065] | 0.216 | 0.639 | 0.453 | 8.00 |
| chain | P2 | 2160 | 110 | 3.612 | 4.031 [3.759, 4.320] | 0.293 | 0.513 | 0.297 | 8.00 |

## Within-question phase contrasts

Negative `P2 - P1` means P2 has lower accepted length. The CI and 
sign fraction use questions as the independent unit.

| Arm | Contrast | Questions | Mean difference [95% CI] | Fraction below zero |
|---|---|---:|---:|---:|
| chain | P2 - P1 | 110 | -0.722 [-0.951, -0.489] | 0.736 |
| chain | Miss (fresh JIT) - P1 | 116 | -1.343 [-1.679, -1.035] | 0.828 |
| chain | P2 - Miss (fresh JIT) | 110 | 0.633 [0.333, 0.968] | 0.327 |
