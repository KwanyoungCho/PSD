# K1=K2=8 phase-difficulty summary

Primary AL includes the one correction/recovery token. Confidence 
intervals resample questions, so long generations do not become 
independent pseudo-replicates.

| Arm | Source | Events | Questions | Event-pooled AL | Question-mean AL [95% CI] | P(AL=1) | P(AL≥3) | P(AL≥5) | Mean valid-k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | Miss (fresh JIT) | 1855 | 120 | 3.302 | 3.381 [3.120, 3.653] | 0.330 | 0.480 | 0.250 | 8.00 |
| chain | P1 | 7190 | 116 | 4.766 | 4.806 [4.569, 5.065] | 0.216 | 0.639 | 0.453 | 8.00 |
| chain | P2 | 2160 | 110 | 3.612 | 4.031 [3.759, 4.320] | 0.293 | 0.513 | 0.297 | 8.00 |
| tree | Miss (fresh JIT) | 2065 | 120 | 3.325 | 3.167 [2.910, 3.442] | 0.361 | 0.463 | 0.264 | 8.00 |
| tree | P1 | 6458 | 110 | 5.280 | 5.087 [4.805, 5.376] | 0.115 | 0.733 | 0.530 | 12.00 |
| tree | P2 | 1907 | 107 | 3.719 | 3.971 [3.709, 4.243] | 0.186 | 0.631 | 0.268 | 11.09 |

## Within-question phase contrasts

Negative `P2 - P1` means P2 has lower accepted length. The CI and 
sign fraction use questions as the independent unit.

| Arm | Contrast | Questions | Mean difference [95% CI] | Fraction below zero |
|---|---|---:|---:|---:|
| chain | P2 - P1 | 110 | -0.722 [-0.958, -0.484] | 0.736 |
| chain | Miss (fresh JIT) - P1 | 116 | -1.343 [-1.673, -1.030] | 0.828 |
| chain | P2 - Miss (fresh JIT) | 110 | 0.633 [0.332, 0.955] | 0.327 |
| tree | P2 - P1 | 107 | -1.051 [-1.298, -0.812] | 0.813 |
| tree | Miss (fresh JIT) - P1 | 110 | -1.723 [-2.091, -1.375] | 0.909 |
| tree | P2 - Miss (fresh JIT) | 107 | 0.578 [0.313, 0.864] | 0.336 |
