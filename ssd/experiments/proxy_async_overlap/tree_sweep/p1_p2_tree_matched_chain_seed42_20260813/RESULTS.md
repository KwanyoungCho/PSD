# DUET matched chain vs. tree (full Spec-Bench, seed 42)

- Both arms use the latest DUET engine and the same 560 turns.
- Fixed: seed 42, output 1,024, K1/K2=8/4, exit layer 56, proxy top-k 28, P1 fanout 3, P2 budget 15, N1/M1=14/12, N2/M2=8/8.
- Changed: only the candidate topology policy, chain (`P1/P2 tree=off`) vs. tree (`P1/P2 tree=on`).
- Decode TPS and AL/hit metrics are question-level means; MT-Bench turns are merged before averaging. Step latency is weighted by verification-step count.

## Overall

| Method | Questions (turns) | Decode TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step ms | Target verify ms | Outside verify ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DUET-chain (matched) | 480 (560) | 66.353 | 4.332 | 0.812 | 0.650 | 4.101 | 0.162 | 1.986 | 66.178 | 61.598 | 4.580 |
| DUET-tree | 480 (560) | 65.136 | 4.490 | 0.790 | 0.616 | 4.367 | 0.174 | 2.204 | 69.803 | 63.251 | 6.553 |
| Tree − chain | — | -1.217 | +0.158 | -0.021 | -0.034 | +0.266 | +0.012 | +0.218 | +3.625 | +1.653 | +1.972 |

## Decode TPS by subtask

| Method | mt_bench | translation | summarization | qa | math_reasoning | rag | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| DUET-chain (matched) | 65.923 | 72.890 | 56.095 | 70.193 | 63.425 | 69.590 | 66.353 |
| DUET-tree | 64.572 | 67.642 | 55.354 | 72.321 | 63.165 | 67.762 | 65.136 |

## Accepted length by subtask

| Method | mt_bench | translation | summarization | qa | math_reasoning | rag | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| DUET-chain (matched) | 4.279 | 4.754 | 3.688 | 4.574 | 4.125 | 4.572 | 4.332 |
| DUET-tree | 4.446 | 4.643 | 3.836 | 4.978 | 4.327 | 4.712 | 4.490 |

## Integrity

- Both files validated as exactly 480 questions / 560 turns.
- Dataset UID and order must match exactly.
- All fixed configuration fields and required metrics are validated row by row.

Machine-readable table: `/home/eslab/chokwans99/PSD/ssd/experiments/proxy_async_overlap/tree_sweep/p1_p2_tree_matched_chain_seed42_20260813/comparison.csv`
