# DUET P1+P2 tree full Spec-Bench local sweep

모든 arm은 전체 Spec-Bench 480 questions / 560 turns를 사용한다. MT-Bench의
두 turn은 먼저 한 question으로 결합하며, TPS는 decode-only tokens/time을
question마다 계산한 뒤 평균한다. 모든 arm은 seed 42, output 1,024다.

## Sweep design

| Case | C | N1 | M1 | P1 start/conf threshold | Changed from reference |
|---|---:|---:|---:|---:|---|
| original_seed42 | 2 | 14 | 12 | 0 / 0 | 2026-08-11 reference raw |
| reference_repeat | 2 | 14 | 12 | 0 / 0 | exact independent repeat |
| n1_12 | 2 | 12 | 12 | 0 / 0 | generated nodes/root 14→12; removes on-hit rerank |
| c3 | 3 | 14 | 12 | 0 / 0 | branch width 2→3 only |
| threshold_mild | 2 | 14 | 12 | 0.001 / 0.01 | mild P1 pruning only |

`M1=12`와 P2 설정은 전 arm에서 고정한다. 따라서 C/N1 비교는 target에 보내는
P1 verification node cap을 바꾸지 않는다. 공통 P2는 budget/root/N2/M2
=15/10/8/8, threshold=0.01/0.01이다.
실행은 target-step 증가 원인 분석을 위해 일시 중단했다. `N1=12` arm은
분석에서 확인한 on-hit 14→12 rerank 비용을 없애면서 M1을 고정하는 진단 arm이다.

## Completion status

- `original_seed42`: 560/560 turns
- `reference_repeat`: 0/560 turns
- `n1_12`: 0/560 turns
- `c3`: 0/560 turns
- `threshold_mild`: 0/560 turns

## Overall

| Case | Questions (turns) | Decode TPS | AL | P1 AL | P2 AL | Hit | P1 hit | P2 hit | Target step ms | Target verify ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original_seed42 | 480 (560) | 63.945 | 4.490 | 4.367 | 2.204 | 0.790 | 0.616 | 0.174 | 71.028 | 62.950 |

## Decode TPS by subtask

| Case | mt_bench | translation | summarization | qa | math_reasoning | rag | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_seed42 | 63.551 | 66.301 | 54.207 | 71.064 | 62.146 | 66.400 | 63.945 |

## Accepted length by subtask

| Case | mt_bench | translation | summarization | qa | math_reasoning | rag | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_seed42 | 4.446 | 4.643 | 3.836 | 4.978 | 4.327 | 4.712 | 4.490 |

## P1 conditional AL by subtask

| Case | mt_bench | translation | summarization | qa | math_reasoning | rag | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_seed42 | 4.199 | 4.475 | 4.235 | 4.630 | 4.073 | 4.573 | 4.367 |

## Fixed configuration

- Target/draft: LayerSkip Llama-2-70B / TinyLlama-1.1B
- K1/K2=8/4, exit layer 56, P1 fan-out/roots-per-position=3/3
- proxy top-k 28; P1 allocation `backbone`; P1/P2 tree on
- P2 budget/root/N2/M2=15/10/8/8; P2 thresholds 0.01/0.01
- raw prompt, temperature 0.7, top-p 1.0, context 4,096, draft RoPE extension

Machine-readable overall table: `/home/eslab/chokwans99/PSD/ssd/experiments/proxy_async_overlap/tree_sweep/p1_p2_tree_full_local_sweep_seed42_20260812/overall.csv`
