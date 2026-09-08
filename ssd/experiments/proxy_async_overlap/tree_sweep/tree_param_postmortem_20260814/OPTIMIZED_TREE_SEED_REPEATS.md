# Optimized DUET-tree full-seed repeats

REPORT의 최종 권장 설정을 seed 42, 1, 123에서 full Spec-Bench로 반복한
결과다. 설정은 `K1/K2=8/5`, exit 49, P1 `N/M=14/12`, P2 `N/M=10/10`,
P2 proxy/confidence threshold `0.01/0.01`, 양쪽 tree on, native context
2,048, output cap 1,024, profiler off다. TPS는 decode-only이며 MT-Bench 두
turn은 question 단위로 먼저 결합했다.

## Native-2,048 full results

각 실행은 480 questions/560 turns를 완주했고 fatal scan은 비어 있다. Context
경계에서 decode token이 없는 question이 seed마다 하나 있어 latency는 유효한
verification step만 가중 집계했다.

| Seed | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step ms | Verify ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 67.646 | 4.604 | 0.760 | 0.623 | 5.375 | 0.137 | 3.531 | 68.939 | 62.324 |
| 1 | 66.030 | 4.495 | 0.746 | 0.609 | 5.300 | 0.137 | 3.497 | 69.024 | 62.108 |
| 123 | 66.960 | 4.581 | 0.750 | 0.612 | 5.374 | 0.138 | 3.521 | 69.377 | 62.353 |
| 3-seed mean | 66.879 | 4.560 | 0.752 | 0.614 | 5.350 | 0.137 | 3.516 | 69.113 | 62.262 |

세 seed 범위는 TPS 66.030–67.646, AL 4.495–4.604, hit 0.746–0.760이다.
Target step 범위는 68.939–69.377 ms로 0.438 ms이며, AL/TPS 변동의 주된
원인은 verification latency보다 sampling 결과 차이다.

## Common comparison with the current best-subtask composite

기존 best-subtask와 공정하게 비교하기 위해 모든 optimized seed에도 동일한
사전 기준 `prefill_total_tokens + max_new_tokens <= 2048`을 적용했다. 각 결과는
456 questions/536 turns이며, 제외된 24개는 모두 summarization이다.

| Result | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step ms | Verify ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Optimized seed 42 | 67.845 | 4.616 | 0.763 | 0.626 | 5.388 | 0.137 | 3.540 | 68.829 | 62.322 |
| Optimized seed 1 | 66.637 | 4.535 | 0.755 | 0.617 | 5.307 | 0.138 | 3.514 | 68.914 | 62.108 |
| Optimized seed 123 | 67.858 | 4.640 | 0.758 | 0.621 | 5.419 | 0.137 | 3.531 | 69.267 | 62.353 |
| Optimized 3-seed mean | 67.447 | 4.597 | 0.759 | 0.621 | 5.371 | 0.137 | 3.528 | 69.003 | 62.261 |
| Current best-subtask composite | 68.036 | 4.676 | 0.798 | 0.636 | 5.536 | 0.163 | 3.265 | 69.575 | 63.149 |
| Expanded best-subtask candidate | 69.503 | 4.765 | 0.789 | 0.638 | 5.592 | 0.150 | 3.423 | 69.294 | 62.735 |

`Current best-subtask`와 `Expanded best-subtask candidate`는 모두 결과를 본 뒤
subtask별 source를 선택한 composite이며 단일 seed 결과가 아니다. Expanded
candidate는 이번 optimized 세 seed까지 후보군을 넓힌 더 강한 post-hoc
cherry-pick이므로 paper headline으로 자동 승격하지 않는다.

## Decode TPS by subtask on the common subset

| Result | mt_bench | translation | summarization | qa | math_reasoning | rag | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Optimized seed 42 | 67.883 | 73.250 | 56.192 | 70.256 | 68.992 | 67.002 | 67.845 |
| Optimized seed 1 | 64.599 | 73.253 | 58.106 | 68.706 | 67.458 | 65.140 | 66.637 |
| Optimized seed 123 | 64.401 | 77.455 | 58.255 | 72.202 | 66.811 | 65.144 | 67.858 |
| Optimized 3-seed mean | 65.628 | 74.653 | 57.518 | 70.388 | 67.754 | 65.762 | 67.447 |
| Current best-subtask | 65.592 | 73.137 | 59.190 | 72.321 | 67.241 | 68.082 | 68.036 |
| Expanded best candidate | 67.883 | 77.455 | 59.190 | 72.321 | 68.992 | 68.082 | 69.503 |

## Accepted length by subtask on the common subset

| Result | mt_bench | translation | summarization | qa | math_reasoning | rag | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Optimized seed 42 | 4.610 | 4.968 | 3.857 | 4.774 | 4.660 | 4.598 | 4.616 |
| Optimized seed 1 | 4.403 | 4.978 | 3.995 | 4.624 | 4.570 | 4.480 | 4.535 |
| Optimized seed 123 | 4.407 | 5.259 | 4.025 | 4.916 | 4.552 | 4.498 | 4.640 |
| Optimized 3-seed mean | 4.474 | 5.068 | 3.959 | 4.771 | 4.594 | 4.525 | 4.597 |
| Current best-subtask | 4.487 | 4.973 | 4.172 | 4.978 | 4.562 | 4.731 | 4.676 |
| Expanded best candidate | 4.610 | 5.259 | 4.172 | 4.978 | 4.660 | 4.731 | 4.765 |

## Cache hit rate by subtask on the common subset

| Result | mt_bench | translation | summarization | qa | math_reasoning | rag | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Optimized seed 42 | 0.802 | 0.769 | 0.599 | 0.801 | 0.841 | 0.716 | 0.763 |
| Optimized seed 1 | 0.799 | 0.780 | 0.603 | 0.781 | 0.820 | 0.699 | 0.755 |
| Optimized seed 123 | 0.790 | 0.791 | 0.587 | 0.806 | 0.807 | 0.716 | 0.758 |
| Optimized 3-seed mean | 0.797 | 0.780 | 0.596 | 0.796 | 0.823 | 0.711 | 0.759 |
| Current best-subtask | 0.825 | 0.805 | 0.653 | 0.850 | 0.860 | 0.753 | 0.798 |
| Expanded best candidate | 0.802 | 0.791 | 0.653 | 0.850 | 0.841 | 0.753 | 0.789 |

## Per-subtask winner against the current composite

| Subtask | Current best TPS / AL | Best optimized source | Optimized TPS / AL | Selected for expanded candidate |
|---|---:|---|---:|---|
| mt_bench | 65.592 / 4.487 | seed 42 | 67.883 / 4.610 | optimized seed 42 |
| translation | 73.137 / 4.973 | seed 123 | 77.455 / 5.259 | optimized seed 123 |
| summarization | 59.190 / 4.172 | seed 123 | 58.255 / 4.025 | current best-subtask |
| qa | 72.321 / 4.978 | seed 123 | 72.202 / 4.916 | current best-subtask |
| math_reasoning | 67.241 / 4.562 | seed 42 | 68.992 / 4.660 | optimized seed 42 |
| rag | 68.082 / 4.731 | seed 42 | 67.002 / 4.598 | current best-subtask |

Optimized 설정은 MT-Bench, translation, math reasoning의 TPS와 AL을 함께
갱신했다. Current best-subtask는 summarization, QA, RAG에서 유지됐다.
Expanded candidate는 current best 대비 TPS +1.467 tok/s, AL +0.089이지만,
후보군 확대 후 다시 선택한 결과이므로 selection bias가 더 크다.

## Artifacts

- Runner: `run_full_winner_seed_repeats.sh`
- Seed 42: `full/winner_k8_k5_e49_n10m10/`
- Seed 1: `full/winner_k8_k5_e49_n10m10_s1/`
- Seed 123: `full/winner_k8_k5_e49_n10m10_s123/`
- Expanded candidate builder: `build_expanded_best_candidate.py`
- Expanded candidate and manifest: `comparison_best_subtask/`

세 raw output의 동일 hash는 seed 42/1에서 6/560, seed 1/123에서 7/560,
seed 42/123에서 9/560뿐이어서 세 sampler 결과가 실제로 독립적임을 확인했다.
