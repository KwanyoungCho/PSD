# DUET tree parameter postmortem results

이 문서는 tree overhead 및 native-context 버그 수정 이후의 파라미터
사후 분석과 검증만 기록한다. Screening 결과는 후보 선택용이며 논문 수치가
아니다. 최종 비교는 full Spec-Bench, output cap 1,024에서 수행한다.

## 결론 요약

- 기존: `K1/K2=8/4`, exit 56, `N1/M1=14/12`, `N2/M2=8/8`,
  `R=10`, `W=15`, roots-per-position 3.
- Screening winner: `K1/K2=8/5`, exit 49, `N1/M1=14/12`,
  `N2/M2=10/10`, `R=10`, `W=15`, roots-per-position 3.
- Winner의 P2 late rate는 12/1,068 (1.124%), p99 overrun은
  0.148 ms로 기준 tail gate를 통과했다.
- P2는 깊이만 늘리면 이득이 없었지만, 검색/검증 노드를 8→10으로 함께
  늘리자 추가 깊이가 실제 AL로 연결됐다.
- P2 active roots 10→12는 같은 W=15 예산을 더 많은 prefix에 분산시켜
  전체 AL을 낮췄다.
- P1은 N1=16, M1=10, K1=9를 각각 분리해 확인했으나 모두 탈락했다.
- P2 confidence 0.02는 3-seed subset에서는 좋아 보였지만 full에서
  TPS -2.11%, AL -1.89%로 재현되지 않아 탈락했다. 최종 권장 threshold는
  기존 `proxy/confidence=0.01/0.01`이다.

## 사후 구조 진단

Trace는 구조 진단용이며 TPS 측정에 사용하지 않았다.

| 진단 | 기존 K2=4 | K2=5, exit 49 | 해석 |
|---|---:|---:|---|
| P1 max-depth 도달 | 25.76% | 20.53% | 깊이 여지는 있으나 overlap이 부족함 |
| P2 max-depth 도달 | 28.47% | 18.57% | K2+1을 쓸 수 있는 경로가 존재함 |
| P2 alternative-sibling tree | 20.80% | 16.86% | 분기 자체가 실제 수락에 기여함 |
| P2 branch-assisted accepted share | 17.68% | 15.32% | chain 외 형제 노드가 유효함 |
| P2 생성 노드/8 | 7.55 | 7.63 | N2=8이 사실상 포화됨 |
| P2 마지막 두 root-rank hit | 11.95% | 16.29% | root 경계 압력은 있으나 R=12 실험은 실패 |
| P1 세 번째 local root-rank hit | — | 13.56% | RPP 축소 근거가 없음 |

## Overlap tail

양의 signed p01은 draft가 deadline보다 먼저 끝났음을 뜻한다. Late rate와
p99 overrun은 평균이 아니라 aligned step 분포에서 계산했다.

| Arm | P1 late | P1 p01 (ms) | P1 p99 overrun | P2 late | P2 p01 (ms) | P2 p99 overrun |
|---|---:|---:|---:|---:|---:|---:|
| 기존 K8/K4, exit 56 | 6/1,191 (0.504%) | +5.851 | 0.000 | 13/1,127 (1.154%) | -0.159 | 0.159 |
| K2=5, exit 56 | 3/1,109 (0.271%) | +6.286 | 0.000 | 907/1,043 (86.961%) | -4.425 | 4.425 |
| K2=5, exit 52 | 7/1,083 (0.646%) | +3.724 | 0.000 | 196/1,019 (19.235%) | -2.299 | 2.299 |
| K2=5, exit 50 | 4/1,129 (0.354%) | +1.784 | 0.000 | 48/1,062 (4.520%) | -0.953 | 0.953 |
| K2=5, exit 49, N2/M2=8/8 | 6/1,089 (0.551%) | +2.097 | 0.000 | 3/1,025 (0.293%) | +0.659 | 0.000 |
| Winner, N2/M2=10/10 | 5/1,133 (0.441%) | +0.914 | 0.000 | 12/1,068 (1.124%) | -0.148 | 0.148 |
| K1=9 rejection | 449/1,113 (40.341%) | -2.678 | 2.678 | 101/1,047 (9.647%) | -1.846 | 1.846 |

Gate는 기존 대비 late rate +0.5%p 및 p99 overrun +0.3 ms 이내다.
Winner는 P1/P2 모두 통과한다. Exit 50 이상과 K1=9는 통과하지 못한다.

## Fixed screening comparison

동일한 72 questions/84 turns, seed 42, output cap 256에서 profiler를 끄고
측정했다. TPS/AL은 question-level이고 MT-Bench 두 turn은 한 question으로
합쳤다. Latency는 verify-step weighted 값이다.

| Arm | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 기존 K8/K4, N2/M2=8/8 | 59.307 | 4.031 | 0.769 | 0.583 | 4.778 | 0.186 | 3.099 | 70.127 | 62.102 |
| K2=5, exit49, N2/M2=8/8 | 56.269 | 3.831 | 0.738 | 0.565 | 4.452 | 0.173 | 3.132 | 70.043 | 61.402 |
| K2=5, N2/M2=10/8 | 58.756 | 4.003 | 0.719 | 0.560 | 4.719 | 0.159 | 3.238 | 70.248 | 61.543 |
| **K2=5, N2/M2=10/10** | **62.151** | **4.252** | 0.751 | 0.592 | 4.790 | 0.159 | **3.433** | 71.018 | 62.519 |
| 위 설정 + R=12 | 62.000 | 4.221 | 0.718 | 0.569 | 5.001 | 0.149 | 3.443 | 70.415 | 61.621 |
| 위 설정 + N1=16 | 58.024 | 3.956 | 0.728 | 0.551 | 4.577 | 0.178 | 3.337 | 70.162 | 61.630 |
| 위 설정 + M1=10 | 59.154 | 3.992 | 0.741 | 0.573 | 4.584 | 0.169 | 3.196 | 69.661 | 61.217 |

Winner는 기존 대비 AL +5.48%, decode TPS +4.80%, target step +1.27%다.
추가 target 입력의 latency 증가보다 AL 증가가 크다.

상세 question-paired 및 subtask 표는 `SCREEN_RESULTS.md`에 있다.

## Full Spec-Bench

Native context 2,048, output cap 1,024, seed 42, profiler off 조건에서 두
arm 모두 480 questions/560 turns를 완주했다. 두 arm 모두 zero-decode
context stop이 1 question이고 fatal scan은 비어 있다.

| Arm | Questions (turns) | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 기존 K8/K4, exit56, N2/M2=8/8 | 480 (560) | 63.351 | 4.339 | 0.759 | 0.597 | 5.278 | 0.161 | 3.237 | 69.397 | 62.967 |
| **K8/K5, exit49, N2/M2=10/10** | **480 (560)** | **67.646** | **4.604** | **0.760** | **0.623** | **5.375** | 0.137 | **3.531** | **68.939** | **62.324** |

Winner는 기존 대비 decode TPS +6.78%, AL +6.11%다. Target step은
0.66%, target verify는 1.02% 감소했다. P2 hit는 2.42%p 감소했지만 P2
AL은 9.08% 증가했고, P1 hit와 P1 AL도 각각 2.53%p, 1.84% 증가했다.
따라서 전체 hit는 사실상 동일한 상태에서 hit된 tree의 품질과 accepted
length가 개선됐다.

| Subtask | 기존 TPS | Winner TPS | 기존 AL | Winner AL |
|---|---:|---:|---:|---:|
| MT-Bench | 65.059 | 67.883 | 4.446 | 4.610 |
| Translation | 67.791 | 73.250 | 4.643 | 4.968 |
| Summarization | 56.354 | 58.378 | 3.879 | 4.006 |
| QA | 66.513 | 70.256 | 4.541 | 4.774 |
| Math reasoning | 62.138 | 68.992 | 4.229 | 4.660 |
| RAG | 62.168 | 67.002 | 4.290 | 4.598 |

여섯 subtask 모두 TPS와 AL이 증가했다. 상세 phase 지표와 question-paired
비교는 `FULL_RESULTS.md`에 있다.

## Full 결과 기반 2차 사후 탐색

첫 full 검증 뒤에는 앞선 tuning question 132개와 겹치지 않는 별도
48-question 진단 subset과 60-question screening subset을 만들었다. Full
winner의 P2 tree는 생성 노드가 평균 9.535/10으로 다시 N2 cap에 닿았고,
허용 폭을 10에서 12로 넓힌 counterfactual coverage는 depth 1--3에서
각각 3.05%p, 3.37%p, 4.81%p 증가했다. 따라서 P2만 독립적으로
`N2=12`를 검사했다.

| Arm | TPS | AL | P1 AL | P2 AL | Target step (ms) | Verify (ms) |
|---|---:|---:|---:|---:|---:|---:|
| N2/M2=10/10 | 57.351 | 3.901 | 4.564 | 3.178 | 69.765 | 61.619 |
| **N2/M2=12/10** | **59.278** | **4.027** | **4.786** | 3.376 | **69.777** | **61.604** |
| N2/M2=12/12 | 58.640 | 4.022 | 4.674 | **3.441** | 70.649 | 62.582 |

`12/10`은 검색 폭만 늘려 target 입력 수는 그대로 유지한다. 기준 대비
AL +3.25%, TPS +3.36%였고 target step은 +0.012 ms로 동일했다. 반면
`12/12`는 P2 AL 증가가 조금 더 컸지만 target step +0.872 ms, verify
+0.978 ms가 발생해 탈락시켰다. 이 결과는 더 많이 생성하는 것과 더 많이
검증하는 것을 분리해야 함을 보여준다.

`12/10`의 clean timing run에서도 P1 late 4/844 (0.474%), P2 late
3/793 (0.378%)였고 양쪽 p99 overrun은 모두 0 ms였다. 따라서 평균 overlap뿐
아니라 초과 빈도와 tail도 통과했다.

그러나 full Spec-Bench에서는 `12/10`이 기존 full winner보다 낮았다.

| Full arm | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **N2/M2=10/10** | **67.646** | **4.604** | **0.760** | **0.623** | 5.375 | 0.137 | **3.531** | **68.939** | **62.324** |
| N2/M2=12/10 | 66.652 | 4.551 | 0.749 | 0.611 | **5.394** | **0.138** | 3.516 | 69.181 | 62.438 |

`12/10`은 TPS -1.47%, AL -1.16%, target step +0.35%이므로 최종
후보에서 탈락한다. MT-Bench와 RAG는 좋아졌지만 summarization과 math의
하락이 컸다. Question별로는 AL 246/479, TPS 245/479에서 오히려 이겼으나
소수의 큰 하락이 평균을 뒤집었다. 같은 sampler seed에도 동일 output hash는
6/560 turns뿐이어서, tree 변경 뒤 sampling trajectory가 갈라지는 변동이
크다는 것도 확인했다. 따라서 1% 수준의 N2 차이는 한 seed로 정하지 않고,
균형 subset의 `N2=10/11/12` 3-seed 평균으로 중간값을 추가 판정한다.

3-seed medium 결과에서 `11/10`은 모든 seed에서 `10/10`보다 낮아
제외됐다. `12/10`은 medium seed 평균으로 TPS +1.71%, AL +1.77%였지만
seed 표준편차가 각각 1.399, 0.096으로 커졌고, subtask 방향도 full 결과와
일치하지 않았다. 더 큰 480-question full 결과가 하락했으므로 고정 N2=12도
승격하지 않는다. 이 단계의 권장값은 `N2/M2=10/10` 유지다.

## P2 confidence threshold 후속 후보

현재 N2=10 trace에서 depth 5 이전 P2 확장 후보 3,893개를 사후 분석했다.

| Confidence floor | 차단 대상 | 전체 유용 확장 중 손실 | 해석 |
|---:|---:|---:|---|
| 0.01 (현재) | 621 (15.95%) | 0/650 (0.00%) | 안전하지만 보수적 |
| 0.02 | 790 (20.29%) | 7/650 (1.08%) | 안전 후보 |
| 0.03 | 896 (23.02%) | 12/650 (1.85%) | 기존 calibration의 balanced 값 |

Threshold는 이미 생성한 node를 삭제하지 않고 그 node 아래의 다음 확장만
막는다. 따라서 target verify cap 10은 바뀌지 않는다. 현재 tree 형상에서
0.02/0.03을 각각 3 seeds로 A/B했다.

| Confidence | TPS (3-seed) | AL (3-seed) | P2 AL | Target step (ms) | Verify (ms) |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 57.496 ± 0.679 | 3.906 ± 0.041 | 3.240 ± 0.157 | 69.732 ± 0.056 | 61.602 ± 0.024 |
| **0.02** | **59.385 ± 2.160** | **4.038 ± 0.145** | 3.308 ± 0.081 | 69.929 ± 0.082 | 61.603 ± 0.060 |
| 0.03 | 57.089 ± 2.030 | 3.878 ± 0.136 | **3.423 ± 0.141** | **69.690 ± 0.095** | **61.513 ± 0.044** |

0.02는 0.01 대비 seed 평균 TPS +3.29%, AL +3.38%였고 여섯 subtask의
TPS/AL 평균이 모두 증가했다. Seed 1에서는 중립, seed 42와 123에서는
개선됐다. 0.03은 P2 AL은 높지만 seed 42의 전체 AL/TPS가 크게 하락해
제외한다. 따라서 0.02 하나만 full Spec-Bench로 최종 검증한다.

최종 full 검증에서는 이 subset 이득이 재현되지 않았다. 두 arm은 동일한
seed 42, native context 2,048, output cap 1,024에서 각각 480 questions/560
turns를 완주했고 UID 누락·중복과 fatal error가 없었다.

| P2 confidence | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.01** | **67.646** | **4.604** | **0.760** | **0.623** | **5.375** | 0.137 | **3.531** | **68.939** | **62.324** |
| 0.02 | 66.219 | 4.517 | 0.743 | 0.605 | 5.344 | **0.138** | 3.524 | 69.151 | 62.381 |

0.02는 TPS -1.427 tok/s(-2.11%), AL -0.087(-1.89%)이고 target step도
+0.212 ms 증가했다. Paired question 기준 TPS 230/479, AL 227/479에서만
이겼다. MT-Bench는 TPS +0.135, AL +0.012로 소폭 증가했지만 Translation,
Summarization, QA, Math, RAG는 모두 하락했으며 Math 하락이 TPS -4.927,
AL -0.310으로 가장 컸다. 따라서 confidence 0.02는 탈락시키고 full winner의
`P2 proxy/confidence threshold=0.01/0.01`을 유지한다. 상세 비교는
`FULL_THRESHOLD_RESULTS.md`에 있다.

## 재현 파일

- `PLAN.md`: 데이터 분리, 후보 순서, overlap gate
- `run_profile_arm.sh`: tail/trace 진단 실행
- `run_screen_arm.sh`: profiler-off 고정 screening 실행
- `run_full_arm.sh`: native-2,048 full Spec-Bench 실행
- `compare_arms.py`: question-level 및 step-weighted 비교
- `FULL_RESULTS.md`: full Spec-Bench overall/subtask 직접 비교
- `FULL_POSTFULL_RESULTS.md`: full winner와 N2=12 search-only 후보 비교
- `POSTFULL_N12_RESULTS.md`: 독립 post-full subset의 N2/M2 비교
- `run_postfull_multiseed.sh`: N2=10/11/12의 3-seed 중간 규모 검증
- `POSTFULL_MULTISEED_RESULTS.md`: N2=10/11/12의 seed별/평균 결과
- `run_postfull_threshold_multiseed.sh`: P2 confidence 0.02/0.03 A/B
- `POSTFULL_THRESHOLD_RESULTS.md`: confidence threshold 3-seed 결과
- `compare_threshold_multiseed.py`: threshold seed 평균/paired 집계
- `postfull_forensic_subset.jsonl`, `postfull_screening_subset.jsonl`,
  `postfull_subset_manifest.csv`: 기존 tuning question과 겹치지 않는 2차 데이터
- `forensic_subset.jsonl`, `screening_subset.jsonl`, `subset_manifest.csv`:
  고정된 진단/선별 데이터
