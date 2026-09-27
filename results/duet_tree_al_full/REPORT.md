# Full Spec-Bench tree AL 평가 — 완료

실행: 2026-09-21 22:44–2026-09-22 00:25 KST. 사전 설계 동결: 2026-09-21.

**평균 차이의 95% 신뢰구간이 0을 포함하므로 새 점수의 AL 우위는 확정하지 못했다.**

주 지표는 **2.0651 → 2.1003 tokens/hit**이며, 차이는 **+0.0352** (**+1.70%**), 문항 단위 paired bootstrap 95% CI는 **[-0.0025, +0.0721]**다.

**Root 후보 선정은 cache hit, tree 생성은 AL**을 목표로 구분했다. TPS는 이번 정책의 선택 기준이 아니다. 기존 root 정책·forward/depth/node 예산을 고정하고 tree 확장 점수만 비교했다.

## 실행 범위와 검증

- Spec-Bench **480문항/560턴 전체 × 2 policies × 3 seeds = 3,360턴** 완료.
- 6 tasks 각각 80문항, seed 1/42/123. 누락·입력 절단 없음. 두 번째 turn은 해당 정책이 실제 생성한 앞선 응답을 포함한다.
- **출력 상한 128**, T=0.7, EOS에서 종료. Full은 문항 범위 전체를 뜻한다. 1,024-token cap 또는 공식 chat-template 프로토콜의 재현 결과가 아니다.
- Target LayerSkip Llama2 70B AWQ TP4, draft TinyLlama 1.1B BF16 TP1, batch 1, exit56, K1/K2=4/2, C=3, P1 context당 root 3개 / P2 root 상한15개, P1/P2 G=M=8/6.
- P1의 context bucket 3/7/9에 따른 round 폭은 각각 (9,9,9,9), (21,15,15,15), (27,15,15,15)다. 동일한 구성 규칙을 사용하되 실제 context/phase 구성이 달라질 수 있으므로 전체 실행의 총 root 수나 총 forward cell 수가 같다고 주장하지 않는다.
- Root source는 기존 legacy proxy residual로 고정했다. 지난 후보 개선식 e(1−q)를 추가 적용한 결과나 Mirror-SD 전체 시스템과의 비교가 아니다.
- 새 점수의 phase/sibling/q-bin scalar calibration은 기존 8 prompts에서 만든 파일을 동결했다. 이번 full corpus로 refit하지 않았고 exact prompt overlap은 0이다. 새 neural head 학습은 없다. 과거 사용한 연구 benchmark이므로 globally unseen이라고 부르지 않는다.
- 모든 입력/history hash, EOS/cap, 수락 길이, node cap, 실행 명령, score hook, raw 결과 checksum을 독립 검사했다. 세부 내용은 [audit.json](audit.json).

## AL 정의와 주 결과

AL은 hit한 tree에서 수락된 descendant 수다. Correction root와 마지막 recovery/bonus를 제외한다. 종료로 잘리는 마지막 verification step은 주 집계에서 제외한다. 먼저 task별 hit-conditional AL을 구하고 6 tasks를 균등 평균했다. Bootstrap은 task 안에서 original question 80개를 재표집하며 두 turn과 세 seed를 같이 유지한다(5,000회). 신뢰구간은 지정한 세 seed 결과에 조건부인 문항 불확실성이다.

이 엔진의 비종료 step에서 root를 포함한 진행 token 수는 여기의 descendant AL + 1이다. 따라서 root/bonus를 포함하는 다른 AL 관례와 비교할 때 절대값과 상대 개선율을 그대로 섞지 않는다. 같은 집계의 두 정책 간 절대 차이는 +1 여부에 영향을 받지 않는다.

| 지표 | 기존 q-path | 새 calibrated reach | 차이 | 95% CI (차이) |
|---|---:|---:|---:|---|
| Hit당 AL — 주 지표 | 2.0651 | 2.1003 | +0.0352 | [-0.0025, +0.0721] |
| P1 hit당 AL (최대 4) | 2.2792 | 2.3256 | +0.0464 | [+0.0039, +0.0880] |
| P2 hit당 AL (최대 2) | 1.3641 | 1.3772 | +0.0132 | [-0.0075, +0.0334] |
| P1/P2를 50:50으로 둔 AL | 1.8216 | 1.8514 | +0.0298 | [+0.0048, +0.0541] |
| 종료 step 포함 raw AL — 사전 지정 민감도 | 2.0787 | 2.1160 | +0.0373 | [+0.0002, +0.0742] |
| 실제 출력에 남은 descendant의 전체 step AL — 추가 진단 | 2.0461 | 2.0811 | +0.0350 | [-0.0018, +0.0714] |

실출력 AL은 suffix가 현재 root + accepted descendants로 구성된다는 엔진 계약과 실제 출력 길이로 마지막 step에서 잘린 descendant를 제거한 값이다. 이는 계측 감사를 통해 추가한 진단이며 사전 지정 주 지표를 대체하지 않는다.

Task마다 동일한 비중을 준 값이므로 모든 hit를 단순 합산한 AL과 다를 수 있다. 모든 hit를 합산한 값은 2.0439 → 2.0855다.

## 과제별·seed별 결과

| Task | 기존 AL | 새 AL | 차이 | 95% CI |
|---|---:|---:|---:|---|
| MT-Bench | 1.9758 | 2.0413 | +0.0655 | [-0.0003, +0.1312] |
| Translation | 2.2382 | 2.2207 | -0.0174 | [-0.1379, +0.1047] |
| Summarization | 1.9734 | 1.9895 | +0.0161 | [-0.0791, +0.1112] |
| QA | 2.0341 | 2.0589 | +0.0247 | [-0.0427, +0.0934] |
| Math reasoning | 1.9272 | 1.9468 | +0.0196 | [-0.0463, +0.0877] |
| RAG | 2.2419 | 2.3446 | +0.1026 | [-0.0115, +0.2156] |

과제별 CI는 탐색적 분해이며 다중 비교 보정을 적용하지 않았다. 전체 주 지표의 판정과 분리해서 해석한다.

| Seed | 기존 AL (6 tasks 균등) | 새 AL | 차이 |
|---|---:|---:|---:|
| 1 | 2.0506 | 2.0840 | +0.0333 |
| 42 | 2.0655 | 2.1340 | +0.0685 |
| 123 | 2.0803 | 2.0834 | +0.0031 |

## Phase 구성과 node 이용량

대칭적 mixture 항등식으로 주 AL 차이를 분해하면 phase 안의 AL 변화 항은 +0.0386, phase hit 비중 변화 항은 -0.0034다. 둘의 합은 +0.0352다. 이는 **관측 평균의 대수적 분해**이며 동일 context에서의 인과 효과 분해는 아니다.

| 모든 hit 합산 진단 | 기존 | 새 점수 |
|---|---:|---:|
| P1 비중 | 0.7559 | 0.7522 |
| Hit당 검증 node 수 | 6.9124 | 6.9090 |
| P1 hit당 검증 node 수 | 7.3102 | 7.3119 |
| P2 hit당 검증 node 수 | 5.6808 | 5.6862 |
| Miss 포함 모든 비종료 step의 descendant AL | 1.8651 | 1.8937 |

고정한 것은 nominal cap이다. 실제 이용한 node 수는 위와 같으며, 미적중 root까지 포함한 총 생성 node/forward 수는 이번 무관측 실행에 계측되지 않았다. 이 수치가 정확히 같았다고 주장하지 않는다.

Root 정책은 고정했지만 실제 관측 cache hit rate(6 tasks 균등 평균)는 82.4107% → 81.7271%였다. 차이는 -0.6835%p이고 95% CI는 [-1.2624, -0.1025]%p다. Tree 변경 후 생성 경로와 cache 상태도 달라지므로, 이는 같은 prefix에서 root 후보 수식의 품질이 떨어졌다는 증거와는 구별한다. Root의 hit 목표와 tree의 AL 목표를 따로 추적해야 하는 실측 사례다.

## 수락 길이와 종료 처리

| Phase / 모든 hit 합산 tail | 기존 | 새 점수 |
|---|---:|---:|
| P1: P(AL ≥ 1) | 0.8868 | 0.8839 |
| P1: P(AL ≥ 2) | 0.6227 | 0.6201 |
| P1: P(AL ≥ 3) | 0.4093 | 0.4458 |
| P1: P(AL ≥ 4) | 0.3486 | 0.3731 |
| P2: P(AL ≥ 1) | 0.8264 | 0.8303 |
| P2: P(AL ≥ 2) | 0.5258 | 0.5344 |

| 실행 | 완료 턴 | EOS / cap | 즉시 EOS | 마지막 step의 출력 절단 |
|---|---:|---:|---:|---:|
| q_path, seed 1 | 560 | 69 / 491 | 16 | 319 |
| phase_sibling_q_bin, seed 1 | 560 | 83 / 477 | 20 | 331 |
| phase_sibling_q_bin, seed 42 | 560 | 70 / 490 | 12 | 358 |
| q_path, seed 42 | 560 | 80 / 480 | 13 | 334 |
| q_path, seed 123 | 560 | 72 / 488 | 9 | 313 |
| phase_sibling_q_bin, seed 123 | 560 | 68 / 492 | 7 | 329 |

양 정책에서 conditional AL이 정의된 공통 문항 480개의 question-macro AL은 2.0935 → 2.1294다. 양측 공통으로 정의되지 않은 문항은 0개다. 이 문항들을 corpus에서 제외한 것이 아니라 이 보조 macro 비율에서만 undefined로 처리했다.

## 이번 결과가 의미하는 것과 다음 단계

평균 차이의 95% 신뢰구간이 0을 포함하므로 새 점수의 AL 우위는 확정하지 못했다.

이번 full 결과의 구체적인 해석은 다음과 같다.

- 세 seed 모두 평균 AL 차이는 양수지만 seed 123에서는 차이가 작다. 이전 16 prompts의 +9.82%를 full dataset의 효과 크기로 일반화할 수 없다. 이번 주 지표의 효과는 +1.70%이며, EOS/history 처리도 이전 소규모 실행과 다르다.
- P1의 보조 분석에서는 양의 차이가 관측되지만 P2의 신뢰구간은 0을 포함한다. Phase별 분석도 다중 비교 보정을 하지 않은 보조 분석이며, 전체 주 지표의 우위 판정을 대체하지 않는다.
- 모든 hit를 합산한 P1의 AL≥3 비율은 40.93%→44.58%, AL≥4는 34.86%→37.31%다. 반면 AL≥1/2는 소폭 낮았다. 평균 증가가 깊은 수락 경로에서 관측된다는 진단이며, 동일 context에서 breadth/depth 배분의 인과 효과를 확정한 결과는 아니다.
- 종료 후 잘린 token도 세는 raw 로그 집계에서는 CI 하한이 아주 조금 양수지만, 사전 지정 주 지표와 실제 출력만 센 지표는 모두 0을 포함한다. 유리한 집계만 골라 전체 AL 우위를 주장하지 않는다.

이 비교는 고정된 legacy root 정책 아래 q-path와 frozen reach 점수를 평가한다. Node reach는 AL 목적함수와 연결되지만, 실제 leaf 확장의 완전한 가치는 root prior × reach × 다음 확장의 기대 수락 이득 g이다. 이번 prototype은 g까지 모델링한 최적 tree 알고리즘이 아니다. 결과에 맞춰 이번 full corpus에서 계수를 다시 골라 성능 검증으로 제시하지 않는다.

다음 개발에서는 별도 calibration/validation에서 g 및 깊이/형제 간 예산 배분을 설계하고, 같은 자원 제약의 AL을 비교한다. 이미 생성한 token의 점수를 보고 선별하는 기존 G>M rerank의 분포 보존 문제는 별도로 해결해야 한다. 이번 G=M 실험은 그 rerank를 통한 node 절단을 사용하지 않으며, 이것만으로 엔진 전체의 losslessness가 새로 인증된 것은 아니다.

128-token cap, raw base-model continuation, 고정된 4/2 depth와 8/6 node cap이라는 범위를 넘는 결론은 유보한다. 1,024-token cap, 다른 예산/모델, 개선 root와의 조합 및 Mirror-SD 전체 비교는 별도 실험이다.

## 재현 자료

- 목적함수 해설: [OBJECTIVE.md](OBJECTIVE.md), 사전 고정 설계: [PLAN.md](PLAN.md), [plan.json](plan.json).
- 전체 통계: [analysis.json](analysis.json), 문항별 합계: [question_counts.json](question_counts.json), 실행/측정 감사: [audit.json](audit.json).
- 원시 결과: `runs/s{seed}_{policy}/records.jsonl`; 질문 순서/생성 token/metrics 포함.
- 요약 수치: [NUMBERS.txt](NUMBERS.txt), 과제별 표: [task_results.csv](task_results.csv), 그림: [comparison.pdf](comparison.pdf), [comparison.png](comparison.png).

```bash
ssd/.venv/bin/python results/duet_tree_al_full/validate.py
ssd/.venv/bin/python results/duet_tree_al_full/analyze.py
ssd/.venv/bin/python results/duet_tree_al_full/make_report.py
```

Dataset 원본: [공식 Spec-Bench](https://github.com/hemingkx/Spec-Bench). 로컬 원본과 공식 480문항의 내용 일치를 확인하고 원본 파일 및 SHA256을 보존했다.
