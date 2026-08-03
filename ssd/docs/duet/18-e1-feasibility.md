# 18 — E1 feasibility 진행 기록 (offline, 엔진 무수정)

**지위** (설계 v6 §9): 실제 기법의 일부가 아닌 **타당성 사전 점검** —
설계 결정을 위임받지 않으며, 결과는 종합 지표로 보고하고 선택은
사용자가 한다. 입력: E0 trace (17번 §3에서 무결성 검증 완료).

## Step 1 — baseline replay 검증 (v6 요구: ±1% 재현 실패 시 counterfactual 불신)

**모델**: 상태별(step status: P1-hit/P2-hit/miss) 주기 평균 × step
상태열 → 총 시간; 커밋 토큰 합 → predicted TPS.
(스크립트: `experiments/proxy_async_overlap/e1_feasibility/e1_baseline_replay.py`)

| 검증 | 타이밍 출처 | 상태열 | 기준 TPS | 오차 |
|---|---|---|---|---:|
| 1차 (cross-run) | champion profile 런 | E0 run1 | verdict-mean 81.91 | **−3.12%** |
| 1차 분해: 자기-일관성 | champion profile 런 | 같은 런 mix | 그 런 실측 82.56 | **+0.47%** |
| **2차 (same-run, 엄밀)** | **짝런** (E0+PROFILE 동시 ON) | **같은 런** | **같은 런 실측 78.62** | **−0.21% ✅** |

**판독**: 모델 자체의 정확도는 +0.47%/−0.21%로 ±1% 안. 1차의 −3.12%는
모델 오차가 아니라 **런-간 주기 편차**(verdict 유효주기 50.16ms vs
profile 런 51.85ms — 실측 TPS 밴드 ±1.5-2와 같은 규모)와 데이터셋
tok/step 운(4.086 vs 4.108)의 합. **결론(보고): same-run 기준으로
baseline replay 검증 통과 — counterfactual 비교의 전제 성립.** 단,
counterfactual 절대값에는 런-간 편차(±2-3%)가 그대로 실리므로, 트리
정책 비교는 **같은 timing 모델을 공유하는 상대 비교**로 읽어야 한다.

**짝런 세부**: champion 형상, E0+PROFILE 동시 ON (이중 계측 — TPS
78.62는 replay 기준값으로만 사용, 성능 보고 금지). 상태별 주기: miss
51.44 / P1-hit 56.45 / P2-hit 44.48ms (n=4620/14010/6199 — B>1 아님,
같은 형상 4셋×50seq). E0 상태열 24,653 step, tok/step 4.120.

## Step 2 — 비교군 구축 (예정)

설계 v6 비교군: ⓐ 현행 체인(=Step 1 baseline) / ⓑ 정적 π̂-비례 배분 /
ⓒ 동적+level / ⓓ 동적+frontier / ⓔ oracle 상한 (+ coverage 변형
[전 노드/top-M/backbone/verify-only], priority β 변형, target-first
참고군). 각 arm은 E0 trace의 wire(P_iv)·outcome으로 P2-tree hit/AL을
재계산하고, 상태별 주기 + 트리 증분 비용(2-pool slack — §1.2 실측)으로
predicted TPS 산출. 진행하며 이 문서에 단계별 기록.
