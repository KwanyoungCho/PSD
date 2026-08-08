# P1+P2 on/on 파라미터 캘리브레이션 — 수집·사후분석 (2026-08-08)

수집: `tools/duet_calibration/collect_tree_thresholds.sh` (P1/P2 tree ON,
threshold 4종 전부 0, G=M — P1 18/18, P2 8/8, K1=9 K2=4 R=W=10 C=3,
seeds 42/123, eslab18 GPU0-4, load는 command.txt에 기록).
**이 런의 TPS는 trace 진단값 — 성능 수치로 인용 금지.**

## A. Champion 프로파일 귀속 (timeline_phase_matrix_champion_20260808 재분석)

draft step +15.9ms (53.13→69.03) 분해: glue +2.75, P1 tree +3.4
(32.2 vs chain 28.8), proxy_wait +1.1, P2 tree +4.7 (16.1 vs 11.4),
recv +0.65. target step +10.5ms 중 자체 연산 +4.0 (graph_pre +2.2,
tree_verify +1.3, post +0.6), **spec_wait +5.1 = draft 대기**.

핵심 구조 사실:

1. **on/on에서는 draft(69ms)가 시스템 임계경로** (target 연산 ~62-64ms가
   draft를 기다림 — spec_wait 9.06ms).
2. **라운드당 비용 ≈ 3.1-3.2ms로 폭-불감** (P1 폭16-20: 3.21/라운드,
   P2 폭10: 3.06/라운드; chain 폭16: 2.44 + 트리 부기 ~0.77).
   1.1B draft는 launch-bound — **폭 축소는 latency를 못 줄인다.
   라운드 수(K1/K2)만 draft latency를 움직인다** (~3.2ms/라운드).
3. 캡처 그래프 폭이 고정이므로 **threshold도 latency 불변 —
   threshold는 AL(선택 품질) 축**이다. latency 축은 라운드 수·M(target
   verify row, ~0.24ms/row)·glue 폭뿐.

## B. P1AL 열세(3.69-3.79 vs chain 4.43)의 원인 — 이분법 (seed_42, n=2226)

| 서빙 트리 | 비중 | max depth | AL |
|---|---|---|---|
| 만재(18노드) | 81.6% | 전부 9 | **4.31 ≈ chain 4.43** |
| 미니(3-9노드, 대부분 3) | 18.4% | 1-3 | **0.91** |

- 예산 분산·형태 문제 아님: 만재 트리는 chain과 동급.
- 미니트리 = 전역 선택에서 밀려 확장 0인 root(step내 start-rank 5-20위)가
  그래도 hit된 경우. 수락 기여 4.5%. chain이라면 대부분 miss였을 커버리지
  이득의 대가 — 제거 대상이 아니라 **개선 대상**.
- **ceil 33%**: 수락이 트리 깊이 천장에서 끝남 — 미니트리가 depth 2-3만
  돼도 AL 즉증 여지. start floor가 사망-root(하위분위)의 확장 lane을 회수해
  중위 root 깊이로 재분배하는 것이 정확히 이 지점을 겨냥.

## C. 도구 업데이트 (이번 커밋)

- `analyze_thresholds.py`: P1 축 추가 — conf는 CALIB trace의 phase 분리,
  start는 topo `.draft.jsonl`(전수 root prior=선택기 비교값과 동일한
  `piv`) + serve/walk의 depth≥2 수락 라벨. per-slot 실현 확률 의미론을
  P2 proxy와 일치시켜 risk 한계 재사용. threshold.env는
  `TREE_PROXY/TREE_CONF/P1_START/P1_CONF_THRESHOLD` 4변수
  (run_p1_p2_tree_formal 입력명과 일치). P1 축은 "어떤 floor도 위험 한계
  통과 못함 → 0.0 유지"를 정상 결론으로 처리. 합성 fixture 재검산 통과,
  구 P2-전용 호출 하위호환.
- `collect_tree_thresholds.sh`: 구 CLI(`--duet_tree_policy eagle`) →
  새 phase-CLI 전면 교체 + topo/CALIB/E0 3-trace 동시 수집 + 분석기 자동
  실행.
- `analyze_round_widths.py` (신규): depth별 생성 수/수락 rank(전역 score
  경쟁 rank)/**확장-필수 rank**(비말단 수락 노드 — 폭의 진짜 요구)와
  depth-도달 CDF. 축소 방향만 유효(수집 폭 초과는 관측 불가).

## D. 사후분석 최종 수치 (seeds 42+123: P1 4407 / P2 2060 served trees)

- 깊이 CDF (P1): d5 32.4% / d7 25.0% / d9 19.7%. K1 9→7 = draft
  −6.4ms, AL −0.45/hit-tree. 현 draft-임계 상태에서 **K1 절단 단독은
  TPS 본전** (T·S 동시 감소로 상쇄) — M 절단·AL 회복과 결합해야 유효.
- **확장-필수 rank 커버리지** (비말단 수락 노드 기준, 폭의 진짜 요구):
  P1 d1-d3에서 폭16 = 96.7~100% (d2 max 38 = 예외 꼬리), 폭12 =
  90-93%; d4+는 폭10으로 96-99%. **폭 축소는 latency 무득이므로 폭
  유지가 옳고**, 남는 lane은 threshold의 재분배 대상.
  P2는 d1 폭10 = 100% — R=W=10 적정.
- M rerank replay (acc 보존): P1 14→**99.2%** / 12→97.9 / 10→95.5;
  P2 8→**100%**(평균 7.73) / 7→98.3.
- P1 root prior: start_score AUC **0.813** > local_q 0.764 > reach
  0.645 — reach×q 곱이 옳은 prior. 후보 root ~38/step 중 상위 20이
  round-0 진입, per-root hit 2.3%.

## E. Threshold 판독 (calibration.txt / calibration.json)

| 축 | 현행 | 권고 | 근거 |
|---|---|---|---|
| P2 proxy | 0.01 | **0.01 유지** | 게이트는 표본(15,330 slots) 탓에 upper95 미달로 인증 거절 — 0.01의 손실기여 3.05%는 기존 A/B 검증과 모순 없음 |
| P2 conf | 0.03 | **0.01 하향** | 0.03 useful-contrib 1.28% (한계 1.5%의 85%, upper95 초과), 0.01 = 0.28%로 통과 — 현행 과억제 |
| P1 start | 0 | **safe 0.001 / bal 0.01** | 0.01: 빌드 root 63.9% 확장 차단(예산 회수) vs deep 수락 손실 4.96%; 0.001: 47.0% vs 0.65% |
| P1 conf | 0 | **0.01** | 점유 26.2% 회수, useful-contrib 0.47% |

권고값은 `threshold.recommended.env` (P2 proxy는 기존 검증치 유지 명시).

## F. 검증 실험 (실행: run_threshold_ab_20260808.sh → threshold_ab_20260808/)

paired on/on 전용, formal runner, 3-seed(42/123/2024) × 4-arm 순서회전,
ns=10 outlen=384, eslab18 상대비교 (절대치 아님):

- A_base: champion (proxy .01 / conf .03 / p1 0/0, M 14/8)
- B_safe: conf .01 + p1 .001/.01 (M 동일) — threshold 단독 효과
- C_bal: conf .01 + p1 .01/.01 (M 동일)
- D_balM: C_bal + M1 12 / M2 7 — target-row 절단 결합
- 판정: Decode TPS(1차)·tok/step·P1AL(미니트리 개선 확인). 승자는 17
  클린박스 full-scale로 절대치 확정 후 채택.

### 결과 (2026-08-08, eslab18 상대비교 완료)

| arm | TPS 평균 | vs A paired | 부호 | tok/step | P1AL | P2AL |
|---|---|---|---|---|---|---|
| A_base | 59.41 | — | — | 3.967 | 3.880 | 1.860 |
| B_safe | 58.86 | −0.55 | ++− | 3.973 | 3.810 | 1.937 |
| C_bal | 59.35 | −0.06 | −+− | 3.923 | 3.830 | 1.857 |
| **D_balM** | **61.48** | **+2.07 (+3.5%)** | **+++** | 3.983 | 3.847 | 1.900 |

- **판정: D_balM 승** — 유일한 3-seed 부호 일관, D−C +2.13 / D−B +2.62.
- 이득의 주동력은 threshold가 아니라 **M1 12 / M2 7 (target row 22→19)**:
  exit-proxy 조기 발송 → draft-임계경로의 proxy_wait 단축이라는 §A
  메커니즘과 정합. threshold 단독(B/C)은 이 표본(ns=10, 공유 18번)의
  잡음(σ~1.5-2) 안.
- start floor 0.01의 P1AL 우려는 실측에서 미발현 (3.847 vs 3.880 —
  예산 재분배가 deep 손실 상쇄).
- 다음: 17 클린박스 full-scale(D vs A, 3-seed paired)로 절대치 확정 후
  formal 기본값 갱신 (`run_threshold_confirm17.sh` 준비됨, 17 점유 대기).

### 메커니즘 실측 (mech_prof/, D vs A 프로파일 쌍, seed 42 진단런)

| 구간 | A | D | Δ |
|---|---|---|---|
| target graph_pre | 34.64 | 31.55 | **−3.08** |
| draft proxy_wait | 3.99 | 1.39 | **−2.61** |
| draft step | 65.41 | 62.45 | −2.96 |
| target full step | 71.27 | 67.04 | −4.23 |
| p1/p2_graph_replay | 28.90/12.24 | 28.98/12.27 | ±0.07 |

예측 사슬(M row 절단 → verify 단축 → exit-proxy 조기 → proxy_wait 단축)
그대로. draft 그래프 시간 불변 — M은 draft 연산에 무영향 확인.

**후속 캠페인 함의**: proxy_wait 1.39ms로 랑데부(P1 종료 ≈ proxy 도착)
재균형 — 이 지점부터 **K1 라운드 절단은 단독 무득**(P1이 일찍 끝나도
proxy-bound). spec_wait 8.4ms가 가리키는 다음 병목은 **draft 꼬리
(P2 4라운드 12.3ms + glue)** — K2/P2 축이 다음 탐색 대상.
