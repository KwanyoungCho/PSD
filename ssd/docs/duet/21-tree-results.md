# P2-Tree — 결과 리포트 및 기법 서술 (T5)

> 상태: **작성 중** — 구현 완료(docs/duet/20), E2E 검증 진행.
> §1–§3은 논문/발표자료에 바로 쓸 수 있도록 자족적으로 서술했다.
> §4 이후는 sweep/비교 결과가 나오는 대로 채운다.
> 설계 원문: docs/duet/15 (v6). 구현·이슈 로그: docs/duet/20.

## 1. 기법 개요 — 무엇을 하는 것인가

DUET의 Phase 2는 target의 early-exit proxy 분포 P_iv로 "다음 step에
target이 실제로 회복할 토큰"을 예측해 draft 캐시를 채운다. 기존
구현은 proxy가 고른 seed마다 **체인**(K2-깊이 일렬 continuation)을
만들었다. P2-tree는 이 체인을 **트리**로 바꾼다:

- **왜 트리인가**: 검증은 한 번의 target forward로 트리 전체를 판정할
  수 있고 (tree attention), 수락 보행이 첫 기각에서 형제 가지로
  갈아탈 수 있으므로, 같은 노드 예산에서 기대 수락 길이(AL)가
  체인보다 길다. AL은 TPS의 지배 항이다 (E1 실측: 트리화 AL 이득
  +3.5~3.8%, 형제 상관 λ=0.51~0.54 — docs/duet/18).
- **무엇이 어려운가**: (a) 수락-보존(losslessness) — 트리에서 형제를
  순서대로 제안하면 뒤 형제의 제안분포는 앞 형제 기각에 조건화된
  잔차 분포가 되어야 target 분포가 정확히 보존된다. (b) 엔진 통합 —
  체인 전제가 깔린 glue/fork/캐시 키/KV 배치/검증 창을 트리로
  일반화해야 한다.

## 2. 알고리즘 — 어떻게 동작하는가

한 step의 draft 측 Phase 2 (rollout; 설계 v6 §6):

1. **Root forest**: proxy가 고른 seed (노드, 토큰) 후보들이 트리의
   root가 된다. root별 자식 예산은 라이브 P_iv에 비례해 draw 전에
   확정한다 (⑤v2: budget_r ∝ P_iv^β, 상한 N_v — 정체성 비조건화
   원칙 D10: "누굴 뽑을지"가 "무엇이 뽑혔는지"에 의존하지 않는다).
2. **F_total번의 W-폭 forward**: 각 forward에서 정책(level=깊이
   동기 / frontier=priority 상위 W)이 확장할 노드를 고르고, 노드당
   C_tensor개를 **비복원(WOR)** 추출한다. WOR은 exponential-race
   top-k로 구현 — race 점수 내림차순이 순차 비복원 추출 순서와
   동일하며, C=1이면 기존 Sampler와 op·RNG 소비까지 bit-identical.
   형제 순서(sib_order)는 그 자체가 잔차 사다리의 좌표다.
3. **priority** = log π̂(root P_iv) + Σ_경로 log c_raw. c_raw는
   **재정규화 전** 원본 확률 (결정②) — "이 경로가 실제로 수락될
   확률"의 로그 추정치로, frontier 정책과 예산 prefix 배분의 기준.
4. **응답 뷰**: root당 상위 N_v 노드의 고정-패딩 뷰 (tok /
   parent_local / sib_order / parent_q 참조). 생성-시점 캡 덕에
   절단이 발생하지 않는다.

검증 (target 측, v6 §7.2 — **무손실의 핵심**):

- 창 = [recovery 토큰 | 뷰 노드들(생성 순서)]. 한 번의 tree-attention
  forward로 모든 노드 위치의 target 분포 p를 얻는다.
- **잔차 사다리 보행**: 컨텍스트 ctx에서 자식 형제들 {x_1..x_m}을
  sib_order 순으로 시도한다. j번째 형제의 수락 확률은
  a_j = min(1, R_j(x_j) / D_j(x_j)) — R_j는 앞 형제 기각에 조건화된
  target 잔차, D_j는 draft 제안의 잔차 (앞 형제 토큰 제거 후
  재정규화). 수락되면 그 노드로 내려가 반복; 전원 기각이면 잔차
  분포에서 recovery를 뽑고, 수락된 잎에서는 plain p 보너스 샘플.
- **증명**: 비복원 추출 순서의 전수 열거 × 사다리 해석해 합성이
  target p와 1e-9 오차 내 일치 (30 케이스 전수 검증 —
  tests/test_p2_tree_alloc.py TestLosslessExhaustive) + 프로덕션
  텐서 보행이 참조 보행과 동일-코인 50시행 완전 동등.
- 검증 q(parent_q)는 draft 샘플측과 **동일 함수**(q_probs_from_logits)
  로 빌드 — 수락-보존의 전제인 "같은 logits → 같은 q"를 코드로 강제.

## 3. 엔진 통합 — 체인 기계의 트리 일반화 (구현 아키텍처)

핵심 발견 세 가지가 통합 비용을 크게 줄였다:

1. **응답 운반**: out_tokens 행에 뷰 토큰(생성 순서), valid_k에 유효
   노드 수를 실으면 기존 extend/prepare 기계가 트리 행을 그대로
   나른다 — 검증 창 = [rec]+뷰, scratch 셀은 선형 그대로. target
   측 신규 작업은 rope(pos0+1+depth) 덮어쓰기 + 조상 custom mask +
   FlashInfer plan뿐이다. topology와 parent_q logits는 응답 wire에
   고정 크기 블록으로 동승한다 (miss면 zero — 크기·호출 순서 불변).
2. **KV 재실체화 = 셀 복사** (D14의 등가 실행): 트리 rope가
   pos0+1+depth라서 **수락 경로의 rope는 canonical 위치와 정확히
   일치**한다. 따라서 수락 후 필요한 재실체화는 forward가 아니라
   "뷰-순서 셀 → 경로-순서 셀"의 KV 복사 (겹침 대비 gather→scatter)
   로 끝난다. target은 verifier가 전 TP rank에 commit_tree_kv를
   브로드캐스트하고 (B=1에선 SHM 명령 순서가 ack를 대체), draft는
   다음 요청 수신 직후 로컬 복사한다.
3. **종단 노드 id = 캐시 키 좌표**: 체인의 "수락 길이 p" 좌표는
   트리에서 "보행 종단 노드 id" (0=root-종단, 1+j=노드 j)로
   일반화되고, 체인은 그 퇴화 사례로 정확히 포섭된다 (chain accept-p
   ↔ 노드 id p). P1 fork는 [rec+전 노드] = N_v+1개 종단 컨텍스트에서
   일어나고 (결정④ ⓑ), populate의 fan_idx가 그대로 다음 요청 키가
   된다.

트리-hit step의 draft 파이프라인 (모두 mask-override 재사용):
TREE_GLUE(뷰를 조상-mask 단일 W-폭 forward로 실체화, KV는 canonical
scratch 셀) → 노드-fork P1 (컨텍스트별 fanout 재배분, 조상 비트맵
mask, depth rope, K1-step CG 재사용) → 노드-seed P2 rollout (proxy
위치축이 노드 id 축으로 — wire 형식 불변, ⓒ).

무손실 하드 게이트 외 자동 문턱은 없다 (v6 판정 철학): 종합 지표를
보고하고 판정은 사용자가 한다.

## 4. E2E 검증 (2026-08-03, 박스 CPU 오염 load~82 — TPS 해석 금지 구간)

| 런 | 정책 | 완주 | Decode tok/s | AL(+rec) | hit | hit-AL | P1/P2 hit |
|---|---|---|---|---|---|---|---|
| e2e1_off_r3 | off | ✓ | 52.13 | 3.53 | 0.78 | 3.80 | — |
| e2e2_level_r1 | level | ✓ | 2.60 | **3.71 (+5.1%)** | 0.78 | **4.03 (+6.1%)** | 0.538 / 0.240 |
| e2e2_frontier_r1 | frontier | ✓ | 10.42* | 3.46 (-2.0%) | 0.74 | 3.74 | 0.499 / 0.237 |

\* frontier 런은 롤아웃 GPU-상주 최적화(4f5b9a3) 적용 후 실행 —
level 2.60과의 4× 차이는 최적화 효과와 오염 변동이 혼재. 정책 간
AL 우열은 단일런 잡음 범위 — T4 sweep에서 판정.

- 트리 라이브 루프 (서빙→트리 verify→보행→commit→종단 키→재실체화→
  노드-fork 빌드) 무크래시 완주. 출력 텍스트 건전 (코드 프롬프트
  정상 완성; 사다리 D_j=0 가드 미발화).
- AL 이득 +5.1%는 E1 예측(+3.5~3.8%) 상회 — 단일런·상이 코인 경로
  참고치. 확정은 T4/T5의 반복 실측.
- Decode 2.60은 v1 미최적화(트리 verify eager, 보행 CPU, 매 스텝
  rollout 오버헤드) × CPU 오염 복합 — 성능 판정 구간 아님. 최적화
  1차(롤아웃 GPU 상주)는 적용 완료, 스팬 분해는 PROFILE 런에서.

## 5. T4 sweep (예정 — 한산 가드)

그리드: 정책{level,frontier} × R{8,10} × N_v{6,8} × β{0.5,1.0}
(tree_sweep/run_tree_sweep.sh; load<24 && GPU 유휴 대기 가드).

## 6. T5 비교 (예정)

AR / async-SD(C: k=7 f=6) / DUET-chain(E9K24_jit) / DUET-tree(챔피언
sweep 구성), 5-rep 인터리브. 지표: Decode TPS(제1지표), AL, cache
hit(phase 분해), miss 분포, timeline 정렬 그래프 (plot_duet_aligned_
timeline.py — 수 step 창 확대 렌더).
