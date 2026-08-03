# 15 — P2 동적 트리 (P2-Tree) 설계 문서

**작성**: 2026-08-02 v3 → 2026-08-03 v4 (외부 리뷰 1차) →
**2026-08-04 v5** (외부 리뷰 2차 + 사용자 결정 ①②③: 정책 스위치·
D11 single-shot·priority 확정·π̂=라이브 P_iv·pack 필수화·G0/G1 게이트·
E1 비교군 5종·5-rep·그래프 분리·§12 TODO). **상태: 설계 단계 — 구현 착수 금지.**
사용자 지시: "모든 건 완벽하게 설계가 이루어진 이후에 시작." §9의 관문
실험(E0-E2)이 전부 green이고 설계가 승인되기 전에는 엔진 코드를
수정하지 않는다. **단 하나의 예외**: E0가 요구하는 계측 덤프
게이트(§9 E0)는 **별도 승인을 받아 관문 전 선행 구현**한다(§10 P0).
이 문서가 단일 소스다.

**v4 변경 요약** (v3 대비): ① D1을 "고정폭 W forest + frontier(보류)
확장 + parent_idx/동적 조상 mask"로 재서술하고 "새 CG 0개" 주장 폐기,
② 자식 생성은 결정론 top-c가 아니라 **q_eff에서 WOR 샘플** (무손실
조건) + 장부 4규칙, ③ proxy score는 chosen_tok int64 상위 비트에
pack (wire 항목 수 불변 확인), ④ 채산 상수를 final_rematch 공인
수치로 교체 (계수 0.12→0.156) + N_v별 breakeven 표, ⑤ verify는
SGLang-참조 설계 (custom tree mask + N_v bucket + slot 전략) — 이
엔진의 EAGLE은 체인이라 tree verify가 전면 신규임을 확인.

**읽는 법**: 처음 보는 사람은 §1(배경)→§2(목표)부터. 설계 리뷰어는
§4(확정 결정)→§5(사실 확인)→§6-8(설계)→§9(관문). `[파일:라인]`은
2026-08-02 HEAD 기준.

---

## 1. 배경 — 이 작업이 왜 존재하는가

### 1.1 기본 시스템: async 추측 디코딩 (SSD)

이 저장소는 **SSD**(Speculative Speculative Decoding — 검증 결과까지
추측한다는 뜻의 원 프로젝트명) 엔진의 fork다. **target**(layerskip-
llama2-70B, AWQ W4A16, TP4 — GPU 0-3)과 **draft**(TinyLlama-1.1B AWQ —
GPU 4 전용 프로세스)가 병렬로 돈다. 일반 추측 디코딩과 달리 **async**:
draft는 target이 현재 step을 검증하는 동안 **다음 step의 검증 결과를
미리 추측**해서, 가능한 결과(어느 위치에서 거부되고 어떤 recovery
토큰이 나올지)마다 이어질 추측 체인을 **트리 캐시**에 채워 둔다. 검증이
끝나 실제 결과가 나오면 draft는 캐시를 조회해 — **hit**이면 저장된
체인을 즉시 응답하고, **miss**면 즉석 생성(JIT)해 응답한다. 베이스라인
**C** = 이 async 엔진 원형 (노브: 체인 깊이 k, 결과-예측 폭 f). 처리량
수치는 모두 **decode TPS (tok/s)**.

### 1.2 DUET이 더한 것: early-exit proxy 2단 draft (split-K1/K2)

**DUET**의 아이디어: target이 verify forward를 도는 **도중**, exit
layer(현 B=1 최적 구성인 champion — E9K24_jit, §1.3 — 은 56/80)의 중간
logits로 "이번 검증이 어떻게 끝날지"를 target 스스로 예측해 draft에
미리 알려준다. draft는 두 단계로 캐시를 채운다:

- **Phase 1 (P1)**: draft 자신의 체인에서 dfo 폭으로 fan-out하며 K1번
  forward — draft가 스스로 예상한 결과들의 트리 (valid_k=K1).
- **Phase 2 (P2)**: target이 보낸 proxy 예측(seed)에서 출발해 K2번
  forward — target의 시각으로 본 결과들의 체인 (valid_k=K2). **이
  문서의 대상.**

**용어 사전** (신입용):

| 용어 | 뜻 |
|---|---|
| K1 / K2 | P1 / P2의 forward 횟수 = 각 행의 체인 깊이. K1+K2 = speculate_k [config.py:348-352], K2 ≤ K1 [draft_runner.py:203에서 강제] |
| dfo / pfo | draft fan-out / proxy fan-out. pfo는 직접 폭이 아니라 budget 산식의 계수 |
| exit layer | proxy를 뽑는 target 중간 레이어 (`duet_exit_layer`, 기본 2L/3) |
| seed (root) | P2 트리의 시작 (위치, 토큰) — proxy가 고른 예상 결과. v4에서는 root로도 부름 |
| budget (=B_s) | P2 seed 총수 = `pfo × (K1+1)` [config.py:227-241]. **상수** — 매 step, 모든 seq 동일 |
| valid_k | 캐시 행의 실제 깊이 ∈ {K1, K2}. verify/glue CG 버킷 선택 기준 |
| glue decode | 응답 확정된 체인을 draft의 정식 KV에 실체화하는 varlen forward |
| JIT | miss 시 즉석 체인 생성 (`jit_speculate`; JIT-short 게이트 시 K2 깊이) |
| wire | target rank0 → draft NCCL 페이로드: `(chosen_pos, chosen_tok)` 점수 내림차순 `wire_N`개 — **wire_N은 config 상수, step/valid_k와 무관** [config.py:131-170] |
| Policy B | P2 seed 선택 정책 (아래 수식) |
| **CG** | **CUDA Graph** — 커널 시퀀스를 고정 shape으로 capture해 replay하는 최적화. replay는 텐서 **shape이 상수**여야 하므로 모든 동적성은 "shape 고정, 내용(mask/index 값)만 변동"으로 구현 |
| band-clear | A/B 판정: 3-rep 인터리브에서 한쪽의 최악 rep > 상대의 최고 rep (분산 구간 비겹침). 겹치면 동률 |
| spec_wait | target이 검증 결과 송신 후 다음 추측 응답을 받을 때까지의 블로킹 |
| proxy_wait | draft가 P1을 끝내고 exit proxy 도착을 기다리는 블로킹 |
| q_eff | draft의 **실제 제안 분포** — temperature 등 sampler 처리가 적용된 뒤의 분포. 무손실 검증의 기준 분포 |
| WOR | without-replacement — 이미 뽑힌 후보를 제외하고 재정규화해 이어 샘플 |

**Policy B 수식** [verifier.py `_compute_and_send_proxy`; docs/duet/05]:
verify 도중 exit에서 p^E(early-exit 분포), p^D(draft 분포)를 얻어

- 위치별 근사 수락확률 **α̂_i = min(1, p_i^E(y_i)/p_i^D(y_i))**
- 첫-거부 분포 **ĥ**: h_0 = 1−α̂_0, h_i = (∏_{j<i}α̂_j)(1−α̂_i),
  h_K = ∏α̂_j (전부-수락)
- 위치별 보정 분포 **corr**: i<K는 residual **[p^E − p^D]_+** (검증 중
  토큰 제외) top_k; **i=K(전부-수락)는 p^E[K,:] 전체 분포의 top_k**
  [verifier.py:437-441]
- **P_iv = ĥ_i × corr[i,v]** → 전 위치 flatten 후 글로벌
  topk(wire_N) → 송신. draft는 P1 트리와 dedup 후 **앞에서부터 정확히
  B_s개**를 root로 채택 [draft_runner.py `_select_proxy_sourced_tokens_unified`].

**캐시 키** = (seq_id, fan_idx, recovery_token). hit = 실제 검증 결과와
일치하는 키의 행 존재 — 그 행(v4에선 그 root의 서브트리)이 다음 verify
입력이 된다.

**한 step의 타임라인** (B=32 실측 그림:
`experiments/proxy_async_overlap/b_gt1/bscale32/overlap_profile/duet_k2x2_prof/timeline_step121_mixed.png`):

```
target(GPU0-3): [spec_wait]→[graph_pre: 층0..exit]→[exit+proxy send]→[graph_post]→[sample]
draft (GPU4)  : [respond(조회/JIT)]→[glue]→[P1 rollout]→[proxy_wait...]→[P2 rollout]→[cache]
```

**균형 2조건** (사용자 규칙): ① P1 종료 ≈ proxy send 도착, ② draft
종료 ≈ target 종료. B=1 champion은 proxy_wait ~9ms로 거의 균형점.
(B=32에선 조건①이 137ms 깨져 있었으나 소비처 부재로 회복이 무의미했다
[balance32] — 본 설계의 트리는 후속 B>1 일반화 때 그 소비처가 될 수
있다. 본 설계의 타겟은 B=1이다, §2.)

### 1.3 현재 성적 — 우리가 어디에 서 있나

(상세: docs/duet/12, 13, bscale32/REPORT.md; 단위 decode tok/s)

- **B=1**: champion **E9K24_jit** (K1=9 K2=4, exit=56, dfo=2 pfo=1,
  jit-short) **81.91 vs C 81.52 (+0.5%)** — 동률(band-clear 아님).
  B∈{2,4} 동률과 함께 지지 않는 세 지점 중 하나이며 champion 형상이
  유지되는 유일한 지점.
- **B>1 공정화 반전**: C에도 B별 재튜닝을 허용하자 기존 band-clear
  승리가 소멸 — B∈{2,4} 동률, **B∈{8,16,32} C-opt에 band-clear 패배**
  (−3.7/−2.5/−4.1%). 깊이-폭 frontier(폭 비용 B-선형, 깊이 가치
  B-불변)는 시스템 불변.
- **matched-shape**: 같은 형상이면 tok/step C와 동일 — 수락 로직 무결.
  남은 격차는 verify 내부 시간 (graph_pre +19%/layer, 별도 트랙).
- **finding 5a**: P2 hit 시 수락 길이(**L_p2**) **1.76** [final_rematch
  RESULTS.md:53]이 breakeven 2.6 미달 — champion의 마지막 큰 레버.
  **이것이 본 작업의 직접 동기.**

### 1.4 문제 정의 — P2 hit의 수락 길이가 낮다

(E[AL|P2 hit] = L_p2로 표기)

| 지표 | 값 | 의미 |
|---|---|---|
| L_p2 | **1.76** (champion, final_rematch; m6_fix 계열 1.63~1.81) | 상한 K2=4의 절반 이하 |
| P2 위치당 수락률 | ≈0.6~0.7 (K2=1 실측 0.61~0.68; 역산 ≈0.70. E0에서 정밀 실측) | P1(≈0.7~0.8)보다 낮음 |
| P2 hit rate | 0.269 (m6_fix, B≤4) / **0.24~0.25 (final_rematch champion 레짐)** — 채산표(§2)는 0.25 기준 | **hit은 충분히 높다 — 낮은 것은 hit 시 길이** |
| finding 1 | hit↑ ≠ tok↑ | hit 자체는 토큰을 나르지 않음 |

**원인**: 1) **무분기 체인** (구조적 — 공략 대상): P2 rollout은 seed
선택에서만 분기, 이후 행당 1토큰 직진 [draft_runner.py:1292-1299,
1250-1253; mask도 행별 대각선, cudagraph_helpers.py:397]. 2) off-policy
연속. 3) 선택 효과 (가설 — E0로 검증).

---

## 2. 목표와 성공 기준

- **정량 목표**: B=1 champion 레짐에서 **L_p2 1.76 → 2.6+**로 올려
  동률(+0.5%)을 band-clear 우위로 전환.
- **채산 부등식** (v4 — final_rematch 공인 상수: TPS 81.91 =
  0.0819 tok/ms, tok/step 4.108, T_target 51.44ms [final_rematch
  RESULTS.md:76-78]; verify 행당 한계비용 c_row — 체인 스윕 실측
  1.9ms/행 [E10 프로브: K1 9→10 = +1행 → +1.7ms], **트리 행에도
  동일하다는 것은 가정이므로 E2①에서 T_verify(N_v)로 직접 재실측**):

  `breakeven ΔL_p2 = TPS × c_row × Δrows ≈ 0.156 × Δrows`

  트리 응답은 P2-hit step에만 나가므로 verify 비용·이득 양쪽에서
  hit율이 소거된다. **단 draft 쪽 비용(forest 구성·동적 mask·샘플링)은
  매 step 발생** [phase2_build는 hit 무관, draft_runner.py:1828] —
  이는 조건①의 proxy_wait 여유(B=1 ~9ms)로 흡수하고, 부족하면 exit
  위치 재조정을 병행한다 (§11.1).

  | 응답 노드 N_v | 추가 verify 행 | breakeven L_p2 | TPS +3%에 필요한 L_p2 (P2hit≈0.25) |
  |---|---|---|---|
  | 6 | +2 | 2.07 | ≈2.57 |
  | 8 | +4 | 2.38 | ≈2.89 |
  | 10 | +6 | 2.69 | ≈3.21 |

  → **N_v=10은 공격적. 6·8부터 탐색** (외부 리뷰 검증 완료 — 산술
  재검산 일치).
- **비목표**: B>1 최적화 (호환만), P1 트리화 (후속), 수락 분포 변경
  (무손실성 절대 조건).

---

## 3. 제안 개요

```
현재 P2:  root₁ ── t ── t          root별 1행 무분기 체인
          root₂ ── t ── t          (B_s = root 수 = 상수)

제안 P2:  root₁ ─┬ t ─┬ t         매 forward 행수 W=10 완전 고정.
                 │    └ t         "누구를 확장하나"만 frontier에서
                 └ t ── t         value 순으로 동적 선택 —
          root₂ ── t ── (보류)    이번에 안 뽑힌 노드는 버리지 않고
          root₃ ── (확장 0)       보류: 다음 forward에서 선택될 수 있음
```

**관련 연구 좌표** (검증 완료): **DFVG** (ASPLOS'26 — confidence 임계
sparse 동적 트리 + 오버랩; FPGA라 shape 자유 — 아이디어만), **EAGLE-2**
(arXiv:2406.16858 — 노드 가치 V=∏confidence로 위상 선택, 레벨당 top-k
확장 후 rerank; log 미사용 — raw 곱), **DySpec** (arXiv:2410.11744 —
draft 확률-수락률 상관), **Sequoia** (WOR 다후보 검증의 무손실 수학 —
§6 샘플링 규칙의 근거), **SGLang EAGLE 구현** (§7 참조 설계의 원형).

**우리의 차별점**: ① 루트가 proxy가 고른 **root 숲**, ② root에 draft
confidence보다 강할 수 있는 사전점수(ĥ×P_iv — E0로 실증할 가설),
③ frontier 확장은 EAGLE-2의 레벨-단위보다 일반적 (얕은 보류 노드가
나중에 역전 선택 가능).

---

## 4. 확정 설계 결정 (v4 — 사용자 문답 2026-08-02/03 반영)

- **D1(v5) — 고정폭 W forest + 확장 정책 스위치** (결정 2026-08-04):
  매 forward 입력 행수 **W = B_s = 10 완전 고정** (depth별 축소 없음 —
  v3의 W=[10,8,6,4]는 같은 그래프 반복 replay 구조[draft_runner.py:
  1255-1300]와 모순이라 폐기). 확장 대상 선택은 **하나의 구현에 정책
  스위치** (`--duet_tree_policy`): `level` = 최신 depth 노드만 선택
  (level-synchronous — frontier의 퇴화 경로), `frontier` = 미평가 노드
  풀 전체에서 priority 상위 선택 (안 뽑힌 노드는 **보류**되어 다음
  forward에서 경쟁). level-sync는 frontier의 특수한 경우이므로 코드는
  하나다. **기본값은 미정** — E1(두 정책 모두 시뮬레이션) + 실측
  (L_p2 증가폭, frontier 고유 오버헤드)으로 최종 결정한다.
  행별 depth/rope/조상 mask/slot은 전부 per-행 데이터라 shape 동일.
  신규 필요물: parent_idx 관리 + **동적 조상 mask 생성기** (draft CG
  family 추가는 없으나 mask 내용 생성기는 신규). target에는 tree-verify
  bucket 신규 (§7) — **"전체 신규 CG 0개" 주장은 폐기.**
- **D2(v5) — π̂ = 라이브 P_iv, score 비트-pack 필수** (결정 ③으로
  조건부→필수 승격): log P_iv를 16비트 고정소수점(범위 log₁₀P ∈
  [−6,0])으로 양자화해 **chosen_tok int64의 비트 15~30에 pack** —
  V=32,000이라 토큰은 15비트만 사용(vocab ≤ 32768 assert + 버전 비트
  1개), 부호 비트 불사용. wire 항목 수는 valid_k와 무관한 config
  상수라 장애 없음 (config.py:131-170; 고정 ring 송신 verifier.py:
  497-501). NCCL 호출 수·통신량 증가 0. rank prior/경험 테이블은
  폐기 — E0의 역할은 P_iv의 **calibration 검증**(위치별 신뢰도
  곡선)으로 전환.
- **D3 — log 공간 유지**: value 비교는 log-합 (top-k 선택에 raw 곱과
  동치, 수치 안전).
- **D4 — 동적 선택 오버헤드 최소**: forward당 frontier topk 1회, 신규
  GPU→CPU 동기화 0회 원칙.
- **D8(신규) — 자식 생성은 샘플링** (무손실 조건, §6): 자식 토큰의
  정체는 **q_eff에서 비복원 샘플** (WOR — 이미 뽑힌 토큰을 제외하고
  재정규화해 이어 뽑기) — 결정론 top-c 금지. 몇 개를 어느 부모에게
  줄지(fanout 배분)는 value로 결정론적으로 정해도 됨 (자식 정체와
  무관하므로 분포 보존과 무관).
- **D11(신규, 2026-08-04) — single-shot fanout**: 노드가 평가(forward)
  되는 순간 그 노드의 fanout을 확정하고 형제들을 **그 자리에서 한
  번에** 비복원 샘플한다. **평가 완료된 노드에는 이후 형제를 추가하지
  않는다** (재확장 금지 — 사용자 결정). 효과: 비복원 커서(부모 분포
  보관·뽑은 집합·순서)가 forward를 넘어 생존할 필요가 없어져 frontier
  모드의 장부 비용이 소멸. 대가: 유망해진 부모의 폭을 나중에 넓힐 수
  없고 깊이로만 투자 가능.

## 5. 사실 확인 (코드/논문/프레임워크 검증; 라인은 2026-08-02 HEAD)

- **F1 — 트리 어텐션 기계는 draft 전용**: packed bitmask →
  FlashInfer `_custom_mask_buf` 주입 [cudagraph_helpers.py:318-410,
  456]. target verify capture는 cu_seqlens causal 체인 — custom mask
  인자 없음 [:1065-1120].
- **F2 — 무손실 트리 수락은 신규**: 현행 verify()는 위치당 후보 1개
  체인 [utils/verify.py:37, 119-132]. 트리는 형제 WOR 검증(§7.2) 필요.
- **F3 — KV slot 경합**: 같은 depth 형제 행이 같은 position slot 경합
  → §7.3 (SGLang 대비 포함).
- **F4 — wire**: 현행 logits 응답은 speculate_k=13 폭 ≈832KB/step
  [speculator_async.py:109, draft_runner.py:732]. 트리 응답(N_v~8)은
  이보다 작다 — wire 크기는 리스크 아님 (E2에서 확인만).
- **F5 — 응답 절단 규칙**: 절단은 **조상 폐포 + 형제-prefix 보존**
  (§6 장부 규칙 ③) — v3의 "value 상위 N_v 임의 절단"은 무손실성
  위반으로 폐기.
- **F6 — 기존 trace는 shape만** [verifier.py:119, 463] — E0 덤프
  게이트 신규 (유일 예외).
- **F7 — top_k 자동상향 산식**: `max(total_budget + p1_max + 2,
  ceil(wire_N/(K_min+1)))` [config.py:445-449] — budget 의미 변경 시
  두 항 + scheduler 예약 [scheduler.py:58] 일괄 재검토.
- **F8(신규) — 이 엔진의 EAGLE은 체인**: sync speculator도
  speculations [B,K+1] 체인 [speculator_sync.py:30-69], eagle은 draft
  모델 종류(activations 전달)일 뿐 트리 없음. **"EAGLE이 있으니 tree
  verify도 있겠지"는 이 코드베이스에서 성립하지 않는다** — target-측
  tree verify와 트리 KV 관리는 전면 신규다.

## 6. 트리 구성 알고리즘 (draft 쪽, v4)

**frontier 확장 루프** (forward F_total = K2회, 행수 W = B_s 고정):

```
pool ← proxy가 고른 B_s개 root (미평가; prior = rank/packed-score 기반 π̂)
for f in 1..F_total:
    expand_set ← 선택 정책(D1)에 따라 pool에서 W개    # GPU topk 1회
        level:    최신 depth의 미평가 노드만
        frontier: 미평가 노드 전체에서 priority 상위
    forward(expand_set)                                # W행, 항상 같은 CG
    각 평가 노드 x에서 (D11 — 이 순간 1회뿐):
        fanout_x 확정 (자식 정체 관측 전 — D10)
        fanout_x개 자식을 q_eff에서 비복원 샘플, 순서 기록
    자식들 pool에 추가; x는 평가 완료로 전환 (재방문 없음)
    depth = K2 도달 노드는 pool에서 제외 (버퍼 [N,K] 캡)
샘플된 모든 노드는 candidate tree에 남는다 (확장 안 해도 잎 — D10)
```

**확장 우선순위 (결정 ②, 2026-08-04)**:

```
priority(n) = log π̂(root(n)) + Σ_경로 log c_raw
```

- **π̂(root) = 이번 step의 라이브 P_iv** (결정 ③, 2026-08-04):
  P_iv = ĥ×corr는 구조적으로 "검증 결과가 (pos, tok)일 확률"의
  추정치이므로 root hit 확률 그 자체다 — 경험 테이블/rank prior는
  폐기하고 step별 라이브 값을 wire 비트-pack으로 수신해 사용 (사용자
  결정: 문맥-평균 테이블은 step별 신호를 버림). 상대 배분에는 균일
  편향이 상쇄되므로(모든 root에 같은 log-상수) 무해하고, **비균일
  (위치별) 편향만** E0의 calibration 곡선으로 **측정**한다. 보정계수는
  **선제 구현하지 않는다** (사용자 결정 2026-08-04) — 먼저 라이브
  P_iv 그대로 구현·측정하고, 실측이 "P_iv가 hit 빈도를 잘 대변하지
  못하는 비균일 편향"을 보일 때만 후속 도입 (§12 TODO). budget 사전
  분배는 보류 (동적 배분 확정; E1 비교군에 참고용으로만 유지).
- **c_raw = 그 토큰의 "부모 원본 q_eff에서의" 확률.** 비복원 샘플링은
  재정규화된 나머지 분포로 뽑지만, value 기록은 반드시 **원본 확률**로
  한다 — 형제 순서 효과가 자동 내장되기 때문: 형제2의 검사 확률
  (1−q(a))와 검사 시 수락 추정 (q(b)/(1−q(a)))의 곱에서 (1−q(a))가
  약분되어 원본 q(b)만 남는다 (수치 검증: R=0.6, q(a)=0.7, q(b)=0.2
  → value 0.12 = 실제 기여 0.6×0.3×0.667 = 0.12). 재정규화 값을
  기록하면 1/(1−q(a))배 과대평가가 생겨 그때만 별도 할인이 필요해짐
  — 원본 규약으로 그 문제를 원천 제거.
- 동률 시 얕은 depth 우선 (tie-break; 비용 0).
- **깊이 보정·한계비용·명시적 형제 할인 항은 두지 않는다** (사용자
  결정 2026-08-04 — acceptance rate(평균 수락 길이)에 집중, 비용
  최적화는 후속). 비용 통제는 우선순위가 아니라 위상 수준(root별
  생성-시 캡 D10, bucket 선택은 E1)에서 처리.
- E1 검증 항목: (a) 이 priority가 oracle(사후 최적 배분) 대비 남기는
  격차, (b) "c_raw ≈ 실제 기여 확률" calibration.

**장부 4규칙 (무손실 조건 — Sequoia/SpecInfer 계열)**:

1. 같은 부모의 형제들은 **q_eff에서 WOR로 이어 샘플** (독립 재샘플
   금지 — 중복·보정식 불일치).
2. **형제 뽑은 순서 기록** — verify가 같은 순서로 보정 재적용.
3. **폐기/절단은 형제 그룹의 뒤에서부터만** (형제 3을 남기고 1을 빼면
   보정 재구성 불가). 확장 안 하는 것(위상 선택)은 자유.
4. 보류-재확장으로 형제가 여러 forward에 걸쳐 생기면 **부모별 WOR
   체인(뽑힌 집합+순서)을 이어간다.**

**체인 퇴화(fast path, 핵심 회귀 기준)**: fanout=1/root, R=B_s로 두면
현행 무분기 체인과 동일해야 한다 — 기존 sampler 호출을 그대로 쓰는
별도 fast path로 만들어 **RNG 소비까지 bit-identical** 보장 (§10
go/no-go ②).

**B>1 호환**: per-seq (B_s, W, F_total) 동일 상수 — budget-합-상수
불변량의 일반화. seq별 선택은 dim-1 topk (docs/duet/13 M3 패턴).

## 7. verify 측 설계 (v4 — SGLang 참조)

**SGLang EAGLE 구현에서 가져오는 구조** (조사: docs.sglang.io
speculative_decoding, sglang-jax 문서):

| SGLang 개념 | 내용 | 우리 대응 |
|---|---|---|
| `speculative_num_steps` / `eagle_topk` / `num_draft_tokens` | 깊이 / 스텝당 분기 / **rerank 후 최종 검증 용량(고정)** | F_total / fanout 배분 / **N_v** — "만드는 트리"와 "검증받는 트리"를 분리하고 후자를 고정 용량으로 잘라내는 구조가 동일. 단 우리 절단은 형제-prefix 규칙(§6-③) 준수 |
| tree verify = 평탄화 토큰 + **custom tree mask** | 트리를 1차원으로 펴고 조상 관계를 mask로 전달; **dense 명시 mask는 느리므로** packed/bitmask 커널 필요 | draft가 이미 쓰는 FlashInfer packed bitmask 경로를 verify capture에 이식 (F1) — sm_86에서 검증된 동일 커널 패밀리. 외부 전용 커널(DeFT 등)은 SwiftSpec 전례(sm_90 전용) 있어 2차 옵션 |
| KV: 추측 토큰마다 **slot 선할당**, 기각 토큰은 PADDING_SLOT_ID(-1)로 쓰기 억제, 수락 경로만 유지 | 토큰-단위 paged pool이라 가능 | 우리 pool은 **블록(32토큰)-단위**라 토큰-단위 유지 불가 → **scratch slot 검증 후 accepted path만 canonical 위치로 복사** (D5-a). 복사량 ≈ 수락경로 ≤5토큰 × 80층 × rank당 1KB/층 ≈ 400KB/rank — 수십 µs (E2③ 실측) |

### 7.1 트리 어텐션과 응답 스키마
hit 응답 = 명중 root의 서브트리:

```
tok[N_v], parent_idx[N_v], sibling_order[N_v],
parent_q_ref[N_v]  (각 노드가 어느 부모 분포에서 샘플됐는지의 색인),
parent_q_logits[U, V]  (고유 부모 분포 U개 — 중복 전송 회피)
```

- **q_eff 재구성 공유-함수 원칙**: verifier가 수락 검정에 쓰는 q_eff는
  draft가 샘플에 쓴 것과 동일해야 하며, temperature·sampler_x 처리를
  **production sampler와 같은 함수로** 재구성한다 (현행 체인 verify가
  apply_sampler_x_rescaling을 공유하는 것과 동일 원칙 — sampler_x
  구현의 F+1 처리 [async_spec_helpers.py:131]까지 함수 공유로 자동
  일치). E2⑦ parity 검사로 확인 (temperature별, sampler_x on/off,
  비복원 첫 샘플 = 기존 단일 샘플 경로 일치).
- verify rows = N_v+1 (+1은 recovery). mask 값은 parent_idx에서 매
  step 계산해 packed bitmask 버퍼에 주입 — capture는 **N_v bucket별**
  (§7.4).

### 7.2 무손실 트리 수락
깊이별·형제-순서 순차 기각 샘플링:

```
pos p의 형제 x₁..x_m (기록된 순서; q = q_eff, p = target 분포):
  for j in 1..m:
    r ~ U(0,1);  r < p(x_j)/q(x_j) → x_j 수락, x_j의 자식들로 진행
    else: p ← norm((p − q)_+);  q ← q에서 x_j 제거·재정규화
  전원 기각 → recovery ~ p (보정된 분포)   # 무손실 보존
```
검증: 작은 vocab 전수(exhaustive) 테스트로 출력 분포 = target 분포
정확 일치 (§10 go/no-go ①) + 체인 퇴화에서 현행 verify()와 동일.

### 7.3 KV (D5 — v4 잠정 확정: scratch + 복사)
SGLang식 토큰-단위 slot 유지는 우리 블록-단위 pool과 안 맞으므로,
형제 행들은 고유 scratch slot에 KV를 쓰고 (`kvcache_block_size ≥
2k+2` 여유 활용) 수락 확정 후 accepted path만 canonical 연속 위치로
복사한다. 복사 비용은 E2③ 실측으로 최종 확정.

### 7.4 verify bucket (N_v 가변성 대응)
재배분 때문에 root별 서브트리 크기가 다르다 (1노드~W노드). 단일 최대
N_v 패딩은 작은 hit에 행당 1.9ms 낭비 → **bucket capture**:

```
P1 체인:      K1+1 행  (기존 duet_verify_k1 — 불변)
P2 체인:      K2+1 행  (기존 duet_verify_k2 — **그대로 보존, fast path**)
P2 트리:      duet_verify_tree_n4 / n6 / n8  (신규 family)
              (N_v=10은 E1/E2가 명확히 지지할 때만)
```

**기존 체인 그래프를 트리 그래프로 대체하지 않는다** — 같은 행수라도
트리(branch mask)와 체인(causal)은 다른 그래프다. 검사 두 겹: gate
OFF에서 기존과 bit-identical + gate ON에서 체인-퇴화 topology를 넣은
트리 그래프가 체인 fast path와 RNG·출력 동일. custom mask × CG 패딩
조합의 OOB 사례가 보고된 바 있으므로 wrapper batch ≠ real batch인
bucket 경계 검사를 T3 테스트에 포함.
기존 valid_k 기반 k1/k2 이중 bucket dispatch의 축을 늘리는 구조.
캐시 생성 시 root별 응답 view(절단 결과)를 미리 계산해 hit 임계
경로에서 트리 정렬을 하지 않는다. capture 메모리 증가는 E2④ 실측.

## 8. 미결 설계 결정

| ID | 질문 | 상태 |
|---|---|---|
| D5 | KV 처리 | **잠정 (a) scratch+복사** (§7.3, SGLang 대비 근거) — E2③ 실측으로 확정 |
| D6 | 응답 절단 | value 우선 + **조상 폐포·형제-prefix 보존** (규칙 ③) — bucket {4,6,8} 중 선택 |
| D7 | 탐색 공간 초기값 | W=10 고정, F_total=D=4 우선(D=5 후속), R ∈ {4,6,8,10}, N_v ∈ {4,6,8} — E1이 결정 |
| ~~D9~~ | ~~value의 π̂/â 형태~~ | **해소 (결정 ③)**: π̂ = 라이브 P_iv (위치별 calibration은 E0 검증, 필요시 라이브 보정계수) |

## 9. 관문 실험 — 2단 게이트 (G0 연구 타당성 / G1 채택)

**G0 (구현 전, 전부 green이어야 T1 착수)**: E0 계측(유일한 엔진 예외
P0) + E1 standalone 시뮬레이터 + E2 **standalone** 마이크로벤치 —
E2의 mask/KV/수락 프로토타입은 **엔진 밖 별도 하네스**(FlashInfer·
torch 직접 사용)로 만들어 "엔진 코드 금지" 원칙과 충돌하지 않는다.
**G1 (T1-T3 구현 후, 채택 판정)**: §10 go/no-go 4조건.

- **E0 — calibration (이중 trace)** (유일 예외 — 별도 승인 후 선행
  구현, §10 P0): target 30줄만으로는 부족 — **hit 판정은 dedup 후
  살아남은 root 기준**이므로 [draft_runner.py:1592-1597] draft 쪽
  trace 병행:
  - target: step, seq, valid_k, temp, 전체 wire 후보의
    (pos, tok, P_iv, rank), 실제 결과 (reject_pos, recovery_tok)
  - draft: dedup 후 retained roots + 원 wire rank + P1 중복 여부;
    다음 응답의 phase_source·accepted length·명중 root rank
  - offline 재생용 prompt ID/토큰 prefix
  **판정 지표**: root coverage Recall@R (R∈{4,6,8,10}), rank별 hit
  확률, score/rank 캘리브레이션 (ECE·Brier), rank별 L_p2, {proxy
  prior, rank prior, uniform, confidence-only}의 기대-TPS 직접 비교.
- **E1 — offline 트리 시뮬레이션** (엔진 무수정): E0 덤프 + HF 재생.
  objective는 L_p2가 아니라 **predicted TPS** = `기대 출력 토큰 /
  파이프라인 주기(target·draft·NCCL 임계경로 max)` (draft 시간 모델
  포함 — 균형조건 ①②). **비교군 5종 필수**: ⓐ 현행 체인, ⓑ 사전
  고정 π̂-비례 배분(정적), ⓒ 동적+level, ⓓ 동적+frontier, ⓔ oracle
  상한(사후 최적 배분). 판정 규칙: ⓓ−ⓒ < 1%p면 level을 기본값으로
  (결정 ①). **full step-status trace replay** — R(root 수) 변경은
  P2 hit뿐 아니라 P1/miss/JIT 전이 전체를 바꾸므로 조건부-P2만
  계산하지 않는다. 컨트롤러 전체(위상 할당·절단 포함)를 작은 vocab
  전수 열거로 무손실 검증 (D10류 편향은 verifier 단독 테스트로는
  안 잡힘).
- **E2 — 마이크로벤치** (6항목): ① T_verify(N_v), N_v∈{4,6,8,10}
  직접 실측 (§2의 c_row 가정 재확정), ② packed-mask verify capture
  프로토타입 replay 시간 + tree sample/accept 단계의 GPU·CPU sync
  비용, ③ scratch→canonical KV 복사 비용 (TP rank별), ④ bucket 추가
  CG capture 메모리, ⑤ 동적 조상 mask 생성 비용 (W=10), ⑥ score
  비트-pack 시 wire 변화 확인 (기준선 832KB — F4), ⑦ q_eff parity —
  draft 샘플측과 verifier 재구성측의 분포 일치 (temperature별,
  sampler_x on/off, 비복원 첫 샘플 = 기존 단일 샘플 경로; §7.1의
  공유-함수 원칙 검증). **판정**: 합계 오버헤드 < E1 기대 이득의 1/2
  (안전계수 2 — 노이즈·모델 오차 흡수). 전 항목 standalone (G0).

## 10. 마일스톤과 go/no-go

**P0 (관문 전, 별도 승인 대상)**: E0 이중 trace 게이트 — 덤프 스키마
유닛테스트, OFF 시 TPS 무영향. **이것만이 관문 전 허용 코드.**

**T1-T5 (관문 green + 설계 승인 후)**: 방식은 docs/duet/13 M1-M6 관례
(단계별 유닛테스트 + B=1 회귀 스모크 + 상세 커밋; 가드는 `python -O`
생존형 명시 raise — docs/duet/14 R1).

| 단계 | 내용 | 검증 |
|---|---|---|
| T1 | frontier rollout + WOR 샘플 + 장부 + 캐시 구조 | CPU 참조 대비 동일성; **fast path(fanout=1/root, R=B_s) RNG까지 bit-identical** |
| T2 | 응답 view/wire (tok+parent+순서+logits, score pack) + F7 산식 재검토 | wire 왕복 테스트; payload 실측 |
| T3 | verify: packed-mask bucket capture + 트리 수락 + scratch KV | **작은 vocab 전수 분포-일치 테스트**; 체인 퇴화 동일성 |
| T4 | B=1 E2E + champion A/B (**5-rep 인터리브** — final_rematch 관례; ±1.5 tok/s 노이즈에서 +3% 판정에 3-rep band-clear는 불안정) | §2 목표 판정 |
| T5 | B>1 호환 | B=2 스모크; 기존 테스트(M1-M6 38 + jit_subset 5) 회귀 |

**go/no-go (엔진 구현 착수 아닌 **채택** 기준 — 외부 리뷰 수용)**:
① 작은 vocab 전수 테스트에서 target 분포와 정확 일치, ② chain fast
path가 RNG 소비까지 bit-identical, ③ N_v≤8에서 draft 오버헤드 포함
predicted TPS ≥ champion +3%, ④ 마이크로벤치 기대 이득 ≥ 오버헤드×2.
**N_v≤8에서 미달 시**: 트리 확대 대신 P2-context distillation /
proxy-conditioned draft adapter로 전환 검토 — 단 이는 **training이
필요**해 지금까지의 training-free 노선을 벗어나는 스코프 결정이므로
그 시점에 별도 판단.

## 11. 위험 목록

1. **B=1 proxy_wait 여유 ~9ms** — draft 쪽 매-step 비용(forest·mask·
   샘플링) 증가가 조건②를 깨면 spec_wait 증가. 완화: exit 위치 병행
   재조정; E1이 draft 시간 모델 포함.
2. **wire/top_k 산식 연쇄** (F7) — T2에서 일괄 재검토.
3. **선택 효과** (원인 3, 가설) — E0의 문맥별 수락률로 비중 실측.
4. **graph_pre 이상(+19%/layer) 간섭** — c_row 가정은 E2①의
   T_verify(N_v) 직접 실측으로 대체.
5. **체인 퇴화 동일성** — fast path가 T1/T3의 핵심 회귀 기준.
6. ~~WOR 장부 복잡도~~ — **D11(single-shot fanout)로 해소** (2026-08-04):
   비복원 커서가 forward를 넘지 않으므로 부모별 체인 유지 불필요.
7. **트리 행 한계비용 미확정** — 1.9ms/행은 체인 스윕에서 온 값.
   E2①이 트리 행에서 재실측 (더 싸면 채산 문턱 하락 — 유리).

## 12. 후속 과제 (TODO — 본 설계 범위 밖, 잊지 않기 위해 기록)

1. **P1 트리화** (사용자 지정, 2026-08-04): P1도 현재 "위치별 seed
   fan-out 후 무분기 연속" 구조라 P2와 같은 한계를 가진다. P2-tree의
   기계(priority·비복원 샘플·tree verify·bucket)가 검증되면 P1에
   일반화한다 — P1은 hit의 2/3를 담당하므로 (P1 hit 0.53 vs P2 0.27)
   기대 효과가 P2보다 클 수 있다. 착수 조건: P2-tree G1 통과 후.
2. **위치별 calibration 보정계수**: 라이브 P_iv 구현·실측 후, E0/실측
   calibration 곡선이 비균일 편향을 보일 때만 (결정 ③의 유보 사항).
3. **확장 priority의 한계비용 항** (bucket 경계 비용): acceptance
   rate 우선 방침(결정 ②)에 따라 보류 — E1 oracle 대비 격차가 크면
   재검토.
4. **budget 사전 고정 배분**: E1 비교군 ⓑ의 성적이 동적 배분과
   대등하면 단순성을 위해 재고할 수 있음.
5. **B>1 일반화**: per-seq 상수 (B_s, W, F_total, N_v bucket)로
   불변량 일반화 — T5의 호환 확인 후, 최적화는 별도 캠페인.
