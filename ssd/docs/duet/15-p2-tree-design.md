# 15 — P2 동적 트리 (P2-Tree) 설계 문서

**작성**: 2026-08-02 (v3 — 검증 패스 반영: 수치 대조 7건 + 신입 독자
리뷰 12건 수정). **상태: 설계 단계 — 구현 착수 금지.** 사용자 지시:
"모든 건 완벽하게 설계가 이루어진 이후에 시작. 충분한 검증·설계·실험이
진행된 이후에 시작할 수 있도록." §9의 관문 실험(E0-E2)이 전부 green이고
설계가 승인되기 전에는 엔진 코드를 수정하지 않는다. **단 하나의 예외**:
E0 자체가 요구하는 계측 덤프 게이트(§9 E0, ~30줄)는 관문 이전에
필요하므로 **별도 승인을 받아 선행 구현**한다(§10의 P0) — 이 예외
외에는 게이트류도 금지. 이 문서가 단일 소스다.

**읽는 법**: 이 프로젝트를 처음 보는 사람은 §1(배경)→§2(목표)부터.
설계 리뷰어는 §4(확정 결정)→§5(사실 확인)→§6-8(설계)→§9(관문). 각주의
`[파일:라인]`은 2026-08-02 HEAD 기준.

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
체인을 즉시 응답하고(대기 최소), **miss**면 그 자리에서 체인을
만들어(JIT) 응답한다. 베이스라인 **C** = 이 async 엔진 원형으로, 노브는
체인 깊이 k와 결과-예측 폭 f뿐이다. 이 문서의 처리량 수치는 모두
**decode TPS (tok/s)** — 이 프로젝트의 절대 지표다.

### 1.2 DUET이 더한 것: early-exit proxy 2단 draft (split-K1/K2)

**DUET**(이 fork가 더한 기법의 프로젝트명)의 아이디어: target이 verify
forward를 도는 **도중**, exit layer(현 B=1 최적 구성인 champion —
E9K24_jit, §1.3 — 은 56/80)의 중간 logits로 "이번 검증이 어떻게
끝날지"를 target 스스로 예측해 draft에 미리 알려준다. draft는 두
단계로 캐시를 채운다:

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
| seed | P2 체인의 시작 (위치, 토큰) — proxy가 고른 예상 결과 |
| budget (=B_s) | P2 seed 총수 = `pfo × (K1+1)` [config.py:227-241]. **상수** — 매 step, 모든 seq 동일. 본 문서에서 B_s로 표기 |
| valid_k | 캐시 행의 실제 깊이 ∈ {K1, K2}. verify/glue CG 버킷 선택 기준 |
| glue decode | 응답 확정된 체인을 draft의 정식 KV에 실체화하는 varlen forward |
| JIT | miss 시 즉석 체인 생성 (`jit_speculate`; JIT-short 게이트 시 K2 깊이) |
| wire | target rank0 → draft NCCL 페이로드: `(chosen_pos, chosen_tok)` 점수 내림차순 `wire_N`개 |
| Policy B | P2 seed 선택 정책 (아래 수식) |
| **CG** | **CUDA Graph** — 커널 시퀀스를 고정 shape으로 capture해 두고 replay하는 최적화. replay는 텐서 **shape이 상수**여야 하므로, 이 시스템의 모든 동적성은 "shape은 고정, 내용(mask/index 값)만 변동"으로 구현된다. D1의 존재 이유 |
| band-clear | A/B 판정 기준: 3-rep 인터리브 반복에서 **한쪽의 최악 rep이 상대의 최고 rep보다 좋아** 분산 구간이 겹치지 않는 승리. 겹치면 "동률" |
| spec_wait | target이 검증 결과를 draft에 보내고 다음 추측 응답을 받을 때까지 블로킹된 시간 (wire 왕복 + draft 응답 처리) |
| proxy_wait | draft가 P1을 끝내고 target의 exit proxy 도착을 기다리는 블로킹 시간 |

**Policy B 수식** [verifier.py `_compute_and_send_proxy`; docs/duet/05]:
verify 도중 exit에서 p^E(early-exit 분포), p^D(draft 분포)를 얻어

- 위치별 근사 수락확률 **α̂_i = min(1, p_i^E(y_i)/p_i^D(y_i))**
- 첫-거부 분포 **ĥ**: h_0 = 1−α̂_0, h_i = (∏_{j<i}α̂_j)(1−α̂_i),
  h_K = ∏α̂_j (전부-수락)
- 위치별 보정 분포 **corr**: i<K는 residual **[p^E − p^D]_+** (검증 중
  토큰 제외) top_k; **i=K(전부-수락 위치)는 residual이 아니라
  p^E[K,:] 전체 분포의 top_k** [verifier.py:437-441]
- **P_iv = ĥ_i × corr[i,v]** → 전 위치 flatten 후 글로벌
  topk(wire_N) → 송신. draft는 P1 트리와 dedup 후 **앞에서부터 정확히
  B_s개**를 seed로 채택 [draft_runner.py `_select_proxy_sourced_tokens_unified`].

**캐시 키** = (seq_id, fan_idx, recovery_token). hit = 실제 검증 결과와
일치하는 키의 행이 존재 — 그 행의 체인이 다음 verify 입력이 된다.

**한 step의 타임라인** (B=32 실측 그림:
`experiments/proxy_async_overlap/b_gt1/bscale32/overlap_profile/duet_k2x2_prof/timeline_step121_mixed.png`):

```
target(GPU0-3): [spec_wait]→[graph_pre: 층0..exit]→[exit+proxy send]→[graph_post]→[sample]
draft (GPU4)  : [respond(조회/JIT)]→[glue]→[P1 rollout]→[proxy_wait...]→[P2 rollout]→[cache]
                        ▲ 응답이 먼저, glue는 그 뒤     ▲ 조건① 랑데부      ▲ 조건② 랑데부
```

**균형 2조건** (사용자 규칙, feedback-duet-pipeline-balance): ① P1 종료
≈ proxy send 도착, ② draft 종료 ≈ target 종료. B=1 champion은
proxy_wait ~9ms로 거의 균형점이다. (B=32에선 조건①이 137ms 깨져
있었고, exit를 당겨 회복해도 풀린 시간을 사줄 소비처 — P2 비중 — 가
없어 TPS가 불변이었다 [balance32]. 본 설계의 트리는 **후속으로 B>1에
일반화할 때** 그 소비처가 될 수 있으나, 본 설계의 타겟은 B=1이다, §2.)

### 1.3 현재 성적 — 우리가 어디에 서 있나

2026-07까지의 캠페인 결론 (상세: docs/duet/12, 13, bscale32/REPORT.md;
수치 단위는 decode tok/s):

- **B=1**: champion **E9K24_jit** (K1=9 K2=4, exit=56, dfo=2 pfo=1,
  jit-short) **81.91 vs C 81.52 (+0.5%)** — 동률(band-clear 아님,
  5-rep 중 4승). B∈{2,4}의 동률과 함께 지지 않는 세 지점 중
  하나이며, **champion 형상이 유지되는 유일한 지점**.
- **B>1 공정화 반전**: C에게도 B별 형상 재튜닝을 허용하자(k* 7→5→3→
  3→2→2) 기존 "+6.9/+14.8/+26.9% (B=2/4/8) band-clear 승리"가 소멸 —
  B∈{2,4} 동률, **B∈{8,16,32}는 C-opt에 band-clear 패배**
  (−3.7/−2.5/−4.1%).
- 메커니즘 — **깊이-폭 frontier는 시스템 불변**: verify 폭(행수)의
  시간 비용은 B에 선형, 깊이의 토큰 가치는 B-불변이라는 비용/가치
  곡선이 C와 DUET에 동일하게 적용된다 — 형상을 공정하게 튜닝하면 둘 다
  같은 곡선 위에 올라선다.
- **matched-shape 검증**: 같은 형상이면 tok/step이 C와 동일 (2.38 vs
  2.40) — 수락 로직 무결. 남은 격차는 전부 verify 내부 시간
  (graph_pre가 C 대비 +19%/layer — 원인 미해결, 별도 트랙).
- **finding 5a (남은 알고리즘 레버)**: P2 hit 시 수락 길이(**L_p2** —
  P2 캐시 행이 검증에서 연속 수락되는 평균 토큰 수) ≈ 1.8이 breakeven
  2.6에 미달 — champion의 마지막 큰 레버로 지목된 상태. **이것이 본
  작업의 직접 동기다.**

### 1.4 문제 정의 — P2 hit의 수락 길이가 낮다

실측 (docs/duet/12 m6_fix, bscale32; E[AL|P2 hit] = L_p2로 표기):

| 지표 | 값 | 의미 |
|---|---|---|
| L_p2 | **1.73 / 1.81 / 1.63** (B=1/2/4, champion 형상) | 상한 K2=4의 절반 이하; breakeven 2.6 미달 |
| P2 위치당 수락률 | **≈0.6~0.7** (실측: K2=1 형상에서 0.61~0.68 [RESULTS_balance32]; champion L_p2 1.73 역산 ≈0.70. 정밀값은 E0에서 실측) | P1(≈0.7~0.8)보다 낮음 |
| P2 hit rate | 0.269 (B≤4, B-불변) | 전체 hit(0.82)의 1/3을 P2가 담당 |
| finding 1 (wrong currency) | hit↑ ≠ tok↑ | hit 자체는 토큰을 나르지 않음 — **hit 시 길이**가 화폐 |

**원인 3가지**:

1. **무분기 체인** (구조적, 이 설계의 공략 대상): P2 rollout은 seed
   선택에서만 분기하고, 이후 K2 depth 동안 행마다 토큰 1개씩만 잇는다
   — `_decode_tree`가 `for depth in range(K2)` 루프에서 행당 1 토큰
   샘플 [draft_runner.py:1292-1299, 1250-1253], tree-decode mask도
   행별 identity 대각선뿐 [cudagraph_helpers.py:396-397]. 2번째
   토큰이 틀리면 그대로 끝.
2. **off-policy 연속**: seed 이후는 draft 단독 rollout이라 target
   시각(proxy)의 이점이 seed 한 칸에서 끝난다.
3. **선택 효과** (가설 — E0 덤프의 문맥별 수락률로 검증 예정): P2
   hit은 "방금 거부가 난" 문맥에서 발화하므로 본질적으로 수락이
   어려운 상태의 표본일 수 있다.

원인 1이 구조적이고 고칠 수 있는 부분이다: **2번째 이후 토큰을 트리로
헤징하면** 첫 연속이 틀려도 형제 가지가 살아남는다.

---

## 2. 목표와 성공 기준

- **정량 목표**: B=1 champion 레짐에서 **L_p2 1.73 → 2.6+
  (breakeven)** 로 올려 TPS 동률(+0.5%)을 band-clear 우위로 전환한다.
  breakeven 2.6의 뜻 [docs/duet/12 finding 5a]: P2 행들이 점유하는
  verify 행 비용을 P2가 나르는 토큰 가치가 정확히 상쇄하는 수락 길이.
- **채산 부등식** (go/no-go 판정식, B=1 기준 수치로 구체화). 트리
  응답은 P2 hit step에서만 나가므로 **추가 verify 행 비용과 수락 길이
  이득이 같은 step 집합에서 발생 → hit율이 소거**되고 식이 단순해진다:

  `ΔL_p2 × v_tok > Δrows × c_row`   즉   `ΔL_p2 > Δrows × (c_row/v_tok) ≈ Δrows × 0.12`

  여기서 **v_tok** = 토큰 1개의 시간 가치 ≈ t_step/tok_per_step ≈
  56ms/3.6 ≈ **15.5ms/tok** (B=1 champion 실측), **c_row** = verify
  행 1개의 한계 비용 — B=1 실측 **1.9ms/행** [docs/duet/12 frontier
  finding; B≥4에서는 별도 실측 B×2.25ms/행이 있으나 본 설계는 B=1
  타겟이므로 1.9를 쓴다]. 예: P2-hit 응답을 K2+1=5행에서 N_v+1=11행
  으로 6행 늘리면 **ΔL_p2 > 6×0.12 ≈ +0.74**를 벌어야 채산 — 상한
  여유(1.73→K2 상향 포함 시 cap ~4+) 안이며, E1이 이 식으로 형상을
  고른다. (전체 TPS 환산 시에는 P2hit율 0.269가 이득·비용 양쪽에
  곱해질 뿐 판정은 불변.)
- **draft 시간 예산**: 트리 rollout의 draft 시간 증가는 조건①의
  proxy_wait 여유(B=1 ~9ms)로 흡수하되, 부족하면 **exit layer 위치를
  함께 재조정**해 조건①②를 재균형한다 (§11.1) — 여유가 작으므로
  E1이 draft 시간 모델을 포함해야 한다.
- **비목표**: B>1 최적화 (구현 호환만 — B≥16은 대부분 miss 레짐이라
  async 이점 소멸, 사용자 결정), P1 트리화 (후속 검토), 수락 분포
  변경 (무손실성 유지는 절대 조건).

---

## 3. 제안 개요

```
현재 P2:  seed₁(s=0.4) ── t ── t          seed별 1행, 무분기 체인
          seed₂(s=0.3) ── t ── t          (B_s = seed 수 = 상수)
          seed₃(s=0.2) ── t ── t

제안 P2:  seed₁(s=0.4) ─┬ t ─┬ t         레벨별 행수 W_l 고정(CG),
                        │    └ t         어떤 노드를 확장할지는
                        └ t ── t         value = log s_seed + Σlog c
          seed₂(s=0.3) ── t ── t         로 매 forward 전체-레벨 top-W 선택
          seed₃(s=0.2) ── (확장 0)       → 점수 낮은 seed는 자식 0개
```

**관련 연구와 좌표** (2026-08-02 검증):

- **DFVG** (ASPLOS'26, Draft-on-FPGA/Verify-on-GPU): confidence 임계
  ε 기반 **sparse 동적 트리** + draft/verify 오버랩. 우리와 정신이
  가장 가깝지만 FPGA라 shape 제약이 없다 — 아이디어만 이식.
- **EAGLE-2** (arXiv:2406.16858): 노드 가치 **V = ∏ confidence**
  (경로 수락확률 근사 — draft가 well-calibrated라는 관찰), 레벨당
  top-k(=10) 확장, depth 6, 총 48-60 노드, 이후 rerank(가치 단조성이
  연결성 보장). **log는 원문에 등장하지 않음** — raw 곱 사용. 우리
  방식의 CG-호환 원형.
- **DySpec** (arXiv:2410.11744): draft 확률-수락률 상관을 근거로 한
  greedy 동적 확장 — D2의 이론 근거.

**우리의 차별점**: ① 루트가 단일 체인이 아니라 **proxy가 고른 seed
숲** (결과-예측별 트리), ② seed에 draft confidence보다 강한 사전점수
(target 자신의 exit 분포 기반 ĥ×P_iv)가 있음 — 단 이 우위는 **E0로
실증해야 할 가설**, ③ 레벨 폭 고정이라 rerank 불필요.

---

## 4. 확정 설계 결정 (사용자, 2026-08-02)

- **D1 — shape 완전 고정, 재배분 동적**: 매 forward 입력 행수는 CG
  재사용을 위해 상수. seed별 확장 수는 점수로 재배분 (seed 1이 2개,
  seed 2가 0개 식). **가능함이 확인됨** — P2 seed 레벨이 이미 같은
  기계다: fan_out 합=상수, fan_idx/mask는 값만 변동, GPU→CPU 동기화는
  기존 `.tolist()` 1회 [draft_runner.py:1520-1532]. 이 기계를 레벨마다
  반복하면 새 CG family 0개.
- **D2 — value = proxy score × confidence**: 단 proxy score의 hit
  예측력이 실증 안 된 상태 — "budget 때문에 들어간 낮은 점수 seed가
  실제로도 hit 확률이 낮은가"를 **E0가 먼저 판정**한다. 낮으면
  confidence 단독으로 후퇴.
- **D3 — log 공간**: EAGLE-2 원문은 raw 곱(∏c)이고 log는 등장하지
  않음 (원문 확인). log-합은 top-k 선택에 **동치**(단조변환)이며
  수치 안전 + proxy 점수와의 합성이 깔끔: `value(n) = log s_seed +
  Σ_path log c_j`. 채택 (비용 0; beam search 관례).
- **D4 — 동적 선택 오버헤드 최소**: 레벨별 선택은 GPU topk 1회, 신규
  GPU→CPU 동기화 0회 원칙 (M1-M3 독트린 = B>1 batch화 때 확립된
  "동기화 횟수 불변" 규칙, docs/duet/13).

## 5. 사실 확인 (2026-08-02 코드/논문 검증; file:line은 당시 HEAD)

- **F1 — "verify tree-attention 이미 있음"은 절반만 참**: 트리 어텐션
  기계(`get_custom_mask`, packed mask)는 **draft tree-decode 전용**
  [mask_helpers.py:248; cudagraph_helpers.py:318-410 (np.packbits
  사전계산), :456 (FlashInfer `_custom_mask_buf` 주입)]. target의
  `capture_duet_verify_cudagraph`는 cu_seqlens 기반 **causal varlen
  체인 전용 — custom mask 인자 자체가 없다** [cudagraph_helpers.py:
  1065, 1107-1120]. 트리 verify = draft의 packed-mask 기계를 verify
  CG로 **이식**하는 신규 공사 (발명은 아님).
- **F2 — 무손실 트리 수락은 신규 알고리즘**: 현행 verify()는
  speculations [B, K+1] — **위치당 후보 1개인 체인** — 의 순차 ratio
  test [utils/verify.py:37, 119-132] + 거부 시 residual recovery
  [:174-180]. 트리는 같은 위치의 sibling들을 **기각할 때마다 q를
  without-replacement 보정하며 순차 검증** (SpecInfer/EAGLE 계열)해야
  분포가 보존된다. §7.2에 의사코드.
- **F3 — KV slot 경합**: 같은 depth의 sibling verify 행들이 같은
  position slot을 경합. 후보안 (a) scratch 검증 후 accepted path만
  복사, (b) `kvcache_block_size ≥ 2k+2` 여유로 sibling slot 예약.
  → 미결 D5.
- **F4 — 응답/wire (v3 정정)**: 현행 hit-응답 logits wire는 유효
  깊이(valid_k)가 아니라 **speculate_k 폭으로 전송**된다 — champion
  기준 [B, 13, V=32000] fp16 ≈ **832KB/step** [speculator_async.py:
  109 recv 버퍼; draft_runner.py:732 송신]. 트리 응답(**N_v** = hit
  시 응답하는 서브트리 노드 수, 상수)을 N_v~10으로 잡으면 tok+parent+
  logits ≈ 640KB — **현행보다 오히려 작다**. wire 크기는 리스크가
  아니라 확인 항목 (E2①, 기준선 832KB).
- **F5 — 응답 크기 균일화**: seed별 서브트리 크기가 재배분으로
  달라지므로 hit 응답 트리를 상수 N_v로 절단/패딩 필요 → 미결 D6.
  (절단은 후보 축소일 뿐 수락 분포는 불변 — 무손실성과 무관.)
- **F6 — 기존 trace 게이트는 shape만 로깅** [verifier.py:119, 463]
  — E0에는 신규 덤프 게이트 필요 (§9 E0, 헤더의 유일 예외).
- **F7 — top_k 자동 상향과의 상호작용**: config가 proxy top_k를
  `max(total_budget + p1_max + 2, ceil(wire_N/(K_min+1)))` 로 자동
  상향한다 [config.py:445-449]. 트리화로 budget 의미가 바뀌면 두 항
  모두 재검토 필요 (scheduler 예약 [scheduler.py:58] 포함).

## 6. 트리 구성 알고리즘 (draft 쪽)

- **seed 레벨** (현행 유지): Policy B가 B_s개 seed 선택, 점수 s_i.
- **rollout 레벨 l = 1..K2**: 레벨 폭 W_l (config 상수). 이전 레벨
  생존 노드의 자식 후보(draft logits top-c, c = 노드당 자식 후보 수)에
  대해 `value = log s_seed + Σ_path log c_j` 를 계산, **레벨 전체에서
  top-W_l 선택** (GPU topk 1회) → fan_out 재배분. fan_idx/mask 갱신은
  기존 P2 기계 [draft_runner.py:1488-1532] 재사용.
- **체인 퇴화 조건** (핵심 회귀 기준): **c=1 이고 W_l=B_s**이면 모든
  seed가 자식 1개씩 이어가는 현행 무분기 체인과 완전히 동일해진다 —
  이 설정에서 현행 구현과 bit-identical해야 한다 (§11.5). (W_l=1은
  퇴화가 아니라 전 seed 통합 단일 체인이 됨에 주의.)
- **워크스루 예시** (B_s=3, c=2, W_1=4, K2=2, W_2=4):

```
seed:  s₁=0.5  s₂=0.3  s₃=0.2                       (행 3개, 현행과 동일)
L1 후보: s₁의 자식 c=(0.6,0.3), s₂의 (0.7,0.2), s₃의 (0.4,...)
  value: 0.30, 0.15 | 0.21, 0.06 | 0.08, ...
  top-4 → s₁에서 2개, s₂에서 2개, s₃ 0개        (행 4개 — 폭 고정)
L2 후보: 위 4개의 자식들 → 같은 방식 top-4        (행 4개)
총 노드 N = 3 + 4 + 4 = 11 = 상수 → CG shape 불변
```

- **B>1 호환**: per-seq (B_s, W_l) 동일 상수 → 기존 budget-합-상수
  불변량의 일반화. seq별 선택은 dim-1 topk로 병렬 (docs/duet/13 M3의
  selector 벡터화 패턴).

## 7. verify 측 설계

### 7.1 트리 어텐션 (F1 이식)
hit 응답 = 명중 seed의 서브트리 N_v 노드 (D6 절단). verify rows =
N_v+1 (+1은 recovery/bonus 위치). draft tree-decode의 packed-mask
파이프라인(numpy 사전계산 → packbits → `_custom_mask_buf` copy)을
duet_verify capture에 이식 — mask 값은 응답의 parent 배열에서 매 step
계산 (조상 경로만 attend).

### 7.2 무손실 트리 수락 (F2 신규)
깊이별 sibling 순차 기각 샘플링 (SpecInfer 계열):

```
pos p에서 후보 형제 x₁..x_m (draft 분포 q, target 분포 p):
  for j in 1..m:
    r ~ U(0,1);  if r < p(x_j)/q(x_j): accept x_j → p+1로 진행 (x_j의 자식들로)
    else: p ← normalize(max(p − q, 0));  q ← q에서 x_j 제거·재정규화
  전부 기각되면: recovery ~ p (보정된 분포)  → 무손실 보존
```
현행 verify()의 클램프(valid_k)·recovery 구조는 유지하고 후보 루프만
추가된다. 검증은 CPU 참조 구현 대비 분포 동일성 유닛테스트로 (체인
퇴화 c=1·W_l=B_s에서 현행 verify()와 동일 출력 포함).

### 7.3 KV (D5 미결)
(a) scratch slot 검증 → accepted path만 canonical 복사: 구현 국소적,
복사 비용 ~path길이×토큰 (수 µs 예상, E2③ 실측). (b) sibling별 slot
예약: 복사 없음, block 소요 증가 + 예약 로직 복잡. **초기 권장 (a)**
— 단순성 우선.

## 8. 미결 설계 결정

| ID | 질문 | 옵션 | 결정 근거가 될 것 |
|---|---|---|---|
| D5 | 트리 verify의 KV 처리 | (a) scratch+복사 / (b) slot 예약 | E2③ 복사 비용 실측 |
| D6 | 응답 트리 크기 N_v와 절단 규칙 | value 상위 N_v (연결성은 value 단조성이 보장 — EAGLE-2 rerank 논리) | E1 형상 탐색 |
| D7 | 형상 {B_s, c, W_1..W_K2} 초기값 | champion 기준 예: B_s=10, c=2, W=[10, 8, 6, 4] 등 | E1 시뮬레이션 |

## 9. 관문 실험 (구현 전 — 전부 green이어야 착수)

- **E0 — proxy score calibration** (헤더의 유일 예외 — 덤프 게이트
  `SSD_DUET_SCORE_TRACE=1`, target 쪽 ~30줄을 **별도 승인 후 관문 전
  선행 구현**, §10 P0): step마다 {chosen (pos,tok), P_iv 점수}와 다음
  step의 실제 결과(거부 위치, recovery 토큰)를 JSONL 덤프 → offline
  join. **산출**: 점수 분위별 hit율 곡선, seed rank vs hit, 점수 vs
  hit 후 L_p2 상관, (원인 3 검증용) 문맥별 수락률. **판정**: 곡선
  단조 → D2 채택 / 평평 → confidence 단독. (B=1 champion 형상, ns=20
  out=256. 덤프도 성능 오염이므로 TPS 측정과 절대 병용 금지 — TPS는
  항상 게이트 OFF로 측정하는 저장소 원칙 그대로.)
- **E1 — offline 트리 시뮬레이션** (엔진 무수정): E0 덤프의 실제
  (문맥, seed, 결과) 표본 + HF로 draft/target 재생 (top-level 분석
  스크립트 전통) → "체인 대신 트리였다면 L_p2가 얼마"를 형상 후보별
  계산 + draft 시간 모델 (§2 예산). **판정**: §2 채산 부등식을
  만족하는 {B_s, c, W_l, N_v} 존재.
- **E2 — 마이크로벤치**: ① wire 실측 (기준선 832KB/step, F4) —
  트리 포맷의 실제 변화와 spec_wait 영향, ② packed-mask verify
  capture 프로토타입 replay 시간, ③ D5(a) KV 복사 비용. **판정**:
  합계 오버헤드 < E1 기대 이득의 1/2 (안전계수 2 — 스캔 노이즈와
  모델 오차 흡수용).

## 10. 마일스톤

**P0 (관문 전, 별도 승인 대상)**: E0 덤프 게이트 구현 — 덤프 스키마
유닛테스트, OFF 시 TPS 무영향 확인. **이것만이 관문 전 허용되는 코드
변경이다.**

**T1-T5 (관문 E0-E2 전부 green + 설계 승인 후에만)**. 방식은 B>1
캠페인(docs/duet/13 M1-M6)의 관례를 따른다: 단계별 유닛테스트 + B=1
회귀 스모크 + 상세 커밋; assert는 `python -O`에서 제거되므로 정합성
가드는 명시적 raise로 (docs/duet/14 R1의 교훈).

| 단계 | 내용 | 검증 |
|---|---|---|
| T1 | draft 트리 rollout (레벨별 top-W 재배분) + 캐시 구조 | CPU 참조 구현 대비 선택 동일성; **체인 퇴화(c=1, W_l=B_s) bit-identical**; B=1 스모크 |
| T2 | 응답/wire 트리 포맷 (tok+parent+logits) + F7 산식 일괄 재검토 | wire 왕복 유닛테스트; payload 실측 |
| T3 | verify packed-mask 이식 + 트리 수락 (§7.2) | 분포 보존 유닛테스트 (체인 퇴화 동일성 포함); -O 생존 가드 |
| T4 | B=1 E2E + champion A/B (3-rep 인터리브) | L_p2/TPS 판정 — §2 목표 |
| T5 | B>1 호환 (불변량 일반화, 게이트) | B=2 스모크; 기존 유닛테스트 전체(M1-M6 38개 + jit_subset 5개) 회귀 |

## 11. 위험 목록

1. **B=1 proxy_wait 여유가 9ms뿐** — 트리 rollout 증가가 draft 종료를
   늦추면 조건② 위반으로 spec_wait가 커진다. 완화: exit layer 위치를
   함께 재조정해 조건①②를 재균형 (§2 draft 시간 예산; E1이 시간
   모델 포함).
2. **wire/top_k 산식 연쇄** (F7): budget 의미 변경이 wire_N·top_k
   자동상향·scheduler 예약에 파급 — T2에서 일괄 재검토.
3. **선택 효과는 트리로 안 고쳐짐** (원인 3, 가설): 트리는 원인 1만
   공략 — E0의 문맥별 수락률로 원인 3의 비중을 실측해 E1 기대치를
   보정한다.
4. **graph_pre 이상(+19%/layer) 미해결과의 간섭**: verify 행 추가
   비용이 이상 구간에 얹히면 c_row=1.9ms 가정이 흔들림 — T4 A/B에서
   실측으로 재확인.
5. **체인 퇴화 동일성**: **c=1, W_l=B_s** 설정이 현행 체인과
   bit-identical해야 한다 (B=1 항등성 원칙의 트리판) — T1/T3의 핵심
   테스트. (§6에 명시했듯 W_l=1은 퇴화 조건이 아니다.)
