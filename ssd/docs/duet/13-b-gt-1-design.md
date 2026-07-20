# 13 — DUET split-K1/K2의 B>1 지원: 설계 + 단계별 구현 기록

**작성일**: 2026-07-18. **동기**: docs/duet/12의 finding 5(b) — B>1은
DUET에게 구조적으로 유리한 홈그라운드다. 근거는 두 가지: (i) draft
forward는 latency-bound(연산량이 아니라 커널 왕복 지연이 지배)라서
여러 seq가 한 forward를 거의 공짜로 공유할 수 있고, (ii) DUET은 seq당
verify row가 26개로 SD-best(48개)의 절반 근처라 batch를 키워도 target
연산 벽에 더 늦게 부딪힌다. B=1 가정이 박혀 있던 코드 전수 감사 결과는
아래 설계 항목들과 config.py L308(낡은 "Policy A" 주석 — 실제 blocker는
단일-seq Policy B 파이프라인과 단일-bucket dispatch였다)에 반영되어 있다.

## 핵심 설계: uniform-width batch + per-seq mask (v1)

감사에서 확인된 중심 사실: DUET이 아닌 기본 SSD는 모든 seq가 **하나의
layout을 공유**하기 때문에 이미 batch가 된다. DUET의 이질성(seq마다
다른 것)은 정확히 세 가지다 — per-seq valid_k ∈ {K1, K2}, per-seq
Policy-B fan-out 분포, per-seq proxy wire 내용. v1 설계는 batch의
**shape(모양)은 전부 공유로 복원**하고, seq별 차이는 전부 mask/index
텐서(replay 시점에 값만 바뀌는 버퍼)로 밀어넣는다. 그 결과 **새 CUDA
graph family가 하나도 필요 없다** — 기존 bs-bucket 축만 쓴다.

1. **valid_k 혼합 → batch 전체를 `vk_max = max(valid_k)`로 dispatch.**
   glue 폭, phase-1 layout(long/short) 선택, verify bucket, TP-broadcast
   스칼라가 전부 vk_max 하나를 쓴다(TP 동기화를 위해 스칼라 하나로
   유지). 짧은 seq는 vk_max까지 패딩되고, 실제 폭은 `valid_k [B]`
   텐서로 따로 전달되어 소비자별로 딱 한 지점에서 강제된다:
   - verify(): `accept_until = min(accept_until, valid_k_i)` 클램프
     (벡터화된 한 줄) — 패딩된 위치는 절대 수락될 수 없다.
   - glue/tree mask: 원래부터 seq별로 만든다(cache_hits_list 루프) —
     seq별 tril 폭이 valid_k_i를 따라간다.
   - 비용: 혼합 batch는 전부 K1 폭으로 verify를 돈다(모든 seq가 짧을
     때만 k2 bucket dispatch — max 검사 한 번이면 됨). v1에서 감수하는
     손실이며, all-short 확률은 B가 클수록 줄어든다.
     **[사후 확인: 이 "vk_max 패딩세"가 B=4 격차의 주범이었고(verdict
     실험), K2=K1 shape으로 갭 자체를 0으로 만드는 것이 처방이었다.]**
2. **hit/miss 혼합 (감사 #5)**: JIT/cache 우선순위를 뒤집는다 — miss가
   하나라도 있으면 **전체 row에 대해 batched JIT을 돌리고**(draft는
   latency-bound라 여분 row가 거의 공짜), **그 다음 hit row를 cache
   값으로 덮어쓴다**. per-seq valid_k = hit이면 row 고유값, miss면 K2
   (jit-short). "JIT-all이 hit들을 깔아뭉개는" 버그를 원천 차단 — 이
   버그가 있으면 B=4에서 전체 step의 ~55%가 hit를 날린다.
3. **Policy B batch화 (감사 #2)**: `_compute_and_send_proxy`에 B 축을
   끝까지 관통시킨다 — accept_probs/h/cumprod [B,K], P_iv
   [B,K+1,top_k], seq별 topk(wire_N) → chosen [B, wire_N]. wire와
   irecv/ring 크기는 2·B·wire_N. `duet_policy.policy_b_from_candidates`
   에 batch 차원 추가(두 소비자 모두 그대로 동작). 짧은 seq의 패딩
   위치는 h=0(α̂을 0으로 패딩)이라 P_iv 질량이 0 → chosen이 vk_i
   너머에 떨어질 수 없다.
4. **Phase-2 동역학 (감사 #3, #4)**: 우리를 구하는 핵심 불변량 —
   **seq당 budget 합이 상수**(= duet_proxy_total_budget)다. 따라서
   phase-2 batch shape은 [B × MQ_p2] × K2 forward로 균일하고 CG shape은
   건드릴 필요가 없다. seq마다 달라지는 것은 **분포**뿐이다:
   - `_select_proxy_sourced_tokens_unified`: dedup/cumsum/argsort를 B에
     대해 벡터화 → proxy_forked [B, MQ_p2], fan_out [B, vk_max+1]
     (chosen의 구성상 pos > vk_i인 row는 0).
   - `_update_phase2_layout_inplace`: fan_out_t가 [B, P]가 되고,
     fan_idx는 seq별 결과를 이어붙인 [B·MQ_p2]; position_count =
     vk_max+1 균일. 유일한 .tolist() 동기화가 [B, P] tolist로 바뀔 뿐
     (동기화 횟수는 그대로 1회).
   - mask 빌더: per-b 루프가 원래 있으므로 공유 fan_out_list 대신
     seq별 리스트(리스트의 리스트)를 먹이면 된다.
   - rope_positions: 이미 fan_idx 기반 row 단위 — seq별 fan_idx만 주면
     된다.
5. **Speculator/verifier 배관**: 균일 폭을 가정하고 무너뜨리던 네 곳
   (speculator L145-167 slice, verifier L110-114 unique assert, L137
   스칼라 lookahead, L186 view)을 vk_max + accept 클램프로 교체.
   logits_p view는 vk_max+1로 균일(패딩 row는 낭비 — v1 패딩 비용의
   일부).
6. **v1 범위 밖** (진입 시 assert B==1): exit_topm_gather /
   exit_replica / proxy_on_draft 게이트(전부 champion 밖, 효과 중립
   측정됨), EAGLE 경로. duet_exit_topm의 y_tok slice는 B>1에서 seq
   경계를 넘는다 — 수정하지 않고 가드만 했다.
7. **Config**: DUET의 max_num_seqs==1 assert를 ≤8로 완화; 낡은
   Policy-A 주석 수정; wire_N은 seq당 값 유지(B 배수는 송수신
   지점에서만 적용).

## 왜 빨라야 하는가 (시스템 관점)

- Draft: phase-1은 forward당 B×16 row, phase-2는 B×10 row — B=2면
  정확히 Marlin 타일 2개(32 row, 완전 활용)라 seq당 draft 비용이 거의
  반토막. JIT은 miss row들을 묶어서 batch로 처리. glue도 batch(varlen
  이미 지원).
- Target: verify가 B×(vk_max+1) row — m 차원이 커지며 70B 쪽 GEMM
  활용률도 좋아진다; verify()와 메트릭은 이미 벡터화되어 있다.
- 새 CG family 없음 ⇒ capture 시간/메모리가 기존 bs-bucket 축
  ({1,2,4}로 시작)만큼만 늘어난다.
- Scheduler/wire/step 계층은 이미 B-generic(감사 §b).

## 시스템 최적화 기법 총정리

구현 전반에 흩어져 있는 최적화 기법을 한곳에 모은다. 각각 "무엇을,
왜, 어떤 비용으로"를 명시한다.

| # | 기법 | 핵심 아이디어 | 절약하는 것 |
|---|---|---|---|
| 1 | 새 CG family 제로 | seq별 차이를 shape이 아닌 mask/index 값으로 표현 | CG capture 시간·메모리 폭발 방지 |
| 2 | vk_max 단일 스칼라 dispatch | batch 폭 결정을 GPU→CPU 동기화 1회로 유지 (`valid_k[0].item()` → `valid_k.max().item()`은 동기화 **교체**지 추가가 아님) | step당 hot-path 동기화 횟수 불변 |
| 3 | JIT-all-then-overwrite | miss 하나에 전 row JIT을 돌리고 hit row만 cache로 덮어쓰기 — draft forward가 latency-bound라 여분 row는 공짜 | any-miss 시 hit row의 cache 이익 보존 (miss 증폭 탈출의 핵심) |
| 4 | budget-합-상수 불변량 | phase-2는 seq당 정확히 total_budget개 토큰 → batch shape [B×MQ_p2] 균일 | phase-2 CG shape 불변, 재capture 불필요 |
| 5 | selector 완전 벡터화 | dedup/cumsum/argsort/scatter를 B축 포함 텐서 연산으로 — boolean-index 후 `view(B, total_budget)` 복원 | seq 루프 제거; 동기화 추가 0회 |
| 6 | 단일 .tolist() 유지 | layout 갱신의 유일한 GPU→CPU 동기화가 [B,P] tolist로 커질 뿐 횟수는 1회 | 동기화 횟수 불변 |
| 7 | per-seq nested mask + 캐시 키 | glue mask를 seq별 블록으로 조립하되 `_cached_fol`이 전체 seq별 구조를 키로 캐시 | B=1에서 기존 캐시 적중 유지; B>1은 분포 변화 시에만 재빌드 |
| 8 | verify 클램프 한 줄 | 패딩 위치 오수락 방지를 `min(accept_until, valid_k)` 벡터 연산 하나로 | 분기/루프 없이 정합성 보장 |
| 9 | h=0 패딩 (proxy) | 짧은 seq의 α̂ 패딩 열을 0으로 → P_iv 질량 0 → budget이 vk_i 너머로 새지 않음 | phase-2 budget 낭비 방지 (M6에서 실제 구현) |
| 10 | K2=K1 shape (사후) | vk_max 갭을 shape 차원에서 0으로 → 패딩세 자체를 소멸 | B=4에서 +17~21ms/step 회수 |

각 기법의 상세:

- **(1)+(4)**: CUDA graph는 텐서 shape이 고정이어야 replay가 가능하다.
  seq 조합(긴/짧은, hit/miss, fan-out 분포)마다 family를 만들면
  조합 폭발이 난다. v1은 "shape은 최악폭으로 통일, 내용은 버퍼 값"
  전략으로 family 수를 B=1과 동일하게 유지했다. phase-2가 특히
  위험했는데, budget 합이 상수라는 알고리즘 불변량이 shape 균일성을
  공짜로 보장했다.
- **(2)+(6)**: async 파이프라인에서 GPU→CPU 동기화는 draft/target
  겹침을 끊는 주범이다. B>1 전환에서 동기화 **횟수**가 늘어난 곳은
  한 군데도 없다 — 기존 동기화의 피연산자만 [B] 텐서로 커졌다.
- **(3)**: 기본 async-SD의 구조적 약점(하나라도 miss면 전체 batch가
  fallback)을 DUET이 그대로 물려받지 않도록 하는 장치. B=8에서 C는
  step의 92%가 JIT 저하 상태(any-miss 부담 1−0.73^8=0.92)인 반면
  DUET은 miss난 row만 대가를 치른다. finding 5b 증폭의 실체가 이
  코드다.
- **(5)**: row-major 순서 + "정확히 total_budget개" 불변량 덕분에
  boolean-index 결과를 reshape만으로 seq별 그룹으로 복원할 수 있다 —
  정렬/카운트 추가 없이 pre-M3와 동일한 연산 하나로 끝난다.
- **(7)**: mask 빌드는 CPU(numpy) 작업이라 GPU 동기화와 무관하지만
  step당 반복되면 오버헤드가 된다. B=1에서는 같은 분포가 반복될 때
  캐시가 계속 적중하고(기존과 동일), B>1에서는 분포가 매 step 달라지는
  것이 정상이라 재빌드가 기본값이다(프로파일 확인 결과 +0.2ms 수준 —
  숨은 비용 아님).

## 단계별 계획 (단계마다 커밋 + 스모크)

| 단계 | 내용 | 검증 |
|---|---|---|
| M1 | Verifier: batched Policy B + wire 2·B·wire_N; draft irecv/unpack B축; speculator vk_max slicing; verifier 균일 assert → vk_max + accept 클램프 | B=1 회귀 스모크 (경로가 바이트 동일해야 함) |
| M2 | Draft: vk_max glue/layout dispatch, per-seq valid_k 배관, mixed-JIT 수정 (JIT-all-then-cache-overwrite) | B=1 스모크 + 클램프/패딩 산술 단위 테스트 |
| M3 | Batched selector + per-seq phase-2 fan-out (fan_out_t [B,P], per-seq fan_idx/mask/rope) | B=1 selector 참조 대비 CPU 단위 테스트 |
| M4 | Config 게이트 완화 (≤8); B=2 end-to-end 스모크; 정합성 검사 (B=2 tok/step ≈ B=1 ± 샘플링 노이즈; hit rate 붕괴 없음) | ns=8 B=2 vs 2× B=1 |
| M5 | 성능: B ∈ {1,2,4} champion vs SD-best B-sweep, 같은 GPU 세트, 인터리브 | regime-win 측정 (docs/duet/12 finding 5b) |

리스크: (a) 패딩 위치의 per-seq fan_out 0-row가 repeat_interleave/mask
산술과 일관돼야 함(M3 단위 테스트); (b) TP bucket 동기화는 rank 0과
draft가 vk_max를 동일하게 유도한다는 가정에 의존 — 기존 valid_k wire에
실려 draft 쪽에서 한 번 계산됨; (c) B>1에서 scheduler admission이
lookahead 예약을 배수로 늘림 — preemption 감시(B=0 가드는 이미 있음).

## M1 — 구현 완료 (2026-07-19)

설계 §1/§5대로 반영: `_compute_and_send_proxy`의 batched Policy B
(h/cumprod/P_iv에 B 축, seq별 topk(wire_N) → chosen [B,wire_N], wire를
2·B·wire_N로 평탄화; ring은 max_num_seqs 기준), draft
`_irecv_duet_proxy`/`_unpack_duet_proxy` B축 ([B,wire_N] view; selector는
M3 전까지 seq 0 소비), speculator의 seq-0 `_vk_scalar` → vk_max,
verifier `torch.unique` assert → `valid_k.max()` (동기화 교체, 추가
아님), `verify(valid_k=...)` per-seq accept 클램프
(`accept_until = min(accept_until, valid_k)`), champion 밖 게이트
(topm/replica/proxy-on-draft, §6)에 B==1 가드.

검증: 단위 테스트 ssd/tests/test_b_gt1_m1.py — 9/9 OK (B=1/2/3에서
batched 산술 ≡ 단일-seq 참조; 클램프 의미론). B=1 GPU 회귀 스모크
(champion config, ns=4/out=128): 71.50 tok/s, L_p1 3.46, cache 0.80,
오류 없음 — 기존 ns=4 노이즈 밴드(58.8–75.8) 안. B=1 wire 길이
2·1·wire_N은 불변; 모든 reshape는 B=1에서 no-op.

## M2 — 구현 완료 (2026-07-18)

설계 §1/§2대로 반영 (draft 쪽, `ssd/engine/draft_runner.py` +
`utils/async_helpers/async_spec_helpers.py`):

1. **vk_max dispatch** (§1): `hit_cache_and_respond`의 dispatch 스칼라
   `_vk_scalar`가 seq-0 캡처 `valid_k[0].item()`에서
   `valid_k.max().item()`으로 — 동기화 **교체**다(step당 GPU→CPU 동기화
   여전히 정확히 1회). 이 스칼라가 유일한 batch 폭으로,
   `make_glue_decode_input_ids` slicing,
   `prepare_glue_decode_ctxt`/`_glue_decode` bucket dispatch,
   `_build_tree_batch_split_k1k2`의 phase-1 long/short layout 선택을
   전부 시그니처 변경 없이 구동한다. per-seq `valid_k [B]`는 응답
   wire와 `partial_tree_decode_args`에 그대로 실린다. vk_i < vk_max인
   row는 vk_max 폭 glue에서 vk_i 너머 구간에 필러가 들어간다(짧은
   hit는 cache 패딩 0, JIT-short miss는 vocab 안 랜덤 초기값) — phase-1
   fork 선택이 layout별로 위치를 slice하고 M1 verify 클램프가 짧은
   row의 vk_i 너머 fork를 도달 불가능한 cache row로 만들기 때문에
   안전하다(v1 패딩 비용일 뿐 정합성 문제 아님). 혼합 K1/K2 batch는
   long bucket으로, all-short batch만 K2로 dispatch(싼 max 검사).
2. **hit/miss 혼합 수정** (§2, JIT-all이 hit를 깔아뭉개는 버그):
   `hit_cache_and_respond`의 채움 로직을 "all-hit → cache 채움, 아니면
   JIT-all"에서 "**miss가 하나라도 있으면 JIT-all**(기존 batched 호출
   그대로 — latency-bound라 여분 row 공짜), **그 다음 hit row를 cache로
   덮어쓰기**"로 재구성. hit row는 cache의 tokens/logits/valid_k/
   phase_source를 유지하고, miss row는 JIT 출력 + JIT 기본
   valid_k(`SSD_DUET_JIT_SHORT`면 K2, 아니면 K_max) + phase_source 0을
   가진다. `.any()/.all()`의 __bool__ 동기화는 기존 분기 조건이 이미
   내던 동일한 동기화의 대체 — hot path에 새 GPU 동기화 없음.

**B>1에서 이 hit/miss 혼합 수정이 중요한 이유** (finding 5b, 사용자
가설): 기본 async-SD는 miss 하나에 **batch 전체**가 멈춘다 —
P(step 저하) = 1 − hit_rate^B이므로 B=4, hit≈0.80이면 step의 ~59%.
M2 이전 DUET도 정확히 같은 실패 모드였다: miss 하나가 모든 hit row의
캐시 트리를 버렸다(B=4에서 step의 ~55%가 hit를 날림). M2 이후에는
miss가 자기 row의 cache 이익만 잃는다; hit row들은 캐시된 K1/K2
트리를 그대로 verify한다. DUET의 높은 hit rate(0.80 vs SD-best 0.76,
게다가 JIT-short로 miss도 저렴)가 B에 의해 파괴되는 대신 **B와 함께
복리로 쌓이게** 만드는 것이 바로 이 수정이다 — DUET이 구조적 B>1
이점을 실현하게 해주는 지점.

B=1 항등성: B=1이면 batch가 all-hit 아니면 all-miss라서 JIT 분기와
cache 채움이 M2 이전과 정확히 같게 동작하고(all-hit → 채움만,
all-miss → JIT만, 덮어쓰기 없음), `max(valid_k) == valid_k[0]`.

감사 잔여물 (§4): `_construct_tree_decode_args`는 비-DUET 전용(방치);
`_build_tree_batch_split_k1k2`에 M1/M2 범위의 B=1 스칼라 인덱싱 없음
(`_step_valid_k` = vk_max가 설계임; `duet_proxy[...][0]` seq-0 붕괴와
selector의 `assert B == 1`은 M3 범위; `_policy_b_from_raw_proxy`의
`out_logits[0]`은 B==1 가드된 champion 밖 raw-proxy 게이트 뒤).

검증: 단위 테스트 ssd/tests/test_b_gt1_m2.py — 8/8 OK (진짜
`hit_cache_and_respond`를 CPU에서 stub DraftRunner로: B=3 hit/miss/hit
에서 hit row의 캐시 tokens/valid_k 유지 + miss row의 JIT 출력 + K2;
JIT-long 기본값; all-hit는 JIT 생략; all-miss/빈 캐시는 덮어쓰기 생략;
all-short는 vk_max=K2 dispatch; B=1 hit/miss 항등성). M1 테스트 여전히
9/9 OK. B=1 GPU 회귀 스모크 (champion config, ns=4/out=128): 71.35
tok/s, L_p1 3.49, cache 0.83, Traceback 0건 — M1의 71.50/3.46/0.80과
노이즈 안에서 일치 (ns=4 밴드 58.8–75.8). 로그:
experiments/proxy_async_overlap/b_gt1/m2_smoke/run.log.

## M3 — 구현 완료 (2026-07-18)

설계 §4대로 반영 (batched selector + per-seq phase-2 fan-out;
`ssd/engine/draft_runner.py` + `engine/helpers/cudagraph_helpers.py` +
`engine/helpers/tree_layout.py`):

1. **Batched selector**: `_select_proxy_sourced_tokens_unified`가
   `assert B==1`을 버리고 B에 대해 벡터화 — dedup은 seq별 advanced
   indexing `draft_forked[b_idx, chosen_pos]` ([B,N,max_fo]; [P,max_fo]
   mask는 broadcast — Phase 1 fan_out_list가 seq 내 균일이라 seq 간
   공유 가능), budget cumsum은 dim 1을 따라
   `take = valid & (rank <= total_budget)`을 seq별로, boolean-index +
   `view(B, total_budget)` (row-major 순서 + 정확히-total_budget
   불변량이 seq별 그룹을 복원 — pre-M3와 같은 boolean-index 연산
   하나, 새 동기화 없음), `scatter_add_` dim 1 → fan_out [B, K_rank+1]
   (각 row 합 = total_budget), seq별 stable argsort + gather로
   위치-그룹 결과 [B, MQ_p2]. Fix-③ underfill 가드는 seq별
   (`(take.sum(1) == total_budget).all()`, 여전히 `__debug__` 전용).
   호출부: M1의 seq-0 붕괴 제거; raw-proxy 모드(B==1 가드, §6)는 1-D
   텐서를 [1, wire_N]로 승격.
2. **Per-seq layout**: `_update_phase2_layout_inplace`가 fan_out [B, P]
   를 받는다; `fan_idx = arange(P).repeat(B).repeat_interleave(
   fan_out.reshape(-1))` — seq별 repeat_interleave를 이어붙인 [B·MQ_p2]
   (B=1에서 bit-identical); `fan_out_list`는 기존의 **단 하나뿐인**
   `.tolist()` 동기화로 seq별 리스트의 리스트가 된다(이제 B×(≤9)
   원소); position_count = K_rank+1 균일. 새 TreeLayout 플래그
   `fan_idx_per_seq` (기본 False, 런타임 변형되는 split_k2 layout만
   True)가 소비자에게 fan_idx가 이미 전 seq를 포괄함을 알린다:
   `_build_tree_decode_args_for_layout` (j_idx/rope_positions)와
   `_merge_and_populate_cache` (proxy_k cache key)가 seq별 재-cat 없이
   직접 사용. metadata F는 int 유지 (seq 0의 첫 원소 — B=1에서 pre-M3와
   같은 값).
3. **Per-seq mask**: `run_fi_tree_decode_cudagraph`의 glue-mask 빌드가
   중첩 fan_out_list를 감지하면 seq별로 `np.repeat(_tril, fol_b)`
   블록을 만든다; per-b mask 루프는 `glue[b]`를 인덱싱(CG-bucket 패딩
   row는 마지막 실제 seq의 블록 재사용 — 출력은 slot_map -1로 폐기).
   `_cached_fol` 캐시는 **전체 seq별 구조**를 키로 잡아 분포가 조금만
   달라져도 재빌드(B>1의 정상 상태); B=1에서 같은 분포 반복은 pre-M3
   그대로 캐시 적중. 평평한 리스트(split_k1/full/비-DUET)는 공유-glue
   경로 그대로. Phase 1은 무변경 (§4 — seq 내 균일 리스트라 이미
   batch 동작).

B=1 항등성: 모든 batched 연산이 B=1에서 pre-M3 단일-seq 연산으로
퇴화한다 (한 row에 대한 새 dim의 indexing/cumsum/scatter/argsort;
fan_idx 값 동일; 중첩 `[[...]]` glue 빌드도 같은 numpy 블록 생성).

검증: 단위 테스트 ssd/tests/test_b_gt1_m3.py — 7/7 OK (batched
selector vs 테스트 안에 복사해 둔 pre-M3 원본 selector의 seq별 루프,
B=1/2/3, champion shape K_rank ∈ {4,9}, 심어놓은 dedup 충돌 0/5/12,
혼합 short-seq batch vk=[9,4,4]에서 vk_i 너머 fan_out 0 확인 — 설계
리스크 (a); per-seq fan_idx 공식 ≡ pre-M3 seq별 repeat_interleave의
concat (패딩 위치 0-row 포함); per-seq glue 블록 ≡ pre-M3 공유 빌드의
seq별 결과). M1 9/9, M2 8/8 OK (개별 실행 기준; m1+m2를 한 unittest
프로세스에서 돌리면 5개 실패 — 수정 전 트리에서도 재현되는 기존
env-baking 순서 아티팩트, M3 회귀 아님). B=1 GPU 회귀 스모크
(champion config, ns=4/out=128): 71.47 tok/s, L_p1 3.45, cache 0.81,
Traceback 0건 — M2의 71.35/3.49/0.83과 노이즈 안 일치. 로그:
experiments/proxy_async_overlap/b_gt1/m3_smoke/run.log.

## M4 — 구현 완료 (2026-07-18)

설계 §7대로 반영 (`ssd/ssd/config.py` + `ssd/CLAUDE.md` 불변량 라인):

1. **게이트 완화**: DUET의 `max_num_seqs == 1` assert가 `<= 8`로 (v1
   상한 — 기존 bs-bucket 축만 사용, 새 CG family 없음). "Policy A
   accept_probs[0]" 탓을 하던 낡은 주석을 실제 역사로 교체: 제약의
   진짜 원인은 단일-seq Policy B 파이프라인(proxy wire / selector /
   phase-2 layout)이었고 M1-M3에서 batch화됐다.
2. **B==1 전용 게이트 가드** (§6): `duet_enabled` + `max_num_seqs > 1`
   에서 `SSD_DUET_EXIT_TOPM_GATHER` / `SSD_DUET_EXIT_REPLICA` /
   `SSD_DUET_PROXY_ON_DRAFT` 중 하나라도 설정되면 config 시점
   ValueError로 즉시 실패 — 실행 중 M1의 `assert B == 1`에 걸려
   터지는 대신 fail-fast.

엔진 쪽 수정은 불필요했다: 예상했던 지뢰 두 개가 터지지 않았다.
bs=2 CG bucket이 모든 family에서 깨끗하게 capture됐고
(`fi_tree_decode`, `split_k1_long/short`, `split_k2`,
`duet_verify_k1/k2` — max_num_seqs=2에서 bucket 축 {1,2}), 두 배가 된
lookahead 예약에도 scheduler preemption이 없었다 (2048-토큰 seq,
517-block KV pool).

검증: 단위 테스트 ssd/tests/test_b_gt1_m4.py — 6/6 OK (진짜
Config.__post_init__를 CPU에서, champion shape: B ∈ {1,2,8} 생성 성공,
B=9 거부, 각 게이트가 B=2에서 raise + B=1에서는 셋 다 생성 가능).
M1 9/9, M2 8/8, M3 7/7 개별 실행 여전히 OK.

GPU 스모크 (champion E9K24_jit, out=128, temp 0.7, GPU 0-4):
B=2 `--b 2 --numseqs 8` (m4_smoke_b2/run.log), B=1 회귀
`--b 1 --numseqs 4` (m4_smoke_b1/run.log). 둘 다 exit 0, Traceback 0건.

| 지표 | B=1 (ns=4) | B=2 (ns=8) |
|---|---|---|
| Decode TPS (합산) | 70.92 | 75.40 |
| 평균 Cache Hits | 0.82 | 0.80 |
| 평균 Phase 1 Accepted Len | 3.33 | 3.12 |
| 평균 Tokens/step (recovery 포함) | 3.61 | 2.75 |
| Phase 1 (draft) hit rate | 0.535 | 0.376 |
| Phase 2 (proxy) hit rate | 0.285 | 0.419 |
| 평균 Phase 2 Accepted Len | 1.69 | 0.92 |
| 평균 target full step (ms) | 55.95 | 80.54 |
| 평균 draft step (ms) | 44.83 | 66.46 |

(P1 + P2 hit rate가 cache-hit rate를 분할: B=1에서 0.535+0.285=0.82,
B=2에서 0.376+0.419=0.80.)

B=1 기준선 (TPS ≥ 68, L_p1 ≥ 3.0): PASS — 70.92/3.33/0.82는 M3의
71.47/3.45/0.81과 ns=4 노이즈 안 일치. B=2 기준선 (Traceback 0건,
hits ≥ 0.70, L_p1 ≥ 3.0): PASS — 0.80/3.12, decode 75.40 합산.

설계의 "tok/step ≈ B=1" 기대 대비 정합성 판독: hit rate는 붕괴하지
않고(0.80 vs 0.82) L_p1도 유지(3.12 vs 3.33)되지만, tok/step이
3.61 → 2.75 (−24%)로 떨어지며 proxy-sourced row에 집중된다: hit 구성이
draft-sourced(0.535 → 0.376)에서 proxy-sourced(0.285 → 0.419)로
이동하고 P2 accepted len이 반토막(1.69 → 0.92). 그래도 두 seq가 한
step에 verify되므로 합산 decode TPS는 +6.3%. P2 이동이 진짜 B=2
효과인지 ns=8 프롬프트-믹스 노이즈인지 — 그리고 seq당 성능 스토리 —
는 M5 스윕의 몫 (B ∈ {1,2,4} vs SD-best, 인터리브).
**[사후 확인: 이 "P2 이동"은 실제로는 M6 verify-window 버그의 첫
징후였다.]**

## M5 — 측정 (2026-07-18)

스윕: B ∈ {1,2,4}, DUET champion vs SD-best C (k7 f6), B별 인터리브,
셀당 1회 실행, ns=20 out=256 seed 42, GPU 0-4, 포트 12900-12905.
전체 표 + 분해: `experiments/proxy_async_overlap/b_gt1/m5_sweep/RESULTS.md`.

**finding 5b 가설은 v1에서 기각(처럼 보였다).** 합산 decode TPS:

| B | DUET | C | 격차 |
|---|---|---|---|
| 1 | 71.86 | 77.90 | −7.8% |
| 2 | 89.22 | 109.86 | −18.8% |
| 4 | 108.87 | 150.31 | −27.6% |

C는 B1→B4 ×1.93 스케일 vs DUET ×1.52. DUET의 재료는 전부 착지했는데
— B=4에서 hit 0.84 vs 0.74, any-miss 부담 0.50 vs 0.70 — 그래도
졌다: (i) M4의 P2 신호는 진짜 단조 B-효과(L_p2 1.64 → 0.85 → 0.49;
hit 중 P2 비중 35% → 53%)로 tok/step −10% 상당인데 C의 tok/step은
평평; (ii) DUET의 step 시간이 모든 축에서 C보다 빨리 큼 (T_verify
×2.39 vs ×2.01, T_draft ×2.26 vs ×1.93) — seq당 26 vs 48 row인데도
그랬다 (vk_max 패딩, mid-verify DUET 블록, B×16 row로 타일 절벽을
넘는 13회 직렬 draft forward); (iii) any-miss JIT 스톨은 B ≤ 4에서
커지는 항이 아니었고(C는 0.70의 any-miss 부담을 tok/step 변화 없이
흡수), 그래서 hit-rate 이점이 증폭할 대상이 없었다; JIT-short miss는
DUET miss row를 1.48 tok으로 캡(B>1에서 토큰 부채). 향후 레버 순위:
RESULTS.md §4.

**⚠ 2026-07-18 (M6): 위 M5의 DUET 수치는 버그였다.** 이 표의 "P2
희석"과 miss-row 붕괴는 DUET 알고리즘이 아니라 B>1 정합성 버그가
원인이었다. 교정된 스윕 + 수정된 판정: 아래 §M6과
`m5_sweep/RESULTS.md`(교정 표).

## M6 — B>1 short-row verify-window 버그: 근본 원인 + 수정 (2026-07-18)

**증상** (M4/M5): L_p2가 B에 대해 단조 붕괴(1.64 → 0.85 → 0.49),
miss-row 토큰도 마찬가지(2.57 → 1.98 → 1.48), 그런데 P2 hit **rate**는
상승(0.28 → 0.445) — "키는 맞는데 체인이 쓰레기". seq별 독립
rollout에서는 물리적으로 불가능한 조합 — batch화된 무언가가 잘못된
데이터를 먹이고 있다는 뜻.

**근본 원인** (감사의 용의자 #6 — M1/M3 batch 산술이 아니라;
full-chain CPU 테스트가 row 단위로 무죄를 입증했다): target verify
입력 window. `prepare_decode_tensors_from_seqs` (runner_helpers.py)가
각 seq의 verify row를

    pos0 = seq.num_tokens - (k + 1)        # k = _duet_step_lookahead = vk_max

로 만드는데, 이 k는 **batch 균일 vk_max**인 반면 speculator는 각
`seq.token_ids`를 **seq별 vk_i**만큼만 연장했다. 혼합 batch(vk_max =
K1 = 9)의 짧은 row(vk_i = K2 = 4)는 pos0가 `vk_max − vk_i = 5` 토큰
일찍 떨어진다: 10-row window가 `[rec | t1..t9]`가 아니라 `[낡은 문맥
5토큰 | rec | t1..t4]`가 된다. 그 seq의 모든 `logits_p` row가 5칸씩
밀리므로, ratio test는 P2/JIT 체인을 "이미 알려진 문맥"에 대한 모델
예측과 비교하고(위치 0에서 거의 확정 거부), recovery 토큰은 밀린
위치에서 샘플링된다(모델이 옛 문맥 토큰을 다시 뱉음 — 성능 손실이
아니라 **출력 corruption**). 이를 지키던 assert
(`num_cached_tokens == pos0`, runner_helpers.py L88)는 모든 벤치가
쓰는 `python -O`에서 제거된다.

**B=1이 면역이었던 이유**: 단일-seq batch는 항상 균일 — vk_max = vk_i
— 이라 window가 정렬된다. M1-M3의 B=1 회귀 스모크가 전부 통과한
이유가 정확히 이것.

**"seq 0만 정상"처럼 보였던 이유**: 실체는 "**혼합 batch의 짧은
row가 corruption**"이다. 긴 row(P1 hit, vk_i = K1 = vk_max)는 절대
밀리지 않는다 → L_p1은 유지/상승. 짧은 row(모든 P2 hit과 모든
JIT-short miss row)는 batch에 긴 row가 하나라도 있으면 corruption —
그 확률은 B와 함께 상승. 동역학은 끌개(attractor)다: corruption된
짧은 row는 accept≈0 + 퇴화 recovery를 받고, 다음 step의 요청 키
(seq, 0, rec)가 P2 pos-0 후보 fan에 걸려 **또 P2 hit**(rate 상승)이
되는데 그 체인도 다시 corruption(L_p2 ≈ 0.1) — P1 hit이 구출할 때까지
seq가 짧은 상태를 맴돈다. M5의 "이상한" L_p1 상승(3.54 → 5.07, P1
hit rate는 하락)도 설명된다: corruption된 row가 만드는 퇴화 반복
텍스트는 깊게 speculate하기 쉽고, P1 hit은 건강한 long-state seq로
자기선택된다.

**수정** (3부, 전부 B=1/균일 batch에서 no-op):

1. `speculator_async.py` — `extend_seqs_for_verify` (함수로 추출, 단위
   테스트): **모든** seq의 token_ids를 vk_max만큼 연장(짧은 row는
   패딩 꼬리를 지참) → 모든 seq에서 pos0 = num_cached_tokens. 패딩
   꼬리는 절대 수락될 수 없고(M1 클램프) 연장 전체는 SpecDecodeStep의
   상태 복원으로 롤백된다. `num_draft_cached_tokens`는 seq별 vk_i + 1
   전진 유지.
2. `verifier.py _compute_and_send_proxy(valid_k=...)`: 짧은 seq의 패딩
   열에서 α̂을 0으로 → h[b, vk_i]가 전-실토큰-수락 질량을 온전히
   담고 vk_i 너머 h는 0, 따라서 chosen이 도달 불가능한 위치에 떨어지지
   않는다 (M1 설계 §3의 주장이었으나 실제로는 **미구현 상태**였다 —
   짧은 seq가 P2 budget을 vk_i 너머로 누수시키고 있었다).
3. `utils/verify.py`: residual (p−q)+ recovery 보정을
   `min(K, valid_k)` 아래에서만 적용 — 클램프된 전체-수락 짧은 row는
   위치 vk_i에서 plain p를 샘플링(B=1의 같은 사건과 동일)하며,
   0-패딩 logits에서 엉터리 uniform q를 빼지 않는다.

**검증**: 단위 테스트 `ssd/tests/test_b_gt1_m6_verify_window.py`
8/8 OK — (a) draft쪽 전체 체인(wire unpack → batched selector →
layout 갱신 → args 빌드 → merge key)을 B=3, seq별 상이 데이터로:
모든 row의 (seq, k_idx, position, rope, seed) 튜플 ≡ 같은 seq의 B=1
실행 (용의자 1-5 무죄); (b) 진짜 extend_seqs_for_verify +
prepare_decode_tensors_from_seqs를 혼합 batch에: window ≡
[rec]+spec[:vk_max] (seq마다), M6 이전의 seq별 연장은 짧은 row의
window를 미는 것이 증명됨(prepare 자체 assert 발화); (c) verify():
ratio로 수락될 뻔한 패딩 열이 vk_i 클램프에 잘리고 recovery가
logits_p[b, vk_i]에서 나옴; (d) proxy h-패딩: 짧은 seq의 chosen_pos가
vk_i를 넘지 않음(수정 전엔 6+ 누수). M1 9/9, M2 8/8, M3 7/7, M4 6/6
여전히 OK.

GPU B=2 스모크 (m4-smoke 인자, 포트 12910,
`experiments/proxy_async_overlap/b_gt1/m6_fix/b2_smoke/`): 시그니처의
모든 원소가 반전 — L_p2 **1.75** (버그 시 0.92; B=1 1.69), miss 토큰
**2.56** (버그 시 1.98; B=1 2.57), P2 hit rate 0.286 (버그성 부풀림
0.419 소멸), P1 hit 0.529 / L_p1 3.81, tok/step 3.80, hits 0.82,
decode TPS 101.5 (같은 셀 버그 시 75.4), Traceback 0건.

**교정된 M5** (DUET 셀 재실행, 같은 인자/GPU, 포트 12911-13,
`experiments/proxy_async_overlap/b_gt1/m6_fix/duet_b{1,2,4}/`; C 셀은
그대로 — DUET 게이트가 없으므로 무영향):

| B | DUET TPS | C TPS | 격차 (버그 시) | L_p2 | miss-tok | P2 hit |
|---|---|---|---|---|---|---|
| 1 | 74.69 | 77.90 | −4.1% (−7.8%) | 1.73 | 2.59 | 0.269 |
| 2 | 104.59 | 109.86 | −4.8% (−18.8%) | 1.81 | 2.71 | 0.269 |
| 4 | 118.00 | 150.31 | −21.5% (−27.6%) | 1.63 | 2.68 | 0.274 |

L_p2 / miss-tok / P2-hit이 이제 **B-불변** — 원래 M5의 "단조 P2 희석"
은 100% 버그 아티팩트였다. B=2는 tok/step 동률(3.89 vs 3.90)의
근접-동률. 살아남은 B=4 격차는 ~85%가 **시간 쪽**(T_draft ×2.32 vs C
×1.93; T_verify ×2.46 vs ×2.01): finding 5b는 아직 미확인이지만 이유가
토큰/hit이 아니라 step-시간 형상(B당 draft forward 수/폭, vk_max 패딩
verify, mid-verify 블록)에 있다. 전체 교정 표 + 수정된 판정:
`m5_sweep/RESULTS.md` 교정 섹션; docs/duet/12 B>1 섹션도 그에 맞춰
재작성.

## Verdict 실험 — 버그냐 물리냐? (2026-07-18)

전체 기록: `experiments/proxy_async_overlap/b_gt1/verdict/RESULTS.md`.
M6 교정 후 남은 B=4 격차(−21.5%)를 겨냥한 결정적 실험 두 개:

**Exp1 — B=4 PROFILE 포렌식** (champion 인자 + SSD_PROFILE_DUET=1,
포트 12920, 126.69 tok/s): 모든 프로파일 라벨을 각각의 구조적 B×rows
모델과 대조. **전부 일치**: phase1_replay 9 × 5.26 ms (64 row = Marlin
m-타일 4개; 16-row 2.47 ms 대비 ×2.13 — 타일-선형 상한 5.79보다 아래),
phase2_replay 4 × 4.45 (40 row), glue replay ×2.0, prep/빌드/merge는
평평 (seq별 중첩-mask 빌드 +0.2 ms — M3 기계장치는 숨은 비용이 아님),
all-hit cache 채움 0.89 ms ≡ B=1, batched any-miss JIT 8.64 vs B=1의
8.00 ms (M2의 latency-bound 주장이 B×rows에서 확인됨), 양쪽 프로세스
wall이 라벨로 완전 회계(동기화 폭풍 없음). **버그 판정: 남은 B>1 버그
없음.**

구조적 발견: (1) **target이 병목** — draft idle이 6.0 → 34.5 ms/step
(작업 46.1 → 87.6 vs target wall 122.6)으로 커져, 13회 직렬 draft
forward는 hit-step 임계 경로에 앉지 않는다(hit-step spec_wait 3.0 ≈
B=1의 2.7); (2) 폭 분포: step의 93.3%가 K1-폭 verify를 dispatch
(all-short K2는 5.8%, 이론값 0.447^4와 일치)하는데 실제 긴 row는
55%뿐 → **vk_max 패딩 = 17-21 ms/step** (낭비 8.3 row × verify 한계
비용 2.23 ms/row) — 시간 쪽 지배항; (3) finding-5b의 miss-stall 증폭
항은 실재하고 B와 함께 커지지만(any-miss 부담 0.57 vs C 0.70; 빈도
이점 13pt — B=1의 6pt에서 증가; 스톨당 7.8 ms 측정) B=4에선 +1..+5
ms/step 가치. 분해 합(+12..+16 ms)이 측정된 ΔT_target = C 대비
+16.1 ms와 맞아떨어진다.

**Exp2 — fat-shape 재튜닝 프로브** (B=4, PROFILE=0, 포트 12921-2):

| 셀 | shape | 직렬 fwd | verify rows | TPS | vs C | tok/step | t_step (ms) |
|---|---|---|---|---|---|---|---|
| champion | K1=9 K2=4 list [2×6,1×4] | 13 | 40 | 118.00 | −21.5% | 3.63 | 123.1 |
| fat7 | K1=7 K2=4 dfo=2 uniform | 11 | 32 | 144.72 | −3.7% | 3.71 | 102.5 |
| fat5 | K1=5 K2=4 dfo=3 uniform (--f 4) | 9 | 24 | **155.12** | **+3.2%** | 3.41 | 87.9 |
| C | k=7 f=6 | 7 | 32 | 150.31 | — | 3.99 | 106.2 |

fat7은 T_verify를 C와 정확히 동률로 착지시키고(91.97 vs 91.42 — 같은
32-row 폭) step은 이미 C보다 빠르다; fat5 — **DUET의 첫 B>1 측정
승리** — 는 토큰(C의 0.855배)을 step 시간(C의 0.827배)과 맞바꾼다.
B=1 champion의 deep-narrow shape은 타일-절벽 아티팩트였고, B=4에선
step의 93%에서 K1-폭 verify 패딩만 물어냈다; 타일 절벽 자체는 B=4에서
seq들로 분할상환된다(fat5의 72-row phase-1 forward가 그래도 이긴다).
finding 5b: v1에서 **부분 확인** — shape을 B별로 재튜닝하면 B>1은
DUET의 승리 레짐이다. 주의: 셀당 1회 실행(토큰 노이즈 ±4%; +3.2%
단독으론 band-clear 아님 — 강건한 결과는 fat-beats-deep +10..+31%),
fat5는 --f 4 필요(더 넓은 miss JIT), fat5는 B ∈ {1,2} 미측정; B=1
champion은 여전히 E9K24_jit.

## B별 shape 스윕 + 확정 승리 (pb_sweep, 2026-07-18/19)

질문: fat5/fat7(첫 감)은 정말 B별 최적이었나?
전체 기록: `experiments/proxy_async_overlap/b_gt1/pb_sweep/RESULTS.md`.
B별 그리드(제약 K2≤K1, 균일 dfo, f=dfo+pfo, ns=12 셀당 1회, C 앵커
재실행): B=4 9개 shape, B=2 5개 shape. 전 셀 rc=0 — 그리드 전체가 v1
제약 집합 안이다.

**답: 아니었다.** B=4에서 표면은 K1이 그리드 가장자리까지 떨어질수록
계속 상승: k3x3_d4p1 (K1=K2=3, dfo=4, k=6 f=5) 165.50 vs fat5의
149.72 (+10.6%). 응답 표면(스캔):

- **K1 = verify 폭이 지배 노브, 순수 시간 효과** — T_verify는 K1의
  거의 순수 함수(K1=3/4/5/6에서 60.1/71.5/79/87 ms = B×2.25 ms/row,
  verdict의 한계-row 물리)인 반면 K1 한 칸은 ~0.3 tok/step만 사준다.
- **K2=K1은 공짜 토큰** (vk_max 갭 제로 → 패딩 제로): 고정 K1에서
  k4x4 > k4x3, k7x6 > k7x4, T_verify 불변.
- **pfo=2**는 중간 그리드 +2.7% (draft idle이 비용 부담), 승자에선
  중립 (k3x3_d4p2 166.27 ≈ 165.50).
- **dfo**는 B=4에서 평평; B=2의 주 노브 (dfo 2→3: +3.6%, hit
  0.81→0.84). B=2 표면은 평평한 능선(±1.8%, 모든 DUET 셀이 C_b2를
  이김): 승자 k6x5_d3p1 114.35 vs k5x4_d3p1 114.22는 동전던지기;
  확정 단계로 간 것은 k6x5_d3p1.

**확정 단계 (ns=20 out=256, B별 3-rep 인터리브 DUET/C) — 두 승자 모두
band-clear:**

| B | shape | DUET 평균 (범위) | C 평균 (범위) | 판정 |
|---|---|---|---|---|
| 4 | k3x3_d4p1 (k=6 f=5) | **169.42** (167.24-171.89) | 147.53 (142.48-151.28) | **+14.8%, band-clear** |
| 2 | k6x5_d3p1 (k=11 f=4) | **114.09** (112.82-115.77) | 106.73 (105.45-108.36) | **+6.9%, band-clear** |

B=4 메커니즘: tok/step 2.85 vs 3.94 (0.723배) × step 시간 67.4 vs
106.9 ms (1.586배) — verify 16 row vs C의 32, hit 0.87 vs 0.73. 캠페인
전체의 B별 최적 추세: B 1 → 2 → 4에서 K1 9 → 6 → 3, f 3 → 4 → 5.
**finding 5b: 확인 — shape을 B별로 재튜닝하면 DUET 승리가 B와 함께
증폭된다 (+0.5% → +6.9% → +14.8%).**

주의: 스캔은 ns=12 1회(중간 그리드 순위 미해결); C 앵커 + 확정은
DUET 스캔 셀 하루 뒤 실행(run-script argparse 버그가 원래 C 셀을 모델
로드 전에 죽임; 확정은 내부 인터리브라 drift-안전); K1=3은 그리드
가장자리(K1=2, K2>K1, B=8 미측정); B=4 승리는 100% step-시간 — 토큰이
더 비싼 레짐에선 최적점이 다시 깊이 쪽으로 이동. B별 권장 config:
docs/duet/12 "B>1 recommended configs". **[07-19: 그리드-가장자리
주의사항은 bscale 캠페인으로 해소 — 다음 섹션.]**

## B-scaling 캠페인 완결: B=8 + 가장자리 + 그래프 (bscale, 2026-07-19)

최종 갭 메우기
(`experiments/proxy_async_overlap/b_gt1/bscale/REPORT.md`): 외삽 최적
주변의 B=8 그리드(K1 ∈ {2,3,4}, K1=K2), pb_sweep 그리드에 없던 B=4
K1=2 가장자리 셀, B=1 동일-레짐 앵커, B=8 3-rep 인터리브 확정. 스캔
11셀 + 확정 6셀, 전부 rc=0, Traceback 0건 — B=8(M4 게이트 상한)이
전체 DUET 파이프라인을 깨끗하게 돌린다, CG bucket 축 {1,2,4,8}.

**B=8 판정: k2x2_d5p1 (K1=K2=2, dfo=5 pfo=1, k=4 f=6)이 C를 band-clear
로 이김 — 210.39 vs 165.85 (+26.9%)**, 범위 209.74-211.11 vs
162.64-169.61 (최악 DUET rep이 최고 C rep보다 +23.7% 위). 메커니즘:
tok/step 2.38 vs 3.83 (0.621배) × t_step 90.4 vs 184.7 ms (2.044배) →
R = 1.269. DUET verify는 B×(K1+1) = 24 row vs C의 64 (T_verify 80.7
vs 160.1 ms); hit 0.89 vs 0.73 → any-miss 부담 0.62 vs 0.92 — B=8에서
C는 step의 92%를 JIT-저하 상태로 돈다(DUET이 이 운명을 피하는 것이
M2의 hit/miss 혼합 수정). C는 폭 축에서 포화(B=4→8 겨우 +12.4%, step
시간 107 → 185 ms)하는 반면 DUET은 계속 스케일(+24.2%).

**완성된 증폭 곡선 (finding 5b, 최종):**

| B | 승자 shape | vs C | 상태 |
|---|---|---|---|
| 1 | E9K24_jit (K1=9 K2=4) | +0.5% (헤드라인) / +0.6% (동일-레짐 앵커) | 동률 |
| 2 | k6x5_d3p1 | +6.9% | band-clear |
| 4 | k3x3_d4p1 | +14.8% | band-clear |
| 8 | k2x2_d5p1 | +26.9% | band-clear |

**shape 법칙**: K1 9 → 6 → 3 → 2 (B 2배마다 그리드 한 칸), K2 → K1
(균일 폭, vk_max 패딩 제로), f 3 → 4 → 5 → 6, seq당 verify row
10 → 7 → 4 → 3. bscale의 B=4 가장자리 셀이 이것이 "작을수록 좋다"가
아니라 **움직이는 내부 최적**임을 증명한다: K1=2는 B=4에서 진다
(157.3 vs 165.5 — step 시간 7 ms 절약이 0.41 tok/step 손실을 못 덮음)
그리고 B=8에서 이긴다(같은 절약이 19 ms가 되므로). 깊이의 토큰 가치는
B-불변인데, 폭의 시간 비용은 B에 선형이기 때문이다.

산출물: `bscale/REPORT.md` (전체 표, 메커니즘, 주의사항) + 그래프
5종 (`bscale/figs/fig1..5`): TPS-vs-B, 증폭 곡선, shape 법칙, 양
가장자리를 포함한 B=4 응답 표면, seq당 처리량/지연 트레이드오프.
기억할 주의사항: B=8에서 ns가 8의 배수가 아님(꼬리 step은 전체 폭
미만 — 양쪽 동일 조건); K1=1과 B>8 미측정(v1 상한); B≥4 승리는 100%
step-시간이므로 토큰이 더 비싼 레짐에선 최적점이 깊이 쪽으로 회귀.
