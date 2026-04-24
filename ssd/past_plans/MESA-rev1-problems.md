# MESA Rev1 Problems — 수정 대상과 계획

이 문서는 MESA Rev1에서 **실제로 수정할 항목만** 남긴 버전이다. 현재 B=1 + 단일 run 실험 환경에서 영향이 0에 가까운 항목 (`run_mesa_verify_cudagraph` padding cat, `fan_idx` helper, hot-path import, profiling flush wiring, phase build label 세분화)은 제외했다.

수정 순서:
1. #3 B=1 assert 추가 (correctness, 3 LoC)
2. #4 proxy_top_k 확대 + draft fallback 제거 (design + correctness)
3. #D `_decode_tree()` 진입 setup 버퍼 pre-allocate (매 step 8 MB 재할당 제거)
4. #1 Proxy selection 벡터화 (fallback 제거된 상태에서 pure GPU)
5. #8 Dead code 제거

**제외(defer/제거)**:
- #2 TreeLayout LRU cache — **제거**: Policy A의 `fan_out_list`가 step마다 바뀌어 hit rate 낮을 가능성 높음, 기대 이득도 작음(~0.2 ms)
- #5 `topk_probs` — **defer**: Policy B 도입 시 활성화
- #6 verify padding cat — **defer**: B=1 전제 하에 trigger 안 됨

---

## #1 Proxy selection: Python loop + GPU→CPU sync 제거 (Critical)

### 위치
- `ssd/engine/draft_runner.py:1009` (`_select_proxy_sourced_tokens`)
- `ssd/engine/draft_runner.py:1054` (`_select_proxy_sourced_tokens_policy_a`, Rev1 실사용 경로)

### 현재 코드 문제
```python
# 1026-1028: GPU→CPU sync 3회
draft_cpu = draft_forked[:, :K, :].cpu().tolist()
proxy_cpu = proxy_topk_ids.cpu().tolist()
fallback_cpu = fallback_topk[:, :K, :].cpu().tolist()

# 1031-1044: Python 이중 for loop + set dedup
for b in range(B):
    for pos in range(K):
        draft_set = set(draft_cpu[b][pos])
        selected = [t for t in proxy_tokens if t not in draft_set]
        ...
```

### 영향
- `phase2_build` 라벨 내부 ~0.8 ms가 이 경로
- `.tolist()`가 implicit GPU sync → stream stall
- 특히 Policy A는 `proxy_recv_work.wait()` 직후 → draft critical path 직결

### 수정 계획 (#4 이후 전제 — pos<K는 proxy-only, pos==K만 draft all-accept path)

**pos < K 경로: proxy-only 벡터화**

Inputs (모두 GPU tensor):
- `draft_forked: [B, K+1, draft_fan_out]`
- `proxy_topk_ids: [B, K, proxy_top_k]` — #4로 proxy_top_k가 충분히 큼 (`pfo*(K+1) + dfo + 2`)

의사코드:
```python
# 1. Proxy pool 자체의 내부 dedup (prefix duplicate mask)
#    proxy_topk_ids[..., i]가 proxy_topk_ids[..., :i]에 이미 있으면 제외
#    in_prev = (proxy[..., i:i+1] == proxy[..., :i]).any(dim=-1)  # triu 스타일

# 2. draft_forked와의 겹침 mask
#    in_draft = (proxy[..., None] == draft[:, :K, None, :]).any(dim=-1)  # [B, K, P]

# 3. 유효 토큰 mask
#    valid_mask = ~in_prev & ~in_draft  # [B, K, P]

# 4. fan_out_list[pos] 만큼 앞에서 pick
#    rank = valid_mask.cumsum(dim=-1) - 1          # 각 valid 토큰의 누적 순위
#    taken_mask = valid_mask & (rank < fan_out_tensor[pos, None])
#    result[b, pos] = proxy[b, pos][taken_mask]    # scatter by offset
```

**pos == K (all-accept) 경로: draft top-k만 사용 (이게 유일한 "fallback 유지 지점")**
```python
# draft_forked[b, K, :]가 이미 top-dfo, but 이걸로 proxy의 all-accept 위치 채움
# 추가 top-k from draft logits at pos K, excluding draft_forked
# #4의 설계대로 "all-accept 위치는 correction 불필요"라 proxy 없음
logits_k = glue_logits[:, K, :].clone()
logits_k.scatter_(1, draft_forked[:, K, :], float('-inf'))
_, all_accept_topk = torch.topk(logits_k, fan_out_list[K], dim=-1)  # [B, fo_K]
result[b, K_offset:] = all_accept_topk
```

**edge cases**:
- `fan_out_list[pos] == 0` → 해당 position skip (cumsum mask로 자연스레 0개 pick)
- `fan_out_list[K] == 0` → all-accept path 자체 skip
- `underfill`: #4로 proxy_top_k 충분 커서 안 생김. assert만 debug build에 남김

### 작업량
- 메인 구현: ~40 LoC (fallback 제거된 상태라 단순)
- Non-Policy A (`_select_proxy_sourced_tokens`)는 #8 Dead code 제거로 자연 정리
- 예상 소요: 2-3 시간

### 예상 이득
- `.tolist()` 호출 제거 → GPU sync 0
- Python for loop 제거 → Python 오버헤드 0
- 실측 phase2_build 1.4 ms → **~0.4 ms** 예상
- 전체 MESA step ~1 ms 단축 → throughput **+1.5%**

---

## #3 B=1 only 강제 (Correctness)

### 위치
- `ssd/config.py:100` (MESA 검증 블록)
- `ssd/engine/verifier.py:227` (`accept_probs[0]`)
- `ssd/engine/draft_runner.py:1171` (단일 fan_out_list 사용)

### 현재 코드 문제
```python
# verifier.py:227
cumprod = torch.cumprod(accept_probs[0], dim=0)  # [K] (B=1 scope)
# ↑ 0번째 seq만 사용. 주석엔 "B=1 scope"라 돼있지만 런타임 assert 없음

# draft_runner.py: fan_out_list를 batch 전체에 적용
# B>1일 때 seq 0의 h_i 분포를 seq 1, 2에 강제 적용 → 잘못된 token 선택
```

### 영향
- 현재 실험 전부 B=1 → correctness 문제 발현 안 됨
- 하지만 누군가 `--b 2` 이상으로 MESA 돌리면 silent correctness bug
- 결과 metric은 나오지만 proxy가 잘못 작동 → accept rate 하락

### 수정 계획

`ssd/config.py` MESA 검증 블록에 추가 (기존 assert 옆):
```python
if self.mesa_enabled:
    ...  # 기존 assertions
    assert self.max_num_seqs == 1, \
        "MESA Rev1 only supports B=1 (max_num_seqs=1); " \
        "Policy A uses accept_probs[0] as a single h_i distribution for the whole batch"
```

### 작업량
- **3 LoC**

### 예상 이득
- correctness 보장. 미래 B>1 MESA는 별도 설계 (Policy A 확장 or per-seq fan_out_list 전송)

---

## #4 Underfill: Target proxy_top_k 확대 + **draft fallback 제거** (Design + Correctness)

### 왜 fallback이 문제인가

현재 `_select_proxy_sourced_tokens_policy_a` (draft_runner.py:1054-1102)는 두 소스에서 토큰을 뽑음:
1. **Proxy** — target의 residual `(p_E - p_D).clamp(min=0)`의 top-k. "target 분포에서 유의미하지만 draft는 놓친" 토큰.
2. **Fallback** — draft logits 자체의 top-k. dedup 후 부족하면 여기서 채움.

MESA의 철학은 **"target이 draft에게 유용한 correction을 알려준다"**. 그런데 fallback은:
- Draft가 이미 argmax/top-k로 뽑아둔 것과 **같은 분포**에서 추가로 뽑는 것
- 즉 "draft가 두 번째로 예측하는 것" — target 정보 0
- 이런 토큰을 tree에 넣어봤자 target이 선호할 가능성은 proxy보다 현저히 낮음
- 결과: 실효 tree 유효 branch 수가 줄어듦 → accept rate 저하

**근본 원인**: target의 `mesa_proxy_top_k` (default 3) 가 너무 작아서 draft가 부족분을 fallback으로 채울 수밖에 없는 구조.

### 올바른 설계

**Target이 넉넉한 residual top-k를 보낸다 → draft는 fallback 필요 없다.**

| 항목 | 현재 | 제안 |
|------|:---:|:---:|
| `mesa_proxy_top_k` default | 3 | **15** (또는 `max(async_fan_out) * (K+1) + 4`) |
| Target residual.topk compute | top-3 | top-17 (~+50-100 µs GPU, 실질 무시) |
| NCCL payload (K=6) | 216 bytes | ~720 bytes (무시 가능) |
| Draft fallback path | proxy 부족 시 draft logits 사용 | **제거** |
| Dedup worst case 후 unique proxy 수 | ≤ 3 - 1 = 2 | ≤ 15 - 1 = 14 (충분) |

### 필요한 proxy_top_k 하한

dedup 후 각 position에서 `fo` 개의 unique 토큰이 필요. Worst case overlap:
- `draft_set` 크기 = `mesa_draft_fan_out` (보통 1)
- 겹침 최대 = `min(draft_fan_out, proxy_top_k)`
- 최악: draft의 모든 토큰이 proxy top-k 안에 있음 → unique proxy = `proxy_top_k - draft_fan_out`

조건: `proxy_top_k - draft_fan_out ≥ max(fan_out_list)`  
⇒ `proxy_top_k ≥ max(fan_out_list) + draft_fan_out`

실험 세팅에서 max(fan_out_list)는 보통 `sum(fan_out_list) = pfo × (K+1) = 2×7 = 14` 가 상한 (모든 예산이 한 position에 몰릴 때). 안전 여유 +2 → `proxy_top_k = 16` 권장.

간단히 **`proxy_top_k = pfo × (K+1) + draft_fan_out + 2`** 로 자동 계산 (config에서).

### 수정 계획

**1단계 — `mesa_proxy_top_k` 기본값 재산정 (config.py)**
```python
# ssd/config.py (MESA validation 블록)
if self.mesa_enabled:
    ...
    # Rev1 default: proxy_top_k 이 dedup 후 worst case에서 부족하지 않도록 자동 설정
    pfo = self.mesa_proxy_fan_out  # 이미 계산된 값
    K_plus_1 = self.speculate_k + 1
    max_possible_fo = pfo * K_plus_1  # fan_out_list 전체 예산이 한 position에 몰린 worst case
    required_top_k = max_possible_fo + self.mesa_draft_fan_out + 2  # +2 safe margin
    if self.mesa_proxy_top_k < required_top_k:
        print(f'[Config] mesa_proxy_top_k raised from {self.mesa_proxy_top_k} '
              f'to {required_top_k} (to eliminate draft fallback)')
        self.mesa_proxy_top_k = required_top_k
```

K=6, pfo=2 → `max_possible_fo = 14`, `required_top_k = 14+1+2 = 17`. 현재 default 3보다 훨씬 큼.

**2단계 — draft fallback 로직 제거 (draft_runner.py:1062-1093)**

```python
# 제거 대상:
logits_fb = glue_logits.clone()
logits_fb[:, :-1, :] = logits_fb[:, :-1, :].scatter(...)
total_need = max(...)
_, fallback_topk = torch.topk(logits_fb, total_need, dim=-1)
fallback_cpu = fallback_topk.cpu().tolist()

# for pos in range(K):
#     ... proxy dedup ...
#     if len(selected) < fo:
#         used = draft_set | set(selected)
#         fb = [t for t in fallback_cpu[b][pos] if t not in used]
#         selected.extend(fb[:fo - len(selected)])
```

교체:
```python
# Proxy only: target의 residual top-k에서 draft 제외하고 뽑음
# proxy_top_k가 충분히 크므로 dedup 후에도 fo개 확보 가능 (config에서 보장)
for pos in range(K):
    proxy_tokens = proxy_cpu[b][pos]
    draft_set = set(draft_cpu[b][pos])
    selected = [t for t in proxy_tokens if t not in draft_set][:fo]

# pos == K (all-accept): 여기는 proxy 없음 → draft logits top-k 그대로 OK
# (All-accept position은 정의상 target이 전부 수락 예상 → correction 불필요)
# 기존 로직 유지 (draft-sourced fallback)
```

**3단계 — underfill assert (탐지, debug)**
```python
# 극히 드문 edge case (proxy_top_k 설정보다 fan_out_list가 커진 경우) 탐지
if __debug__:
    assert len(selected) >= fo, \
        f"MESA underfill: pos={pos} fo={fo} got={len(selected)} " \
        f"(proxy_top_k={proxy_top_k}, needed ≥ {fo + draft_fan_out})"
```

**4단계 — 테스트 / 검증**
- `mesa_proxy_top_k` 자동 산정 결과 로그 확인
- Underfill assert가 트리거되지 않는지 100 step 이상 run
- accept rate 전후 비교 (fallback 제거 효과)

### `all-accept position` (pos=K) 처리

Pos=K는 "앞의 모든 position이 accept되면" 도달하는 가상 position. Proxy residual 없음 (target은 accept 성공 전제). 여기는 **draft logits top-k를 그대로 써도 무방** — 이건 fallback이 아니라 올바른 동작.

따라서 fallback 제거는 `pos < K` 케이스에만. `pos == K`는 기존 로직 유지.

### 작업량
- Config 자동 산정: ~10 LoC
- Fallback 로직 제거: **-15 LoC** (순 감소)
- Assert 추가: 3 LoC
- **합계 순 -2 LoC** (코드 더 짧아짐)

### 예상 이득
- **Correctness**: tree의 모든 Phase-2 branch가 target-informed — MESA 설계 취지 그대로 반영
- **Accept rate 개선 예상**: 현재 fallback으로 채워지던 slot들이 실제로는 낭비 branch였음. 이들이 진짜 correction으로 교체되면 accept 상승
- **NCCL payload**: 216 B → ~720 B (무시)
- **Target compute**: residual topk k 증가 ~+50-100 µs GPU (실질 무시)
- **Draft compute**: fallback topk 제거 → -30 µs
- **코드 단순화**: fallback 분기 사라짐, #1 (Python loop 벡터화) 도 더 쉬워짐

### #1과의 관계

`#1` (proxy selection 벡터화) 구현 시 fallback 분기가 있으면 2 경로 (proxy/fallback 혼합) 벡터화가 복잡해짐. **Fallback 먼저 제거 → #1 벡터화 단순**.

따라서 작업 순서: **#4 → #1**.

---

## #D `_decode_tree()` 진입 setup 비용: spec 버퍼 pre-allocation (High — 새 항목)

### 위치
- `ssd/engine/draft_runner.py:879-904` (`_decode_tree`)

### 현재 문제

매 `_decode_tree()` 호출(= Phase 1, 2 각 1회/step)마다:

```python
# Line 888-895
spec_tokens = torch.zeros((N, K), dtype=torch.int64, device=self.device)
spec_logits = torch.zeros((N, K, V), dtype=self.hf_config.torch_dtype, device=self.device)
spec_activations = torch.zeros((N, K, hidden_size), ...) if use_eagle else None

# Line 902-904
_, step_rope_positions, step_context_lens, step_slot_maps = self._compute_step_positions_and_slot_maps(...)
```

할당 크기 (K=6, V=32000, fp16, CodeLlama-34B):
- Phase 1 (MQ_LEN=7): `spec_logits = 7 × 6 × 32000 × 2 = 2.7 MB`
- Phase 2 (MQ_LEN=14): `spec_logits = 14 × 6 × 32000 × 2 = 5.4 MB`
- **매 step 8 MB new allocation + zero-fill**

`_compute_step_positions_and_slot_maps`는 `torch.arange` + `%` + `//` + `gather` 연쇄 GPU ops.

### 측정된 영향

Timeline에서 관찰:
| 구간 | 측정 gap |
|------|:---:|
| phase1_build end → phase1_prep start | **0.32 ms** |
| phase2_build end → phase2_prep start | **0.49 ms** |
| **매 step 총 gap** | **~0.8 ms** |

이 0.8 ms 중 상당 부분이 `_decode_tree` 진입 직후 spec 버퍼 할당 + setup compute.

### 수정 계획

**Rev1 불변식 (static prealloc이 dynamic fan_out_list에도 안전한 이유)**:

Target의 `_compute_and_send_proxy` 가 `fan_out_list`를 `sum(fan_out_list) == mesa_proxy_fan_out × (K+1)` 로 항상 맞춰서 전송함 (`verifier.py:226, 242-247`에서 redistribute로 보장). 따라서:
- `step_proxy_layout.MQ_LEN == self.proxy_layout.MQ_LEN` (static class attr) **항상**
- fan_out_list의 **per-position 분포**는 바뀌지만 **총량은 고정**

즉 Policy A runtime layout이 dynamic이어도 **proxy buffer 총 크기는 static MQ_LEN 예산으로 충분**. 나중에 "dynamic fanout인데 static buffer로 가능한가?" 재의심 방지를 위해 이 불변식을 코드 주석에도 넣어야 함.

**1단계: DraftRunner에 pre-allocated 버퍼 (`_init_prealloc_buffers` 확장)**

```python
# draft_runner.py __init__ / _init_prealloc_buffers에 추가
# Rev1 불변식: sum(fan_out_list) 항상 고정 (proxy_fan_out × (K+1))
# → proxy_layout.MQ_LEN이 runtime max_N의 상한
mq_list = [self.draft_layout.MQ_LEN, self.proxy_layout.MQ_LEN]
if hasattr(self, 'full_layout'):
    mq_list.append(self.full_layout.MQ_LEN)
max_mq = max(mq_list)
max_N = self.config.max_num_seqs * max_mq  # B * MQ_LEN upper bound

K = self.config.speculate_k
V = self.hf_config.vocab_size
H = self.hf_config.hidden_size

self._spec_tokens_buf = torch.empty((max_N, K), dtype=torch.int64, device=self.device)
self._spec_logits_buf = torch.empty((max_N, K, V), dtype=self.hf_config.torch_dtype, device=self.device)
if self.config.use_eagle:
    self._spec_activations_buf = torch.empty((max_N, K, H), dtype=self.hf_config.torch_dtype, device=self.device)
```

**2단계: `_decode_tree` 버퍼 재사용**
```python
def _decode_tree(self, payload, layout=None):
    _layout = layout or self.full_layout
    B, K, F, N = payload["metadata_ints"]
    V = self.hf_config.vocab_size

    # 기존: torch.zeros(...) 새로 할당
    # 수정: 미리 할당된 버퍼 슬라이스, 필요 시 zero_ 또는 그냥 덮어쓰기
    spec_tokens = self._spec_tokens_buf[:N, :K]
    spec_logits = self._spec_logits_buf[:N, :K, :V]
    # spec_tokens/logits는 매 iter에서 full-overwrite되므로 zero_ 불필요
    # (단, scatter/indexing으로 일부만 쓰는 경우 zero_ 필요 — 확인 필요)
    spec_activations = self._spec_activations_buf[:N, :K, :H] if self.config.use_eagle else None
    
    # 나머지는 동일
    ...
```

**3단계: `_compute_step_positions_and_slot_maps`는 layout별로 캐시 가능한 부분 확인**
- `step_pos_offsets`, `step_rope_offsets`는 이미 `layout.step_pos_offsets` / `layout.step_rope_offsets`로 pre-computed (tree_layout.py에)
- `step_positions = initial_positions[None, :] + _layout.step_pos_offsets` — layout 고정이면 initial_positions만 바뀜
- `step_slot_maps` 계산은 dbt에 의존 — 매 step 재계산 불가피
- 하지만 일부 intermediate tensor (arange, batch_indices)는 cache 가능
- 개선폭 작을 듯. 1,2단계 먼저.

### 주의사항

- `spec_logits`을 `torch.empty`로 쓰고 안 초기화해도 되는지 확인 필요 — `_decode_tree_step` 내부에서 `spec_logits[:, depth, :] = logits_flat` 로 per-depth 전체 덮어쓰기 이루어지는지 검증
- EAGLE 경로는 spec_activations도 동일하게 처리
- Graph pool 캡처 경로와 상호작용 체크 (CudaGraph가 이 버퍼를 캡처 범위에 포함하는지)

### 작업량
- pre-allocated 버퍼: ~15 LoC (init) + ~5 LoC (교체)
- **총 ~20 LoC**

### 예상 이득
- 매 step **−0.5~0.8 ms** (phase1_build→phase1_prep 및 phase2_build→phase2_prep gap 제거)
- 전체 MESA step **−1~2%**

### 왜 이 항목이 실질 중요한가

`_decode_tree` 진입 setup은 **Phase 2-only 구조적 비용**이 아니라 **Phase 1에도 동일하게 걸리는 고정 오버헤드**. 즉 MESA뿐 아니라 baseline에도 있는 비용이지만, MESA는 Phase 1/2를 **2번** 거치므로 오버헤드가 2배. #1 (Python loop 벡터화)과 함께 Phase 2 비-replay 영역의 주요 제거 대상.

---

## #5 `topk_probs` — **DEFER** (이번 Rev1에서 손대지 않음)

### 위치
- `ssd/engine/verifier.py:207-211, 222, 252-255`
- `ssd/engine/draft_runner.py:998-999`

### 현재 상태
- Target: `residual.topk` → `topk_probs` 계산 + normalize + NCCL send
- Draft: `_unpack_mesa_proxy`에서 dict에 들어가지만 Policy A selection에서 **미사용**

### 결정: defer

- Rev1만 보면 dead payload, 제거하면 compute/comm 소폭 절약
- 하지만 **Policy B (joint `h_i × r_i(v)` 기반 selection)** 도입 예정
- 지금 제거했다가 Policy B 시 다시 추가는 churn
- 현재 overhead는 target compute ~0.2 ms, NCCL +72 B/step — Rev1에서 무시 가능

### 조치
이번 Rev1 수정 범위에 포함 안 함. 코드에 주석만 1줄 추가하여 Policy B 계획 표시 (선택 사항):

```python
# verifier.py:209 위 선택적 주석
# NOTE: topk_probs currently unused by Policy A. Kept for Policy B (joint r_i(v) weighted selection).
```

주석도 안 달아도 무방 (이 문서가 해당 역할).

---

## #8 Dead code: `get_forked_recovery_tokens_from_logits(..., mesa_proxy=...)` 제거

### 위치
- `ssd/utils/async_helpers/async_spec_helpers.py:26` (함수 시그니처)
- `ssd/utils/async_helpers/async_spec_helpers.py:57-90` (mesa_proxy 분기)

### 현재 상태
- `draft_runner.py:786`은 MESA off / async non-MESA 경로에서 `mesa_proxy=None`으로 호출
- MESA 경로는 `_build_tree_batch_mesa` → `_select_proxy_sourced_tokens_policy_a` 직접 사용
- 따라서 `async_spec_helpers.py:57-90`의 `if mesa_proxy is not None:` 블록은 **dead**

### 수정 계획

**Option A (보수적)**: mesa_proxy 분기 및 파라미터 제거
```python
# 기존
def get_forked_recovery_tokens_from_logits(config, logits, cache_hits, returned_tokens, tokenizer, mesa_proxy=None):
    ...
    if mesa_proxy is not None:
        # 30+ lines of dead code
        ...

# 수정
def get_forked_recovery_tokens_from_logits(config, logits, cache_hits, returned_tokens, tokenizer):
    ...  # mesa_proxy 분기 완전 삭제
```

**Option B (주석만)**: 
```python
def get_forked_recovery_tokens_from_logits(..., mesa_proxy=None):
    ...
    # DEPRECATED: MESA Rev1은 _select_proxy_sourced_tokens_policy_a를 직접 사용함
    # 이 분기는 legacy fallback용. Rev2 정리 시 제거 예정.
    if mesa_proxy is not None:
        ...
```

**추천 Option A**. 진짜 dead이고 (`grep`으로 호출부 0건 확인), 주석만 남기면 또 잊어버림.

### 작업량
- `async_spec_helpers.py`: ~35 LoC 삭제 + signature 정리
- `tests.py` 확인 (mesa_proxy kwarg 쓰는지) — 호출부 0

---

## 전체 요약

| # | 항목 | 유형 | LoC | 예상 이득 |
|:-:|------|------|:---:|:---:|
| 3 | B=1 assert | correctness | 3 | correctness 보장 |
| **4** | **proxy_top_k 확대 + fallback 제거** | design/correctness | **-2 (순)** | tree 전체가 target-informed → accept ↑ |
| **D** | **`_decode_tree` spec 버퍼 pre-allocate** | perf | ~20 | step −0.5~0.8 ms |
| 1 | Proxy selection 벡터화 | perf/critical path | ~40 (fallback 제거로 단순해짐) | step −0.8 ms |
| 8 | Dead code 제거 | cleanup | -35 | maintainability |
| 2 | ~~TreeLayout LRU cache~~ | **제거**: hit rate 낮을 것, 이득 작음 | — | — |
| 5 | `topk_probs` | **defer** (Policy B 예정) | 0 | — |
| 6 | verify padding cat | **defer** (B=1 조건부) | 0 | — |

**총 작업량**: 약 63 LoC 추가 + 35 LoC 제거 = **순 +28 LoC**  
**예상 MESA throughput 개선**: **+2-4%** (주로 #D, #1)  
**Accept rate 개선 예상**: +수 %p (#4 — fallback 제거, 모든 Phase-2 branch가 target-informed)  
**Correctness 보강**: B=1 불변, underfill 제거

## 작업 순서

1. **#3 B=1 assert** (3 LoC, 즉시 — 가장 간단)
2. **#4 proxy_top_k 확대 + fallback 제거** (design change, #1 전에 해야 벡터화 단순)
3. **#D `_decode_tree` spec 버퍼 pre-allocate** (매 step 8 MB 재할당 제거, 가장 큰 perf 이득)
4. **#1 Proxy selection 벡터화** (fallback 제거된 상태에서 pure GPU 로 재작성)
5. **#8 Dead code 제거** (최종 clean-up)

## #6 `run_mesa_verify_cudagraph` padding 5× `torch.cat` — **DEFER** (조건부)

### 위치
- `cudagraph_helpers.py:1079-1092`

### 상태
```python
if wrapper_bs > orig_bs:
    input_ids = torch.cat([input_ids, torch.zeros(...)])
    positions = torch.cat([...])
    slot_mapping = torch.cat([...])
    block_tables = torch.cat([...])
    context_lens = torch.cat([...])
```
- **B=1 + `graph_bs_list[0]==1`이면 `wrapper_bs == orig_bs`** → 해당 path 실행 안 됨 (현재 모든 실험 조건)
- B>1 세팅이나 bucket이 1부터 시작 안 하면 매 step 5× cat 발생
- MESA 경로에만 복제된 오버헤드라 기존 SSD 공통 이슈와 별도

### 결정: defer
- 현재 실험 조건 (B=1)에선 dead path — 성능 영향 0
- B>1 확장 시 반드시 수정 대상 → 그때 처리
- Rev1 마무리엔 포함 안 함

### Rev2/B>1 시 수정 방향
- `graph_vars["input_ids"]` 등에 직접 pad 영역 write
- `torch.cat`로 새 tensor 만들지 말 것

---

## 제외된 항목 (성능 영향 사실상 0, defer도 불필요)

- **`fan_idx` helper vectorize** — B=1에서 comprehension iter 1회. 영향 없음
- **Hot-path 내부 import** — Python import cache로 사실상 비용 0 (첫 호출 후 µs 이하)
- **MESA profiling flush wiring** — 1 generate / process 구조에서 이미 동작
- **`phase1_build` / `phase2_build` label 세분화** — 측정 품질 개선이지만 Rev1 마무리엔 불필요

이 4개는 B>1 / multi-run / 정밀 분석이 필요해질 때 재검토. 현재는 maintainability 이슈로만 간주.
