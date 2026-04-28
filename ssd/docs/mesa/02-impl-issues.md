# MESA-SSD 구현 이슈 트래커

원본은 `IMPL_ISSUE.md` (v1 구현 이슈), `MESA-rev1-problems.md` (Rev1 수정
항목), `MESA-PHASE2-HYBRID-ISSUE.md` (Phase 2 hybrid 이슈) 세 파일이다.
시간순으로 통합.

> **읽는 순서**: Part 1 (v1) → Part 2 (Rev1) → Part 3 (Phase 2 Hybrid).
> Phase 2 Hybrid 이후 발견된 회귀 (split 경로 sync, per-depth label,
> phase2_hybrid_build 비효율) + 그에 대한 fix 도 Part 3 안에 시간순으로 기록.

---

## Part 1. v1 구현 이슈 (계획 → 실제)

### 해결된 이슈

#### ISSUE-001: attention.py + context.py layout-aware

**해결**: `context.py` 에 `active_mq_len` / `active_wrappers` 추가.
`attention.py` 의 tree_decode 분기에서 context 기반 wrapper selection
구현. 계획 3 단계에서 예정했으나 7 단계 구현 시 함께 적용.

#### ISSUE-002: Budget split 2-pass

**해결**: 초기에는 proxy token swap (full_layout 단일 decode) 으로
우회했으나, ISSUE-003 해결 후 계획대로 `draft_layout + proxy_layout`
2-pass decode 로 구현.

#### ISSUE-003: Layout CudaGraph 캡처 hang

**해결**. 4 가지 원인을 모두 수정:

1. `capture_fi_tree_decode_cudagraph` 에서 `prefill_wrappers[bs]` 가 항상
   full layout wrapper 를 가져옴 → layout 별 wrapper 선택으로 변경
2. `set_context` 에 `active_mq_len` / `active_wrappers` 전달 (캡처 시에도)
3. `outputs` / `logits` 텐서의 dtype 미명시 → `dtype=hf_config.torch_dtype`
   추가
4. 모듈 전역 `cache` dict 가 layout 간 공유 → `cache.clear()` on MQ_LEN
   change

### 미해결 이슈

#### ISSUE-004: Throughput 하락 (-43%)

**실측 결과** (비어있는 GPU 기준, 10 seqs × 256 tokens):

- Baseline: 149 tok/s
- MESA: 84 tok/s (-43%)

**주 원인**: 2-pass tree decode 의 구조적 비용

1. 2× CudaGraph replay (K steps × 2 passes) — 각 replay 마다 고정 overhead
2. 2× FlashInfer wrapper.plan() — batch-dependent KV metadata 재구성
3. 2× batch-dependent packed mask 재구성 — context_lens, block_tables,
   cache_hits 에 의존하여 매 pass 필수
4. `_build_tree_batch()` full layout tree args 중복 생성 (~2 ms)

**이전 분석 정정**: mask cache clear 가 주 원인이라고 했으나,
layout-independent 한 것은 `glue_hit_np` / `glue_miss_np` (~0.1 ms) 뿐.
나머지 precompute 는 batch 의존이라 layout 분리로 절약 불가.

#### ISSUE-005: Llama2-13B/70B triton 에러

SSD KV cache copy 커널에서 `tl.arange(0, D)` — `D` 가 non-power-of-2
이면 에러. Llama2-13B (hidden=5120, heads=40) 가 TP 분할 후 해당. MESA 와
무관한 SSD 자체 이슈. Llama2-7B (hidden=4096, heads=32) 로 대체.

### 계획 대비 달라진 점

#### 1. Token swap → 2-pass (계획 실현)

계획은 처음부터 2-pass 였으나 구현 과정에서 ISSUE-003 으로 인해
일시적으로 token swap (full_layout 단일 decode) 으로 우회. ISSUE-003 해결
후 계획대로 2-pass 구현.

#### 2. irecv 위치

계획: `_build_tree_batch` 내부에서 glue decode 와 fork token 사이에 irecv.

실제: `_build_tree_batch_mesa` 시작 시 irecv 를 가장 먼저 post (line
1030). Target send 와의 overlap 최대화.

**결과**: `proxy_wait = 0.0 ms` (완벽한 overlap).

#### 3. _build_tree_batch 리팩토링 불완전

계획: glue decode 로직을 분리하여 MESA 전용 경로로 재사용.

실제: `_build_tree_batch` 를 그대로 호출하고 `_mesa_glue_logits` /
`_mesa_gd_for_fork` 를 tree_decode_args 에 추가하여 반환. Full layout
tree_decode_args 가 불필요하게 구축됨 (~2 ms 낭비). Rev1 에서 `_glue_decode`
분리로 정리.

#### 4. Global cache 문제 발견 및 해결

계획에 없던 이슈. `cudagraph_helpers.py` 의 `cache = {}` (모듈 전역) 가
draft / proxy pass 간 공유. Layout 변경 시 `cache.clear()` 추가로 해결.

#### 5. FlashInfer wrapper 초기화 순서

계획: TreeLayout 생성 후 wrapper 생성.

실제: `_init_flashinfer_wrappers()` 가 `ModelRunner.__init__()` 안에서
호출되므로, config 값 (`mesa_draft_fan_out`, `mesa_proxy_fan_out`) 으로
직접 MQ_LEN 계산하여 wrapper 생성. Layout 객체는 이후 `_init_prealloc_buffers()`
에서 생성.

#### 6. run_model() tree decode dispatch

계획: `_decode_tree` 에 layout 을 전달하면 자동으로 올바른 CudaGraph 사용.

실제: `_decode_tree` → `_decode_tree_step` → `set_context(active_mq_len=...)`
→ `run_model` → context 에서 layout 읽어 `graph_vars[layout.graph_key]`
dispatch. 간접적이지만 동작.

#### 7. Llama2 지원

계획: Llama-only assert 후 Llama2 자동 지원.

실제: sentencepiece 설치 필요 + `use_fast=False` fallback 추가 + 13B 는
triton 에러로 7B 사용.

#### 8. mesa_budget_mode 파라미터 미구현

계획: `token_swap` / `h_redistribute` / `outcome_posterior` 모드.

실제: 2-pass budget split 만 구현. 모드 선택 파라미터 없음. Rev1 에서
Policy A 로 `h_redistribute` 만 추가.

#### 9. _select_proxy_sourced_tokens Python loop

계획: vectorized 구현.

실제: `for b in range(B): for pos in range(K):` Python loop. B=1 에서는
무시 가능 (~0.9 ms). Rev1 #1 에서 벡터화 예정.

#### 10. Hot path overhead 수정

`mesa_enabled=False` 일 때 `_decode_tree_step` 과 `run_model` tree decode
분기에서 불필요한 layout 체크 코드가 매 step 실행됨. `if self.config.mesa_enabled:`
가드를 추가하여 기존 경로 보호.

#### 11. 타이밍 코드 sync overhead

`_build_tree_batch_mesa` 의 `torch.cuda.synchronize()` 8 개가 GPU
파이프라인을 깨뜨림. 제거 완료.

---

## Part 2. Rev1 변경사항 (요약)

| 항목 | 내용 |
|------|------|
| Rev1-1 | Glue decode 분리 (`_glue_decode()` 함수 추출) — full tree args 구축 없이 glue decode 만 호출 |
| Rev1-2 | Target 측 `ĥ_i` + `fan_out_list` 계산 (`_compute_and_send_proxy`). Draft 에서 `ĥ_i` 계산 불필요. 전송 포맷: `[fan_out_list(K+1), topk_ids(B*K*top_k), topk_probs(B*K*top_k)]` |
| Rev1-3 | Policy A 동적 fan_out — `_select_proxy_sourced_tokens_policy_a()`, runtime `create_tree_layout(fan_out_list=...)`, `Context.active_layout` 으로 전달 |

### Rev1 실험 결과 (간단 요약, 자세한 수치는 03-results.md)

- Llama3-8B: Policy A 가 v1 (고정) 보다 약간 하락 (accept 0.83 → 0.79)
- Llama2-7B: Cache hit 0.61 → 0.80 (+31%), accept 0.58 → 0.61 (+5%)
- Throughput: 두 모델 모두 -29~44% (2-pass 구조적 비용, v1 과 동일)

---

## Part 3. Rev1 수정 대상 (`MESA-rev1-problems.md`)

### 작업 순서 결정

수정 순서:

1. #3 B=1 assert 추가 (correctness, 3 LoC)
2. #4 proxy_top_k 확대 + draft fallback 제거 (design + correctness)
3. #D `_decode_tree()` 진입 setup 버퍼 pre-allocate (매 step 8 MB 재할당
   제거)
4. #1 Proxy selection 벡터화 (fallback 제거된 상태에서 pure GPU)
5. #8 Dead code 제거

### 제외 / Defer

- #2 TreeLayout LRU cache — **제거**: Policy A 의 `fan_out_list` 가
  step 마다 바뀌어 hit rate 낮을 가능성 높음, 기대 이득도 작음 (~0.2 ms)
- #5 `topk_probs` — **defer**: Policy B 도입 시 활성화
- #6 verify padding cat — **defer**: B=1 전제 하에 trigger 안 됨

---

### #3 B=1 only 강제 (Correctness)

**위치**:

- `ssd/config.py:100` (MESA 검증 블록)
- `ssd/engine/verifier.py:227` (`accept_probs[0]`)
- `ssd/engine/draft_runner.py:1171` (단일 fan_out_list 사용)

**문제**:

```python
# verifier.py:227
cumprod = torch.cumprod(accept_probs[0], dim=0)  # [K] (B=1 scope)
# ↑ 0번째 seq만 사용. 주석엔 "B=1 scope" 라 돼있지만 런타임 assert 없음

# draft_runner.py: fan_out_list를 batch 전체에 적용
# B>1 일 때 seq 0의 ĥ_i 분포를 seq 1, 2 에 강제 적용 → 잘못된 token 선택
```

**영향**:

- 현재 실험 전부 B=1 → correctness 문제 발현 안 됨
- 누군가 `--b 2` 이상으로 MESA 돌리면 silent correctness bug
- 결과 metric 은 나오지만 proxy 가 잘못 작동 → accept rate 하락

**수정**:

```python
# config.py MESA 검증 블록
if self.mesa_enabled:
    ...
    assert self.max_num_seqs == 1, \
        "MESA Rev1 only supports B=1 (max_num_seqs=1); " \
        "Policy A uses accept_probs[0] as a single h_i distribution for the whole batch"
```

**작업량**: 3 LoC

---

### #4 Underfill: Target proxy_top_k 확대 + draft fallback 제거 (Design + Correctness)

#### 왜 fallback 이 문제인가

현재 `_select_proxy_sourced_tokens_policy_a` (draft_runner.py:1054-1102)
는 두 소스에서 토큰을 뽑음:

1. **Proxy** — target 의 residual `(p_E - p_D).clamp(min=0)` 의 top-k.
   "target 분포에서 유의미하지만 draft 는 놓친" 토큰.
2. **Fallback** — draft logits 자체의 top-k. dedup 후 부족하면 여기서 채움.

MESA 의 철학은 **"target 이 draft 에게 유용한 correction 을 알려준다"**.
그런데 fallback 은:

- Draft 가 이미 argmax / top-k 로 뽑아둔 것과 **같은 분포** 에서 추가로
  뽑는 것
- 즉 "draft 가 두 번째로 예측하는 것" — target 정보 0
- 이런 토큰을 tree 에 넣어봤자 target 이 선호할 가능성은 proxy 보다 현저히 낮음
- 결과: 실효 tree 유효 branch 수가 줄어듦 → accept rate 저하

**근본 원인**: target 의 `mesa_proxy_top_k` (default 3) 가 너무 작아서
draft 가 부족분을 fallback 으로 채울 수밖에 없는 구조.

#### 올바른 설계

**Target 이 넉넉한 residual top-k 를 보낸다 → draft 는 fallback 필요 없다.**

| 항목 | 현재 | 제안 |
|------|:---:|:---:|
| `mesa_proxy_top_k` default | 3 | **15** (또는 `pfo*(K+1) + dfo + 2`) |
| Target residual.topk compute | top-3 | top-17 (~+50-100 µs GPU) |
| NCCL payload (K=6) | 216 B | ~720 B |
| Draft fallback path | proxy 부족 시 draft logits 사용 | **제거** |
| Dedup worst case 후 unique proxy | ≤ 3 - 1 = 2 | ≤ 15 - 1 = 14 |

#### 필요한 proxy_top_k 하한

dedup 후 각 position 에서 `fo` 개의 unique 토큰이 필요. Worst case overlap:

- `draft_set` 크기 = `mesa_draft_fan_out` (보통 1)
- 겹침 최대 = `min(draft_fan_out, proxy_top_k)`
- 최악: draft 의 모든 토큰이 proxy top-k 안에 있음 → unique proxy =
  `proxy_top_k - draft_fan_out`

조건: `proxy_top_k - draft_fan_out ≥ max(fan_out_list)`
⇒ `proxy_top_k ≥ max(fan_out_list) + draft_fan_out`

실험 세팅에서 `max(fan_out_list)` 는 보통 `sum(fan_out_list) = pfo × (K+1)`
가 상한 (모든 예산이 한 position 에 몰릴 때). 안전 여유 +2 →
`proxy_top_k = pfo × (K+1) + dfo + 2`.

#### 수정 계획

**1단계 — 자동 산정 (config.py)**

```python
if self.mesa_enabled:
    pfo = self.mesa_proxy_fan_out
    K_plus_1 = self.speculate_k + 1
    max_possible_fo = pfo * K_plus_1
    required_top_k = max_possible_fo + self.mesa_draft_fan_out + 2
    if self.mesa_proxy_top_k < required_top_k:
        print(f'[Config] mesa_proxy_top_k raised from {self.mesa_proxy_top_k} '
              f'to {required_top_k} (to eliminate draft fallback)')
        self.mesa_proxy_top_k = required_top_k
```

`K=6, pfo=2` → `max_possible_fo = 14`, `required_top_k = 14+1+2 = 17`.

**2단계 — fallback 로직 제거 (draft_runner.py:1062-1093)**

제거 대상:

```python
logits_fb = glue_logits.clone()
logits_fb[:, :-1, :] = logits_fb[:, :-1, :].scatter(...)
total_need = max(...)
_, fallback_topk = torch.topk(logits_fb, total_need, dim=-1)
fallback_cpu = fallback_topk.cpu().tolist()

for pos in range(K):
    ...
    if len(selected) < fo:
        used = draft_set | set(selected)
        fb = [t for t in fallback_cpu[b][pos] if t not in used]
        selected.extend(fb[:fo - len(selected)])
```

교체:

```python
for pos in range(K):
    proxy_tokens = proxy_cpu[b][pos]
    draft_set = set(draft_cpu[b][pos])
    selected = [t for t in proxy_tokens if t not in draft_set][:fo]

# pos == K (all-accept): 기존 draft top-k 로직 유지 (correction 불필요한 위치)
```

**3단계 — underfill assert (debug)**

```python
if __debug__:
    assert len(selected) >= fo, \
        f"MESA underfill: pos={pos} fo={fo} got={len(selected)} " \
        f"(proxy_top_k={proxy_top_k}, needed ≥ {fo + draft_fan_out})"
```

#### `all-accept position` (pos=K) 처리

Pos=K 는 "앞의 모든 position 이 accept 되면" 도달하는 가상 position.
Proxy residual 없음 (target 은 accept 성공 전제). 여기는 **draft logits
top-k 를 그대로 써도 무방** — 이건 fallback 이 아니라 올바른 동작.

따라서 fallback 제거는 `pos < K` 케이스에만. `pos == K` 는 기존 로직 유지.

#### 작업량

- Config 자동 산정: ~10 LoC
- Fallback 로직 제거: **-15 LoC** (순 감소)
- Assert 추가: 3 LoC
- **합계 순 -2 LoC** (코드 더 짧아짐)

#### 예상 이득

- **Correctness**: tree 의 모든 Phase-2 branch 가 target-informed
- **Accept rate 개선 예상**: 현재 fallback 으로 채워지던 slot 들이 실제로는
  낭비 branch 였음
- **NCCL payload**: 216 B → ~720 B (무시)
- **Target compute**: residual topk k 증가 ~+50-100 µs GPU (실질 무시)
- **Draft compute**: fallback topk 제거 → -30 µs
- **코드 단순화**: fallback 분기 사라짐, #1 (Python loop 벡터화) 더 쉬움

#### #1 과의 관계

`#1` (proxy selection 벡터화) 구현 시 fallback 분기가 있으면 2 경로
(proxy/fallback 혼합) 벡터화가 복잡해짐. **Fallback 먼저 제거 → #1
벡터화 단순**. 따라서 작업 순서: **#4 → #1**.

---

### #D `_decode_tree()` 진입 setup 비용: spec 버퍼 pre-allocation (High)

**위치**: `ssd/engine/draft_runner.py:879-904` (`_decode_tree`)

#### 현재 문제

매 `_decode_tree()` 호출 (= Phase 1, 2 각 1회/step) 마다:

```python
spec_tokens = torch.zeros((N, K), dtype=torch.int64, device=self.device)
spec_logits = torch.zeros((N, K, V), dtype=self.hf_config.torch_dtype, device=self.device)
spec_activations = torch.zeros((N, K, hidden_size), ...) if use_eagle else None

_, step_rope_positions, step_context_lens, step_slot_maps = \
    self._compute_step_positions_and_slot_maps(...)
```

할당 크기 (K=6, V=32000, fp16, CodeLlama-34B):

- Phase 1 (MQ_LEN=7): `spec_logits = 7 × 6 × 32000 × 2 = 2.7 MB`
- Phase 2 (MQ_LEN=14): `spec_logits = 14 × 6 × 32000 × 2 = 5.4 MB`
- **매 step 8 MB new allocation + zero-fill**

`_compute_step_positions_and_slot_maps` 는 `torch.arange` + `%` + `//` +
`gather` 연쇄 GPU ops.

#### 측정된 영향

Timeline 에서 관찰:

| 구간 | 측정 gap |
|------|:---:|
| `phase1_build` end → `phase1_prep` start | 0.32 ms |
| `phase2_build` end → `phase2_prep` start | 0.49 ms |
| **매 step 총 gap** | **~0.8 ms** |

#### 수정 계획

**Rev1 불변식**: Target 의 `_compute_and_send_proxy` 가 `fan_out_list`
를 `sum(fan_out_list) == mesa_proxy_fan_out × (K+1)` 로 항상 맞춰서 전송.
따라서:

- `step_proxy_layout.MQ_LEN == self.proxy_layout.MQ_LEN` (static class
  attr) **항상**
- `fan_out_list` 의 **per-position 분포** 는 바뀌지만 **총량은 고정**

즉 Policy A runtime layout 이 dynamic 이어도 **proxy buffer 총 크기는
static MQ_LEN 예산으로 충분**.

**1단계: pre-allocated 버퍼 (`_init_prealloc_buffers`)**

```python
# Rev1 불변식: sum(fan_out_list) 항상 고정 (proxy_fan_out × (K+1))
mq_list = [self.draft_layout.MQ_LEN, self.proxy_layout.MQ_LEN]
if hasattr(self, 'full_layout'):
    mq_list.append(self.full_layout.MQ_LEN)
max_mq = max(mq_list)
max_N = self.config.max_num_seqs * max_mq

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

    # 미리 할당된 버퍼 슬라이스
    spec_tokens = self._spec_tokens_buf[:N, :K]
    spec_logits = self._spec_logits_buf[:N, :K, :V]
    spec_activations = self._spec_activations_buf[:N, :K, :H] if self.config.use_eagle else None
    ...
```

#### 주의사항

- `spec_logits` 을 `torch.empty` 로 쓰고 안 초기화해도 되는지 확인 — 
  `_decode_tree_step` 내부에서 `spec_logits[:, depth, :] = logits_flat`
  로 per-depth 전체 덮어쓰기 이루어지는지 검증
- Graph pool 캡처 경로와 상호작용 체크 (CudaGraph 가 이 버퍼를 캡처
  범위에 포함하는지)

#### 예상 이득

- 매 step **−0.5~0.8 ms** (gap 제거)
- 전체 MESA step **−1~2%**

#### 왜 이 항목이 실질 중요한가

`_decode_tree` 진입 setup 은 **Phase 2-only 구조적 비용** 이 아니라
**Phase 1 에도 동일하게 걸리는 고정 오버헤드**. 즉 MESA 뿐 아니라 baseline
에도 있는 비용이지만, MESA 는 Phase 1/2 를 **2 번** 거치므로 오버헤드가
2 배.

---

### #1 Proxy selection: Python loop + GPU→CPU sync 제거 (Critical)

**위치**:

- `ssd/engine/draft_runner.py:1009` (`_select_proxy_sourced_tokens`)
- `ssd/engine/draft_runner.py:1054` (`_select_proxy_sourced_tokens_policy_a`,
  Rev1 실사용 경로)

#### 현재 코드 문제

```python
draft_cpu = draft_forked[:, :K, :].cpu().tolist()
proxy_cpu = proxy_topk_ids.cpu().tolist()
fallback_cpu = fallback_topk[:, :K, :].cpu().tolist()

for b in range(B):
    for pos in range(K):
        draft_set = set(draft_cpu[b][pos])
        selected = [t for t in proxy_tokens if t not in draft_set]
        ...
```

#### 영향

- `phase2_build` 라벨 내부 ~0.8 ms 가 이 경로
- `.tolist()` 가 implicit GPU sync → stream stall
- 특히 Policy A 는 `proxy_recv_work.wait()` 직후 → draft critical path 직결

#### 수정 계획 (#4 이후 전제)

**pos < K 경로: proxy-only 벡터화**

```python
# 1. Proxy pool 자체 dedup (prefix duplicate mask)
in_prev = (proxy[..., None] == proxy[..., :i]).any(dim=-1)

# 2. draft_forked 와의 겹침 mask
in_draft = (proxy[..., None] == draft[:, :K, None, :]).any(dim=-1)

# 3. 유효 토큰 mask
valid_mask = ~in_prev & ~in_draft

# 4. fan_out_list[pos] 만큼 앞에서 pick
rank = valid_mask.cumsum(dim=-1) - 1
taken_mask = valid_mask & (rank < fan_out_tensor[pos, None])
result[b, pos] = proxy[b, pos][taken_mask]
```

**pos == K (all-accept) 경로: draft top-k 만 사용**

```python
logits_k = glue_logits[:, K, :].clone()
logits_k.scatter_(1, draft_forked[:, K, :], float('-inf'))
_, all_accept_topk = torch.topk(logits_k, fan_out_list[K], dim=-1)
```

#### Edge cases

- `fan_out_list[pos] == 0` → 해당 position skip (cumsum mask 로 자연스레
  0 개 pick)
- `fan_out_list[K] == 0` → all-accept path 자체 skip
- `underfill`: #4 로 proxy_top_k 충분 커서 안 생김. assert 만 debug build 에 남김

#### 작업량 / 예상 이득

- 메인 구현: ~40 LoC
- 예상 소요: 2-3 시간
- `phase2_build` 1.4 ms → **~0.4 ms**
- 전체 MESA step ~1 ms 단축 → throughput **+1.5%**

---

### #5 `topk_probs` — DEFER

**위치**:

- `ssd/engine/verifier.py:207-211, 222, 252-255`
- `ssd/engine/draft_runner.py:998-999`

**현재 상태**: Target 은 계산 + send, draft 는 dict 에 들어가지만 Policy
A selection 에서 미사용.

**결정**: defer (Policy B 도입 시 활성화). 코드에 1 줄 주석만 추가:

```python
# NOTE: topk_probs currently unused by Policy A. Kept for Policy B
#       (joint r_i(v) weighted selection).
```

---

### #6 `run_mesa_verify_cudagraph` padding 5× `torch.cat` — DEFER

**위치**: `cudagraph_helpers.py:1079-1092`

**상태**:

```python
if wrapper_bs > orig_bs:
    input_ids = torch.cat([input_ids, torch.zeros(...)])
    positions = torch.cat([...])
    slot_mapping = torch.cat([...])
    block_tables = torch.cat([...])
    context_lens = torch.cat([...])
```

- B=1 + `graph_bs_list[0]==1` 이면 `wrapper_bs == orig_bs` → 해당 path
  실행 안 됨 (현재 모든 실험 조건)
- B>1 세팅이나 bucket 이 1 부터 시작 안 하면 매 step 5× cat 발생

**결정**: defer. Rev2 / B>1 시 `graph_vars["input_ids"]` 등에 직접 pad
영역 write (`torch.cat` 으로 새 tensor 만들지 말 것).

---

### #8 Dead code 제거: `get_forked_recovery_tokens_from_logits(..., mesa_proxy=...)`

**위치**:

- `ssd/utils/async_helpers/async_spec_helpers.py:26` (함수 시그니처)
- `ssd/utils/async_helpers/async_spec_helpers.py:57-90` (mesa_proxy 분기)

**현재 상태**:

- `draft_runner.py:786` 은 MESA off / async non-MESA 경로에서
  `mesa_proxy=None` 으로 호출
- MESA 경로는 `_build_tree_batch_mesa` → `_select_proxy_sourced_tokens_policy_a`
  직접 사용
- 따라서 `async_spec_helpers.py:57-90` 의 `if mesa_proxy is not None:`
  블록은 **dead**

**수정 (Option A — 보수적)**: `mesa_proxy` 분기 및 파라미터 완전 제거.

**작업량**: ~35 LoC 삭제 + signature 정리.

---

## Part 4. 전체 요약

| # | 항목 | 유형 | LoC | 예상 이득 |
|:-:|------|------|:---:|:---:|
| 3 | B=1 assert | correctness | 3 | correctness 보장 |
| **4** | **proxy_top_k 확대 + fallback 제거** | design / correctness | **-2 (순)** | tree 전체가 target-informed → accept ↑ |
| **D** | **`_decode_tree` spec 버퍼 pre-allocate** | perf | ~20 | step −0.5~0.8 ms |
| 1 | Proxy selection 벡터화 | perf / critical path | ~40 | step −0.8 ms |
| 8 | Dead code 제거 | cleanup | -35 | maintainability |
| 2 | ~~TreeLayout LRU cache~~ | **제거** | — | — |
| 5 | `topk_probs` | **defer** (Policy B 예정) | 0 | — |
| 6 | verify padding cat | **defer** (B>1 시) | 0 | — |

**총 작업량**: 약 63 LoC 추가 + 35 LoC 제거 = **순 +28 LoC**

**예상 MESA throughput 개선**: **+2-4%** (주로 #D, #1)

**Accept rate 개선 예상**: +수 %p (#4 — fallback 제거, 모든 Phase-2
branch 가 target-informed)

**Correctness 보강**: B=1 불변, underfill 제거.

### 작업 순서

1. **#3 B=1 assert** (3 LoC, 즉시 — 가장 간단)
2. **#4 proxy_top_k 확대 + fallback 제거** (design change, #1 전에 해야
   벡터화 단순)
3. **#D `_decode_tree` spec 버퍼 pre-allocate** (매 step 8 MB 재할당 제거)
4. **#1 Proxy selection 벡터화** (fallback 제거된 상태에서 pure GPU 로
   재작성)
5. **#8 Dead code 제거** (최종 clean-up)

### 제외된 항목 (성능 영향 사실상 0)

- `fan_idx` helper vectorize — B=1 에서 comprehension iter 1 회. 영향 없음
- Hot-path 내부 import — Python import cache 로 사실상 비용 0
- MESA profiling flush wiring — 1 generate / process 구조에서 이미 동작
- `phase1_build` / `phase2_build` label 세분화 — 측정 품질 개선이지만
  Rev1 마무리엔 불필요

이 4 개는 B>1 / multi-run / 정밀 분석이 필요해질 때 재검토.

---

# Part 5. Phase 2 Hybrid 구현 이슈

`MESA-PHASE2-HYBRID-ISSUE.md` 의 issue tracker + 구현 중 발견된 모든 이슈를
시간순으로 통합. Step 1..9D 단위.

## 5.1 Phase 3b sub-staging (2026-04-26)

Plan 의 Phase 3 ("Phase 1 K1 split + hybrid Phase 2") 가 내부적으로 6 단계
복합 변경:
1. Phase 1 forward depth shortening (K_long → K1)
2. Phase 2 continuation (Phase 1 leaf 를 K2 만큼 추가 확장)
3. Phase 2 proxy depth shortening (K_long → K2)
4. continuation + proxy 단일 hybrid forward 통합
5. per-row attention plumbing (custom block tables, mask builder)
6. long-base CUDAGraph capture for hybrid path

각 commit 가 reviewable 하도록 sub-stage 로 분리:
- **3b.1** (Phase 1 K1 split): `phase1_layout_long` (forward_depth=K1) 추가.
  Phase 1 K1, Phase 2 continuation K2, Phase 2 proxy 는 K_long 유지. Cache
  emit 모양 무변동 (모든 row K_long).
- **3b.2** (Phase 2 proxy at K2): proxy forward depth K_long → K2.
  Proxy-sourced row suffix 가 K_short = K2. Cache emit 두 row class 분기.
- **3b.3** (single hybrid forward): cont + proxy 통합. Per-row attention
  plumbing + custom mask builder 가 여기서 lands. 가장 high-risk.

## 5.2 구현 단계별 이슈

### Step 1-7 — 기반 + parity harness

- **Step 4 mask builder** (`build_hybrid_packed_mask`): per-row j_idx-aware
  visibility (continuation: persistent + glue + own Phase 1 + own A_tail
  prefix; proxy: persistent + glue + own B_proxy prefix). FlashInfer 의
  LSB-first byte 단위 packing, padding 된 행은 zero-filled.

- **Step 5 5-region scratch slot layout**: cont rows 의 A_tail base =
  `phase1_kv_base + K1*MQ_p1`; proxy rows 의 B_proxy base = `a_tail_base +
  K2*MQ_p1`. depth-major 배치로 cont row i at depth d 는
  `cont_initial[i] + d*MQ_p1` slot 에 write.

- **Step 6 `_decode_phase2_hybrid` (eager)**: env-var gate 로 default 는
  split, `SSD_FORCE_HYBRID_PHASE2=1` 일 때 hybrid 사용. `cu_seqlens_q=None`
  으로 routing 해서 prefill_wrapper.run() 의 custom mask 가 attention
  branch 에 도달.

- **Step 7 parity harness**: split eager mirror 와 hybrid eager 의 self-
  consistency + cont oracle (`_decode_correct_split_cont`, plan-correct
  mask) + proxy oracle (`fresh_proxy_oracle` = use_cuda_graph=False
  wrapper) 3-way diff. 100% drift-consistent first-mismatches, 100% 자기-
  자기 deterministic, e2e text byte-identical.

### Step 7 root cause (2026-04-...)

- **`cu_seqlens_q` routing 버그**: `set_context(cu_seqlens_q=...)` 으로
  값 설정 시 attention.py 가 `flash_attn_with_kvcache` 로 분기 → 우리의
  custom_mask 무시되고 causal-only attention. Hybrid mask 의 의미가 무시
  되며 잘못된 KV 영역 attend.
- **해결**: hybrid 경로에서는 `cu_seqlens_q=None` 강제 → tree_decode 분기로
  routing → `prefill_wrapper.run()` 가 plan() 이 받은 custom_mask 적용.
- **active_mq_len=1**: hybrid 의 query 는 row 당 1 개라 wrapper.plan()
  이 사용하는 q layout 도 1.

### Step 8 — hybrid default 전환

- Hot path 흐름: Phase 1 → proxy recv/wait → phase2_build → hybrid →
  merge.
- 기본 = hybrid (when `mesa_phase1_k != None`), fallback `SSD_FORCE_SPLIT_PHASE2=1`.
- Parity harness `SSD_HYBRID_PARITY=1` 디버그용 보존.
- Native hybrid eager TPS 69.07 (split 86.94 대비 -20.6%) — CG 작업이
  필수임을 확인. → Step 9B 동기.

### Step 9A — runtime `valid_k` bucket dispatch

- `partial_tree_decode_args` 에 `valid_k` 추가 (cache hit row 의 valid_k).
- `phase1_layout_long`/`_short` dispatch.
- proxy step layout `position_count = valid_k+1`.
- `_select_proxy_sourced_tokens_policy_a(valid_k=...)` slice 적용.
- `_compute_hybrid_bool_mask_for_depth` `K_long → K_step` rename.
- `_decode_phase2_hybrid` long-only assert lift.
- `_merge_and_populate_cache(draft_layout=...)` 파라미터 추가.
- `run_fi_tree_decode_cudagraph` mask `K_for_mask = layout.position_count - 1`.
- `phase1_layout_short` CG + wrapper 추가.

### Step 9B — hybrid CG capture + dispatch

- **9B-0 glue_short**: `make_glue_decode_input_ids(valid_k=...)`,
  `prepare_glue_decode_ctxt(valid_k=...)`, `Context.glue_valid_k`,
  `capture_verify_cudagraph(k_plus_1=K_short+1)`. `run_model` 의
  `glue_valid_k` 분기로 verify_long / verify_short.
- **9B-1 hybrid CG infra**: `phase2_hybrid_long`/`_short` (eager) +
  `phase2_hybrid_long_cg`/`_short_cg` (use_cuda_graph=True) wrapper family.
  `capture_phase2_hybrid_cudagraph(bucket, K_step)`: bucket 별 single-depth
  CG, separate graph_pool. `run_phase2_hybrid_cudagraph`: K2 회 replay +
  per-depth metadata 를 `low_level_packed_plan` 으로 baked buffer 에 write.
- **9B-2 runtime dispatch**: 기본 = CG hybrid, `SSD_FORCE_EAGER_HYBRID_PHASE2=1`
  → eager hybrid fallback, `SSD_FORCE_SPLIT_PHASE2=1` → split fallback.
- **9B-3 short bucket oracle / parity**: oracle 함수에 `K_step = plan.valid_k`
  generalize + bucket tag 출력.

### 9B Tolerance gate (Step 8 entry rule)

8 prompts × 128 tok = 131 spec steps:

| 항목 | gate | 측정값 |
|---|---|---|
| self-consistency strict 0 | = 0 | 0/72 (drop-in deterministic) |
| proxy first-divergence drift-consistent (2*log_d bound) | 100% | 60/60 |
| cont first-divergence drift-consistent | 100% | 22/22 |
| proxy mismatch token rate | ≤ 3% | 2.10% |
| cont mismatch token rate | ≤ 0.6% | 0.51% |
| end-to-end text byte-identical | 100% | 8/8 |
| accept rate delta vs split | ≥ 0 | +11.3% |
| first-divergence max o_margin | ≤ 0.2 | 0.125 |

### Step 9C — hot path sync 제거

- **Phase 9C**: K2 회 `.item()` sync 가 `run_phase2_hybrid_cudagraph` depth
  loop 안에 있던 것을 `_build_phase2_hybrid_plan` 단계의 1회 `.item()` 로
  통합 (plan 에 Python int 캐싱). per-depth sync 0.
- **mask 중복 작업 제거**: per-depth packed mask 를 build 단계에서 이미
  완성된 형태로 `plan.per_depth_packed_masks` 에 저장 → runtime 은 slice
  만 사용.

## 5.3 Post-9B 발견 — split 경로 잔존 회귀 (2026-04-28)

Phase 2 hybrid 작업 중 split 경로에 의도치 않게 끼어든 overhead 가 있다고
판단된 측정 (200 prompts × output_len=256, both AWQ 70B):

| metric | (A) pre-hybrid split | (B) current split fallback (post-hybrid 코드) | Δ |
|---|---:|---:|---:|
| TPS | 66.71 | **63.32** (sync fix 전 dense draft 기준 58.32 / -4.4%) | -3.4 ~ -4.4% |
| draft_step_time | 53.87 ms | 56.21 ms (dense) / 43.75 ms (AWQ) | + |
| `phase1_build` | 0.16 | 0.62 | +298% |
| `hit_cache_respond` | 4.12 | 4.65 | +12.9% |
| `target_spec_wait` | 11.57 | 14.43 | +24.8% (cascade) |

**원인**: hybrid 의 bucket dispatch 를 위한 `.item()` GPU→CPU sync 가 split
경로에서도 fire 하도록 되어있었음:
- `draft_runner.py:567` (in `hit_cache_and_respond`): `_vk_scalar = int(valid_k[0].item())`
- `draft_runner.py:862` (in `_glue_decode`): `_vk_glue = int(...item())`
- `draft_runner.py:1449` (main hot path): `_step_valid_k = int(...item())`

매 step **3 회 sync** = CUDA stream flush. split 에서는 의미없는 동기화.

### 3.3.1 Sync fix 적용

`hit_cache_and_respond` 가 `_vk_scalar` 를 한 번만 sync 하고 (hybrid mode 일
때만), 8-tuple 반환 으로 caller 에 전달. caller 는 `partial_tree_decode_args`
에 `valid_k_scalar` 키로 저장. `_glue_decode` / 메인 hot path 는 dict 에서
Python int 를 읽음 (sync 없음).

```python
# After: hit_cache_and_respond
if self.config.mesa_phase1_k is not None and valid_k.numel() > 0:
    _vk_scalar = int(valid_k[0].item())   # 1 sync, hybrid only
else:
    _vk_scalar = None                      # split path

# Caller stashes it
partial_tree_decode_args["valid_k_scalar"] = valid_k_scalar

# Downstream readers — no further sync
_vk_glue = partial_tree_decode_args.get("valid_k_scalar") or self.config.speculate_k
_step_valid_k = partial_tree_decode_args.get("valid_k_scalar") or self.config.speculate_k
```

| | sync/step (split) | sync/step (hybrid) |
|---|---:|---:|
| Before | 3 | 3 |
| After | **0** | **1** |

검증 (200×256, both AWQ): split TPS 회귀 -4.4% → -0.2% (사실상 닫힘).

## 5.4 Post-9B 발견 — `phase2_hybrid_replay` 라벨 misplaced (2026-04-28)

profiling 라벨이 `run_phase2_hybrid_cudagraph` **외부** 를 감싸서 K2 회
graph.replay() 가 1 개 event 로 합쳐졌음. split 의 `phase2_replay × 5` 와
비교가 왜곡 (단위 mismatch).

### 3.4.1 라벨 재배치

CG 경로 (`run_phase2_hybrid_cudagraph` in `cudagraph_helpers.py`):
```python
for d in range(K2):
    _mev_php = mesa_record(f"phase2_hybrid_prep_{bucket}")
    # KV plan + buffer copies
    mesa_close(f"phase2_hybrid_prep_{bucket}", _mev_php)

    _mev_phr = mesa_record(f"phase2_hybrid_replay_{bucket}")
    graph.replay()
    mesa_close(f"phase2_hybrid_replay_{bucket}", _mev_phr)
```

eager fallback (`_decode_phase2_hybrid`): 동일 패턴 (`_eager_prep_*`,
`_eager_replay_*`).

### 3.4.2 효과

split 의 `phase2_replay × 5` 와 hybrid 의 `phase2_hybrid_replay_long × ~1.74`
+ `_short × ~0.26` 가 동일 단위 (per-replay) 로 직접 비교 가능. K2=2 이고
long-hit 비율 ~87% / short-hit ~13% 로 events_per_step 1.74 + 0.26 = 2.0
forwards/step 검증.

## 5.5 Post-9B 발견 — `phase2_hybrid_build` 자체가 큰 비용 (2026-04-28)

Both AWQ 환경에서 측정:
- `phase2_hybrid_build` = 3.36 ms / step (Python tensor metadata 채우기)
- `phase1_replay` (TinyLlama AWQ, K1=3) = 2.97 ms / forward
- → **build 가 forward 보다 더 오래** 걸림

Forward 보다 비싼 metadata 채우기는 비정상. 분석:
- `_build_hybrid_packed_mask_inplace` 가 K2 회 × ~30 GPU kernel launch
  (전체의 ~80%). 매 depth 마다 `torch.arange(L)`, `bit_weights = (1 << arange(8))`,
  `cont_idx_t = arange(cont_count)` 등 상수성 연산을 새로 alloc.
- bool mask `[total, L]` 만들고 (~50 × 1500 = 75 KB) 후 packed bytes 로
  변환 — 중간 산출물이 최종 결과의 8 배.
- 3 개 별도 K2 루프 (slot_maps, context_lens, kv_indptr).

## 5.6 Phase 9D — build hot-path 최적화 (2026-04-28)

`HybridPhase2Plan.create()` 에 pre-allocated workspace 추가 (engine init 1
회):
- `bit_weights_u8` [8] uint8 — 완전 상수
- `cont_idx_arange` [max_cont] / `proxy_idx_arange` [max_proxy] int64
- `kv_pos_arange` [max_L] int64 (max_L = max_model_len + worst-case extra)
- `full_mask_buf` [max_total, max_L_padded] bool — 매 step in-place reuse
- `total_arange_p1` (GPU/CPU 둘 다) [max_total + 1] int32 — kv_indptr /
  mask_indptr multiply 용

`_build_hybrid_packed_mask_inplace` 변경:
- `kv_pos_2d = plan.kv_pos_arange[:L].view(1, L)` — view, alloc 없음
- `mask_out = plan.full_mask_buf[:total, :L_padded]` — view
- `cont_out.copy_(persistent_vis.expand(...))` + `bitwise_or_(...)` —
  in-place
- `torch.mul(arange, stride, out=...)` — out= 으로 alloc 회피

`_build_phase2_hybrid_plan` 의 K2 루프도 pre-allocated arange 사용.

**예상 효과**:
- 매 step alloc ~10 → ~2
- kernel launch ~60 → ~40
- `phase2_hybrid_build` 3.43 ms → 추정 2.0~2.5 ms (30~40% 절감)

위험도: 낮음 (alloc 위치만 옮김, semantic 무변동, in-place op 는 PyTorch
표준). 측정 결과는 `03-results.md` Part 5 에 기록.
