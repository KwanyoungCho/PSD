# MESA-SSD 설계 문서

이 문서는 MESA-SSD 의 전체 설계 — TreeLayout 추상화, Budget Split,
Split CudaGraph, Rev1 (Policy A/B) — 를 한 곳에 모은다. 원본은
`MESA-IMPL-PLAN.md` (v6 구현 계획), `MESA-BREAKDOWN-PLAN.md` (per-phase
profiling), `MESA-rev1.md` (Rev1 동적 budget allocation) 셋이다.

---

## 0. 설계 원칙

1. **CudaGraph 성능 유지** — Target verify CudaGraph 를 pre/post 로 분리.
2. **TP > 1 호환** — 모든 TP rank 가 동일한 NCCL collective 패턴 실행.
3. **Mid-forward proxy 전송** — target verify 의 ~2/3 지점에서 proxy 를
   draft 에 전송.
4. **TreeLayout 추상화** — `MQ_LEN` 전역 의존 제거. Layout 별로
   pre-computed 버퍼, CudaGraph, FlashInfer wrapper 를 관리.
5. **Budget split** — draft-sourced branches 즉시 decode + proxy-sourced
   branches 도착 후 decode. KV scratch 재사용 — proxy pass 가 draft pass
   의 KV positions 를 덮어써도 안전 (draft 결과는 이미 spec_tokens /
   logits 에 추출, proxy attention mask 는 proxy 자신의 데이터만 참조).
6. **Token dedup** — proxy-sourced 우선, 부족분은 logits fallback 에서
   draft / proxy 모두 제외한 토큰으로 refill. Draft tree 와 proxy tree
   간 중복 branch 없음.
7. **Llama only** — Qwen3, EAGLE 미지원.

---

## 0.1 전체 타이밍 다이어그램

```
TARGET (rank 0, all TP ranks)            DRAFT (rank N)
─────────────────────────────            ──────────────────
1. speculate() 호출                       1. recv_cmd() [blocking]
   → cmd=0, cache_keys, etc 전송
                                         2. _service_spec_request()
                                            - cache lookup / JIT speculate
                                            - SEND response [기존 SSD 와 동일]
2. response 수신
3. verify() 호출                          3. _reset_tree_cache_tensors()
   ┌ graph_pre.replay()                   4. _build_tree_batch_mesa()
   │  layers [0 .. exit_layer]               - glue decode → draft logits
   └ → exit_buffer                           - draft-sourced fork tokens 선택
                                             - irecv() 걸어둠 (non-blocking)
   [CudaGraph 밖, ALL TP ranks]:             - _decode_tree(draft_layout) ← 즉시!
   norm(exit_buffer) → lm_head                      ↕ (target send + draft decode 병렬)
   → exit_logits (TP gather)             ┌── send() 즉시 완료 (irecv 걸려있으므로)
   rank 0: proxy SEND ─────────────────→ │
                                         │   draft decode 완료 → irecv.wait()
   ┌ graph_post.replay() ← 즉시 시작!    │   proxy-sourced fork tokens (dedup)
   │  layers [exit_layer+1 .. L-1]       │   _decode_tree(proxy_layout) ← KV scratch 재사용
   │  + final norm                       └──
   └ → outputs                           5. _populate_tree_cache(draft + proxy 합침)
   lm_head(outputs) → final_logits
   verify 알고리즘 [기존과 동일]
4. postprocess
5. 다음 speculate() →                    6. recv_cmd()
```

---

## 1. Config 확장 (`ssd/config.py`)

추가 필드:

```python
mesa_enabled: bool = False
mesa_exit_layer: int | None = None      # None=auto: 2*L//3
mesa_proxy_top_k: int = 3               # proxy correction token 수
mesa_draft_fan_out: int | None = None   # draft-sourced branches per pos (None=auto: fan_out//2)
```

`__post_init__` 검증:

- `draft_async`, `speculate`, `model_type=="llama"` 강제
- EAGLE 금지 (eagle_acts split collection 미구현)
- `mesa_exit_layer` 기본값 `2*L//3`
- `mesa_draft_fan_out` 기본값 `async_fan_out // 2`
- `mesa_proxy_fan_out = async_fan_out - mesa_draft_fan_out` 자동 계산

Rev1 추가 invariant:

- `assert max_num_seqs == 1` — Policy A 가 `accept_probs[0]` 를 batch
  공통 분포로 사용하므로 단일 시퀀스 전용
- `assert jit_speculate` — `jit_speculate=False` 일 때 miss row 의
  `accept_probs=0` 강제로 `ĥ_0=1` 왜곡 발생 → MESA 강제 True
- `mesa_proxy_top_k` 자동 산정: `pfo*(K+1) + dfo + 2` 로 raise (draft
  fallback 제거하려면 worst-case dedup 후 unique proxy 가 충분해야 함)

---

## 2. TreeLayout 추상화

### 2.1 dataclass (`ssd/engine/helpers/tree_layout.py` 신규)

```python
@dataclass
class TreeLayout:
    name: str                          # "full", "draft", "proxy"
    fan_out_list: list[int]            # per-position fan_out [K+1]
    fan_out_list_miss: list[int]
    MQ_LEN: int                        # sum(fan_out_list)
    K: int

    fan_out_t: torch.Tensor
    fan_out_t_miss: torch.Tensor
    fan_idx_hit: torch.Tensor          # arange(K+1).repeat_interleave(fan_out_t)
    fan_idx_miss: torch.Tensor
    arange_mq: torch.Tensor
    step_pos_offsets: torch.Tensor     # arange(K)[:, None] * MQ_LEN
    step_rope_offsets: torch.Tensor
    graph_key: str                     # "fi_tree_decode_{name}"
```

`create_tree_layout(name, fan_out_list, fan_out_list_miss, K, device)` 가
모든 텐서를 device 에 pre-allocate.

### 2.2 DraftRunner 초기화

`_init_prealloc_buffers()` 가 `full_layout` 을 항상 만들고, `mesa_enabled`
면 `draft_layout`, `proxy_layout` 을 추가 생성. 기존 전역 텐서
(`_step_pos_offsets`, `_fan_idx_hit` 등) 는 `full_layout` 으로 위임 →
backward compat 유지.

### 2.3 함수 시그니처 일반화

다음 함수들이 `layout` 파라미터를 받도록 변경:

- `_decode_tree(payload, layout=None)` — None 이면 `self.full_layout`
- `_compute_step_positions_and_slot_maps(..., layout)` — `MQ_LEN`,
  `step_pos_offsets`, `step_rope_offsets` 모두 layout 에서 가져옴
- `_build_tree_decode_args(..., layout)` — full / draft / proxy 모두
  동일 로직, layout 기반 fan_idx 사용

### 2.4 FlashInfer Wrapper layout 별 생성

기존 `prefill_wrappers` (full layout 기반) 를 `prefill_wrappers_by_layout`
dict 로 바꾸고, MESA 면 draft / proxy layout 용 wrapper 추가 생성. 각
layout 마다 별도의 `cu_seqlens_q`, `kv_indptr`, `kv_indices`,
`kv_last_page_len`, `mask_buf`, `mask_indptr_buf` 버퍼.

### 2.5 Attention layer layout-aware

`ssd/layers/attention.py` 의 tree decode 분기:

```python
elif tree_decode:
    context = get_context()
    mq_len = getattr(context, 'active_mq_len', self.F * (self.K + 1))
    bs = q.shape[0] // mq_len
    wrappers = getattr(context, 'active_wrappers', self.prefill_wrappers)
    wrapper_bs = next(b for b in sorted(wrappers.keys()) if b >= bs)
    prefill_wrapper = wrappers[wrapper_bs]
```

`set_context()` 에 `active_mq_len`, `active_wrappers`, `active_layout`
optional 필드 추가 (Rev1).

### 2.6 CudaGraph capture / replay 일반화

`capture_fi_tree_decode_cudagraph(model_runner, layout=None)` 가 layout
파라미터를 받음. layout 없으면 backward compat (`config.fan_out_list` /
`config.MQ_LEN` 사용). MESA draft 는 full / draft / proxy 세 layout 의
CudaGraph 를 캡처. `cudagraph_helpers.py` 의 모든 `MQ_LEN` 참조와
`fan_out_list` 참조를 layout 기반으로 치환.

---

## 3. LlamaModel split forward (`ssd/models/llama3.py`)

```python
def forward(
    self,
    input_ids,
    positions,
    start_layer: int = 0,
    end_layer: int | None = None,
    init_hidden_states: torch.Tensor | None = None,
    init_residual: torch.Tensor | None = None,
):
    if init_hidden_states is not None:
        hidden_states, residual = init_hidden_states, init_residual
    else:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

    actual_end = end_layer if end_layer is not None else len(self.layers)

    for layer_idx in range(start_layer, actual_end):
        hidden_states, residual = self.layers[layer_idx](positions, hidden_states, residual)

    if end_layer is None:
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    else:
        return hidden_states, residual
```

`LlamaForCausalLM.forward` 도 동일한 split 파라미터 (start_layer,
end_layer, init_*) 를 받아 `self.model` 에 위임.

---

## 4. Split CudaGraph (Target Verify)

### 4.1 `capture_mesa_verify_cudagraph()`

- `graph_pre`: layers `[0, exit_layer]` → `exit_hidden`, `exit_residual`
- `graph_post`: layers `[exit_layer+1, L-1]` + final norm → `outputs`
- 두 graph 가 graph_pool 을 공유 (메모리 절약)
- `graph_vars` 에 input_ids / positions / slot_mapping / context_lens /
  block_tables / cu_seqlens_q / exit_hidden / exit_residual / outputs 모두 포함

### 4.2 `run_mesa_verify_cudagraph()`

```python
graph_pre.replay()                    # layers [0..exit_layer]

# Mid-forward (CudaGraph 밖, ALL TP ranks compute_logits 실행):
exit_h = graph_vars["exit_hidden"][:flat] + graph_vars["exit_residual"][:flat]
normed = model_runner.model.model.norm(exit_h, None)
exit_logits = model_runner.model.compute_logits(normed, last_only=False)
# rank 0 만 mesa_proxy_fn 등록되어 있음 → exit_logits 받아 proxy 계산 + send
if mesa_proxy_fn is not None:
    mesa_proxy_fn(exit_logits, orig_bs)

graph_post.replay()                   # layers [exit_layer+1..L-1] + norm
logits = model_runner.model.compute_logits(outputs, last_only)
```

`mesa_proxy_fn` 은 Verifier 가 rank 0 의 ModelRunner 에만 set. Rank 1+
는 `_mesa_proxy_fn = None` 이므로 callback 에 진입하지 않음.

### 4.3 model_runner 분기

`run_model()` 에서 verify 경로:

- `mesa_enabled and not is_draft and "mesa_verify" in graph_vars` →
  `run_mesa_verify_cudagraph` (split)
- 그 외 → 기존 `run_verify_cudagraph` (단일)

---

## 5. Verifier — Proxy 계산 및 전송

### 5.1 verify() flow

`mesa_enabled` 이면 `_mesa_proxy_fn` 을 closure 로 set:

```python
def _proxy_fn(exit_logits, orig_bs):
    self._compute_and_send_proxy(
        exit_logits, draft_tokens, logits_q, orig_bs, K,
        async_pg, draft_rank, cache_hits=cache_hits)

self.target_model_runner._mesa_proxy_fn = _proxy_fn
result = self.target_model_runner.call("run", seqs, ...)
self.target_model_runner._mesa_proxy_fn = None
```

### 5.2 `_compute_and_send_proxy()`

```python
p_E = softmax(exit_logits[:, :K, :])        # early-exit 분포
p_D = softmax(logits_q)                     # draft 분포

p_E_y = p_E.gather(2, draft_tokens)
p_D_y = p_D.gather(2, draft_tokens)
accept_probs = (p_E_y / (p_D_y + ε)).clamp(max=1.0)

residual = (p_E - p_D).clamp(min=0)
residual.scatter_(2, draft_tokens, 0.0)
topk_probs, topk_ids = residual.topk(top_k, dim=-1)
topk_probs = topk_probs / topk_probs.sum(-1, keepdim=True).clamp(min=ε)

# Cache-miss 보정 (jit_speculate=False 일 때만)
# Rev1: jit_speculate 강제 True → 이 분기 dead, 제거 가능

send_int64(async_pg, draft_rank, accept_probs, topk_ids, topk_probs)
```

Rev1 변경: target 측에서 `accept_probs` 를 직접 보내지 않고
`fan_out_list` 까지 계산해서 보냄 → draft critical path 단축.

---

## 6. Draft 측 — Budget Split + 2-Pass Tree Decode

### 6.1 `draft_loop()` 분기

```python
glue_input_ids, partial = self._service_spec_request()
self._reset_tree_cache_tensors()

if self.config.mesa_enabled:
    self._build_tree_batch_mesa(partial, glue_input_ids)   # 내부에서 decode + populate 완료
else:
    tree_args = self._build_tree_batch(partial, glue_input_ids)
    tokens, logits, acts = self._decode_tree(tree_args)
    self._populate_tree_cache(tree_args, tokens, logits, ...)
```

### 6.2 `_build_tree_batch_mesa()` (Rev1 — `_glue_decode()` 분리 후)

```python
glue_logits, gd_for_fork, cache_hits, cache_hits_list, dbt, pos_offset = \
    self._glue_decode(partial, glue_input_ids)

# Phase 1: draft-sourced (즉시 시작, proxy 대기 없음)
draft_forked = self._select_draft_sourced_tokens(
    glue_logits, cache_hits, gd_for_fork, draft_fan_out)

# irecv 를 decode 시작 전에 걸어둠 → target send 가 block 안 됨
proxy_recv_work, proxy_buf = self._irecv_mesa_proxy(B, K)

draft_args = self._build_tree_decode_args(
    partial, draft_forked.view(-1), self.draft_layout, ...)
draft_tokens, draft_logits, draft_acts = self._decode_tree(
    draft_args, layout=self.draft_layout)

# proxy 도착 대기 + unpack
proxy_recv_work.wait()
mesa_proxy = self._unpack_mesa_proxy(proxy_buf, B, K)

# Phase 2: proxy-sourced (dedup + Policy A 동적 layout)
fan_out_list = mesa_proxy["fan_out_list"]
step_proxy_layout = create_tree_layout(
    "proxy", fan_out_list, fan_out_list, K, self.device)

proxy_forked = self._select_proxy_sourced_tokens_policy_a(
    glue_logits, draft_forked, mesa_proxy, fan_out_list)

proxy_args = self._build_tree_decode_args(
    partial, proxy_forked.view(-1), step_proxy_layout, ...)
proxy_tokens, proxy_logits, proxy_acts = self._decode_tree(
    proxy_args, layout=step_proxy_layout)

self._merge_and_populate_cache(
    draft_args, draft_tokens, draft_logits,
    proxy_args, proxy_tokens, proxy_logits,
    self.draft_layout, step_proxy_layout)
```

### 6.3 Token selection

**`_select_draft_sourced_tokens`**: draft logits top-`draft_fan_out`
(returned tokens 마스킹).

**`_select_proxy_sourced_tokens` (v1)**: proxy correction 우선, dedup 후
부족분은 draft logits fallback. B × K Python loop + `tolist()` 3회 GPU
sync.

**Rev1 변경**:
- `proxy_top_k` 자동 raise → fallback 불필요 → 모든 Phase-2 branch 가
  target-informed
- pos < K: proxy-only (벡터화 가능)
- pos == K (all-accept): draft top-k 그대로 (correction 불필요한 위치)

### 6.4 `_irecv_mesa_proxy()` + `_unpack_mesa_proxy()`

Non-blocking recv 로 target send blocking 방지:

```python
buf = torch.empty(total_len, dtype=torch.int64, device=self.device)
work = dist.irecv(buf, src=0, group=self.async_pg)
return work, buf
```

안전한 이유:
- Buffer lifetime — caller 가 `buf` 를 참조 유지하므로 GC 안 됨
- Race 없음 — decode 중 `buf` 읽지 않고, `wait()` 후에만 unpack
- Stream ordering — NCCL recv 는 자체 stream, decode 는 default stream

### 6.5 `_merge_and_populate_cache()`

Draft pass 와 proxy pass 의 cache key 를 각자의 layout fan_idx 로 만들어
합침:

```python
draft_keys, _, _, _ = self._populate_tree_cache(
    draft_payload, draft_tokens, draft_logits,
    draft_payload["cache_hits"], draft_layout)

proxy_keys, _, _, _ = self._populate_tree_cache(
    proxy_payload, proxy_tokens, proxy_logits,
    proxy_payload["cache_hits"], proxy_layout)

self.tree_cache_keys = torch.cat([draft_keys, proxy_keys], dim=0)
self.tree_cache_tokens = torch.cat([draft_tokens, proxy_tokens], dim=0)
self.tree_cache_logits = torch.cat([draft_logits, proxy_logits], dim=0)
```

Cache lookup 은 `(seq_id, k_idx, rec_token)` 매칭이므로 layout 별 다른
fan_out 이어도 정상 동작.

---

## 7. Rev1: 동적 Budget Allocation

### 7.1 CudaGraph 와 동적 fan_out 의 양립

CudaGraph 가 캡처하는 것 (고정):

```
model(input_ids, positions) → outputs    # GPU 연산 그래프, 텐서 shape (N 고정)
```

CudaGraph 밖에서 매 step 실행 (동적 가능):

```
wrapper.plan(cu_seqlens, kv_indptr, custom_mask, ...)   # FlashInfer 설정
graph_vars 에 input_ids / positions 복사                # 텐서 값
mask precompute                                         # mask 내용
```

**결론**: CudaGraph 는 `N = B × MQ_LEN` (총 node 수) 만 고정. **`MQ_LEN
= sum(fan_out_list)` 만 유지하면 fan_out 분포는 매 step 바꿀 수 있다.**

```
예: proxy_layout MQ_LEN = 10 (고정), K = 4

Step N: h = [0.70, 0.03, 0.24, 0.02, 0.01]
  → fan_out = [4, 0, 3, 0, 3]  (sum = 10 ✅)
  → CudaGraph replay (N=10 고정)

Step N+1: h = [0.10, 0.60, 0.05, 0.20, 0.05]
  → fan_out = [0, 5, 0, 3, 2]  (sum = 10 ✅)
  → 같은 CudaGraph replay (N=10 고정)
```

### 7.2 Target 측 `ĥ_i` + Budget 배분 (draft → target 이전)

기존: Target sends `[accept_probs, topk_ids, topk_probs]` → Draft computes
`ĥ_i` + budget.

변경: Target computes `ĥ_i` + budget → sends `[fan_out_list, topk_ids,
topk_probs]` → Draft 바로 사용.

이점:
- Draft critical path 에서 ~0.5ms 제거
- Target mid-forward 구간에 ~0.01ms 추가 (무시 가능)
- accept_probs 전송 불필요 → payload 단순화

### 7.3 Runtime layout 전달 경로

Context 에 layout 객체 자체를 전달 (정적 lookup → 동적):

```python
# context.py
@dataclass
class Context:
    ...
    active_layout: object | None = None

# draft_runner._decode_tree_step
set_context(..., active_layout=step_proxy_layout)

# model_runner.run_model
_ctx = get_context()
if _ctx.active_layout is not None:
    _tree_layout = _ctx.active_layout
    _tree_graph_key = _tree_layout.graph_key

# cudagraph_helpers.run_fi_tree_decode_cudagraph
_fan_out_list = layout.fan_out_list if layout else config.fan_out_list
```

CudaGraph 는 기존 proxy graph 재사용, mask / plan 만 동적 layout 기반.

### 7.4 Policy A: `ĥ_i` 기반 동적 Budget

```python
# 1. accept_probs → ĥ_i (B=1)
cumprod = torch.cumprod(accept_probs, dim=1)
h = torch.zeros(1, K + 1, device=accept_probs.device)
h[0, 0] = 1 - accept_probs[0, 0]
h[0, 1:K] = cumprod[0, :-1] * (1 - accept_probs[0, 1:])
h[0, K] = cumprod[0, -1]

# 2. ĥ_i 비례 배분 (sum = proxy_MQ_LEN)
total = proxy_MQ_LEN
if h[0, :K].sum() < 1e-6:
    # 전부 accept 예상 → uniform fallback
    fan_out_list = [total // (K+1)] * (K+1)
    for i in range(total - sum(fan_out_list)):
        fan_out_list[i] += 1
else:
    raw = (h[0] / h[0].sum() * total).floor().int()
    rem = total - raw.sum().item()
    _, sorted_idx = h[0].sort(descending=True)
    for i in range(int(rem)):
        raw[sorted_idx[i]] += 1
    fan_out_list = raw.tolist()
```

### 7.5 Token 선택 (Rev1, fallback 제거 후)

```
Pos 0..K-1:
  proxy correction tokens (topk_ids 에서 draft 제외 후 fan_out 만큼)

Pos K (all-accept):
  draft logits top-k (proxy 없음 — accept 위치는 correction 불필요)
```

`fan_out_list[pos] == 0` 이면 해당 position skip.

**Helper 입력 계약**: `_build_tree_decode_args_for_layout()` 은
`forked_tokens.view(-1)` = `[MQ_LEN]` flat 텐서를 기대. flat 순서는
`layout.fan_idx_hit = [0,0,0,0, 2,2,2, 4,4,4]` 의 position 순서와 정확히
일치해야 함.

### 7.6 Policy B: `P̂(i, v) = ĥ_i · r̂_i(v)` (defer)

Position 뿐 아니라 어떤 correction token 이 유력한지까지 반영:

```python
P = h[0, :K].unsqueeze(-1) * topk_probs[0]   # [K, top_k]
flat_P = P.view(-1)
_, top_indices = flat_P.topk(min(proxy_MQ_LEN, K * top_k))
positions = top_indices // top_k
token_ranks = top_indices % top_k
```

Allocation (target count) 과 fill (actual tokens) 을 분리하고, dedup 후
부족분만 draft fallback. Rev1 에서는 미구현 (Policy B 대신 Policy A 만
검증).

---

## 8. Per-Phase Profiling 계측 (`MESA-BREAKDOWN-PLAN.md`)

기존 함수 body / signature 손대지 않는 최소 추가 ( ~70 LoC).

### 8.1 Helpers (`cudagraph_helpers.py` 하단 append)

```python
PROFILE_MESA = os.environ.get("SSD_PROFILE_MESA", "0") == "1"
_mesa_events = []   # [(step, label, start_ev, end_ev)]

def mesa_record(step, label):
    if not PROFILE_MESA:
        return None
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev

def mesa_close(step, label, start_ev):
    if start_ev is None:
        return
    end_ev = torch.cuda.Event(enable_timing=True)
    end_ev.record()
    _mesa_events.append((step, label, start_ev, end_ev))

def mesa_dump(tag):
    if not _mesa_events:
        return
    torch.cuda.synchronize()
    rows = [{"step": s, "label": l, "ms": a.elapsed_time(b)}
            for s, l, a, b in _mesa_events]
    json.dump(rows, open(f"/tmp/mesa_profile_{tag}.json", "w"))
```

### 8.2 측정 지점 (각 2 줄씩 추가, 기존 body 그대로)

| Process | Label | 위치 |
|---------|-------|------|
| draft | `glue` | `_glue_decode` |
| draft | `phase1_replay` | `run_fi_tree_decode_cudagraph` (layout=draft) |
| draft | `proxy_wait` | irecv `work.wait()` 근처 |
| draft | `phase2_replay` | `run_fi_tree_decode_cudagraph` (layout=proxy) |
| draft | `merge_cache` | `_merge_and_populate_cache` |
| target | `graph_pre` | `run_mesa_verify_cudagraph` |
| target | `proxy_compute_send` | `_compute_and_send_proxy` |
| target | `graph_post` | `run_mesa_verify_cudagraph` |

총 8 × 2 = **16 줄**. `step` 은 함수에 이미 전달되는 argument 사용.

### 8.3 End-of-run dump

`llm_engine.py` METRICS print 뒤, draft_runner loop 종료 뒤:

```python
mesa_dump("target" if self.rank == 0 else "draft")
```

### 8.4 Plot 스크립트 (`bench/plot_mesa_breakdown.py`, 30 LoC)

JSON 두 파일 (target, draft) 읽어서 label 별 mean / median bar plot.

### 8.5 Default off 비용

- `PROFILE_MESA=0`: helper 첫 줄에서 `return None`. ~10 ns × 16
  호출/step → 측정상 0%.
- `PROFILE_MESA=1`: CUDA Event record ~1-2 µs × 16 → ~30 µs / step,
  step 당 ~40 ms 대비 < 0.1%.

---

## 9. 수정 파일 요약 (v6 + Rev1)

| 파일 | 변경 | 규모 |
|------|------|------|
| `ssd/config.py` | mesa params + B=1 / jit_speculate assert + proxy_top_k auto-raise | ~40 |
| `ssd/engine/helpers/tree_layout.py` | **신규**: TreeLayout + create_tree_layout | ~50 |
| `ssd/models/llama3.py` | split forward (start/end_layer + init_*) | ~25 |
| `ssd/engine/helpers/cudagraph_helpers.py` | mesa_verify capture/run + layout 일반화 + profiling | ~180 |
| `ssd/engine/model_runner.py` | mesa CudaGraph 캡처 분기 + run_model 분기 + layout wrapper | ~60 |
| `ssd/layers/attention.py` | layout-aware wrapper selection | ~10 |
| `ssd/utils/context.py` | active_mq_len / active_wrappers / active_layout | ~5 |
| `ssd/engine/verifier.py` | proxy_fn + _compute_and_send_proxy + h_i + fan_out_list | ~80 |
| `ssd/engine/draft_runner.py` | TreeLayout 적용 + 2-pass + Policy A + irecv + merge + buffer prealloc | ~200 |
| `bench/bench.py` | --mesa args + jit_speculate auto | ~15 |
| **총** | | **~665** |

---

## 10. 구현 순서

```
1. Config + feature gating (config.py)
2. TreeLayout 추상화 + backward compat (tree_layout.py, draft_runner._init_prealloc)
3. _decode_tree / _compute_step_positions / run_fi_tree_decode 일반화
4. Split forward (llama3.py)
5. Split CudaGraph capture + replay (cudagraph_helpers, model_runner)
6. Proxy 계산 + send (verifier)
7. Budget split 2-pass + dedup + merge (draft_runner)
8. Profiling 계측 (cudagraph_helpers append + 측정 지점 16 줄)
9. Rev1: glue decode 분리, tolist 최적화, runtime layout, Policy A
10. Rev1: B=1 assert, proxy_top_k 확대 + fallback 제거, _decode_tree 버퍼
    pre-allocate, Python loop 벡터화, dead code 제거
11. bench.py + 테스트
```

---

## 11. Test Checklist

### 11.1 단위 테스트

1. Split forward — `forward(end_layer=E)` + `forward(start_layer=E+1, init_*)` == `forward()`
2. Split CudaGraph — `graph_pre + graph_post` == 단일 graph
3. TreeLayout backward compat — full_layout 으로 기존 경로 결과 동일
4. TreeLayout MQ_LEN — draft + proxy MQ_LEN 합산 정확
5. Proxy 계산 — `accept_probs ∈ [0, 1]`, residual top-k 에 draft token 미포함
6. Token dedup 기본 — proxy ∩ draft == ∅ (모든 position)
7. Token dedup edge — `draft=[A]`, `proxy=[A]`, `proxy_fan_out=2` → proxy
   row 에 `A` 없이 fallback 만
8. Position K dedup — `draft_forked[:, K, :] ∩ proxy[:, K, :] == ∅`
9. Budget split — 2-pass 결과 합쳐서 cache populate 정상
10. KV scratch — 2 번째 pass 가 1 번째 결과 훼손 안 함
11. NCCL pack / unpack round-trip
12. Feature gating — Qwen3 / EAGLE + mesa → assert

### 11.2 Edge case

- TP > 1 (4 GPU TP + 1 draft)
- cache miss + jit_speculate=False / True
- temp > 0
- B > 1 (Rev1: assert 로 차단됨)
- `proxy_top_k < proxy_fan_out` (Rev1: auto-raise 로 차단)

### 11.3 벤치마크

```bash
cd bench

# Baseline
python -O bench.py --llama --size 8 --async --spec --k 4 --f 3 \
    --gpus 2 --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# MESA
python -O bench.py ... --mesa --mesa_exit_layer 21

# Budget split sweep
for DF in 1 2; do
    python -O bench.py ... --mesa --mesa_exit_layer 21 --mesa_draft_fan_out $DF
done

# Exit layer sweep
for EL in 10 16 21 26; do
    python -O bench.py ... --mesa --mesa_exit_layer $EL
done
```

---

## 12. Rev1 예상 효과 (B=1 only)

| 항목 | v1 (고정 fan_out) | Rev1 Policy A | Policy B (defer) |
|------|-------------------|---------------|------------------|
| Throughput | 84 tok/s | 동일 / 소폭 감소 | 동일 / 소폭 감소 |
| Cache hit | 0.87 | ↑ (risky position 집중) | ↑↑ (joint) |
| Accept rate | 0.83 | ↑ | ↑↑ |
| Tok / Step | 4.31 | ↑ | ↑↑ |

Throughput 자체는 2-pass 구조적 비용 + runtime layout 생성 / `ĥ_i` 계산
추가로 동일하거나 소폭 감소. Token efficiency 개선이 주된 이득.

장기적 throughput 개선 방향:

1. **Per-pass overhead 줄이기** — FlashInfer wrapper.plan() 최적화,
   persistent mask cache
2. **Larger model 비율** — 70B 에서는 model forward 가 지배적 → 2-pass
   overhead 비율 감소 (실제 34B / 70B 실험에서 확인)
3. **B > 1 확장** — batch 공통 layout 근사 (`ĥ_i` 평균) 또는 per-sequence
   layout 지원
