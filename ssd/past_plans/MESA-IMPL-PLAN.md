# MESA-SSD 구현 계획서 (v6 — TreeLayout + Budget Split + Split CudaGraph)

## 0. 설계 원칙

1. **CudaGraph 성능 유지**: Target verify CudaGraph를 pre/post로 분리.
2. **TP > 1 호환**: 모든 TP rank가 동일한 NCCL collective 패턴 실행.
3. **Mid-forward proxy 전송**: target verify의 ~2/3 지점에서 proxy를 draft에 전송.
4. **TreeLayout 추상화**: MQ_LEN 전역 의존 제거. layout별로 pre-computed 버퍼, CudaGraph, FlashInfer wrapper 관리.
5. **Budget split**: draft-sourced branches 즉시 decode + proxy-sourced branches 도착 후 decode. KV scratch 재사용 — proxy pass가 draft pass의 KV positions를 덮어써도 안전 (draft 결과는 이미 spec_tokens/logits에 추출, proxy attention mask는 proxy 자신의 데이터만 참조).
6. **Token dedup**: proxy-sourced 우선, 부족분은 logits fallback에서 draft/proxy 모두 제외한 토큰으로 refill. Draft tree와 proxy tree 간 중복 branch 없음.
7. **Llama only**: Qwen3, EAGLE 미지원.

---

## 0.1 전체 타이밍

```
TARGET (rank 0, all TP ranks)            DRAFT (rank N)
─────────────────────────────            ──────────────────
1. speculate() 호출                       1. recv_cmd() [blocking]
   → cmd=0, cache_keys, etc 전송
                                         2. _service_spec_request()
                                            - cache lookup / JIT speculate
                                            - SEND response [기존 SSD와 동일]
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

## 1단계: Config 확장

- [ ] 완료

**파일**: `ssd/config.py`

추가할 필드 (line 45 이후):
```python
# MESA-SSD parameters
mesa_enabled: bool = False
mesa_exit_layer: int | None = None      # None=auto: 2*L//3
mesa_proxy_top_k: int = 3              # proxy에서 전송할 correction token 수
mesa_draft_fan_out: int | None = None   # draft-sourced branches per position (None=auto: fan_out//2)
```

`__post_init__`에 추가 (line 94 앞):
```python
if self.mesa_enabled:
    assert self.draft_async, "MESA-SSD requires draft_async=True"
    assert self.speculate, "MESA-SSD requires speculate=True"
    assert self.hf_config.model_type == "llama", "MESA-SSD only supports Llama models"
    assert not self.use_eagle, "MESA-SSD + EAGLE: not yet implemented (eagle_acts split collection needed)"
    if self.mesa_exit_layer is None:
        L = self.hf_config.num_hidden_layers
        self.mesa_exit_layer = (2 * L) // 3
    assert 0 < self.mesa_exit_layer < self.hf_config.num_hidden_layers
    if self.mesa_draft_fan_out is None:
        self.mesa_draft_fan_out = max(1, self.async_fan_out // 2)
    assert 0 < self.mesa_draft_fan_out < self.async_fan_out, \
        "mesa_draft_fan_out must be in (0, async_fan_out)"
    self.mesa_proxy_fan_out = self.async_fan_out - self.mesa_draft_fan_out
```

---

## 2단계: TreeLayout 추상화

- [ ] 완료

### 2.1 TreeLayout dataclass 정의

**파일**: `ssd/engine/helpers/tree_layout.py` (신규)

```python
from dataclasses import dataclass
import torch

@dataclass
class TreeLayout:
    """Tree decode에 필요한 layout 정보를 캡슐화.
    기존 전역 MQ_LEN 의존을 제거하고, 복수 layout(full, draft, proxy) 지원.
    """
    name: str                          # "full", "draft", "proxy"
    fan_out_list: list[int]            # per-position fan_out [K+1]
    fan_out_list_miss: list[int]       # per-position fan_out for cache miss [K+1]
    MQ_LEN: int                        # sum(fan_out_list)
    K: int                             # speculate_k

    # Pre-computed tensors (device에 올라감)
    fan_out_t: torch.Tensor            # tensor(fan_out_list)
    fan_out_t_miss: torch.Tensor       # tensor(fan_out_list_miss)
    fan_idx_hit: torch.Tensor          # arange(K+1).repeat_interleave(fan_out_t)
    fan_idx_miss: torch.Tensor         # arange(K+1).repeat_interleave(fan_out_t_miss)
    arange_mq: torch.Tensor            # arange(MQ_LEN)
    step_pos_offsets: torch.Tensor     # arange(K)[:, None] * MQ_LEN
    step_rope_offsets: torch.Tensor    # arange(K)[:, None]

    # CudaGraph / FlashInfer 식별용
    graph_key: str                     # "fi_tree_decode", "fi_tree_decode_draft", etc.


def create_tree_layout(name: str, fan_out_list: list[int], fan_out_list_miss: list[int],
                        K: int, device: torch.device) -> TreeLayout:
    """TreeLayout 인스턴스 생성. pre-computed 텐서 할당."""
    MQ_LEN = sum(fan_out_list)
    fan_out_t = torch.tensor(fan_out_list, device=device, dtype=torch.int64)
    fan_out_t_miss = torch.tensor(fan_out_list_miss, device=device, dtype=torch.int64)

    return TreeLayout(
        name=name,
        fan_out_list=fan_out_list,
        fan_out_list_miss=fan_out_list_miss,
        MQ_LEN=MQ_LEN,
        K=K,
        fan_out_t=fan_out_t,
        fan_out_t_miss=fan_out_t_miss,
        fan_idx_hit=torch.arange(K + 1, device=device, dtype=torch.int64).repeat_interleave(fan_out_t),
        fan_idx_miss=torch.arange(K + 1, device=device, dtype=torch.int64).repeat_interleave(fan_out_t_miss),
        arange_mq=torch.arange(MQ_LEN, device=device, dtype=torch.int64),
        step_pos_offsets=torch.arange(K, device=device, dtype=torch.int64)[:, None] * MQ_LEN,
        step_rope_offsets=torch.arange(K, device=device, dtype=torch.int64)[:, None],
        graph_key=f"fi_tree_decode_{name}" if name != "full" else "fi_tree_decode",
    )
```

### 2.2 DraftRunner에서 TreeLayout 생성

**파일**: `ssd/engine/draft_runner.py`

`_init_prealloc_buffers()` (line 112-122) 변경:

```python
def _init_prealloc_buffers(self):
    from ssd.engine.helpers.tree_layout import create_tree_layout
    K = self.config.speculate_k
    d = self.device

    # 기존 텐서 (layout과 무관)
    self._arange_kp1 = torch.arange(K + 1, device=d, dtype=torch.int64)
    self._arange_2kp1 = torch.arange(2 * K + 1, device=d, dtype=torch.int64)

    # full_layout: 기존 SSD용 (non-MESA 경로 + MESA 비활성 시)
    self.full_layout = create_tree_layout(
        name="full",
        fan_out_list=self.config.fan_out_list,
        fan_out_list_miss=self.config.fan_out_list_miss,
        K=K, device=d)

    # 기존 전역 변수를 full_layout으로 위임 (backward compat)
    self._step_pos_offsets = self.full_layout.step_pos_offsets
    self._step_rope_offsets = self.full_layout.step_rope_offsets
    self._fan_idx_hit = self.full_layout.fan_idx_hit
    self._fan_idx_miss = self.full_layout.fan_idx_miss
    self._arange_mq = self.full_layout.arange_mq

    # MESA: draft_layout + proxy_layout
    if self.config.mesa_enabled:
        draft_fo = self.config.mesa_draft_fan_out
        proxy_fo = self.config.mesa_proxy_fan_out

        self.draft_layout = create_tree_layout(
            name="draft",
            fan_out_list=[draft_fo] * (K + 1),
            fan_out_list_miss=[draft_fo] * (K + 1),
            K=K, device=d)

        self.proxy_layout = create_tree_layout(
            name="proxy",
            fan_out_list=[proxy_fo] * (K + 1),
            fan_out_list_miss=[proxy_fo] * (K + 1),
            K=K, device=d)
```

### 2.3 _decode_tree() 시그니처 일반화

**파일**: `ssd/engine/draft_runner.py` (`_decode_tree`, line 763-812)

현재:
```python
def _decode_tree(self, payload):
    B, K, F, N = payload["metadata_ints"]
    # ... self._step_pos_offsets, self.config.MQ_LEN 등 전역 참조 ...
```

변경:
```python
def _decode_tree(self, payload, layout=None):
    """Tree decode. layout이 None이면 self.full_layout 사용 (backward compat)."""
    if layout is None:
        layout = self.full_layout
    B, K, F, N = payload["metadata_ints"]

    # 기존: self.config.MQ_LEN → layout.MQ_LEN
    # 기존: self._step_pos_offsets → layout.step_pos_offsets
    # 기존: self._step_rope_offsets → layout.step_rope_offsets
    # 나머지 로직 동일
    ...
```

### 2.4 _compute_step_positions_and_slot_maps() 일반화

**파일**: `ssd/engine/draft_runner.py` (line 714-731)

현재:
```python
def _compute_step_positions_and_slot_maps(self, ..., MQ_LEN):
    step_positions = initial_positions[None, :] + self._step_pos_offsets
    ...
    step_context_lens = step_positions.view(K, B, MQ_LEN)[:, :, -1] + 1
    b_flat = ... .expand(B, self.config.MQ_LEN).flatten()
```

변경:
```python
def _compute_step_positions_and_slot_maps(self, ..., layout):
    step_positions = initial_positions[None, :] + layout.step_pos_offsets
    step_rope_positions = initial_rope_positions[None, :] + layout.step_rope_offsets
    step_context_lens = step_positions.view(K, B, layout.MQ_LEN)[:, :, -1] + 1
    b_flat = ... .expand(B, layout.MQ_LEN).flatten()
    ...
```

### 2.5 _build_tree_batch()에서 layout 기반 tree_decode_args 구축

**파일**: `ssd/engine/draft_runner.py` (`_build_tree_batch`, line 698-710)

현재 `_build_tree_batch`에서 tree_decode_args를 만들 때 `self._fan_idx_hit`, `self._arange_mq` 등을 사용. 이것을 layout 파라미터로 변경:

```python
def _build_tree_decode_args(self, partial_tree_decode_args, forked_rec_tokens,
                              layout, cache_hits, ...):
    """layout 기반 tree_decode_args 구축. full/draft/proxy 모두 동일 로직."""
    B = ...
    K = layout.K
    MQ_LEN = layout.MQ_LEN

    _pre_b_flat = torch.arange(B, device=self.device)[:, None].expand(B, MQ_LEN).flatten()
    _pre_fkp1_flat = layout.arange_mq.repeat(B)

    _pre_j_idx_flat = torch.cat([
        layout.fan_idx_hit if int(h) else layout.fan_idx_miss
        for h in cache_hits_list
    ])

    _pre_positions = (num_tokens[_pre_b_flat] - 1 + pos_offset) + (K+1) + _pre_fkp1_flat
    _pre_rope_positions = (num_tokens[_pre_b_flat] - 1 + pos_offset) + _pre_j_idx_flat + 1
    _pre_temperatures = partial_tree_decode_args["temperatures"][_pre_b_flat]

    N = _pre_b_flat.shape[0]  # B * MQ_LEN

    tree_decode_args = {
        "metadata_ints": (B, K, layout.fan_out_list[0], N),
        "input_ids": forked_rec_tokens,
        "positions": _pre_positions,
        "rope_positions": _pre_rope_positions,
        "block_tables": dbt,
        "temps": _pre_temperatures,
        "rec_flat": forked_rec_tokens,
        "seq_ids_expanded": ...,
        "cache_hits": cache_hits,
        "cache_hits_list": cache_hits_list,
        "hidden_states": tree_hidden_states,
    }
    return tree_decode_args
```

### 2.6 CudaGraph / FlashInfer Wrapper를 layout별로 캡처

**파일**: `ssd/engine/model_runner.py`, `ssd/engine/helpers/cudagraph_helpers.py`

#### FlashInfer Wrapper — layout별 생성

현재 attention.py의 tree decode 경로:
```python
# attention.py:117-124 (현재 코드)
mq_len = self.F * (self.K+1)  # ← 전역 fan_out 고정!
bs = q.shape[0] // mq_len
prefill_wrapper = self.prefill_wrappers[wrapper_bs]
```

**문제**: `self.F`(=async_fan_out)로 mq_len을 계산하므로, draft_layout(MQ_LEN=5)이나 proxy_layout(MQ_LEN=10)으로 tree decode 시 shape mismatch 발생.

**해결**: layout별 wrapper dict를 생성하고, tree decode 시 active layout에 맞는 wrapper를 사용.

```python
# model_runner.py _init_flashinfer_wrappers 내부:

# 기존 full wrapper (self.prefill_wrappers)
MQ_LEN = self.config.async_fan_out * (self.config.speculate_k + 1)
# ... 기존 wrapper 생성 로직 ...
self.prefill_wrappers_by_layout = {"full": self.prefill_wrappers}

if self.config.mesa_enabled:
    for layout_name, layout_mq_len in [
        ("draft", self.config.mesa_draft_fan_out * (self.config.speculate_k + 1)),
        ("proxy", self.config.mesa_proxy_fan_out * (self.config.speculate_k + 1)),
    ]:
        layout_cu_seqlens_q = torch.empty(max_bs + 1, dtype=torch.int32, device=self.device)
        layout_kv_indptr = torch.empty(max_bs + 1, dtype=torch.int32, device=self.device)
        layout_kv_indices = torch.empty(max_bs * max_num_blocks, dtype=torch.int32, device=self.device)
        layout_kv_last_page_len = torch.empty(max_bs, dtype=torch.int32, device=self.device)
        layout_mask_buf = torch.empty(max_bs * layout_mq_len * self.config.max_model_len,
                                       dtype=torch.uint8, device=self.device)
        layout_mask_indptr_buf = torch.empty(max_bs + 1, dtype=torch.int32, device=self.device)

        layout_wrappers = {}
        for bs in graph_bs_list:
            layout_wrappers[bs] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=layout_cu_seqlens_q[:bs + 1],
                paged_kv_indptr_buf=layout_kv_indptr[:bs + 1],
                paged_kv_indices_buf=layout_kv_indices[:bs * max_num_blocks],
                paged_kv_last_page_len_buf=layout_kv_last_page_len[:bs],
                custom_mask_buf=layout_mask_buf[:bs * layout_mq_len * self.config.max_model_len],
                mask_indptr_buf=layout_mask_indptr_buf[:bs + 1],
            )
        self.prefill_wrappers_by_layout[layout_name] = layout_wrappers
```

#### Attention layer — layout-aware wrapper selection

**파일**: `ssd/layers/attention.py` (line 113-124)

현재 attention은 `self.F * (self.K+1)`로 mq_len을 계산. 이를 context에서 읽도록 변경:

```python
elif tree_decode:
    if self.only_prefill_wrapper is not None:
        prefill_wrapper = self.only_prefill_wrapper
    else:
        # layout-aware: context에 active_mq_len이 있으면 사용
        context = get_context()
        mq_len = getattr(context, 'active_mq_len', self.F * (self.K + 1))
        bs = q.shape[0] // mq_len
        # layout-aware: context에 active_wrappers가 있으면 사용
        wrappers = getattr(context, 'active_wrappers', self.prefill_wrappers)
        wrapper_bs = next(b for b in sorted(wrappers.keys()) if b >= bs)
        prefill_wrapper = wrappers[wrapper_bs]
    o = prefill_wrapper.run(q, (self.k_cache, self.v_cache))
```

#### Context에 active layout 정보 세팅

**파일**: `ssd/utils/context.py`

`set_context`에 optional 필드 추가:
```python
def set_context(..., active_mq_len=None, active_wrappers=None):
    ctx = ...
    ctx.active_mq_len = active_mq_len
    ctx.active_wrappers = active_wrappers
    ...
```

#### _decode_tree_step에서 layout 기반 context 설정

**파일**: `ssd/engine/draft_runner.py` (`_decode_tree_step`)

```python
def _decode_tree_step(self, depth, current_input_ids, ..., layout):
    set_context(
        is_prefill=False,
        slot_mapping=step_slot_maps[depth],
        context_lens=step_context_lens[depth].to(torch.int32),
        block_tables=dbt,
        # layout-aware wrapper selection
        active_mq_len=layout.MQ_LEN,
        active_wrappers=self.prefill_wrappers_by_layout.get(layout.name, self.prefill_wrappers),
    )
    ...
```

#### CudaGraph 캡처 — layout별

`capture_fi_tree_decode_cudagraph`를 layout 파라미터로 일반화:

```python
def capture_fi_tree_decode_cudagraph(model_runner, layout=None):
    """layout 기반 FI tree decode CudaGraph 캡처."""
    if layout is None:
        MQ_LEN = model_runner.config.MQ_LEN
        graph_key = "fi_tree_decode"
    else:
        MQ_LEN = layout.MQ_LEN
        graph_key = layout.graph_key
    # N = max_bs * MQ_LEN으로 버퍼/그래프 캡처
    # context에 active_mq_len=MQ_LEN, active_wrappers=해당 layout wrappers 세팅
    ...
```

DraftRunner 초기화에서 layout별 캡처:

```python
# model_runner.py setup_and_warmup_model_and_cudagraphs:
if self.config.speculate and self.is_draft and self.config.draft_async:
    fi_... = capture_fi_tree_decode_cudagraph(self)  # full_layout
    self.graph_vars["fi_tree_decode"] = ...

    if self.config.mesa_enabled:
        draft_fi_... = capture_fi_tree_decode_cudagraph(self, layout=self.draft_layout)
        self.graph_vars[self.draft_layout.graph_key] = ...

        proxy_fi_... = capture_fi_tree_decode_cudagraph(self, layout=self.proxy_layout)
        self.graph_vars[self.proxy_layout.graph_key] = ...
```

### 2.7 run_fi_tree_decode_cudagraph() + capture_fi_tree_decode_cudagraph() 일반화

**파일**: `ssd/engine/helpers/cudagraph_helpers.py`

현재 `config.fan_out_list` / `config.MQ_LEN`을 직접 참조하는 곳 (모두 `layout.*`으로 변경):

| 위치 (line) | 현재 참조 | 변경 |
|-------------|----------|------|
| 158 | `MQ_LEN = sum(config.fan_out_list)` | `MQ_LEN = layout.MQ_LEN` |
| 160 | `orig_flat % MQ_LEN` | 동일 |
| 225 | `... * MQ_LEN` (cu_seqlens_q) | 동일 |
| 235, 251, 257 | `s * MQ_LEN` (step context_lens) | 동일 |
| 279 | `np.arange(MQ_LEN)` (mask rows) | 동일 |
| 285 | `(s+1) * MQ_LEN + (K+1)` (ttl_added) | 동일 |
| 290, 293 | `MQ_LEN` (mask shape) | 동일 |
| 299 | `blk * MQ_LEN` (mask diag) | 동일 |
| 342 | `wrapper._custom_mask_buf` | layout별 wrapper의 mask_buf |
| 784 | `MQ_LEN = sum(config.fan_out_list)` (capture) | `layout.MQ_LEN` |
| 785-871 | `bs * MQ_LEN` (capture 전체) | 동일 |
| 827 | `... * MQ_LEN` (cu_seqlens_q capture) | 동일 |
| 839 | `bs * MQ_LEN * max_model_len` (mask) | 동일 |

또한 **mask precompute** 내부 (`get_custom_mask` 호출, line 267-299):
- `fan_out_list` → `layout.fan_out_list`
- `fan_out_list_miss` → `layout.fan_out_list_miss`
- mask shape `[MQ_LEN, cols]` → `[layout.MQ_LEN, cols]`

**wrapper selection** (line 342, attention forward):
- `wrapper._custom_mask_buf` → layout별 wrapper의 버퍼

시그니처 변경:

```python
# run 함수
def run_fi_tree_decode_cudagraph(model_runner, input_ids, positions, last_only,
                                  graph_vars, tree_decode_step, cache_hits,
                                  hidden_states=None, layout=None):
    if layout is None:
        MQ_LEN = sum(model_runner.config.fan_out_list)  # backward compat
        graph_key = "fi_tree_decode"
    else:
        MQ_LEN = layout.MQ_LEN
        graph_key = layout.graph_key
    # 이하 모든 MQ_LEN 참조를 위 변수로 대체
    ...

# capture 함수
def capture_fi_tree_decode_cudagraph(model_runner, layout=None):
    if layout is None:
        MQ_LEN = sum(model_runner.config.fan_out_list)
        graph_key = "fi_tree_decode"
    else:
        MQ_LEN = layout.MQ_LEN
        graph_key = layout.graph_key
    # 이하 모든 MQ_LEN, fan_out_list 참조를 layout 기반으로 대체
    ...
```

---

## 3단계: LlamaModel에 split forward 지원

- [ ] 완료

### 3.1 LlamaModel 변경

**파일**: `ssd/models/llama3.py` (LlamaModel.forward, line 248-273)

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    start_layer: int = 0,
    end_layer: int | None = None,
    init_hidden_states: torch.Tensor | None = None,
    init_residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if init_hidden_states is not None:
        hidden_states = init_hidden_states
        residual = init_residual
    else:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

    actual_end = end_layer if end_layer is not None else len(self.layers)
    collected_acts = [] if self.use_eagle else None

    for layer_idx in range(start_layer, actual_end):
        layer = self.layers[layer_idx]
        if collected_acts is not None and layer_idx in self.eagle_layers:
            current_act = hidden_states if residual is None else hidden_states + residual
            collected_acts.append(current_act)
        hidden_states, residual = layer(positions, hidden_states, residual)

    if end_layer is None:
        hidden_states, _ = self.norm(hidden_states, residual)
        if collected_acts:
            eagle_acts = torch.cat(collected_acts, dim=-1)
            return hidden_states, eagle_acts
        return hidden_states
    else:
        return hidden_states, residual
```

### 3.2 LlamaForCausalLM.forward() 변경

**파일**: `ssd/models/llama3.py` (line 325-331)

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    start_layer: int = 0,
    end_layer: int | None = None,
    init_hidden_states: torch.Tensor | None = None,
    init_residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    out = self.model(input_ids, positions,
                     start_layer=start_layer, end_layer=end_layer,
                     init_hidden_states=init_hidden_states,
                     init_residual=init_residual)
    return out
```

---

## 4단계: Split CudaGraph (Target Verify)

- [ ] 완료

### 4.1 capture_mesa_verify_cudagraph()

**파일**: `ssd/engine/helpers/cudagraph_helpers.py`

```python
def capture_mesa_verify_cudagraph(model_runner):
    """MESA-SSD용 split verify CudaGraph.
    graph_pre: layers [0, exit_layer] → exit_hidden, exit_residual
    graph_post: layers [exit_layer+1, L-1] + norm → outputs
    """
    config = model_runner.config
    hf_config = config.hf_config
    max_bs = min(config.max_num_seqs, 512)
    k_plus_1 = config.speculate_k + 1
    exit_layer = config.mesa_exit_layer
    H = hf_config.hidden_size

    input_ids = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    positions = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs * k_plus_1, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, model_runner.max_num_blocks, dtype=torch.int32)
    cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)
    exit_hidden = torch.zeros(max_bs * k_plus_1, H, dtype=hf_config.torch_dtype)
    exit_residual = torch.zeros(max_bs * k_plus_1, H, dtype=hf_config.torch_dtype)
    outputs = torch.zeros(max_bs * k_plus_1, H, dtype=hf_config.torch_dtype)

    base = [1, 2, 4, 8]
    dynamic = list(range(16, max_bs + 1, 16))
    all_b = sorted(set(base + dynamic + [max_bs]))
    all_N = [b for b in all_b if b <= max_bs]

    graphs_pre = {}
    graphs_post = {}
    graph_pool = None

    for bs in reversed(all_N):
        flat = bs * k_plus_1
        seqlen_q = torch.full((bs,), k_plus_1, dtype=torch.int32)
        cu = cu_seqlens_q[:bs + 1]
        cu.zero_()
        cu[1:].copy_(torch.cumsum(seqlen_q, 0))
        context_lens[:bs] = seqlen_q

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping[:flat],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
            cu_seqlens_q=cu,
            max_seqlen_q=k_plus_1,
        )

        # graph_pre
        hs, res = model_runner.model(
            input_ids[:flat], positions[:flat], end_layer=exit_layer + 1)
        exit_hidden[:flat].copy_(hs)
        exit_residual[:flat].copy_(res)

        graph_pre = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_pre, graph_pool):
            hs, res = model_runner.model(
                input_ids[:flat], positions[:flat], end_layer=exit_layer + 1)
            exit_hidden[:flat].copy_(hs)
            exit_residual[:flat].copy_(res)
        if graph_pool is None:
            graph_pool = graph_pre.pool()
        graphs_pre[bs] = graph_pre

        # graph_post
        out = model_runner.model(
            input_ids[:flat], positions[:flat],
            start_layer=exit_layer + 1,
            init_hidden_states=exit_hidden[:flat],
            init_residual=exit_residual[:flat])
        outputs[:flat] = out if not isinstance(out, tuple) else out[0]

        graph_post = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_post, graph_pool):
            out = model_runner.model(
                input_ids[:flat], positions[:flat],
                start_layer=exit_layer + 1,
                init_hidden_states=exit_hidden[:flat],
                init_residual=exit_residual[:flat])
            outputs[:flat] = out if not isinstance(out, tuple) else out[0]
        graphs_post[bs] = graph_post

        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, context_lens=context_lens,
        block_tables=block_tables, cu_seqlens_q=cu_seqlens_q,
        exit_hidden=exit_hidden, exit_residual=exit_residual,
        outputs=outputs,
    )
    return graph_vars, graph_pool, graphs_pre, graphs_post, all_N
```

### 4.2 run_mesa_verify_cudagraph()

```python
@torch.inference_mode()
def run_mesa_verify_cudagraph(model_runner, input_ids, positions, last_only,
                               graph_vars, mesa_proxy_fn=None):
    """Split CudaGraph verify: pre → proxy → post → logits."""
    context = get_context()
    config = model_runner.config
    k_plus_1 = config.speculate_k + 1
    orig_bs = input_ids.size(0) // k_plus_1

    wrapper_bs = next(
        x for x in model_runner.graph_bs_list["mesa_verify"] if x >= orig_bs)
    graph_pre = model_runner.graphs["mesa_verify_pre"][wrapper_bs]
    graph_post = model_runner.graphs["mesa_verify_post"][wrapper_bs]

    for k, v in graph_vars.items():
        if k not in ("outputs", "exit_hidden", "exit_residual"):
            v.zero_()

    # Padding
    if wrapper_bs > orig_bs:
        pad_bs = wrapper_bs - orig_bs
        pad_flat = pad_bs * k_plus_1
        dev = input_ids.device
        input_ids = torch.cat([input_ids, torch.zeros(pad_flat, dtype=input_ids.dtype, device=dev)])
        positions = torch.cat([positions, torch.zeros(pad_flat, dtype=positions.dtype, device=dev)])
        slot_mapping = torch.cat([
            context.slot_mapping,
            torch.full((pad_flat,), -1, dtype=context.slot_mapping.dtype, device=dev)])
        bt = context.block_tables
        cl = context.context_lens
        block_tables = torch.cat([bt, bt[orig_bs-1:orig_bs].expand(pad_bs, -1).contiguous()])
        context_lens = torch.cat([cl, cl[orig_bs-1:orig_bs].expand(pad_bs).contiguous()])
        bs = wrapper_bs
    else:
        slot_mapping = context.slot_mapping
        block_tables = context.block_tables
        context_lens = context.context_lens
        bs = orig_bs

    graph_vars["input_ids"][:bs * k_plus_1] = input_ids
    graph_vars["positions"][:bs * k_plus_1] = positions
    graph_vars["slot_mapping"][:bs * k_plus_1] = slot_mapping
    graph_vars["context_lens"][:bs] = context_lens
    seqlen_q = torch.full(
        (bs,), k_plus_1, dtype=torch.int32, device=graph_vars["cu_seqlens_q"].device)
    cu = graph_vars["cu_seqlens_q"][:bs + 1]
    cu.zero_()
    cu[1:].copy_(torch.cumsum(seqlen_q, 0))
    if block_tables is not None:
        graph_vars["block_tables"][:bs, :block_tables.size(1)] = block_tables

    # graph_pre
    graph_pre.replay()

    # Mid-forward: proxy (CudaGraph 밖, ALL TP ranks가 compute_logits 실행)
    flat = orig_bs * k_plus_1
    exit_h = graph_vars["exit_hidden"][:flat] + graph_vars["exit_residual"][:flat]
    normed = model_runner.model.model.norm(exit_h, None)
    # ALL TP ranks가 compute_logits를 호출하여 gather에 참여.
    # rank 0: exit_logits = [B*(K+1), V] (full vocab logits)
    # rank 1+: exit_logits = None (ParallelLMHead.forward가 gather 후 rank 0에만 반환)
    exit_logits = model_runner.model.compute_logits(normed, last_only=False)

    # mesa_proxy_fn은 Verifier가 rank 0의 ModelRunner에만 설정.
    # rank 1+는 _mesa_proxy_fn = None이므로 이 블록을 skip.
    # → rank 1+에서 exit_logits=None이 callback에 전달되는 일 없음.
    if mesa_proxy_fn is not None:  # rank 0 only
        mesa_proxy_fn(exit_logits, orig_bs)

    # graph_post
    graph_post.replay()

    # Final logits
    outputs = graph_vars["outputs"][:flat]
    logits = model_runner.model.compute_logits(outputs, last_only)
    return logits
```

### 4.3 model_runner.py 변경

**파일**: `ssd/engine/model_runner.py`

`__init__`에 추가:
```python
self._mesa_proxy_fn = None
```

`setup_and_warmup_model_and_cudagraphs()` (line 278-300):
```python
if not self.enforce_eager:
    # decode CudaGraph (항상)
    decode_... = capture_cudagraph(self)
    self.graph_vars["decode"] = ...

    # verify CG
    if self.config.speculate and not (self.is_draft and self.config.use_eagle):
        if self.config.mesa_enabled and not self.is_draft:
            # MESA target: split verify CudaGraph (기존 verify skip → VRAM 절약)
            mesa_gv, mesa_pool, mesa_pre, mesa_post, mesa_bs = \
                capture_mesa_verify_cudagraph(self)
            self.graph_vars["mesa_verify"] = mesa_gv
            self.graph_pools["mesa_verify"] = mesa_pool
            self.graphs["mesa_verify_pre"] = mesa_pre
            self.graphs["mesa_verify_post"] = mesa_post
            self.graph_bs_list["mesa_verify"] = mesa_bs
        else:
            # 기존 단일 verify CudaGraph
            verify_... = capture_verify_cudagraph(self)
            self.graph_vars["verify"] = ...

    # fi_tree_decode CudaGraph (draft only)
    if self.config.speculate and self.is_draft and self.config.draft_async:
        fi_... = capture_fi_tree_decode_cudagraph(self)  # full_layout
        self.graph_vars["fi_tree_decode"] = ...

        # MESA draft: draft_layout + proxy_layout CudaGraph 추가
        if self.config.mesa_enabled:
            draft_fi_... = capture_fi_tree_decode_cudagraph(self, layout=self.draft_layout)
            self.graph_vars["fi_tree_decode_draft"] = ...
            self.graph_bs_list["fi_tree_decode_draft"] = ...

            proxy_fi_... = capture_fi_tree_decode_cudagraph(self, layout=self.proxy_layout)
            self.graph_vars["fi_tree_decode_proxy"] = ...
            self.graph_bs_list["fi_tree_decode_proxy"] = ...
```

`run_model()` (line 595-630):
```python
    # MESA verify: target만 (not is_draft → draft glue decode 보호)
    elif is_mq_kp1 and self.config.mesa_enabled and not self.is_draft \
            and "mesa_verify" in self.graph_vars:
        return run_mesa_verify_cudagraph(
            self, input_ids, positions, last_only,
            self.graph_vars["mesa_verify"],
            mesa_proxy_fn=self._mesa_proxy_fn)
    elif is_mq_kp1:
        return run_verify_cudagraph(...)  # 기존 (draft glue decode 포함)
```

---

## 5단계: Verifier에서 proxy 계산 및 전송

- [ ] 완료

### 5.1 Verifier.verify() 변경

**파일**: `ssd/engine/verifier.py`

```python
def verify(self, seqs, speculate_result, eagle=False):
    B = len(seqs)
    K = self.lookahead
    config = self.target_model_runner.config

    if config.mesa_enabled:
        async_pg = self.target_model_runner.async_pg
        draft_rank = self.target_model_runner.draft_rank
        draft_tokens = speculate_result.speculations[:, 1:]
        logits_q = speculate_result.logits_q
        cache_hits = speculate_result.cache_hits

        def _proxy_fn(exit_logits, orig_bs):
            self._compute_and_send_proxy(
                exit_logits, draft_tokens, logits_q, orig_bs, K,
                async_pg, draft_rank, cache_hits=cache_hits)

        self.target_model_runner._mesa_proxy_fn = _proxy_fn

    result = self.target_model_runner.call("run", seqs, False, False, True)

    if config.mesa_enabled:
        self.target_model_runner._mesa_proxy_fn = None

    # 이하 기존 verify 로직...
```

### 5.2 _compute_and_send_proxy()

```python
def _compute_and_send_proxy(self, exit_logits, draft_tokens, logits_q,
                             B, K, async_pg, draft_rank, cache_hits=None):
    config = self.target_model_runner.config
    top_k = config.mesa_proxy_top_k

    if exit_logits.dim() == 2:
        exit_logits = exit_logits.view(B, K + 1, -1)

    p_E = torch.softmax(exit_logits[:, :K, :].float(), dim=-1)
    p_D = torch.softmax(logits_q.float(), dim=-1)

    gather_idx = draft_tokens.unsqueeze(-1)
    p_E_y = p_E.gather(2, gather_idx).squeeze(-1)
    p_D_y = p_D.gather(2, gather_idx).squeeze(-1)
    accept_probs = (p_E_y / (p_D_y + 1e-10)).clamp(max=1.0)

    residual = (p_E - p_D).clamp(min=0)
    residual.scatter_(2, gather_idx, 0.0)
    topk_probs, topk_ids = residual.topk(top_k, dim=-1)
    topk_sum = topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    topk_probs = topk_probs / topk_sum

    if cache_hits is not None and not config.jit_speculate:
        miss_mask = ~cache_hits.to(torch.bool)
        if miss_mask.any():
            accept_probs[miss_mask] = 0.0
            miss_p_E = p_E[miss_mask].clone()
            miss_p_E.scatter_(2, gather_idx[miss_mask], 0.0)
            miss_topk_probs, miss_topk_ids = miss_p_E.topk(top_k, dim=-1)
            topk_ids[miss_mask] = miss_topk_ids
            topk_probs[miss_mask] = miss_topk_probs / miss_topk_probs.sum(-1, keepdim=True).clamp(min=1e-10)

    from ssd.utils.async_helpers.nccl_pack import send_int64
    send_int64(async_pg, draft_rank,
               accept_probs.view(-1).to(torch.float32).view(torch.int32).to(torch.int64),
               topk_ids.reshape(-1),
               topk_probs.view(-1).to(torch.float32).view(torch.int32).to(torch.int64))
```

---

## 6단계: Draft 측 — Budget Split + 2-Pass Tree Decode

- [ ] 완료

### 6.1 draft_loop() 계약 변경

**파일**: `ssd/engine/draft_runner.py` (`draft_loop`, line 859-907)

현재 호출 계약:
```python
tree_decode_args = self._build_tree_batch(...)     # tree args 반환
tokens, logits, acts = self._decode_tree(tree_decode_args)  # 외부에서 decode
self._populate_tree_cache(tree_decode_args, tokens, logits, ...)  # 외부에서 populate
```

MESA에서는 2-pass decode + populate가 `_build_tree_batch` 내부에서 완료되므로 계약이 다름.
`draft_loop`에서 분기:

```python
# draft_loop() cmd=0 내부:
glue_decode_input_ids, partial_tree_decode_args = self._service_spec_request()
self._reset_tree_cache_tensors()

if self.config.mesa_enabled:
    # MESA: 2-pass decode + populate 내부 완료
    self._build_tree_batch_mesa(partial_tree_decode_args, glue_decode_input_ids)
else:
    # 기존: 외부에서 decode + populate
    tree_decode_args = self._build_tree_batch(partial_tree_decode_args, glue_decode_input_ids)
    tokens, logits, activations = self._decode_tree(tree_decode_args)
    self._populate_tree_cache(tree_decode_args, tokens, logits,
                               tree_decode_args["cache_hits"], activations)
```

### 6.2 _build_tree_batch_mesa()

**파일**: `ssd/engine/draft_runner.py` (MESA 전용, 기존 _build_tree_batch와 별도)

```python
def _build_tree_batch_mesa(self, partial_tree_decode_args, glue_decode_input_ids):
    """MESA 2-pass tree decode. 내부에서 decode + populate까지 완료."""
    # ... 기존 glue decode (step 1-4) — _build_tree_batch와 동일 ...

    # ===== MESA: Budget Split — 2-pass tree decode =====

    # Pass 1: draft-sourced (즉시 시작, proxy 대기 없음)
        draft_forked = self._select_draft_sourced_tokens(
            glue_decode_logits, cache_hits, gd_for_fork,
            self.config.mesa_draft_fan_out)  # [B, K+1, draft_fan_out]

        # irecv를 decode 시작 전에 걸어둠 → target send가 block되지 않음
        proxy_recv_work, proxy_buf = self._irecv_mesa_proxy(B, K)

        draft_tree_args = self._build_tree_decode_args(
            partial_tree_decode_args, draft_forked.view(-1),
            self.draft_layout, cache_hits, ...)
        draft_tokens, draft_logits, draft_acts = self._decode_tree(
            draft_tree_args, layout=self.draft_layout)

        # Pass 중간: irecv 완료 대기 + unpack
        proxy_recv_work.wait()
        mesa_proxy = self._unpack_mesa_proxy(proxy_buf, B, K)

        # Pass 2: proxy-sourced (dedup with draft-sourced)
        proxy_forked = self._select_proxy_sourced_tokens(
            glue_decode_logits, cache_hits, gd_for_fork,
            mesa_proxy, draft_forked,
            self.config.mesa_proxy_fan_out)  # [B, K+1, proxy_fan_out]

        proxy_tree_args = self._build_tree_decode_args(
            partial_tree_decode_args, proxy_forked.view(-1),
            self.proxy_layout, cache_hits, ...)
        proxy_tokens, proxy_logits, proxy_acts = self._decode_tree(
            proxy_tree_args, layout=self.proxy_layout)

        # 결과 합침: layout별 cache key 생성 + 단일 cache populate
        self._merge_and_populate_cache(
            draft_tree_args, draft_tokens, draft_logits,
            proxy_tree_args, proxy_tokens, proxy_logits,
            self.draft_layout, self.proxy_layout,
            draft_acts, proxy_acts)
```

### 6.2 _select_draft_sourced_tokens()

```python
def _select_draft_sourced_tokens(self, logits, cache_hits, returned_tokens, draft_fan_out):
    """Draft logits top-k로 fork tokens 선택."""
    logits = logits.clone()
    logits[:, :-1, :] = logits[:, :-1, :].scatter(
        dim=2, index=returned_tokens[:, 1:].unsqueeze(2), value=float('-inf'))
    _, topk_idx = torch.topk(logits, draft_fan_out, dim=-1)  # [B, K+1, draft_fan_out]
    return topk_idx
```

### 6.3 _select_proxy_sourced_tokens()

```python
def _select_proxy_sourced_tokens(self, logits, cache_hits, returned_tokens,
                                   mesa_proxy, draft_forked, proxy_fan_out):
    """Proxy correction tokens 선택.
    - Draft tree와 중복되지 않는 토큰만 선택 (동일 branch 방지)
    - Proxy 우선, 부족분은 logits fallback에서 채움 (draft_forked 제외)
    """
    B, K = mesa_proxy["accept_probs"].shape
    proxy_topk_ids = mesa_proxy["topk_ids"]  # [B, K, proxy_top_k]

    # Fallback 후보: draft logits top-N에서 returned_tokens 제외
    # draft_forked + proxy 모두에 없는 토큰을 refill용으로 사용
    logits_for_fallback = logits.clone()
    logits_for_fallback[:, :-1, :] = logits_for_fallback[:, :-1, :].scatter(
        dim=2, index=returned_tokens[:, 1:].unsqueeze(2), value=float('-inf'))
    total_need = self.config.async_fan_out  # draft_fan_out + proxy_fan_out
    _, fallback_topk = torch.topk(logits_for_fallback, total_need, dim=-1)  # [B, K+1, total]

    result = torch.zeros(B, K + 1, proxy_fan_out, dtype=torch.int64, device=logits.device)

    # Position 0..K-1
    for b in range(B):
        for pos in range(K):
            draft_set = set(draft_forked[b, pos].tolist())
            proxy_tokens = proxy_topk_ids[b, pos].tolist()

            # proxy 중 draft에 없는 것 우선
            selected = [t for t in proxy_tokens if t not in draft_set]

            # 부족하면 fallback (draft에도 선택된 proxy에도 없는 것)
            if len(selected) < proxy_fan_out:
                used = draft_set | set(selected)
                fallback = [t for t in fallback_topk[b, pos].tolist() if t not in used]
                selected.extend(fallback[:proxy_fan_out - len(selected)])

            for j in range(min(len(selected), proxy_fan_out)):
                result[b, pos, j] = selected[j]

    # Position K (all-accept): draft_forked 제외 후 top-k
    logits_k = logits_for_fallback[:, K, :].clone()
    logits_k.scatter_(1, draft_forked[:, K, :], float('-inf'))  # draft 선택분 제외
    _, all_accept_topk = torch.topk(logits_k, proxy_fan_out, dim=-1)
    result[:, K, :] = all_accept_topk

    return result
```

> **NOTE**: loop는 개념 설명용. 실제 구현 시 vectorized.
> **Dedup 보장**: proxy-sourced tokens ∩ draft-sourced tokens == ∅ (모든 position에서)
> **Underfill 정책**: proxy_top_k < proxy_fan_out이거나 dedup으로 토큰이 부족한 경우,
> `fallback_topk` (draft logits top-N, N=async_fan_out)에서 draft/proxy 모두 미사용 토큰으로 충당.
> V=128256 >> async_fan_out(~5)이므로 underfill이 발생하지 않음이 보장됨.
> Config validation에서 `assert mesa_proxy_top_k >= 1` 추가 (최소 1개 proxy 후보).

### 6.4 _irecv_mesa_proxy() + _unpack_mesa_proxy()

Non-blocking recv로 target send blocking 방지. irecv를 draft decode 시작 전에 걸어두고,
decode 완료 후 wait()로 데이터 도착을 보장.

```python
def _irecv_mesa_proxy(self, B, K):
    """Non-blocking recv를 걸어둠. (work, buffer) 반환."""
    top_k = self.config.mesa_proxy_top_k
    total_len = B * K + B * K * top_k + B * K * top_k
    buf = torch.empty(total_len, dtype=torch.int64, device=self.device)
    work = dist.irecv(buf, src=0, group=self.async_pg)
    return work, buf

def _unpack_mesa_proxy(self, buf, B, K):
    """irecv 완료 후 buffer에서 proxy 데이터 추출."""
    top_k = self.config.mesa_proxy_top_k
    off = 0
    accept_probs = buf[off:off + B*K].to(torch.int32).view(torch.float32).view(B, K)
    off += B * K
    topk_ids = buf[off:off + B*K*top_k].view(B, K, top_k)
    off += B * K * top_k
    topk_probs = buf[off:].to(torch.int32).view(torch.float32).view(B, K, top_k)
    return {"accept_probs": accept_probs, "topk_ids": topk_ids, "topk_probs": topk_probs}
```

**irecv가 isend와 다른 이유 (안전한 이유)**:
- Buffer lifetime: `buf`는 `_irecv_mesa_proxy`에서 생성되어 caller가 참조 유지 → GC 안 됨
- Race condition 없음: decode 중에 `buf`를 읽지 않고, `wait()` 후에만 `_unpack_mesa_proxy`에서 읽음
- Stream ordering: NCCL recv는 자체 stream에서 동작, decode는 default stream → 독립적

### 6.5 _populate_tree_cache() layout-aware로 일반화

현재 `_populate_tree_cache`(draft_runner.py:814-830)는 `self._fan_idx_hit/miss`(full layout)로 key를 생성.
Layout별로 올바른 `fan_idx`를 사용해야 함.

```python
def _populate_tree_cache(self, payload, tokens, logits, cache_hits, layout, activations=None):
    """layout 기반 cache key 생성 + populate."""
    seq_ids_expanded = payload["seq_ids_expanded"].to(torch.int64)
    rec_flat = payload["rec_flat"].to(torch.int64)

    # layout별 fan_idx 사용 (기존: self._fan_idx_hit → layout.fan_idx_hit)
    k_flat = torch.cat([
        layout.fan_idx_hit if hit else layout.fan_idx_miss
        for hit in payload["cache_hits_list"]
    ])

    keys = torch.stack([seq_ids_expanded, k_flat, rec_flat], dim=1).contiguous()
    return keys, tokens, logits, activations
```

### 6.6 _merge_and_populate_cache()

```python
def _merge_and_populate_cache(self, 
                                draft_payload, draft_tokens, draft_logits,
                                proxy_payload, proxy_tokens, proxy_logits,
                                draft_layout, proxy_layout,
                                draft_acts=None, proxy_acts=None):
    """Draft + proxy tree decode 결과의 cache key를 layout별로 생성하고 합침."""

    # 각 pass에서 layout별 fan_idx로 key 생성
    draft_keys, _, _, _ = self._populate_tree_cache(
        draft_payload, draft_tokens, draft_logits, 
        draft_payload["cache_hits"], draft_layout, draft_acts)

    proxy_keys, _, _, _ = self._populate_tree_cache(
        proxy_payload, proxy_tokens, proxy_logits,
        proxy_payload["cache_hits"], proxy_layout, proxy_acts)

    # 합쳐서 단일 cache에 저장
    self.tree_cache_keys = torch.cat([draft_keys, proxy_keys], dim=0)       # [N1+N2, 3]
    self.tree_cache_tokens = torch.cat([draft_tokens, proxy_tokens], dim=0) # [N1+N2, K]
    self.tree_cache_logits = torch.cat([draft_logits, proxy_logits], dim=0) # [N1+N2, K, V]
    if draft_acts is not None:
        self.tree_cache_activations = torch.cat([draft_acts, proxy_acts], dim=0)
```

**Key semantics 보존**: draft pass의 key는 `draft_layout.fan_idx_hit/miss`로 `k_idx`를 만들고, proxy pass는 `proxy_layout.fan_idx_hit/miss`로 만듦. 둘 다 같은 depth 값(0..K)을 사용하되 fan_out 수가 다름. Cache lookup 시 `(seq_id, k_idx, rec_token)` 매칭은 depth 기반이므로, draft와 proxy가 같은 depth에 다른 rec_token으로 cache entry를 갖게 됨 → **정상 동작**.

---

## 7단계: bench.py arguments 및 테스트

- [ ] 완료

### 7.1 bench.py argument 추가

**파일**: `bench/bench.py`

```python
parser.add_argument("--mesa", action="store_true", help="Enable MESA-SSD")
parser.add_argument("--mesa_exit_layer", type=int, default=None,
                    help="Early-exit layer index (default: 2*L//3)")
parser.add_argument("--mesa_proxy_top_k", type=int, default=3,
                    help="Number of correction tokens per position")
parser.add_argument("--mesa_draft_fan_out", type=int, default=None,
                    help="Draft-sourced branches per position (default: fan_out//2)")
```

`create_llm_kwargs()`:
```python
if args.mesa:
    llm_kwargs["mesa_enabled"] = True
    if args.mesa_exit_layer is not None:
        llm_kwargs["mesa_exit_layer"] = args.mesa_exit_layer
    llm_kwargs["mesa_proxy_top_k"] = args.mesa_proxy_top_k
    if args.mesa_draft_fan_out is not None:
        llm_kwargs["mesa_draft_fan_out"] = args.mesa_draft_fan_out
```

### 7.2 단위 테스트

1. Split forward: `forward(end_layer=E)` + `forward(start_layer=E+1, init_*)` == `forward()`
2. Split CudaGraph: graph_pre + graph_post == 기존 단일 graph
3. TreeLayout: full_layout으로 기존 경로 결과 동일 (backward compat)
4. TreeLayout: draft_layout + proxy_layout MQ_LEN이 올바르게 합산
5. Proxy 계산: accept_probs ∈ [0,1], residual top-k에 draft token 미포함
6. Token dedup 기본: proxy-sourced ∩ draft-sourced == ∅ (모든 position)
7. Token dedup edge case: draft=[A], proxy=[A], proxy_fan_out=2 → proxy row에 A 없이 fallback만
8. Position K dedup: draft_forked[:, K, :] ∩ proxy_result[:, K, :] == ∅
7. Budget split: 2-pass decode 결과가 올바르게 합쳐져 cache populate
8. KV scratch 재사용: 2번째 pass가 1번째 결과를 훼손하지 않음
9. NCCL pack/unpack round-trip
10. Feature gating: Qwen3/EAGLE + mesa → assert

### 7.3 Edge case 테스트

- TP > 1 (4 GPU TP + 1 draft)
- cache miss + jit_speculate=False / True
- temp > 0 (non-greedy sampling)
- B > 1 (batch size)
- proxy_top_k < proxy_fan_out (dedup 부족분 → fallback_topk에서 충당되는지 확인)

### 7.4 벤치마크

```bash
cd bench

# Baseline (기존 SSD)
python -O bench.py --llama --size 8 --async --spec --k 4 --f 3 \
    --gpus 2 --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# MESA-SSD (default: draft_fan_out=1, proxy_fan_out=2)
python -O bench.py --llama --size 8 --async --spec --k 4 --f 3 \
    --gpus 2 --b 1 --temp 0 --numseqs 128 --output_len 512 --all \
    --mesa --mesa_exit_layer 21

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

## 수정 파일 요약

| 파일 | 변경 내용 | 규모 |
|------|----------|------|
| `ssd/config.py` | mesa params + budget split + validation + gating | ~30줄 |
| `ssd/engine/helpers/tree_layout.py` | **신규**: TreeLayout dataclass + create_tree_layout() | ~50줄 |
| `ssd/models/llama3.py` | LlamaModel/ForCausalLM split forward | ~25줄 |
| `ssd/engine/helpers/cudagraph_helpers.py` | capture/run_mesa_verify + capture/run_fi_tree_decode layout 일반화 | ~150줄 |
| `ssd/engine/model_runner.py` | mesa CudaGraph 캡처 분기 + run_model 분기 + FlashInfer wrapper layout별 생성 | ~60줄 |
| `ssd/layers/attention.py` | tree decode wrapper selection을 context.active_mq_len/active_wrappers 기반으로 변경 | ~10줄 |
| `ssd/utils/context.py` | set_context에 active_mq_len, active_wrappers 필드 추가 | ~5줄 |
| `ssd/engine/verifier.py` | proxy_fn + _compute_and_send_proxy() | ~55줄 |
| `ssd/engine/draft_runner.py` | TreeLayout 적용 + budget split 2-pass + token selection + layout-aware cache populate | ~130줄 |
| `bench/bench.py` | --mesa arguments | ~15줄 |
| **총** | | **~560줄** |

---

## 구현 순서

```
1단계: Config + feature gating (config.py)
    |
2단계: TreeLayout 추상화 + 기존 경로 backward compat
       (tree_layout.py, draft_runner.py _init_prealloc_buffers)
    |
3단계: _decode_tree / _compute_step_positions / run_fi_tree_decode 일반화
       (draft_runner.py, cudagraph_helpers.py)
    |
4단계: Split forward (llama3.py)
    |
5단계: Split CudaGraph capture + replay — target verify
       (cudagraph_helpers.py, model_runner.py)
    |
6단계: Proxy 계산 + send (verifier.py)
    |
7단계: Budget split 2-pass tree decode + dedup + merge
       (draft_runner.py)
    |
8단계: bench.py arguments + 테스트 + 벤치마크
```
