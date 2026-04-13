# MESA-SSD 구현 계획서 (v3 — Split CudaGraph + TP-safe)

## 0. 설계 원칙

1. **CudaGraph 성능 유지**: eager 모드 fallback 없음. Verify CudaGraph를 pre/post로 분리.
2. **TP > 1 호환**: 모든 TP rank가 동일한 NCCL collective 패턴을 실행.
3. **Mid-forward proxy 전송**: target verify의 ~2/3 지점에서 proxy를 draft에 전송하여 tree decode와 오버랩.

---

## 0.1 전체 타이밍 (Split CudaGraph)

```
TARGET (rank 0, all TP ranks)            DRAFT (rank N)
─────────────────────────────            ──────────────────
1. speculate() 호출                       1. recv_cmd() [blocking]
   → cmd=0, cache_keys, etc 전송
                                         2. _service_spec_request()
                                            - cache lookup / JIT speculate
                                            - SEND response [기존과 동일]
2. response 수신
3. verify() 호출                          3. _reset_tree_cache_tensors()
   ┌ graph_pre.replay()                   4. _build_tree_batch()
   │  layers [0 .. exit_layer]               - glue decode 실행
   └ → exit_buffer에 hidden states 기록       ...glue decode 완료...
                                         ┌─── MESA: proxy 수신 [blocking recv]
   [CudaGraph 밖, ALL TP ranks]:         │
   norm(exit_buffer) → lm_head           │
   → exit_logits (TP gather 포함)        │
   rank 0: proxy 계산 + NCCL SEND ─────→ │    proxy 기반 fork token 선택
                                         └───
   ┌ graph_post.replay()                  5. _decode_tree() [기존과 동일]
   │  layers [exit_layer+1 .. L-1]
   │  + final norm
   └ → outputs
   lm_head(outputs) → final_logits       6. _populate_tree_cache()
   verify 알고리즘 [기존과 동일]
   recovery sampling [기존과 동일]
4. postprocess
5. 다음 speculate() →                    7. recv_cmd()
```

**오버랩**: proxy는 전체 forward의 ~2/3 지점에서 전송. Target의 나머지 ~1/3 forward와 draft의 tree decode가 병렬 실행.

---

## 1단계: Config 확장

- [ ] 완료

**파일**: `ssd/config.py`

추가할 필드 (line 45 이후, debugging 섹션 뒤):
```python
# MESA-SSD parameters
mesa_enabled: bool = False
mesa_exit_layer: int | None = None      # None=auto: 2*L//3
mesa_proxy_top_k: int = 3
mesa_risk_threshold: float = 0.5
```

`__post_init__`에 추가 (line 94 앞):
```python
if self.mesa_enabled:
    assert self.draft_async, "MESA-SSD requires draft_async=True"
    assert self.speculate, "MESA-SSD requires speculate=True"
    if self.mesa_exit_layer is None:
        L = self.hf_config.num_hidden_layers
        self.mesa_exit_layer = (2 * L) // 3
    assert 0 < self.mesa_exit_layer < self.hf_config.num_hidden_layers
    if self.fan_out_list is not None:
        assert self.mesa_proxy_top_k >= max(self.fan_out_list), \
            "mesa_proxy_top_k must be >= max(fan_out_list)"
```

---

## 2단계: LlamaModel/Qwen3Model에 split forward 지원 추가

- [ ] 완료

### 2.1 LlamaModel 변경

**파일**: `ssd/models/llama3.py`

LlamaModel.forward() (line 248-273)를 **start_layer/end_layer + 초기상태** 파라미터 추가로 split 실행 지원.

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    start_layer: int = 0,
    end_layer: int | None = None,         # None = 전체 (L)
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

    # end_layer가 None이면 전체 forward → final norm 적용
    if end_layer is None:
        hidden_states, _ = self.norm(hidden_states, residual)
        if collected_acts:
            eagle_acts = torch.cat(collected_acts, dim=-1)
            return hidden_states, eagle_acts
        return hidden_states
    else:
        # split forward: norm 적용하지 않고 hidden_states, residual 반환
        return hidden_states, residual
```

**start_layer=0, end_layer=None (기존 호출)**: 동작 완전 동일. 기존 코드 호환성 유지.

### 2.2 LlamaForCausalLM.forward() 변경

**파일**: `ssd/models/llama3.py` (line 325-331)

ForCausalLM도 split 파라미터를 받아 내부 LlamaModel에 전달:

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

### 2.3 Qwen3Model, Qwen3ForCausalLM에도 동일 적용

**파일**: `ssd/models/qwen3.py` — 동일 패턴.

---

## 3단계: Split CudaGraph Capture & Replay

- [ ] 완료

이 단계가 가장 핵심. `cudagraph_helpers.py`에 MESA 전용 캡처/실행 함수 추가.

### 3.1 capture_mesa_verify_cudagraph()

**파일**: `ssd/engine/helpers/cudagraph_helpers.py`

기존 `capture_verify_cudagraph()`를 참고하되, **두 개의 CudaGraph**를 캡처:

```python
def capture_mesa_verify_cudagraph(model_runner):
    """MESA-SSD용 split verify CudaGraph 캡처.
    graph_pre: layers [0, exit_layer] → (hidden_states, residual) 출력
    graph_post: layers [exit_layer+1, L-1] + norm → outputs 출력
    """
    config = model_runner.config
    hf_config = config.hf_config
    max_bs = min(config.max_num_seqs, 512)
    k_plus_1 = config.speculate_k + 1
    exit_layer = config.mesa_exit_layer
    H = hf_config.hidden_size

    # 공유 버퍼
    input_ids = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    positions = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs * k_plus_1, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, model_runner.max_num_blocks, dtype=torch.int32)
    cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)

    # Split 전용 중간 버퍼 (graph_pre 출력 → graph_post 입력)
    exit_hidden = torch.zeros(max_bs * k_plus_1, H, dtype=hf_config.torch_dtype)
    exit_residual = torch.zeros(max_bs * k_plus_1, H, dtype=hf_config.torch_dtype)

    # graph_post 출력
    outputs = torch.zeros(max_bs * k_plus_1, H)

    # batch size 버킷 (기존과 동일 패턴)
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

        # --- graph_pre: layers [0, exit_layer] ---
        # warmup
        hs, res = model_runner.model(
            input_ids[:flat], positions[:flat],
            end_layer=exit_layer + 1)
        exit_hidden[:flat].copy_(hs)
        exit_residual[:flat].copy_(res)

        graph_pre = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_pre, graph_pool):
            hs, res = model_runner.model(
                input_ids[:flat], positions[:flat],
                end_layer=exit_layer + 1)
            exit_hidden[:flat].copy_(hs)
            exit_residual[:flat].copy_(res)
        if graph_pool is None:
            graph_pool = graph_pre.pool()
        graphs_pre[bs] = graph_pre

        # --- graph_post: layers [exit_layer+1, L-1] + norm ---
        # warmup
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

### 3.2 run_mesa_verify_cudagraph()

```python
@torch.inference_mode()
def run_mesa_verify_cudagraph(model_runner, input_ids, positions, last_only,
                               graph_vars, mesa_proxy_fn=None):
    """Split CudaGraph verify: pre → proxy → post → logits.
    mesa_proxy_fn: rank 0에서만 non-None. exit_logits [B, K+1, V]를 받음.
    """
    context = get_context()
    config = model_runner.config
    k_plus_1 = config.speculate_k + 1
    orig_bs = input_ids.size(0) // k_plus_1

    wrapper_bs = next(
        x for x in model_runner.graph_bs_list["mesa_verify"] if x >= orig_bs)
    graph_pre = model_runner.graphs["mesa_verify_pre"][wrapper_bs]
    graph_post = model_runner.graphs["mesa_verify_post"][wrapper_bs]

    # 기존과 동일한 padding 로직 (생략 — run_verify_cudagraph과 동일)
    # ... input_ids, positions, slot_mapping, context_lens, block_tables padding ...

    # graph_vars에 입력 복사 (기존 패턴)
    bs = wrapper_bs
    graph_vars["input_ids"][:bs * k_plus_1] = input_ids  # (padded)
    graph_vars["positions"][:bs * k_plus_1] = positions
    # ... slot_mapping, context_lens, block_tables, cu_seqlens_q ...

    # ====== Phase 1: graph_pre.replay() ======
    # layers [0 .. exit_layer] 실행 → exit_hidden, exit_residual에 결과 기록
    graph_pre.replay()

    # ====== Mid-forward: proxy 계산 + 전송 (CudaGraph 밖) ======
    # 모든 TP rank가 실행 → lm_head의 gather에 모두 참여 → TP 정합
    flat = orig_bs * k_plus_1
    exit_h = graph_vars["exit_hidden"][:flat] + graph_vars["exit_residual"][:flat]
    normed = model_runner.model.model.norm(exit_h, None)  # RMSDNorm, 단일 텐서 반환
    exit_logits = model_runner.model.compute_logits(normed, last_only=False)  # TP gather 포함

    if mesa_proxy_fn is not None:  # rank 0만 non-None
        mesa_proxy_fn(exit_logits, orig_bs)

    # ====== Phase 2: graph_post.replay() ======
    # layers [exit_layer+1 .. L-1] + final norm → outputs
    graph_post.replay()

    # ====== Final logits (기존과 동일) ======
    outputs = graph_vars["outputs"][:flat]
    logits = model_runner.model.compute_logits(outputs, last_only)

    return logits
```

**TP 정합 보장**: `compute_logits()` 내부의 `ParallelLMHead.forward()`가 `dist.gather()`를 호출. graph_pre.replay()와 graph_post.replay() 사이에서 ALL TP ranks가 동일하게 `compute_logits`를 호출하므로 gather 횟수가 일치.

### 3.3 model_runner.py 변경

**파일**: `ssd/engine/model_runner.py`

#### setup_and_warmup_model_and_cudagraphs() (line 278-300)

MESA verify CudaGraph 캡처 추가:

```python
if not self.enforce_eager:
    # 기존 decode, verify, fi_tree_decode, glue_decode 캡처...

    # MESA: split verify CudaGraph
    if self.config.mesa_enabled and self.config.speculate and not self.is_draft:
        mesa_gv, mesa_pool, mesa_pre, mesa_post, mesa_bs = \
            capture_mesa_verify_cudagraph(self)
        self.graph_vars["mesa_verify"] = mesa_gv
        self.graph_pools["mesa_verify"] = mesa_pool
        self.graphs["mesa_verify_pre"] = mesa_pre
        self.graphs["mesa_verify_post"] = mesa_post
        self.graph_bs_list["mesa_verify"] = mesa_bs
```

#### run_model() (line 595-630)

MESA verify 분기 추가 (`elif is_mq_kp1:` 앞에):

```python
    elif is_mq_kp1 and self.config.mesa_enabled and "mesa_verify" in self.graph_vars:
        return run_mesa_verify_cudagraph(
            self, input_ids, positions, last_only,
            self.graph_vars["mesa_verify"],
            mesa_proxy_fn=self._mesa_proxy_fn)  # Verifier가 설정
    elif is_mq_kp1:
        return run_verify_cudagraph(...)  # 기존 경로 유지
```

`__init__`에 추가:
```python
self._mesa_proxy_fn = None  # Verifier가 verify 전후로 설정/해제
```

---

## 4단계: Verifier에서 proxy 계산 및 전송

- [ ] 완료

### 4.1 Verifier.verify() 변경

**파일**: `ssd/engine/verifier.py` (line 54-153)

```python
def verify(self, seqs, speculate_result, eagle=False):
    B = len(seqs)
    K = self.lookahead
    config = self.target_model_runner.config

    # MESA: proxy function 정의 및 등록
    if config.mesa_enabled:
        async_pg = self.target_model_runner.async_pg
        draft_rank = self.target_model_runner.draft_rank
        draft_tokens = speculate_result.speculations[:, 1:]  # [B, K]
        logits_q = speculate_result.logits_q  # [B, K, V]

        def _proxy_fn(exit_logits, orig_bs):
            """run_mesa_verify_cudagraph 중간에서 호출됨 (rank 0만)."""
            self._compute_and_send_proxy(
                exit_logits, draft_tokens, logits_q, orig_bs, K,
                async_pg, draft_rank)

        self.target_model_runner._mesa_proxy_fn = _proxy_fn

    # target forward (MESA일 때 run_mesa_verify_cudagraph 경로로 진입)
    result = self.target_model_runner.call("run", seqs, False, False, True)

    # proxy function 해제
    if config.mesa_enabled:
        self.target_model_runner._mesa_proxy_fn = None

    # 이하 기존 verify 로직 동일...
```

### 4.2 _compute_and_send_proxy() (Verifier 신규 메서드)

exit_logits를 직접 받으므로 norm/lm_head 호출 불필요 (Issue 3 해결):

```python
def _compute_and_send_proxy(self, exit_logits, draft_tokens, logits_q,
                             B, K, async_pg, draft_rank):
    """exit_logits로부터 proxy를 계산하고 draft에 NCCL 전송.

    Args:
        exit_logits: [B*(K+1), V] 또는 [B, K+1, V] — norm+lm_head+TP gather 완료
        draft_tokens: [B, K]
        logits_q: [B, K, V]
    """
    config = self.target_model_runner.config
    top_k = config.mesa_proxy_top_k

    if exit_logits.dim() == 2:
        exit_logits = exit_logits.view(B, K + 1, -1)  # [B, K+1, V]

    # p_E, p_D (position 0..K-1)
    p_E = torch.softmax(exit_logits[:, :K, :].float(), dim=-1)  # [B, K, V]
    p_D = torch.softmax(logits_q.float(), dim=-1)                # [B, K, V]

    # Accept probability proxy
    gather_idx = draft_tokens.unsqueeze(-1)  # [B, K, 1]
    p_E_y = p_E.gather(2, gather_idx).squeeze(-1)  # [B, K]
    p_D_y = p_D.gather(2, gather_idx).squeeze(-1)  # [B, K]
    accept_probs = (p_E_y / (p_D_y + 1e-10)).clamp(max=1.0)  # [B, K]

    # Residual proxy
    beta = accept_probs.unsqueeze(-1)  # [B, K, 1]
    residual = (p_E - beta * p_D).clamp(min=0)  # [B, K, V]
    residual.scatter_(2, gather_idx, 0.0)
    topk_probs, topk_ids = residual.topk(top_k, dim=-1)  # [B, K, top_k]
    topk_sum = topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    topk_probs = topk_probs / topk_sum

    # NCCL 전송 (단일 packed 메시지)
    from ssd.utils.async_helpers.nccl_pack import send_int64
    send_int64(async_pg, draft_rank,
               accept_probs.view(-1).to(torch.float32).view(torch.int32).to(torch.int64),
               topk_ids.reshape(-1),
               topk_probs.view(-1).to(torch.float32).view(torch.int32).to(torch.int64))
```

---

## 5단계: Draft 측 — Proxy 수신 및 Tree Cache 토큰 선택

- [ ] 완료

### 5.1 DraftRunner._build_tree_batch() 수정

**파일**: `ssd/engine/draft_runner.py` (`_build_tree_batch`, line 530-711)

step 4 (tree hidden states 구축)과 step 5 (fork tokens) 사이에 proxy 수신:

```python
    # === step 4 완료 ===

    # MESA: proxy 수신
    mesa_proxy = None
    if self.config.mesa_enabled:
        mesa_proxy = self._recv_mesa_proxy(B, K)

    # step 5: fork recovery tokens (proxy 반영)
    forked_rec_tokens = get_forked_recovery_tokens_from_logits(
        self.config, glue_decode_logits, cache_hits,
        gd_for_fork, tokenizer=self.tokenizer,
        mesa_proxy=mesa_proxy,
    ).view(-1)
```

### 5.2 _recv_mesa_proxy() (DraftRunner 신규 메서드)

```python
def _recv_mesa_proxy(self, B, K):
    """Target에서 보낸 MESA proxy를 수신 (단일 packed 메시지)."""
    from ssd.utils.async_helpers.nccl_pack import recv_int64
    top_k = self.config.mesa_proxy_top_k
    total_len = B * K + B * K * top_k + B * K * top_k
    buf = recv_int64(self.async_pg, src=0, total_length=total_len, device=self.device)

    off = 0
    accept_probs = buf[off:off + B*K].to(torch.int32).view(torch.float32).view(B, K)
    off += B * K
    topk_ids = buf[off:off + B*K*top_k].view(B, K, top_k)
    off += B * K * top_k
    topk_probs = buf[off:].to(torch.int32).view(torch.float32).view(B, K, top_k)

    return {"accept_probs": accept_probs, "topk_ids": topk_ids, "topk_probs": topk_probs}
```

### 5.3 get_forked_recovery_tokens_from_logits() 수정

**파일**: `ssd/utils/async_helpers/async_spec_helpers.py` (line 26-78)

`mesa_proxy` 파라미터 추가, risky position에서 residual top-k로 대체:

```python
def get_forked_recovery_tokens_from_logits(
    config, logits, cache_hits, returned_tokens, tokenizer,
    mesa_proxy=None
):
    # ... 기존 로직 (마스킹 + top-k) ...

    # MESA: risky position에서 proxy token으로 대체
    if mesa_proxy is not None:
        accept_probs = mesa_proxy["accept_probs"]
        proxy_topk_ids = mesa_proxy["topk_ids"]
        proxy_top_k = proxy_topk_ids.shape[-1]
        risky = accept_probs < config.mesa_risk_threshold

        replace_ids = topk_idx[:, :K, :].clone()
        replace_ids[:, :, :proxy_top_k] = torch.where(
            risky.unsqueeze(-1).expand_as(proxy_topk_ids),
            proxy_topk_ids,
            replace_ids[:, :, :proxy_top_k])
        topk_idx[:, :K, :] = replace_ids

    # 이하 기존 fan_out 마스킹 동일...
```

---

## 6단계: 테스트 및 벤치마크

- [ ] 완료

### 6.1 단위 테스트

1. Split forward 정합성: `forward(end_layer=E)` + `forward(start_layer=E+1, init_*)` == `forward()` 동일 출력
2. Split CudaGraph: graph_pre + graph_post == 기존 단일 graph와 동일 logits
3. Proxy 계산: accept_probs [0,1], residual top-k에 draft token 미포함
4. NCCL pack/unpack round-trip

### 6.2 벤치마크

```bash
cd bench
python -O bench.py --llama --size 8 --async --spec --k 4 --f 2 \
    --gpus 2 --b 1 --temp 0 --numseqs 128 --output_len 512 --all \
    --mesa --mesa_exit_layer 21
```

메트릭: cache hit rate, tokens/sec, acceptance rate, per-step latency

---

## 수정 파일 요약

| 파일 | 변경 내용 | 규모 |
|------|----------|------|
| `ssd/config.py` | mesa 파라미터 4개 + validation | ~20줄 |
| `ssd/models/llama3.py` | LlamaModel.forward() split 지원 + LlamaForCausalLM.forward() 파라미터 전달 | ~25줄 |
| `ssd/models/qwen3.py` | 동일 패턴 | ~25줄 |
| `ssd/engine/helpers/cudagraph_helpers.py` | capture_mesa_verify + run_mesa_verify | ~120줄 |
| `ssd/engine/model_runner.py` | mesa verify CudaGraph 캡처 + run_model 분기 + _mesa_proxy_fn | ~20줄 |
| `ssd/engine/verifier.py` | proxy_fn 설정 + _compute_and_send_proxy() | ~50줄 |
| `ssd/engine/draft_runner.py` | _recv_mesa_proxy() + _build_tree_batch 수정 | ~25줄 |
| `ssd/utils/async_helpers/async_spec_helpers.py` | mesa_proxy 기반 topk 교체 | ~15줄 |
| **총** | | **~300줄** |

---

## 구현 순서

```
1단계: Config (config.py)
    |
2단계: Split forward 지원 (llama3.py, qwen3.py)
    |
3단계: Split CudaGraph capture + replay (cudagraph_helpers.py, model_runner.py)
    |
4단계: Proxy 계산 + 전송 (verifier.py)
    |
5단계: Proxy 수신 + tree token 교체 (draft_runner.py, async_spec_helpers.py)
    |
6단계: 통합 테스트 + 벤치마크
```

---

## 이전 계획 대비 변경점

| 항목 | v2 (이전) | v3 (현재) |
|------|-----------|-----------|
| **CudaGraph** | target verify만 eager fallback | Split CudaGraph (pre/post) |
| **TP 호환** | rank 0만 callback → 데드락 위험 | 모든 TP rank가 동일 패턴 실행 |
| **proxy 전송 시점** | 기존: eager forward 중간 | CudaGraph pre 직후 (~2/3 지점) |
| **ForCausalLM 변경** | 불필요 (잘못된 판단) | split 파라미터 전달 필요 |
| **Model 변경** | callback 추가 (~3줄) | split forward 지원 (~25줄) |
| **cudagraph_helpers** | 변경 없음 | capture + run 함수 신규 (~120줄) |
| **총 코드량** | ~142줄 | ~300줄 |
| **성능** | verify ~10-20% 느림 (eager) | CudaGraph 성능 유지 |
