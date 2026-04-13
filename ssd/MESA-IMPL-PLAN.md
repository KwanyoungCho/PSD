# MESA-SSD 구현 계획서 (v4 — Split CudaGraph + TP-safe)

## 0. 설계 원칙

1. **CudaGraph 성능 유지**: eager 모드 fallback 없음. Verify CudaGraph를 pre/post로 분리.
2. **TP > 1 호환**: 모든 TP rank가 동일한 NCCL collective 패턴을 실행.
3. **Mid-forward proxy 전송**: target verify의 ~2/3 지점에서 proxy를 draft에 전송하여 tree decode와 오버랩.
4. **Token dedup**: draft-sourced tokens(draft logits 기반)과 proxy-sourced tokens(EE proxy 기반)이 중복되지 않도록 관리.
5. **Llama only**: Qwen3, EAGLE 미지원.

---

## 0.1 전체 타이밍 (Split CudaGraph)

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
   ┌ graph_pre.replay()                   4. _build_tree_batch()
   │  layers [0 .. exit_layer]               - glue decode → draft-sourced tokens
   └ → exit_buffer에 hidden states 기록       ...glue decode 완료...
                                         ┌─── MESA: proxy 수신 [blocking recv]
   [CudaGraph 밖, ALL TP ranks]:         │
   norm(exit_buffer) → lm_head           │    proxy-sourced correction tokens 추출
   → exit_logits (TP gather 포함)        │    (draft-sourced와 dedup 후 배치)
   rank 0: proxy 계산 + NCCL SEND ─────→ │    fork token 최종 결정
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

---

## 1단계: Config 확장

- [ ] 완료

**파일**: `ssd/config.py`

추가할 필드 (line 45 이후, debugging 섹션 뒤):
```python
# MESA-SSD parameters
mesa_enabled: bool = False
mesa_exit_layer: int | None = None      # None=auto: 2*L//3
mesa_proxy_top_k: int = 3              # proxy에서 전송할 correction token 수
# mesa_budget_mode는 token_swap 검증 후 추가 예정 (h_redistribute, outcome_posterior)
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
    # proxy_top_k는 fan_out보다 크게 잡는 것을 권장 (dedup 후 부족분 방지)
    # 부족분은 draft top-k에서 refill되므로 hard failure는 아님
```

---

## 2단계: LlamaModel에 split forward 지원 추가

- [ ] 완료

### 2.1 LlamaModel 변경

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

**start_layer=0, end_layer=None (기존 호출)**: 동작 완전 동일.

### 2.2 LlamaForCausalLM.forward() 변경

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

## 3단계: Split CudaGraph Capture & Replay

- [ ] 완료

### 3.1 capture_mesa_verify_cudagraph()

**파일**: `ssd/engine/helpers/cudagraph_helpers.py`

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

    # Split 전용 중간 버퍼
    exit_hidden = torch.zeros(max_bs * k_plus_1, H, dtype=hf_config.torch_dtype)
    exit_residual = torch.zeros(max_bs * k_plus_1, H, dtype=hf_config.torch_dtype)
    outputs = torch.zeros(max_bs * k_plus_1, H, dtype=hf_config.torch_dtype)

    # batch size 버킷
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

        # --- graph_post: layers [exit_layer+1, L-1] + norm ---
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
    """Split CudaGraph verify: pre → proxy → post → logits."""
    context = get_context()
    config = model_runner.config
    k_plus_1 = config.speculate_k + 1
    orig_bs = input_ids.size(0) // k_plus_1

    wrapper_bs = next(
        x for x in model_runner.graph_bs_list["mesa_verify"] if x >= orig_bs)
    graph_pre = model_runner.graphs["mesa_verify_pre"][wrapper_bs]
    graph_post = model_runner.graphs["mesa_verify_post"][wrapper_bs]

    # padding 로직 (run_verify_cudagraph과 동일 — 생략)
    # ... input_ids, positions, slot_mapping, context_lens, block_tables ...
    bs = wrapper_bs
    # graph_vars에 입력 복사 ...

    # ====== graph_pre.replay() ======
    graph_pre.replay()

    # ====== Mid-forward: proxy 계산 + 전송 (CudaGraph 밖, ALL TP ranks) ======
    flat = orig_bs * k_plus_1
    exit_h = graph_vars["exit_hidden"][:flat] + graph_vars["exit_residual"][:flat]
    normed = model_runner.model.model.norm(exit_h, None)
    exit_logits = model_runner.model.compute_logits(normed, last_only=False)  # TP gather

    if mesa_proxy_fn is not None:  # rank 0만
        mesa_proxy_fn(exit_logits, orig_bs)

    # ====== graph_post.replay() ======
    graph_post.replay()

    # ====== Final logits ======
    outputs = graph_vars["outputs"][:flat]
    logits = model_runner.model.compute_logits(outputs, last_only)
    return logits
```

### 3.3 model_runner.py 변경

**파일**: `ssd/engine/model_runner.py`

#### setup_and_warmup_model_and_cudagraphs() (line 278-300)

```python
if not self.enforce_eager:
    # 기존 decode 캡처 (항상)...

    # verify CG: MESA이면 split graph, 아니면 기존 단일 graph
    if self.config.speculate and not (self.is_draft and self.config.use_eagle):
        if self.config.mesa_enabled and not self.is_draft:
            mesa_gv, mesa_pool, mesa_pre, mesa_post, mesa_bs = \
                capture_mesa_verify_cudagraph(self)
            self.graph_vars["mesa_verify"] = mesa_gv
            self.graph_pools["mesa_verify"] = mesa_pool
            self.graphs["mesa_verify_pre"] = mesa_pre
            self.graphs["mesa_verify_post"] = mesa_post
            self.graph_bs_list["mesa_verify"] = mesa_bs
        else:
            # 기존 단일 verify CudaGraph (MESA 비활성 또는 draft)
            verify_graph_vars, verify_graph_pool, verify_graphs, verify_graph_bs_list = \
                capture_verify_cudagraph(self)
            self.graph_vars["verify"] = verify_graph_vars
            self.graph_pools["verify"] = verify_graph_pool
            self.graphs["verify"] = verify_graphs
            self.graph_bs_list["verify"] = verify_graph_bs_list
```

#### run_model() (line 595-630)

```python
    # MESA verify: target만 진입 (not self.is_draft → draft glue decode 보호)
    elif is_mq_kp1 and self.config.mesa_enabled and not self.is_draft \
            and "mesa_verify" in self.graph_vars:
        return run_mesa_verify_cudagraph(
            self, input_ids, positions, last_only,
            self.graph_vars["mesa_verify"],
            mesa_proxy_fn=self._mesa_proxy_fn)
    elif is_mq_kp1:
        return run_verify_cudagraph(...)  # 기존 경로 (draft glue decode 포함)
```

`__init__`에 추가:
```python
self._mesa_proxy_fn = None
```

---

## 4단계: Verifier에서 proxy 계산 및 전송

- [ ] 완료

### 4.1 Verifier.verify() 변경

**파일**: `ssd/engine/verifier.py`

```python
def verify(self, seqs, speculate_result, eagle=False):
    B = len(seqs)
    K = self.lookahead
    config = self.target_model_runner.config

    if config.mesa_enabled:
        async_pg = self.target_model_runner.async_pg
        draft_rank = self.target_model_runner.draft_rank
        draft_tokens = speculate_result.speculations[:, 1:]  # [B, K]
        logits_q = speculate_result.logits_q                 # [B, K, V]
        cache_hits = speculate_result.cache_hits              # [B] or None

        def _proxy_fn(exit_logits, orig_bs):
            self._compute_and_send_proxy(
                exit_logits, draft_tokens, logits_q, orig_bs, K,
                async_pg, draft_rank, cache_hits=cache_hits)

        self.target_model_runner._mesa_proxy_fn = _proxy_fn

    result = self.target_model_runner.call("run", seqs, False, False, True)

    if config.mesa_enabled:
        self.target_model_runner._mesa_proxy_fn = None

    # 이하 기존 verify 로직 동일...
```

### 4.2 _compute_and_send_proxy()

```python
def _compute_and_send_proxy(self, exit_logits, draft_tokens, logits_q,
                             B, K, async_pg, draft_rank, cache_hits=None):
    """exit_logits에서 proxy 계산 → draft에 NCCL 전송.

    전송 내용:
      accept_probs [B, K]: position별 accept probability proxy
      topk_ids [B, K, top_k]: correction token 후보 (residual [p_E - p_D]+ 기반)
      topk_probs [B, K, top_k]: 해당 확률 (향후 budget allocation용)
    """
    config = self.target_model_runner.config
    top_k = config.mesa_proxy_top_k

    if exit_logits.dim() == 2:
        exit_logits = exit_logits.view(B, K + 1, -1)

    # p_E (early-exit proxy), p_D (draft)
    p_E = torch.softmax(exit_logits[:, :K, :].float(), dim=-1)  # [B, K, V]
    p_D = torch.softmax(logits_q.float(), dim=-1)                # [B, K, V]

    # Accept probability proxy: â_i = min(1, p_E(y_i) / p_D(y_i))
    gather_idx = draft_tokens.unsqueeze(-1)  # [B, K, 1]
    p_E_y = p_E.gather(2, gather_idx).squeeze(-1)
    p_D_y = p_D.gather(2, gather_idx).squeeze(-1)
    accept_probs = (p_E_y / (p_D_y + 1e-10)).clamp(max=1.0)  # [B, K]

    # Residual proxy: [p_E - p_D]_+ (verifier의 [p-q]_+와 동일 형태)
    residual = (p_E - p_D).clamp(min=0)  # [B, K, V]
    residual.scatter_(2, gather_idx, 0.0)  # draft token y_i 제외
    topk_probs, topk_ids = residual.topk(top_k, dim=-1)  # [B, K, top_k]
    topk_sum = topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    topk_probs = topk_probs / topk_sum

    # cache miss + jit_speculate=False: miss row는 p_E 단독 사용
    if cache_hits is not None and not config.jit_speculate:
        miss_mask = ~cache_hits.to(torch.bool)
        if miss_mask.any():
            accept_probs[miss_mask] = 0.0
            # miss row: p_E의 top-k 사용 (p_D 무관), draft token y_i 제외
            miss_p_E = p_E[miss_mask].clone()
            miss_p_E.scatter_(2, gather_idx[miss_mask], 0.0)  # reject 토큰 제외
            miss_topk_probs, miss_topk_ids = miss_p_E.topk(top_k, dim=-1)
            topk_ids[miss_mask] = miss_topk_ids
            topk_probs[miss_mask] = miss_topk_probs / miss_topk_probs.sum(-1, keepdim=True).clamp(min=1e-10)

    # NCCL 전송 (단일 packed 메시지, blocking send — 280 bytes ~3μs로 overhead 무시 가능)
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

```python
    # === step 4 (tree hidden states 구축) 완료 ===

    # MESA: proxy 수신
    mesa_proxy = None
    if self.config.mesa_enabled:
        mesa_proxy = self._recv_mesa_proxy(B, K)

    # step 5: fork recovery tokens
    forked_rec_tokens = get_forked_recovery_tokens_from_logits(
        self.config, glue_decode_logits, cache_hits,
        gd_for_fork, tokenizer=self.tokenizer,
        mesa_proxy=mesa_proxy,
    ).view(-1)
```

### 5.2 _recv_mesa_proxy()

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

**핵심 변경**: 모든 position에서 proxy-sourced correction tokens 사용. Draft-sourced tokens와 중복 방지 (dedup). Position K(all-accept)는 draft top-k 유지.

```python
def get_forked_recovery_tokens_from_logits(
    config, logits, cache_hits, returned_tokens, tokenizer,
    mesa_proxy=None
):
    B, _, V_actual = logits.shape
    K = config.speculate_k
    fan_out_list = config.fan_out_list
    fan_out_list_miss = config.fan_out_list_miss

    # 기존: clone + returned tokens 마스킹
    logits = logits.clone()
    logits[:, :-1, :] = logits[:, :-1, :].scatter(
        dim=2, index=returned_tokens[:, 1:].unsqueeze(2), value=float('-inf'))

    k_max = max(max(fan_out_list), max(fan_out_list_miss))
    _, topk_idx = torch.topk(logits, k_max, dim=-1)  # [B, K+1, k_max]

    # MESA: position 0..K-1에 proxy-sourced correction tokens 적용
    # 슬롯 배치: [proxy-sourced (deduped) | draft-sourced (refill)]
    # proxy-sourced가 우선 배치, 부족분은 draft-sourced에서 refill
    if mesa_proxy is not None:
        proxy_topk_ids = mesa_proxy["topk_ids"]       # [B, K, proxy_top_k]
        proxy_top_k = proxy_topk_ids.shape[-1]
        draft_sourced = topk_idx[:, :K, :]            # [B, K, k_max] — draft logits 기반

        for b in range(B):
            for pos in range(K):
                draft_list = draft_sourced[b, pos].tolist()
                draft_set = set(draft_list)
                proxy_tokens = proxy_topk_ids[b, pos].tolist()

                # proxy-sourced: draft-sourced와 중복되지 않는 것만
                deduped_proxy = [t for t in proxy_tokens if t not in draft_set]

                # draft-sourced: proxy-sourced와 중복되지 않는 것만 (refill용)
                proxy_set = set(proxy_tokens)
                deduped_draft = [t for t in draft_list if t not in proxy_set]

                # 최종 슬롯: proxy 우선 + draft refill (총 k_max개)
                merged = deduped_proxy + deduped_draft
                for j in range(min(len(merged), k_max)):
                    topk_idx[b, pos, j] = merged[j]

        # Position K (all-accept bonus): draft-sourced top-k 유지 (변경 없음)

    # 이하 기존 fan_out 마스킹 로직 (hit_counts, miss_counts, mask, masked_select)...
    # ...
```

> **NOTE**: 위 loop는 개념 설명용. 실제 구현 시 vectorized 버전으로 최적화.
> **NOTE**: `topk_probs`는 현재 `token_swap` 모드에서 미사용. `h_redistribute`/`outcome_posterior` 모드에서 budget 배분에 활용 예정.

---

## 6단계: bench.py arguments 및 테스트

- [ ] 완료

### 6.1 bench.py argument 추가

**파일**: `bench/bench.py` (parse_arguments 함수)

```python
# MESA-SSD configuration
parser.add_argument("--mesa", action="store_true", help="Enable MESA-SSD")
parser.add_argument("--mesa_exit_layer", type=int, default=None,
                    help="Early-exit layer index (default: 2*L//3)")
parser.add_argument("--mesa_proxy_top_k", type=int, default=3,
                    help="Number of correction tokens per position")
# mesa_budget_mode는 token_swap 검증 후 추가 예정
```

**create_llm_kwargs()에 추가**:
```python
if args.mesa:
    llm_kwargs["mesa_enabled"] = True
    if args.mesa_exit_layer is not None:
        llm_kwargs["mesa_exit_layer"] = args.mesa_exit_layer
    llm_kwargs["mesa_proxy_top_k"] = args.mesa_proxy_top_k
    # mesa_budget_mode는 token_swap 검증 후 추가 예정
```

### 6.2 단위 테스트

1. Split forward 정합성: `forward(end_layer=E)` + `forward(start_layer=E+1, init_*)` == `forward()`
2. Split CudaGraph: graph_pre + graph_post 결과 == 기존 단일 graph
3. Proxy 계산: accept_probs ∈ [0,1], residual top-k에 draft token 미포함
4. Token dedup: proxy-sourced ∩ draft-sourced == ∅
5. NCCL pack/unpack round-trip
6. Feature gating: Qwen3 + mesa → assert, EAGLE + mesa → assert

### 6.3 Edge case 테스트

- TP > 1 (4 GPU TP + 1 draft)
- cache miss + jit_speculate=False
- cache miss + jit_speculate=True
- temp > 0 (non-greedy sampling)
- B > 1 (batch size)

### 6.4 벤치마크

```bash
cd bench

# Baseline (기존 SSD)
python -O bench.py --llama --size 8 --async --spec --k 4 --f 2 \
    --gpus 2 --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# MESA-SSD
python -O bench.py --llama --size 8 --async --spec --k 4 --f 2 \
    --gpus 2 --b 1 --temp 0 --numseqs 128 --output_len 512 --all \
    --mesa --mesa_exit_layer 21

# Exit layer sweep
for EL in 10 16 21 26; do
    python -O bench.py ... --mesa --mesa_exit_layer $EL
done
```

메트릭: cache hit rate, tokens/sec, acceptance rate (하락 없어야 함), per-step latency

---

## 수정 파일 요약

| 파일 | 변경 내용 | 규모 |
|------|----------|------|
| `ssd/config.py` | mesa 파라미터 4개 + validation + feature gating | ~25줄 |
| `ssd/models/llama3.py` | LlamaModel split forward + LlamaForCausalLM 파라미터 전달 | ~25줄 |
| `ssd/engine/helpers/cudagraph_helpers.py` | capture_mesa_verify + run_mesa_verify | ~120줄 |
| `ssd/engine/model_runner.py` | mesa CudaGraph 캡처 분기 + run_model 분기 | ~25줄 |
| `ssd/engine/verifier.py` | proxy_fn 설정 + _compute_and_send_proxy() | ~55줄 |
| `ssd/engine/draft_runner.py` | _recv_mesa_proxy() + _build_tree_batch 수정 | ~25줄 |
| `ssd/utils/async_helpers/async_spec_helpers.py` | draft/proxy dedup + proxy 토큰 적용 | ~25줄 |
| `bench/bench.py` | --mesa 관련 arguments + llm_kwargs 전달 | ~15줄 |
| **총** | | **~315줄** |

---

## 구현 순서

```
1단계: Config + feature gating (config.py)
    |
2단계: Split forward (llama3.py)
    |
3단계: Split CudaGraph capture + replay (cudagraph_helpers.py, model_runner.py)
    |
4단계: Proxy 계산 + isend 전송 (verifier.py)
    |
5단계: Proxy 수신 + draft/proxy dedup + tree token 교체 (draft_runner.py, async_spec_helpers.py)
    |
6단계: bench.py arguments + 테스트 + 벤치마크
```

---

## v3 → v4 변경점

| 항목 | v3 | v4 |
|------|----|----|
| **risk_threshold** | 있음 (binary 분류) | **제거** — 모든 position에서 proxy 사용 |
| **Token dedup** | 없음 (중복 가능) | draft-sourced와 proxy-sourced 간 중복 제거 |
| **all-accept branch** | 미정 | draft top-k 유지 (EE proxy 사용 안 함) |
| **budget_mode** | 없음 | `token_swap` (현재) / `h_redistribute` / `outcome_posterior` (향후) |
| **feature gating** | Qwen3 미지원 명시만 | assert: Llama only, no EAGLE |
| **NCCL send** | dist.send (blocking) | dist.send 유지 (280B ~3μs, isend의 GC/stream 위험 불필요) |
| **bench.py** | 미반영 | --mesa, --mesa_exit_layer 등 추가 |
| **fan_out_list_miss** | 검증 누락 | 검증 추가 |
| **테스트 케이스** | 기본만 | TP>1, cache miss, temp>0 등 edge case 추가 |
