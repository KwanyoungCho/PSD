# MESA-SSD 구현 계획서 (v5 — Budget Split + Split CudaGraph + TP-safe)

## 0. 설계 원칙

1. **CudaGraph 성능 유지**: Verify CudaGraph를 pre/post로 분리.
2. **TP > 1 호환**: 모든 TP rank가 동일한 NCCL collective 패턴 실행.
3. **Mid-forward proxy 전송**: target verify의 ~2/3 지점에서 proxy를 draft에 전송.
4. **Budget split**: draft-sourced branches는 EE proxy 대기 없이 즉시 decode. Proxy-sourced branches는 proxy 도착 후 decode. Draft idle time 제거.
5. **Token dedup**: proxy-first union — proxy-sourced 우선, draft-sourced로 refill, 중복 1회만.
6. **Llama only**: Qwen3, EAGLE 미지원.

---

## 0.1 전체 타이밍 (Budget Split + Split CudaGraph)

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
   │  layers [0 .. exit_layer]               - glue decode → draft logits
   └ → exit_buffer                           - draft-sourced fork tokens 선택
                                             - _decode_tree(draft branches) ← 즉시 시작!
   [CudaGraph 밖, ALL TP ranks]:                    ↕ (병렬 실행)
   norm(exit_buffer) → lm_head           ┌── proxy 도착 (draft decode 도중)
   → exit_logits (TP gather)             │
   rank 0: proxy 계산 + NCCL SEND ─────→ │   draft decode 완료 후:
                                         │   proxy-sourced fork tokens 선택 (dedup)
   ┌ graph_post.replay()                 │   _decode_tree(proxy branches)
   │  layers [exit_layer+1 .. L-1]       └──
   │  + final norm                        5. _populate_tree_cache(draft + proxy 합침)
   └ → outputs
   lm_head(outputs) → final_logits
   verify 알고리즘 [기존과 동일]
4. postprocess
5. 다음 speculate() →                    6. recv_cmd()
```

**핵심**: Draft는 proxy를 기다리지 않고 draft-sourced branches를 **즉시** decode 시작.
Proxy 도착 후 proxy-sourced branches를 추가 decode. Idle time 없음.

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
    if self.mesa_draft_fan_out is None:
        self.mesa_draft_fan_out = max(1, self.async_fan_out // 2)
    assert self.mesa_draft_fan_out < self.async_fan_out, \
        "mesa_draft_fan_out must be < async_fan_out (remainder goes to proxy)"
    self.mesa_proxy_fan_out = self.async_fan_out - self.mesa_draft_fan_out
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

(v4와 동일 — graph_pre + graph_post 분리 캡처. 생략.)

### 3.2 run_mesa_verify_cudagraph()

(v4와 동일 — graph_pre.replay() → mid-forward proxy → graph_post.replay(). 생략.)

### 3.3 model_runner.py 변경

(v4와 동일 — mesa_enabled + not is_draft일 때 split graph 캡처/실행. 생략.)

---

## 4단계: Verifier에서 proxy 계산 및 전송

- [ ] 완료

### 4.1 Verifier.verify() 변경

(v4와 동일 — _proxy_fn 설정 → run → 해제. 생략.)

### 4.2 _compute_and_send_proxy()

(v4와 동일 — accept_probs + residual topk + cache miss 처리 + send_int64. 생략.)

---

## 5단계: Draft 측 — Budget Split + Proxy 수신 + 2단계 Tree Decode

- [ ] 완료

**이 단계가 v4 대비 가장 크게 변경됨.**

### 5.1 _build_tree_batch() 재설계

**파일**: `ssd/engine/draft_runner.py` (`_build_tree_batch`, line 530-711)

기존 흐름:
```
glue decode → fork tokens (전부) → tree decode args → return
```

MESA 흐름:
```
glue decode → draft-sourced fork tokens → draft tree decode args
           → _decode_tree(draft branches)  ← 즉시 시작, proxy 대기 없음
           → proxy 수신 (non-blocking check 또는 decode 완료 후 blocking recv)
           → proxy-sourced fork tokens (dedup with draft-sourced)
           → proxy tree decode args
           → _decode_tree(proxy branches)
           → 결과 합쳐서 return
```

```python
def _build_tree_batch(self, partial_tree_decode_args, glue_decode_input_ids):
    # ... 기존 glue decode (step 1-4) ...

    if not self.config.mesa_enabled:
        # 기존 경로: 한번에 fork token 선택 + return
        forked_rec_tokens = get_forked_recovery_tokens_from_logits(
            self.config, glue_decode_logits, cache_hits,
            gd_for_fork, tokenizer=self.tokenizer).view(-1)
        # ... tree decode args 구축 + return ...
    else:
        # === MESA: Budget Split ===
        
        # Step A: draft-sourced fork tokens (proxy 대기 없이 즉시)
        draft_fan_out = self.config.mesa_draft_fan_out  # e.g., 1
        draft_forked = self._select_draft_sourced_tokens(
            glue_decode_logits, cache_hits, gd_for_fork, draft_fan_out)
        # draft_forked: [B, K+1, draft_fan_out] — position K는 all-accept (draft top-k)

        # Step B: draft branches tree decode (즉시 시작!)
        draft_tree_args = self._build_partial_tree_args(
            partial_tree_decode_args, draft_forked, draft_fan_out, ...)
        draft_tokens, draft_logits, draft_acts = self._decode_tree(draft_tree_args)

        # Step C: proxy 수신 (draft decode 중 또는 완료 후)
        mesa_proxy = self._recv_mesa_proxy(B, K)

        # Step D: proxy-sourced fork tokens (draft-sourced와 dedup)
        proxy_fan_out = self.config.mesa_proxy_fan_out  # e.g., 2
        proxy_forked = self._select_proxy_sourced_tokens(
            glue_decode_logits, cache_hits, gd_for_fork,
            mesa_proxy, draft_forked, proxy_fan_out)

        # Step E: proxy branches tree decode
        proxy_tree_args = self._build_partial_tree_args(
            partial_tree_decode_args, proxy_forked, proxy_fan_out, ...)
        proxy_tokens, proxy_logits, proxy_acts = self._decode_tree(proxy_tree_args)

        # Step F: 결과 합침 → tree decode args로 return
        # (draft_tokens + proxy_tokens를 하나의 cache로 populate)
```

### 5.2 _select_draft_sourced_tokens()

```python
def _select_draft_sourced_tokens(self, logits, cache_hits, returned_tokens, draft_fan_out):
    """Draft logits 기반으로 fork tokens 선택 (기존 top-k 로직, fan_out=draft_fan_out)."""
    # 기존 get_forked_recovery_tokens_from_logits와 동일하되
    # fan_out을 draft_fan_out으로 제한
    logits = logits.clone()
    logits[:, :-1, :] = logits[:, :-1, :].scatter(
        dim=2, index=returned_tokens[:, 1:].unsqueeze(2), value=float('-inf'))
    _, topk_idx = torch.topk(logits, draft_fan_out, dim=-1)  # [B, K+1, draft_fan_out]
    return topk_idx
```

### 5.3 _select_proxy_sourced_tokens()

```python
def _select_proxy_sourced_tokens(self, logits, cache_hits, returned_tokens,
                                   mesa_proxy, draft_forked, proxy_fan_out):
    """Proxy 기반 correction tokens 선택. Draft-sourced와 proxy-first union dedup."""
    B, K = mesa_proxy["accept_probs"].shape
    proxy_topk_ids = mesa_proxy["topk_ids"]  # [B, K, proxy_top_k]

    result = torch.zeros(B, K + 1, proxy_fan_out, dtype=torch.int64, device=logits.device)

    # Position 0..K-1: proxy-first union dedup
    for b in range(B):
        for pos in range(K):
            proxy_tokens = proxy_topk_ids[b, pos].tolist()
            draft_tokens = draft_forked[b, pos].tolist()

            # proxy 우선, draft로 refill, 중복 1회만
            seen = set()
            merged = []
            for t in proxy_tokens:
                if t not in seen:
                    merged.append(t)
                    seen.add(t)
            for t in draft_tokens:
                if t not in seen:
                    merged.append(t)
                    seen.add(t)

            for j in range(min(len(merged), proxy_fan_out)):
                result[b, pos, j] = merged[j]

    # Position K (all-accept): draft logits top-k
    logits_clone = logits.clone()
    _, all_accept_topk = torch.topk(logits_clone[:, K, :], proxy_fan_out, dim=-1)
    result[:, K, :] = all_accept_topk

    return result
```

> **NOTE**: 위 loop는 개념 설명용. 실제 구현 시 vectorized 버전으로 최적화.

### 5.4 _recv_mesa_proxy()

(v4와 동일)

### 5.5 _populate_tree_cache() 수정

Draft branches와 proxy branches 결과를 하나의 tree cache에 통합:

```python
# draft_tokens [N1, K], proxy_tokens [N2, K] → combined [N1+N2, K]
combined_tokens = torch.cat([draft_tokens, proxy_tokens], dim=0)
combined_logits = torch.cat([draft_logits, proxy_logits], dim=0)
combined_keys = torch.cat([draft_keys, proxy_keys], dim=0)
self._populate_tree_cache(combined_payload, combined_tokens, combined_logits, ...)
```

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
parser.add_argument("--mesa_draft_fan_out", type=int, default=None,
                    help="Draft-sourced branches per position (default: fan_out//2)")
```

**create_llm_kwargs()에 추가**:
```python
if args.mesa:
    llm_kwargs["mesa_enabled"] = True
    if args.mesa_exit_layer is not None:
        llm_kwargs["mesa_exit_layer"] = args.mesa_exit_layer
    llm_kwargs["mesa_proxy_top_k"] = args.mesa_proxy_top_k
    if args.mesa_draft_fan_out is not None:
        llm_kwargs["mesa_draft_fan_out"] = args.mesa_draft_fan_out
```

### 6.2 단위 테스트

1. Split forward 정합성: `forward(end_layer=E)` + `forward(start_layer=E+1, init_*)` == `forward()`
2. Split CudaGraph: graph_pre + graph_post == 기존 단일 graph
3. Proxy 계산: accept_probs ∈ [0,1], residual top-k에 draft token y_i 미포함
4. Token dedup: proxy-first union이 중복 없이 올바른 순서 유지
5. Budget split: draft_fan_out + proxy_fan_out == async_fan_out
6. NCCL pack/unpack round-trip
7. Feature gating: Qwen3 + mesa → assert, EAGLE + mesa → assert

### 6.3 Edge case 테스트

- TP > 1 (4 GPU TP + 1 draft)
- cache miss + jit_speculate=False / True
- temp > 0 (non-greedy sampling)
- B > 1 (batch size)
- proxy_top_k < proxy_fan_out (dedup으로 부족분 발생)

### 6.4 벤치마크

```bash
cd bench

# Baseline (기존 SSD)
python -O bench.py --llama --size 8 --async --spec --k 4 --f 3 \
    --gpus 2 --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# MESA-SSD (default split: draft 1, proxy 2)
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

메트릭: cache hit rate, tokens/sec, acceptance rate, per-step latency

---

## 수정 파일 요약

| 파일 | 변경 내용 | 규모 |
|------|----------|------|
| `ssd/config.py` | mesa 파라미터 + budget split + validation + gating | ~30줄 |
| `ssd/models/llama3.py` | LlamaModel split forward + ForCausalLM 전달 | ~25줄 |
| `ssd/engine/helpers/cudagraph_helpers.py` | capture_mesa_verify + run_mesa_verify | ~120줄 |
| `ssd/engine/model_runner.py` | mesa CudaGraph 캡처/실행 분기 | ~25줄 |
| `ssd/engine/verifier.py` | proxy_fn + _compute_and_send_proxy() | ~55줄 |
| `ssd/engine/draft_runner.py` | budget split tree decode + proxy recv + token selection | ~80줄 |
| `ssd/utils/async_helpers/async_spec_helpers.py` | (기존 함수는 non-MESA 경로 유지) | ~0줄 |
| `bench/bench.py` | --mesa 관련 arguments | ~15줄 |
| **총** | | **~375줄** |

---

## 구현 순서

```
1단계: Config + feature gating (config.py)
    |
2단계: Split forward (llama3.py)
    |
3단계: Split CudaGraph capture + replay (cudagraph_helpers.py, model_runner.py)
    |
4단계: Proxy 계산 + send 전송 (verifier.py)
    |
5단계: Budget split tree decode + proxy recv + dedup (draft_runner.py)
    |
6단계: bench.py arguments + 테스트 + 벤치마크
```

---

## v4 → v5 변경점

| 항목 | v4 | v5 |
|------|----|----|
| **Tree decode** | proxy 대기 후 1회 decode | **Budget split: draft 즉시 decode + proxy 도착 후 추가 decode** |
| **Draft idle time** | ~10ms (proxy 대기) | **0ms** |
| **Fan_out 분할** | 없음 (전부 proxy 교체) | mesa_draft_fan_out + mesa_proxy_fan_out = async_fan_out |
| **Token dedup** | 대칭차집합 (교집합 소실) | **Proxy-first union** (중복 1회만, 순서 보존) |
| **_build_tree_batch** | 단일 fork token 선택 | draft/proxy 각각 선택 + 2회 _decode_tree |
| **isend 잔존** | 있음 | **제거** (send_int64 통일) |
