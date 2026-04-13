# MESA-SSD 구현 계획서

## 0. 전체 아키텍처 — 현재 vs MESA-SSD

### 현재 Async SSD 타이밍 (한 decode step)

```
TARGET (rank 0)                          DRAFT (rank N)
─────────────────                        ──────────────────
1. speculate() 호출                       1. recv_cmd() [blocking]
   → cmd=0, cache_keys, etc 전송
                                         2. _service_spec_request()
                                            - cache lookup / JIT speculate
                                            - SEND response (tokens + logits_q)
2. response 수신 (tokens, logits_q)
3. verify() 호출                          3. _reset_tree_cache_tensors()
   - target forward (전체 L layers)       4. _build_tree_batch()
   - verify 알고리즘                         - glue decode
   - recovery token sampling                - fork recovery tokens (draft top-k)
                                         5. _decode_tree()
                                            - K sequential forward passes
                                         6. _populate_tree_cache()
4. postprocess
5. 다음 speculate() →                    7. recv_cmd() [blocking, 다음 요청 대기]
```

### MESA-SSD 타이밍

```
TARGET (rank 0)                          DRAFT (rank N)
─────────────────                        ──────────────────
1. speculate() 호출                       1. recv_cmd() [blocking]
   → cmd=0, cache_keys, etc 전송
                                         2. _service_spec_request()
                                            - cache lookup / JIT speculate
                                            - SEND response [기존과 동일]
2. response 수신
3. verify() 호출                          3. _reset_tree_cache_tensors()
   - layers [0, exit_layer] 실행          4. _build_tree_batch()
   ┌─ MESA: early-exit proxy 계산            - glue decode 실행
   │  - norm + lm_head → p_E                 ...glue decode 완료...
   │  - â_i, residual top-k 계산        ┌─── MESA: proxy 수신 [blocking recv]
   │  - NCCL SEND proxy ──────────────→ │    proxy 기반 fork token 선택
   └─ layers [exit_layer+1, L-1] 계속   └─── (기존 draft top-k 대체)
   - verify 알고리즘 [기존과 동일]        5. _decode_tree() [기존과 동일]
   - recovery sampling [기존과 동일]      6. _populate_tree_cache() [기존과 동일]
4. postprocess
5. 다음 speculate() →                    7. recv_cmd()
```

**핵심 변경점**: target의 verify forward pass 중간(~2/3 지점)에서 proxy를 추출하여 NCCL로 draft에 전송. Draft는 glue decode 후 proxy를 수신하여 tree cache의 **어떤 token을 캐시할지** 결정에 활용.

---

## 1단계: Config 확장 및 NCCL 버퍼 사전할당

- [ ] 완료

### 1.1 Config 파라미터 추가

**파일**: `ssd/config.py` (현재 ~95줄)

추가할 필드:
```python
# MESA-SSD parameters
mesa_enabled: bool = False              # MESA-SSD 활성화
mesa_exit_layer: int | None = None      # early-exit layer index (None=auto: 2*L//3)
mesa_proxy_top_k: int = 3              # residual proxy에서 보낼 top-k 개수
mesa_risk_threshold: float = 0.5       # â_i < threshold → risky position
```

`__post_init__`에 추가할 validation:
```python
if self.mesa_enabled:
    assert self.draft_async, "MESA-SSD requires draft_async=True"
    assert self.speculate, "MESA-SSD requires speculate=True"
    if self.mesa_exit_layer is None:
        L = self.hf_config.num_hidden_layers
        self.mesa_exit_layer = (2 * L) // 3  # 기본값: 2/3 지점
    assert 0 < self.mesa_exit_layer < self.hf_config.num_hidden_layers
    # fan_out보다 proxy_top_k가 크거나 같아야 token 중복 없이 대체 가능
    if self.fan_out_list is not None:
        assert self.mesa_proxy_top_k >= max(self.fan_out_list), \
            "mesa_proxy_top_k must be >= max(fan_out_list) to avoid token duplication"
```

### 1.2 Draft 측 NCCL 버퍼 사전할당

**파일**: `ssd/engine/draft_runner.py` (`_init_prealloc_buffers()`, 현재 line 112-122)

```python
if self.config.mesa_enabled:
    B_max = self.config.max_num_seqs
    K = self.config.speculate_k
    top_k = self.config.mesa_proxy_top_k
    # proxy 수신용 단일 packed 버퍼 (accept_probs + topk_ids + topk_probs)
    self._mesa_proxy_buf = torch.zeros(
        B_max * K + B_max * K * top_k + B_max * K * top_k,
        dtype=torch.int64, device=self.device)
```

> **NOTE**: target 측은 proxy를 계산 즉시 전송하므로 별도 사전할당 불필요. 계산 결과 텐서를 직접 pack하여 전송.

---

## 2단계: Target 측 — Early-Exit Proxy 추출 및 전송

- [ ] 완료

이 단계가 가장 핵심이고 변경 범위가 넓음. 3개 파일 수정.

### 2.1 LlamaModel.forward()에 early-exit hook 추가

**파일**: `ssd/models/llama3.py` (LlamaModel.forward, line 248-273)

**현재 코드**:
```python
def forward(self, input_ids, positions):
    hidden_states = self.embed_tokens(input_ids)
    residual = None
    collected_acts = [] if self.use_eagle else None
    for layer_idx, layer in enumerate(self.layers):
        if collected_acts is not None and layer_idx in self.eagle_layers:
            current_act = hidden_states if residual is None else hidden_states + residual
            collected_acts.append(current_act)
        hidden_states, residual = layer(positions, hidden_states, residual)
    hidden_states, _ = self.norm(hidden_states, residual)
    ...
```

**변경**: `mesa_exit_callback` 파라미터 추가. 튜플 `(exit_layer_idx, callable)` 형태.

```python
def forward(self, input_ids, positions, mesa_exit_callback=None):
    hidden_states = self.embed_tokens(input_ids)
    residual = None
    collected_acts = [] if self.use_eagle else None
    for layer_idx, layer in enumerate(self.layers):
        if collected_acts is not None and layer_idx in self.eagle_layers:
            current_act = hidden_states if residual is None else hidden_states + residual
            collected_acts.append(current_act)
        hidden_states, residual = layer(positions, hidden_states, residual)

        # MESA: early-exit proxy 추출
        if mesa_exit_callback is not None and layer_idx == mesa_exit_callback[0]:
            exit_hidden = hidden_states + residual if residual is not None else hidden_states
            mesa_exit_callback[1](exit_hidden)  # callback 호출

    hidden_states, _ = self.norm(hidden_states, residual)
    ...
```

**Qwen3**: `ssd/models/qwen3.py`에도 동일 패턴 적용.

### 2.2 ModelRunner.run_model()에 MESA eager verify 경로 추가

**파일**: `ssd/engine/model_runner.py` (run_model, line 595-630)

**현재 조건 분기**:
```python
if is_prefill or self.enforce_eager:
    # eager path
elif is_tree_decode:
    ...
elif is_mq_kp1 and hidden_states is not None and "glue_decode" in self.graph_vars:
    ...
elif is_mq_kp1:
    return run_verify_cudagraph(...)  # CudaGraph verify
else:
    ...
```

**변경**: `elif is_mq_kp1:` 앞에 MESA 분기 삽입

```python
    elif is_mq_kp1 and self.config.mesa_enabled and self._mesa_exit_callback is not None:
        # MESA verify: CudaGraph 대신 eager 경로 (early-exit hook 필요)
        # NOTE: _mesa_exit_callback은 Verifier.verify()에서 설정/해제됨.
        # 현재 단일 스레드 순차 실행이므로 thread-safety 이슈 없음.
        outputs = self.model(input_ids, positions,
                             mesa_exit_callback=self._mesa_exit_callback)
        logits = self.model.compute_logits(outputs, last_only)
        return logits
    elif is_mq_kp1:
        return run_verify_cudagraph(...)  # 기존 CudaGraph 경로 유지
```

`ModelRunner.__init__`에 추가:
```python
self._mesa_exit_callback = None  # MESA: Verifier가 verify 호출 전후로 설정/해제
```

### 2.3 Verifier.verify()에서 proxy 계산 및 NCCL 전송

**파일**: `ssd/engine/verifier.py` (verify, line 54-153)

> **NOTE**: Verifier에는 `self.config`이 없음. `self.target_model_runner.config`로 접근.
> async_pg도 `self.target_model_runner.async_pg`로 접근 가능 (model_runner.py:259에서 설정됨).
> draft_runner_rank도 `self.target_model_runner.draft_rank`로 접근 가능.
> 따라서 **Verifier.__init__ 시그니처 변경 불필요, llm_engine.py 수정 불필요**.

현재 흐름:
```
verify():
  1. target_model_runner.call("run", ...) → logits_p
  2. verify(logits_p, logits_q, speculations, ...) → new_suffixes, recovery_tokens
  3. return VerifyResult
```

**변경**: step 1의 forward pass 중간에 proxy를 계산하여 draft로 전송

```python
def verify(self, seqs, speculate_result, eagle=False):
    B = len(seqs)
    K = self.lookahead
    config = self.target_model_runner.config

    # MESA: early-exit callback 정의
    mesa_callback = None
    if config.mesa_enabled:
        async_pg = self.target_model_runner.async_pg
        draft_rank = self.target_model_runner.draft_rank
        draft_tokens = speculate_result.speculations[:, 1:]  # [B, K]
        logits_q = speculate_result.logits_q  # [B, K, V] — target device에 이미 존재

        def _mesa_proxy_callback(exit_hidden):
            """Target의 early-exit layer에서 호출됨.
            exit_hidden: [B*(K+1), hidden_size]"""
            self._compute_and_send_proxy(
                exit_hidden, draft_tokens, logits_q, B, K,
                async_pg, draft_rank)

        mesa_callback = (config.mesa_exit_layer, _mesa_proxy_callback)
        self.target_model_runner._mesa_exit_callback = mesa_callback

    # 기존 target forward (MESA일 때 eager 경로로 진입)
    logits_p_flat = self.target_model_runner.call("run", seqs, False, False, True)

    # callback 리셋
    if mesa_callback is not None:
        self.target_model_runner._mesa_exit_callback = None

    # 이하 기존 verify 로직 동일...
```

### 2.4 Proxy 계산 함수 (Verifier 신규 메서드)

**파일**: `ssd/engine/verifier.py`

```python
def _compute_and_send_proxy(self, exit_hidden, draft_tokens, logits_q, B, K,
                             async_pg, draft_rank):
    """Early-exit hidden states로부터 proxy를 계산하고 draft에 NCCL 전송.

    Args:
        exit_hidden: [B*(K+1), hidden_size] - early-exit layer의 출력
        draft_tokens: [B, K] - draft가 생성한 토큰들
        logits_q: [B, K, V] - draft model logits
        async_pg: NCCL process group
        draft_rank: draft GPU rank
    """
    config = self.target_model_runner.config
    model = self.target_model_runner.model  # LlamaForCausalLM
    top_k = config.mesa_proxy_top_k

    # 1) Norm + LM head -> early-exit logits
    #    RMSDNorm.forward(x, residual=None)은 norm_forward를 호출하여 단일 텐서 반환.
    #    tuple이 아니므로 unpacking하지 않음.
    normed = model.model.norm(exit_hidden, None)  # [B*(K+1), hidden]
    # NOTE: TP > 1일 때 compute_logits 내부 ParallelLMHead에서 all-reduce 1회 추가 발생.
    # B=1, K=7 기준 ~10-20us 수준이므로 Level 0에서는 허용.
    exit_logits_flat = model.compute_logits(normed, last_only=False)  # [B*(K+1), V]
    exit_logits = exit_logits_flat.view(B, K + 1, -1)  # [B, K+1, V]

    # 2) p_E, p_D 계산 (position 0..K-1만)
    p_E = torch.softmax(exit_logits[:, :K, :].float(), dim=-1)  # [B, K, V]
    p_D = torch.softmax(logits_q.float(), dim=-1)                # [B, K, V]

    # 3) Accept probability proxy: â_i = min(1, p_E(y_i) / p_D(y_i))
    gather_idx = draft_tokens.unsqueeze(-1)  # [B, K, 1]
    p_E_y = p_E.gather(2, gather_idx).squeeze(-1)  # [B, K]
    p_D_y = p_D.gather(2, gather_idx).squeeze(-1)  # [B, K]
    accept_probs = (p_E_y / (p_D_y + 1e-10)).clamp(max=1.0)  # [B, K]

    # 4) Residual proxy: r_i(v) ~ [p_E(v) - beta_i * p_D(v)]_+, v != y_i
    beta = accept_probs.unsqueeze(-1)  # [B, K, 1]
    residual = (p_E - beta * p_D).clamp(min=0)  # [B, K, V]
    residual.scatter_(2, gather_idx, 0.0)  # draft token 제외
    topk_probs, topk_ids = residual.topk(top_k, dim=-1)  # [B, K, top_k]

    # Normalize
    topk_sum = topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    topk_probs = topk_probs / topk_sum  # [B, K, top_k]

    # 5) NCCL 전송 — 단일 packed 메시지로 전송 (3회 send 대신 1회로 ~5us 절약)
    from ssd.utils.async_helpers.nccl_pack import send_int64
    send_int64(async_pg, draft_rank,
               accept_probs.view(-1).to(torch.float32).view(torch.int32).to(torch.int64),
               topk_ids.reshape(-1),
               topk_probs.view(-1).to(torch.float32).view(torch.int32).to(torch.int64))
```

**성능 분석 (B=1, K=7, H100)**:

| 연산 | 크기 | 예상 시간 |
|------|------|----------|
| RMSNorm | [8, 4096] | ~1 us |
| LM head matmul | [8, 4096] @ [4096, 128256] | ~5 us |
| TP all-reduce (tp>1) | [8, V/tp] | ~10-20 us |
| Softmax x2 | [7, 128256] | ~3 us |
| Residual + top-k | [7, 128256] | ~5 us |
| NCCL send (packed x1) | ~280 bytes | ~3 us |
| **총 오버헤드** | | **~20-40 us** |

전체 verify forward pass ~10-50ms 대비 **<0.4%** 오버헤드.

---

## 3단계: Draft 측 — Proxy 수신 및 Tree Cache 토큰 선택

- [ ] 완료

### 3.1 DraftRunner._build_tree_batch()에 proxy 수신 삽입

**파일**: `ssd/engine/draft_runner.py` (`_build_tree_batch`, line 530-711)

현재 흐름:
```
_build_tree_batch():
  1. glue decode context 준비 (line 539-615)
  2. glue decode forward (line 629-649)
  3. K+1 logits 추출 (line 652-664)
  4. tree hidden states 구축 (line 666-681)
  5. get_forked_recovery_tokens_from_logits() 호출 (line 683-696)  <-- 여기를 수정
  6. tree decode args 구축 (line 698-710)
```

**변경**: step 4와 step 5 사이에 proxy 수신 추가

```python
    # === 기존 step 4: tree hidden states 구축 완료 ===

    # MESA: proxy 수신 (target의 early-exit에서 전송한 것)
    mesa_proxy = None
    if self.config.mesa_enabled:
        mesa_proxy = self._recv_mesa_proxy(B, K)

    # step 5: fork recovery tokens (proxy 반영)
    forked_rec_tokens = get_forked_recovery_tokens_from_logits(
        self.config,
        glue_decode_logits,    # [B, K+1, V]
        cache_hits,            # [B]
        gd_for_fork,           # [B, K+1]
        tokenizer=self.tokenizer,
        mesa_proxy=mesa_proxy,  # NEW
    ).view(-1)
```

### 3.2 Proxy 수신 함수 (DraftRunner 신규 메서드)

**파일**: `ssd/engine/draft_runner.py`

```python
def _recv_mesa_proxy(self, B, K):
    """Target에서 보낸 MESA proxy를 수신 (단일 packed 메시지)."""
    from ssd.utils.async_helpers.nccl_pack import recv_int64
    top_k = self.config.mesa_proxy_top_k

    # 전체 길이: accept_probs(B*K) + topk_ids(B*K*top_k) + topk_probs(B*K*top_k)
    total_len = B * K + B * K * top_k + B * K * top_k
    buf = recv_int64(self.async_pg, src=0, total_length=total_len, device=self.device)

    off = 0
    accept_probs = buf[off:off + B * K].to(torch.int32).view(torch.float32).view(B, K)
    off += B * K
    topk_ids = buf[off:off + B * K * top_k].view(B, K, top_k)
    off += B * K * top_k
    topk_probs = buf[off:off + B * K * top_k].to(torch.int32).view(torch.float32).view(B, K, top_k)

    return {
        "accept_probs": accept_probs,    # [B, K]
        "topk_ids": topk_ids,            # [B, K, top_k]
        "topk_probs": topk_probs,        # [B, K, top_k]
    }
```

**타이밍 분석**:
- NCCL point-to-point은 내부적으로 비동기 (큐에 넣고 반환). 데이터 크기 ~280 bytes로 버퍼 오버플로 위험 없음.
- 70B target, 1B draft: draft glue decode 빨리 끝남 -> draft가 proxy 대기 (~수 ms blocking). Target이 나머지 1/3 layers 실행하는 동안 draft는 tree decode 진행. 오버랩 충분.
- 8B target, 1B draft: target이 더 빨라 proxy가 즉시 도착. 오버헤드 거의 없음.

### 3.3 get_forked_recovery_tokens_from_logits() 수정

**파일**: `ssd/utils/async_helpers/async_spec_helpers.py` (line 26-78)

**변경**: `mesa_proxy` 파라미터 추가, risky position에서 residual top-k로 대체

```python
def get_forked_recovery_tokens_from_logits(
    config, logits, cache_hits, returned_tokens, tokenizer,
    mesa_proxy=None  # NEW
):
    B = logits.shape[0]
    K = config.speculate_k

    # 기존: returned tokens 마스킹
    logits_clone = logits.clone()
    logits_clone[:, :-1, :].scatter_(2, returned_tokens[:, 1:].unsqueeze(2), float('-inf'))

    # 기존: draft logits 기반 top-k
    k_max = max(max(config.fan_out_list), max(config.fan_out_list_miss))
    _, topk_idx = torch.topk(logits_clone, k_max, dim=-1)  # [B, K+1, k_max]

    # MESA: risky position에서 residual proxy top-k로 전체 대체
    # config validation에서 mesa_proxy_top_k >= max(fan_out_list)을 보장하므로
    # proxy가 모든 fan_out 슬롯을 커버. 중복 문제 없음.
    if mesa_proxy is not None:
        accept_probs = mesa_proxy["accept_probs"]      # [B, K]
        proxy_topk_ids = mesa_proxy["topk_ids"]          # [B, K, proxy_top_k]
        proxy_top_k = proxy_topk_ids.shape[-1]
        threshold = config.mesa_risk_threshold

        risky = accept_probs < threshold  # [B, K] bool

        # Position 0..K-1: risky이면 proxy token으로 전체 대체
        replace_ids = topk_idx[:, :K, :].clone()  # [B, K, k_max]
        replace_ids[:, :, :proxy_top_k] = torch.where(
            risky.unsqueeze(-1).expand_as(proxy_topk_ids),
            proxy_topk_ids,
            replace_ids[:, :, :proxy_top_k]
        )
        topk_idx[:, :K, :] = replace_ids
        # Position K (all-accept bonus)는 변경 없이 draft top-k 유지

    # 이하 기존 fan_out 마스킹 로직 동일...
```

**설계 근거**:
- 기존 tree 구조(fan_out_list, MQ_LEN) 완전히 유지 -> position/slot_map 재계산 불필요
- risky position만 선택적으로 토큰 교체 -> minimal diff
- safe position과 all-accept bonus는 기존 draft top-k 그대로
- `mesa_proxy_top_k >= max(fan_out_list)` config validation으로 중복 문제 방지

---

## ~~4단계: Verifier에 async_pg 접근 경로 확보~~ (불필요)

> **삭제됨**: Verifier는 `self.target_model_runner.config`, `self.target_model_runner.async_pg`,
> `self.target_model_runner.draft_rank`로 모든 MESA 관련 정보에 접근 가능.
> Verifier.__init__ 시그니처 변경 불필요, llm_engine.py 수정 불필요.

---

## 4단계: 성능 최적화 경로 (3단계)

- [ ] 완료

### Level 0: 초기 구현 (target verify eager)

- `enforce_eager` 전역이 아닌, MESA verify 전용 eager 경로 사용
- `model_runner.py:run_model()`에서 `mesa_enabled and is_mq_kp1`일 때만 eager
- Draft의 tree decode는 기존 CudaGraph 그대로 유지
- 예상: target verify ~10-20% 느려지지만 cache hit rate 향상이 상쇄 가능
- TP > 1일 때 early-exit에서 LM head all-reduce 추가 1회 (~10-20us)

### Level 1: CUDA stream 오버랩 (중기)

proxy 계산을 별도 CUDA stream에서 수행:
```python
def _mesa_proxy_callback(exit_hidden):
    proxy_stream = torch.cuda.Stream()
    with torch.cuda.stream(proxy_stream):
        # norm + lm_head + proxy 계산 + NCCL send
        ...
    # main stream은 바로 다음 layer 진행
```
proxy ~20-40us가 main forward와 완전히 오버랩 -> 오버헤드 ~0.

### Level 2: Split CudaGraph (장기)

Target verify CudaGraph를 두 개로 분리:
1. `graph_pre_exit`: layers [0, exit_layer]
2. `graph_post_exit`: layers [exit_layer+1, L-1] + norm

```
graph_pre_exit.replay() -> proxy 계산 + NCCL send -> graph_post_exit.replay()
```

수정 필요:
- `capture_verify_cudagraph()` 수정하여 두 그래프 캡처
- `run_verify_cudagraph()` 수정하여 중간에 proxy 처리 삽입
- `LlamaModel.forward()`에 `start_layer`, `end_layer` 파라미터 추가

Level 0에서 효과 검증 후 진행.

---

## 5단계: 테스트 및 벤치마크

- [ ] 완료

### 5.1 단위 테스트

1. Proxy 계산 정확성: accept_probs 범위 [0,1], residual top-k에 draft token 미포함
2. Fan-out 토큰 교체: risky position의 topk_idx가 proxy로 교체되는지
3. NCCL round-trip: 2-GPU에서 proxy send/recv packed 정합성

### 5.2 End-to-end 통합 테스트

```bash
cd bench
python -O chat.py --ssd --spec --async --k 7 --f 3 --gpus 5 \
    --mesa --mesa_exit_layer 42 --metrics
```

확인: 생성 correctness, cache_hits 향상, accepted_suffix_lens 변화

### 5.3 벤치마크

```bash
cd bench
python -O bench.py --llama --size 70 --async --spec --k 7 --f 3 \
    --b 1 --temp 0 --numseqs 128 --output_len 512 --all --gpus 5 \
    --mesa --mesa_exit_layer 42
```

비교:

| 설정 | 설명 |
|------|------|
| `--async --spec` (no mesa) | 기존 SSD baseline |
| `--async --spec --mesa` | MESA-SSD |
| `--async --spec --mesa --mesa_exit_layer X` | exit layer sweep |

메트릭: cache hit rate, tokens/sec, acceptance rate (하락 없어야 함), per-step latency

---

## 수정 파일 요약

| 파일 | 변경 내용 | 규모 |
|------|----------|------|
| `ssd/config.py` | mesa 파라미터 4개 추가 + validation | ~20줄 |
| `ssd/models/llama3.py` | forward()에 mesa_exit_callback 추가 | ~5줄 |
| `ssd/models/qwen3.py` | 동일 패턴 적용 | ~5줄 |
| `ssd/engine/model_runner.py` | run_model()에 mesa eager verify 분기 + _mesa_exit_callback 초기화 | ~12줄 |
| `ssd/engine/verifier.py` | callback 설정 + _compute_and_send_proxy() | ~55줄 |
| `ssd/engine/draft_runner.py` | _recv_mesa_proxy() + build_tree 수정 + 버퍼 사전할당 | ~30줄 |
| `ssd/utils/async_helpers/async_spec_helpers.py` | mesa_proxy 기반 topk 교체 | ~15줄 |
| **총** | | **~142줄** |

> ~~`ssd/engine/llm_engine.py`~~, ~~`ssd/engine/speculator_async.py`~~ 수정 불필요.

---

## 구현 순서

```
1단계: Config (config.py)
    |
2단계: Target forward hook + eager verify + proxy 계산/전송
       (llama3.py, qwen3.py, model_runner.py, verifier.py)
    |
3단계: Proxy 수신 + tree token 교체
       (draft_runner.py, async_spec_helpers.py)
    |
4단계: 성능 최적화 (Level 0 → Level 1 → Level 2)
    |
5단계: 통합 테스트 + 벤치마크 + exit layer sweep
```
