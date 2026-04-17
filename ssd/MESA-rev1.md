# MESA-SSD Revision 1: 최적화 계획

## 현재 상태 요약

### 동작하는 것
- Split CudaGraph (target verify pre/post): overhead < 0.5ms ✅
- Proxy 계산/전송/수신 (NCCL irecv overlap): wait = 0ms ✅
- 2-pass tree decode (draft_layout + proxy_layout): correctness 확인 ✅
- Token efficiency 개선: accept rate +8~28%, cache hit +41% (Llama2) ✅

### 핵심 문제
- **Throughput -43%** (149 → 84 tok/s, Llama3-8B 기준)
- 원인: 2-pass tree decode의 구조적 오버헤드 (2× CudaGraph replay, 2× wrapper.plan(), 2× mask precompute)
- 2-pass는 MESA의 핵심 설계(Phase 1은 proxy 대기 없이 즉시 실행)를 위해 필수

### 미구현
- MESA-SSD.md의 $\hat{h}_i$ 기반 budget allocation (Policy A/B)
- `accept_probs`, `topk_probs`를 전송하지만 draft에서 미사용

### v1 Scope 제약
- **B=1 only**: 동적 fan_out은 batch 전체가 하나의 layout을 공유하므로, B>1에서 sequence별 다른 $\hat{h}_i$를 한 layout으로 처리 불가. Rev1의 Policy A/B 동적 allocation은 **단일 시퀀스(B=1) 전용**.
- **jit_speculate=True 강제**: Config 기본값은 `jit_speculate=False`이지만, bench.py에서 `--backup jit` (기본)로 True가 됨. 직접 LLM 생성 시에는 False가 될 수 있음. `jit_speculate=False`에서 miss row의 `accept_probs=0` 강제 → $\hat{h}_0=1$ 왜곡 발생. **Rev1은 mesa_enabled 시 `assert jit_speculate` 추가로 강제.**

---

## CudaGraph와 동적 fan_out의 양립

### CudaGraph가 고정하는 것 / 고정하지 않는 것

```
CudaGraph가 캡처하는 것 (고정):
  model(input_ids, positions) → outputs    ← GPU 연산 그래프, 텐서 shape (N 고정)

CudaGraph 밖에서 매 step 실행 (동적 가능):
  wrapper.plan(cu_seqlens, kv_indptr, custom_mask, ...)  ← FlashInfer attention 설정
  graph_vars에 input_ids/positions 복사                   ← 텐서 값
  mask precompute (glue_hit_np 기반)                      ← attention mask 내용
```

**결론**: CudaGraph는 N = B × MQ_LEN (총 node 수)만 고정. **MQ_LEN(=sum(fan_out_list))만 유지하면 fan_out 분포는 매 step 바꿀 수 있다.**

```
예: proxy_layout MQ_LEN = 10 (고정), K = 4

Step N: h = [0.70, 0.03, 0.24, 0.02, 0.01]
  → fan_out = [4, 0, 3, 0, 3]  (sum = 10 ✅)
  → mask/plan을 이 fan_out으로 계산 (CudaGraph 밖)
  → CudaGraph replay (N=10 고정)

Step N+1: h = [0.10, 0.60, 0.05, 0.20, 0.05]
  → fan_out = [0, 5, 0, 3, 2]  (sum = 10 ✅)
  → 다른 mask/plan (CudaGraph 밖)
  → 같은 CudaGraph replay (N=10 고정)
```

### Target 측 $\hat{h}_i$ + Budget 배분 (draft에서 target으로 이동)

**변경**: h_i 계산과 fan_out_list 배분을 **target의 mid-forward 구간**에서 수행. Draft는 결과만 수신.

```
기존: Target sends [accept_probs, topk_ids, topk_probs] → Draft computes h_i + budget
변경: Target computes h_i + budget → sends [fan_out_list, topk_ids, topk_probs] → Draft 바로 사용
```

이점:
- Draft critical path에서 ~0.5ms 제거 (h_i 계산 + budget 배분)
- Target mid-forward 구간(graph_pre 후 ~ graph_post 전)에 ~0.01ms 추가 (무시 가능)
- accept_probs 전송 불필요 → 전송 포맷 단순화

### 필수 코드 수정: Runtime layout 전달 경로

**현재 문제**: `run_model()` (model_runner.py:682-684)이 `active_mq_len`으로 정적 `self.proxy_layout`을 선택. Runtime에서 만든 `step_proxy_layout`이 전달되지 않음.

**수정**: Context에 layout 객체 자체를 전달:

```python
# _decode_tree_step에서:
set_context(..., active_mq_len=layout.MQ_LEN,
            active_wrappers=...,
            active_layout=step_proxy_layout)  # ← layout 객체 직접 전달

# run_model에서:
_ctx = get_context()
if _ctx.active_layout is not None:
    _tree_layout = _ctx.active_layout  # ← 동적 layout 사용
    _tree_graph_key = _ctx.active_layout.graph_key
```

이렇게 하면 CudaGraph는 기존 proxy graph 재사용, mask/plan은 동적 layout 기반.

---

## 개선 항목

### 1. Glue decode 분리

**현재 문제**: `_build_tree_batch_mesa()`가 `_build_tree_batch()`를 호출하여 glue decode + full tree args를 모두 실행. MESA는 glue_logits만 필요한데 full tree args 구축(~2ms)이 낭비.

**수정**: `_build_tree_batch()` 내부에서 glue decode 부분만 별도 함수로 분리.

```python
def _glue_decode(self, partial_tree_decode_args, glue_decode_input_ids):
    """Glue decode만 수행 (no-EAGLE scope).
    
    Returns:
        glue_logits [B, K+1, V]: 각 position의 draft logits (fork token 선택용)
        gd_for_fork [B, K+1]: glue decode input tokens (returned token 마스킹용)
        cache_hits [B]: 이번 step의 cache hit 여부 (tensor)
        cache_hits_list [B]: cache_hits의 Python list 버전
        dbt [B, max_blocks]: draft block table
        pos_offset: EAGLE이면 -1, 아니면 0 (no-EAGLE scope에서는 항상 0)
    """
    # 기존 _build_tree_batch의 line 571~700 (context 준비 + model forward + logits 추출)
    return glue_logits, gd_for_fork, cache_hits, cache_hits_list, dbt, pos_offset

def _build_tree_batch(self, ...):
    """기존 SSD: glue decode + full tree args 구축"""
    glue_logits, gd_for_fork, ... = self._glue_decode(...)
    # full tree args 구축 (line 658~752)
    ...

def _build_tree_batch_mesa(self, ...):
    """MESA: glue decode만 + 2-pass"""
    glue_logits, gd_for_fork, ... = self._glue_decode(...)  # glue decode만!
    # Phase 1, Phase 2 ...
```

**효과**: ~2ms/step 절약, 코드 구조 개선.

---

### 2. tolist() GPU→CPU sync 최적화

**현재**: `_select_proxy_sourced_tokens`에서 B×K번 `.tolist()` 호출 (매번 GPU→CPU sync).

```python
# 현재:
for b in range(B):
    for pos in range(K):
        draft_set = set(draft_forked[b, pos].tolist())   # GPU→CPU 매번
        proxy_tokens = proxy_topk_ids[b, pos].tolist()   # GPU→CPU 매번

# 수정: 한 번에 CPU로
draft_cpu = draft_forked[:, :K, :].cpu().numpy()
proxy_cpu = proxy_topk_ids.cpu().numpy()
fallback_cpu = fallback_topk[:, :K, :].cpu().numpy()
for b in range(B):
    for pos in range(K):
        draft_set = set(draft_cpu[b, pos])      # CPU 내에서 (빠름)
        proxy_tokens = proxy_cpu[b, pos].tolist()
```

**효과**: GPU sync 횟수 B×K×3 → 3으로 감소.

---

### 3. Runtime layout 전달 경로 수정

**현재 문제**: `run_model()`이 `active_mq_len`으로 정적 layout을 lookup. 동적 runtime layout이 mask/plan에 반영 안 됨.

**수정**:

context.py:
```python
@dataclass
class Context:
    ...
    active_layout: object | None = None  # ← 추가: TreeLayout 객체 직접 전달

# set_context()도 수정 필요:
def set_context(..., active_mq_len=None, active_wrappers=None, active_layout=None):
    global _CONTEXT
    _CONTEXT = Context(..., active_mq_len, active_wrappers, active_layout)
```

draft_runner.py (_decode_tree_step):
```python
set_context(..., active_layout=step_proxy_layout)  # layout 객체 직접 전달
```

model_runner.py (run_model):
```python
if self.config.mesa_enabled:
    _ctx = get_context()
    if _ctx.active_layout is not None:
        _tree_layout = _ctx.active_layout
        _tree_graph_key = _tree_layout.graph_key
    else:
        _tree_graph_key = "fi_tree_decode"
        _tree_layout = None
    return run_fi_tree_decode_cudagraph(..., layout=_tree_layout)
```

cudagraph_helpers.py (run_fi_tree_decode_cudagraph):
```python
# layout의 fan_out_list로 mask 계산 (정적 config.fan_out_list 대신)
_fan_out_list = layout.fan_out_list if layout else model_runner.config.fan_out_list
```

**효과**: 동적 fan_out의 mask/plan이 실제로 CudaGraph replay에 반영됨.

---

### 4. Policy A: $\hat{h}_i$ 기반 동적 Budget Allocation

#### 동작 원리

1. Draft가 `accept_probs [B, K]`를 수신 (이미 구현됨)
2. $\hat{h}_i$ 계산 (B=1):
   ```python
   cumprod = torch.cumprod(accept_probs, dim=1)        # [1, K]
   h = torch.zeros(1, K + 1, device=accept_probs.device)
   h[0, 0] = 1 - accept_probs[0, 0]
   h[0, 1:K] = cumprod[0, :-1] * (1 - accept_probs[0, 1:])
   h[0, K] = cumprod[0, -1]  # all-accept
   ```
3. $\hat{h}_i$에 비례하여 position별 fan_out 배분:
   ```python
   total_budget = proxy_MQ_LEN  # 예: 10
   h_squeezed = h[0, :K+1]  # [K+1] (all-accept 포함)
   
   # Edge case: accept_probs ≈ 1 전부 → h[:K] ≈ 0, h[K] ≈ 1
   # → reject mass가 거의 없음 → uniform fallback
   if h_squeezed[:K].sum() < 1e-6:
       # 전부 accept 예상 → uniform fan_out으로 복원
       fan_out_list = [total_budget // (K+1)] * (K+1)
       remainder = total_budget - sum(fan_out_list)
       for i in range(remainder):
           fan_out_list[i] += 1
   else:
       raw_alloc = (h_squeezed / h_squeezed.sum() * total_budget).floor().int()
       remainder = total_budget - raw_alloc.sum().item()
       _, sorted_idx = h_squeezed.sort(descending=True)
       for i in range(int(remainder)):
           raw_alloc[sorted_idx[i]] += 1
       fan_out_list = raw_alloc.tolist()  # [K+1], sum = total_budget
   
   # all-accept position (fan_out_list[K]):
   # MESA-SSD.md에 따라 draft 분포 p^D 기반 token 배치
   # h[K]이 크면 (전부 accept 예상) all-accept에 budget 할당됨
   ```
4. **Runtime TreeLayout 생성**:
   ```python
   step_proxy_layout = create_tree_layout(
       name="proxy", fan_out_list=fan_out_list, fan_out_list_miss=fan_out_list,
       K=K, device=self.device)
   ```
5. 이 layout으로 token 선택 + tree decode args 구축
6. CudaGraph replay는 기존 proxy graph 그대로 (N = proxy_MQ_LEN 고정)

#### Token 선택: proxy-prioritized + draft fallback refill

각 position에 할당된 fan_out만큼 token을 채울 때:
- **proxy correction tokens 우선 배치** (residual topk_ids에서, dedup with draft)
- **proxy_top_k(=3)보다 fan_out이 크면**: 초과분은 **draft logits fallback**으로 채움

```
예: mesa_proxy_top_k = 3

Pos 0 (fan_out=4):
  proxy corrections: [world, today, now]  (3개)
  draft fallback: [the]                   (1개 추가, proxy/draft 미중복)
  → [world, today, now, the]

Pos 2 (fan_out=3):
  proxy corrections: [capital, city, town] (3개)
  → [capital, city, town]  (fallback 불필요)

Pos 1 (fan_out=0): 없음
```

#### 출력 순서 계약

`_build_tree_decode_args_for_layout()`이 `layout.fan_idx_hit`으로 position→node 매핑을 수행. Token selection의 출력은 **fan_idx 순서와 일치**해야 함:

```python
# fan_out_list = [4, 0, 3, 0, 3] → fan_idx = [0,0,0,0, 2,2,2, 4,4,4]
# tokens:      [pos0_tok0, pos0_tok1, pos0_tok2, pos0_tok3, pos2_tok0, pos2_tok1, pos2_tok2, pos4_tok0, pos4_tok1, pos4_tok2]
# 총 10개 = MQ_LEN
```

Fan_out=0인 position은 skip. Fan_out>0인 position은 fan_out 개수만큼 연속 배치.

**Helper 입력 계약**: `_build_tree_decode_args_for_layout()`은 `forked_tokens.view(-1)` = `[MQ_LEN]` flat 텐서를 기대 (line 1014, 1019). 이 flat 순서는 `layout.fan_idx_hit` = `[0,0,0,0, 2,2,2, 4,4,4]`의 position 순서와 정확히 일치해야 함. Token selection 함수의 출력은 이 계약을 준수하여 `[B, MQ_LEN]` flat으로 반환.

---

### 5. Policy B: $\hat{P}(i, v) = \hat{h}_i \cdot \hat{r}_i(v)$

Policy A를 확장하여 **position뿐 아니라 어떤 correction token이 유력한지**까지 반영.

1. $\hat{h}_i$ 계산 (Policy A와 동일)
2. $\hat{r}_i(v)$ = `topk_probs [B, K, top_k]` (이미 전송됨)
3. $\hat{P}(i, v) = \hat{h}_i \cdot \hat{r}_i(v)$
4. $\hat{P}(i, v)$가 큰 (position, token) 쌍부터 budget 할당:
   ```python
   P = h[0, :K].unsqueeze(-1) * topk_probs[0]  # [K, top_k]
   flat_P = P.view(-1)  # [K * top_k]
   _, top_indices = flat_P.topk(min(proxy_MQ_LEN, K * top_k))
   positions = top_indices // top_k
   token_ranks = top_indices % top_k
   # positions를 count → fan_out_list 구성
   ```
5. 선택된 (position, token) 쌍으로 fan_out_list 구성 + **해당 token만 배치**

#### Dedup + Refill 규칙 (2단계)

Policy B는 allocation과 fill을 분리:

**1단계 — Allocation (target count)**: $\hat{P}(i,v)$ 상위 → fan_out_list 결정 (position별 slot 수)

**2단계 — Fill (actual tokens)**: 각 position의 할당된 slot을 실제 token으로 채울 때:
- proxy correction tokens 우선 (dedup with Phase 1 draft root)
- 같은 token이 여러 (i,v)에서 중복 선택 → 1회만 포함
- dedup 후 실제 채워진 slot < 할당된 fan_out → **draft logits fallback refill**
- Config에서 `mesa_proxy_top_k * K >= proxy_MQ_LEN`을 권장 (후보 부족 최소화)

#### Policy A vs B 차이 (B=1 only)
```
h = [0.70, 0.03, 0.24, 0.02, 0.01]
topk_probs[pos=0] = [0.5, 0.3, 0.2]
topk_probs[pos=2] = [0.8, 0.1, 0.1]

P[pos=0] = [0.35, 0.21, 0.14]
P[pos=2] = [0.192, 0.024, 0.024]

Policy A: pos 0에 7, pos 2에 3 → pos 0의 3번째 token(P=0.14)도 포함
Policy B: 상위 10개 선택 → pos0-tok0(0.35), pos0-tok1(0.21), pos2-tok0(0.192),
          pos0-tok2(0.14), ... → pos 2의 2,3번째 token(P=0.024)은 대신
          다른 position의 더 유력한 token이 배치될 수 있음
```

---

## 구현 순서 (B=1 only, jit_speculate=True 전제)

```
Step 1: Glue decode 분리
        _glue_decode() 함수 추출
        _build_tree_batch, _build_tree_batch_mesa 모두 _glue_decode 사용
        → backward compat 테스트

Step 2: tolist() 최적화
        한 번에 CPU로 전환

Step 3: Runtime layout 전달 경로 수정
        Context에 active_layout 추가 (active_mq_len/active_wrappers는 유지, 대체 아닌 추가)
        set_context() 시그니처에 active_layout 파라미터 추가
        _decode_tree_step: 양쪽 pass(draft/proxy) 모두 해당 pass의 layout을 active_layout으로 설정
        run_model → run_fi_tree_decode_cudagraph에서 동적 layout 사용
        mask/plan이 동적 fan_out 반영 확인
        _merge_and_populate_cache에 step_proxy_layout 전달: proxy_k를 runtime layout의 fan_idx로 생성

Step 4: Policy A 구현 (h_i 기반 동적 fan_out, B=1 only)
        - config assert: mesa_enabled → jit_speculate=True 강제 (config.py __post_init__)
          bench.py는 --backup jit 기본이라 OK, direct LLM(...) construction도 커버
        - accept_probs → h_i 계산 (sum(h[:K]) < eps fallback 포함)
        - h_i → position별 proxy fan_out 배분 (sum = proxy_MQ_LEN)
        - all-accept position: h[K] > 0이면 draft 분포 기반 token 배치
        - Runtime TreeLayout 생성
        - Token 선택: proxy-prioritized + draft fallback refill
        - fan_idx 순서 계약 준수
        → 벤치마크: vs 현재 고정 fan_out

Step 5: Policy B 구현 (P(i,v) 기반)
        - h_i × topk_probs → P(i,v) 계산
        - (position, token) 단위 budget 배분
        → 벤치마크: vs Policy A

Step 6: 테스트 체크리스트 (B=1)
        - cache hit + Policy A: h_i 기반 fan_out 배분 정상 동작
        - cache miss + jit_speculate=True: miss row의 accept_probs 유효, h_i 정상
        - all-accept에 budget 몰리는 경우: h[:K] ≈ 0 → uniform fallback 정상
        - fan_out=0 position 포함: skip 없이 token flatten + fan_idx 순서 일치
        - Policy B에서 dedup 후 refill 발생: fallback으로 slot 채워지는지 확인
        - runtime proxy layout의 fan_idx로 cache key 생성 (정적 layout 아닌)

Step 7: 최종 벤치마크 (B=1)
        Baseline SSD vs MESA (고정) vs MESA Policy A vs MESA Policy B
        모델: LayerSkip-Llama3-8B + Llama-3.2-1B, LayerSkip-Llama2-7B + TinyLlama
```

---

## 예상 효과 (B=1 only)

| 항목 | 현재 (고정 fan_out) | Policy A ($\hat{h}_i$) | Policy B ($\hat{P}(i,v)$) |
|------|---------------------|------------------------|---------------------------|
| Throughput | 84 tok/s | 같거나 소폭 감소 | 같거나 소폭 감소 |
| Cache hit | 0.87 | ↑ (risky position 집중) | ↑↑ (position+token joint) |
| Accept rate | 0.83 | ↑ | ↑↑ |
| Tok/Step | 4.31 | ↑ | ↑↑ |

**Throughput**: 2-pass 구조 유지 + runtime layout 생성/h_i 계산 추가 → 동일하거나 소폭 감소 가능.
**Token efficiency 개선** → 같은 시간에 더 많은 토큰 생성 → 실질적 속도 향상.

---

## 장기적 Throughput 개선 방향

1. **Per-pass overhead 줄이기**: FlashInfer wrapper.plan() 최적화, persistent mask cache.
2. **Larger model 비율**: 70B에서는 model forward가 지배적 → 2-pass overhead 비율 감소.
3. **B>1 확장**: batch 공통 layout 근사 (h_i 평균) 또는 per-sequence layout 지원.
