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

---

## 핵심 발견: CudaGraph와 동적 fan_out은 양립 가능

### CudaGraph가 고정하는 것 / 고정하지 않는 것

```
CudaGraph가 캡처하는 것 (고정):
  model(input_ids, positions) → outputs    ← GPU 연산 그래프, 텐서 shape

CudaGraph 밖에서 매 step 실행 (동적 가능):
  wrapper.plan(cu_seqlens, kv_indptr, custom_mask, ...)  ← FlashInfer attention 설정
  graph_vars에 input_ids/positions 복사                   ← 텐서 값
  mask precompute (glue_hit_np 기반)                      ← attention mask 내용
```

**결론**: CudaGraph는 `N = B × MQ_LEN` (총 node 수)만 고정. Mask와 plan은 CudaGraph **밖에서** 매 step 설정되므로, **MQ_LEN만 유지하면 fan_out 분포는 매 step 바꿀 수 있다.**

```
예: proxy_layout MQ_LEN = 10 (고정), K = 4

Step N: h = [0.70, 0.03, 0.24, 0.02, 0.01]
  → fan_out = [4, 0, 3, 0, 3]  (sum = 10 = MQ_LEN ✅)
  → mask/plan을 이 fan_out으로 계산 (CudaGraph 밖)
  → CudaGraph replay (N=10 고정)

Step N+1: h = [0.10, 0.60, 0.05, 0.20, 0.05]
  → fan_out = [0, 5, 0, 3, 2]  (sum = 10 = MQ_LEN ✅)
  → 다른 mask/plan 계산 (CudaGraph 밖)
  → 같은 CudaGraph replay (N=10 고정)
```

이를 위해 필요한 것: **매 step runtime에 TreeLayout을 새로 생성**하여 `fan_idx_hit/miss`, `glue_hit_np/miss_np`를 동적으로 만들면 됨.

---

## 개선 항목

### 1. Glue decode 분리

**현재 문제**: `_build_tree_batch_mesa()`가 `_build_tree_batch()`를 호출하여 glue decode + full tree args를 모두 실행. MESA는 glue_logits만 필요한데 full tree args 구축(~2ms)이 낭비.

**수정**: `_build_tree_batch()` 내부에서 glue decode 부분만 별도 함수로 분리.

```python
def _glue_decode(self, partial_tree_decode_args, glue_decode_input_ids):
    """Glue decode만 수행. glue_logits + 메타데이터 반환."""
    # 기존 _build_tree_batch의 line 571~700 (context 준비 + model forward + logits 추출)
    return glue_logits, gd_for_fork, cache_hits, cache_hits_list, tree_hidden_states

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

### 3. Policy A: $\hat{h}_i$ 기반 동적 Budget Allocation

MESA-SSD.md의 핵심. Phase 2에서 proxy_fan_out을 position별로 **동적으로 배분**.

#### 동작 원리

1. Draft가 `accept_probs [B, K]`를 수신 (이미 구현됨)
2. $\hat{h}_i$ 계산:
   ```python
   cumprod = torch.cumprod(accept_probs, dim=1)        # [B, K]
   h = torch.zeros(B, K + 1, device=accept_probs.device)
   h[:, 0] = 1 - accept_probs[:, 0]
   h[:, 1:K] = cumprod[:, :-1] * (1 - accept_probs[:, 1:])
   h[:, K] = cumprod[:, -1]  # all-accept
   ```
3. $\hat{h}_i$에 비례하여 position별 fan_out 배분:
   ```python
   total_budget = proxy_MQ_LEN  # 예: 10
   # h에 비례하여 배분, sum = total_budget 보장
   raw_alloc = (h * total_budget).floor().int()
   remainder = total_budget - raw_alloc.sum()
   # 나머지를 h가 큰 position부터 1씩 추가
   ```
4. **Runtime TreeLayout 생성**: 매 step 동적 fan_out으로 TreeLayout 생성
   ```python
   # 예: h = [0.70, 0.03, 0.24, 0.02, 0.01]
   # → proxy_fan_out_list = [7, 0, 3, 0, 0] (sum = 10 = proxy_MQ_LEN)
   step_proxy_layout = create_tree_layout(
       name="proxy", fan_out_list=[7, 0, 3, 0, 0], ..., K=K, device=d)
   ```
5. 이 layout으로 token 선택 + tree decode args 구축 + mask 계산
6. **CudaGraph replay는 기존 proxy CudaGraph 그대로** (N=10 고정)

#### 예시

```
accept_probs = [0.3, 0.9, 0.2, 0.95]
h = [0.70, 0.03, 0.24, 0.02, 0.01]
proxy_MQ_LEN = 10

Budget 배분:
  Pos 0 (h=0.70): 7 slots → proxy correction tokens 7개
  Pos 1 (h=0.03): 0 slots → proxy 없음
  Pos 2 (h=0.24): 3 slots → proxy correction tokens 3개
  Pos 3 (h=0.02): 0 slots → proxy 없음
  Pos 4 (h=0.01): 0 slots (all-accept)

결과: 10개 proxy-sourced tokens이 risky position에 집중됨
```

#### CudaGraph 호환성

- **CudaGraph**: proxy_layout의 N=B×10으로 캡처 → 항상 N=10으로 replay. **변경 불필요.**
- **FlashInfer wrapper/plan**: CudaGraph 밖에서 매 step 실행. 동적 fan_out의 mask를 반영. **변경 필요: mask 계산에 step_proxy_layout 사용.**
- **TreeLayout**: 매 step `create_tree_layout(fan_out_list=[7,0,3,0,0])`으로 새로 생성. `fan_idx_hit`, `glue_hit_np` 등이 동적으로 변경됨.

#### 수정 필요한 코드

```python
# _build_tree_batch_mesa() Phase 2 부분:

# 기존 (고정 proxy_layout):
proxy_forked = self._select_proxy_sourced_tokens(
    glue_logits, gd_for_fork, mesa_proxy, draft_forked,
    self.config.mesa_proxy_fan_out)
proxy_tree_args = self._build_tree_decode_args_for_layout(
    partial_tree_decode_args, proxy_forked, self.proxy_layout, cache_hits_list)
proxy_tokens, proxy_logits, proxy_acts = self._decode_tree(
    proxy_tree_args, layout=self.proxy_layout)

# Policy A (동적 proxy_layout):
h = compute_h_i(mesa_proxy["accept_probs"])
step_fan_out = allocate_budget(h, self.config.mesa_proxy_fan_out * (K+1))
step_proxy_layout = create_tree_layout(
    name="proxy", fan_out_list=step_fan_out, fan_out_list_miss=step_fan_out,
    K=K, device=self.device)
proxy_forked = self._select_proxy_sourced_tokens_policy_a(
    glue_logits, gd_for_fork, mesa_proxy, draft_forked, step_fan_out)
proxy_tree_args = self._build_tree_decode_args_for_layout(
    partial_tree_decode_args, proxy_forked, step_proxy_layout, cache_hits_list)
proxy_tokens, proxy_logits, proxy_acts = self._decode_tree(
    proxy_tree_args, layout=step_proxy_layout)  # 기존 proxy CudaGraph 재사용!
```

#### `_select_proxy_sourced_tokens_policy_a` 핵심 로직

```python
def _select_proxy_sourced_tokens_policy_a(self, glue_logits, gd_for_fork,
                                            mesa_proxy, draft_forked, fan_out_list):
    """h_i 기반 동적 fan_out으로 proxy tokens 선택."""
    # fan_out_list = [7, 0, 3, 0, 0]
    # Position 0: proxy correction top-7 (dedup with draft)
    # Position 1: 없음 (fan_out=0)
    # Position 2: proxy correction top-3 (dedup with draft)
    # Position 3: 없음
    # Position 4: 없음 (all-accept)
    
    result = []
    for pos in range(K+1):
        fo = fan_out_list[pos]
        if fo == 0:
            continue  # 이 position에 slot 없음
        if pos == K:
            # all-accept: draft logits top-k
            tokens = draft_logits_topk(glue_logits[:, K, :], fo)
        else:
            # proxy correction tokens (dedup with draft)
            tokens = proxy_topk_dedup(mesa_proxy, draft_forked, pos, fo)
        result.append(tokens)
    # flatten → [B, MQ_LEN]
```

---

### 4. Policy B: $\hat{P}(i, v) = \hat{h}_i \cdot \hat{r}_i(v)$ 기반 Budget Allocation

Policy A를 확장하여 **position뿐 아니라 어떤 correction token이 유력한지**까지 반영.

#### 동작 원리

1. $\hat{h}_i$ 계산 (Policy A와 동일)
2. $\hat{r}_i(v)$ = `topk_probs [B, K, top_k]` (이미 전송됨)
3. $\hat{P}(i, v) = \hat{h}_i \cdot \hat{r}_i(v)$:
   ```python
   # h: [B, K+1], topk_probs: [B, K, top_k]
   P = h[:, :K].unsqueeze(-1) * topk_probs  # [B, K, top_k]
   ```
4. $\hat{P}(i, v)$가 큰 (position, token) 쌍부터 우선적으로 budget 할당:
   ```python
   # P를 flatten → sort → 상위 proxy_MQ_LEN개 선택
   flat_P = P.view(B, -1)  # [B, K*top_k]
   _, top_indices = flat_P.topk(proxy_MQ_LEN, dim=-1)  # [B, 10]
   # top_indices → (position, token_rank) 쌍으로 분해
   positions = top_indices // top_k
   token_ranks = top_indices % top_k
   ```
5. 선택된 (position, token) 쌍으로 fan_out_list 구성 + token 배치

#### Policy A vs B 차이

```
accept_probs = [0.3, 0.9, 0.2, 0.95]
h = [0.70, 0.03, 0.24, 0.02, 0.01]
topk_probs[pos=0] = [0.5, 0.3, 0.2]  (correction token별 확률)
topk_probs[pos=2] = [0.8, 0.1, 0.1]

P[pos=0] = 0.70 × [0.5, 0.3, 0.2] = [0.35, 0.21, 0.14]
P[pos=2] = 0.24 × [0.8, 0.1, 0.1] = [0.192, 0.024, 0.024]

Policy A (h_i만): pos 0에 7, pos 2에 3 → pos 0의 3번째 token(P=0.14)도 포함
Policy B (P(i,v)): 상위 10개 = pos0-tok0(0.35), pos0-tok1(0.21), pos2-tok0(0.192),
                    pos0-tok2(0.14), ... → pos 2의 2,3번째 token(P=0.024) 제외

→ Policy B는 pos 2의 불확실한 token 대신 다른 position의 확실한 token을 배치
```

---

## 구현 순서

```
Step 1: Glue decode 분리
        _glue_decode() 함수 추출
        _build_tree_batch, _build_tree_batch_mesa 모두 _glue_decode 사용
        → ~2ms/step 절약 + 코드 구조 개선
        → backward compat 테스트

Step 2: tolist() 최적화
        한 번에 CPU로 전환
        → GPU sync 횟수 감소

Step 3: Policy A 구현 (h_i 기반 동적 fan_out)
        - accept_probs → h_i 계산
        - h_i → position별 proxy fan_out 배분 (sum = proxy_MQ_LEN)
        - Runtime TreeLayout 생성
        - _select_proxy_sourced_tokens_policy_a() 구현
        - mask/plan에 동적 layout 반영
        → 벤치마크: vs 현재 고정 fan_out

Step 4: Policy B 구현 (P(i,v) 기반)
        - h_i × topk_probs → P(i,v) 계산
        - (position, token) 단위 budget 배분
        → 벤치마크: vs Policy A

Step 5: 최종 벤치마크
        Baseline SSD vs MESA (고정) vs MESA Policy A vs MESA Policy B
        메트릭: throughput, accept rate, cache hit rate, tok/step
        모델: LayerSkip-Llama3-8B + Llama-3.2-1B, LayerSkip-Llama2-7B + TinyLlama
```

---

## Phase 1 / Phase 2 타이밍 (변경 없음)

```
Phase 1 (proxy 대기 없이 즉시):
  Draft-sourced tokens 선택 + decode (draft_layout, N=5)
  → 모든 position에 draft top-1 배치 (기존과 동일)
  → CudaGraph: draft_layout graph 사용

Phase 2 (proxy 도착 후):
  h_i 계산 → 동적 fan_out 배분
  Runtime TreeLayout 생성 (fan_out_list = [7, 0, 3, 0, 0])
  Token 선택 + tree decode args 구축
  → CudaGraph: 기존 proxy_layout graph 재사용 (N=10 고정)
  → Mask/plan만 동적 layout 기반으로 변경

합계: Phase 1 (5개) + Phase 2 (10개) = 15개 cache entries
```

---

## 예상 효과

| 항목 | 현재 (고정 fan_out) | Policy A ($\hat{h}_i$) | Policy B ($\hat{P}(i,v)$) |
|------|---------------------|------------------------|---------------------------|
| Throughput | 84 tok/s | ~84 tok/s | ~84 tok/s |
| Cache hit | 0.87 | ↑ (risky position 집중) | ↑↑ (position+token joint) |
| Accept rate | 0.83 | ↑ | ↑↑ |
| Tok/Step | 4.31 | ↑ | ↑↑ |

**Throughput**: 2-pass 구조 유지이므로 변하지 않음. 하지만 Tok/Step 개선으로 **같은 시간에 더 많은 토큰 생성** → 실질적 속도 향상.

---

## 장기적 Throughput 개선 방향

2-pass overhead(-43%)를 줄이는 방향:

1. **Per-pass overhead 줄이기**: FlashInfer wrapper.plan() + mask precompute 최적화. Runtime TreeLayout 생성 비용 최소화.
2. **Larger model에서의 상대적 비율**: 70B 모델에서는 model forward가 지배적이라 2-pass overhead 비율이 줄어듦. 8B에서의 -43%가 70B에서는 훨씬 작을 것.
3. **Phase 1 budget 동적 조절**: Phase 1의 draft_fan_out도 이전 step의 proxy 정보로 최적화 가능 (future work).
