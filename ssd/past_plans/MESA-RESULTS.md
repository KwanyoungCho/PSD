# MESA-SSD 구현 결과 리포트 (v3 — 상세 타이밍 분석)

## 구현 상태

### 해결된 이슈
- ISSUE-001: attention/context layout-aware wrapper selection
- ISSUE-002: 2-pass budget split tree decode (draft_layout → proxy_layout)
- ISSUE-003: Layout CudaGraph 캡처 (dtype fix + wrapper 바인딩 + global cache 분리)

### 미해결
- ISSUE-004: Throughput 하락 (구조적 2-pass 오버헤드)
- ISSUE-005: Llama2-13B/70B triton 에러 (SSD 자체 이슈)

---

## 상세 타이밍 분석 (LayerSkip-Llama3-8B + Llama-3.2-1B, RTX 3090, k=4, f=3, B=1)

### Baseline SSD Step 타이밍

```
Draft:  [_service_spec_request ~5ms] → [_build_tree_batch ~5ms] → [_decode_tree(full, N=15) ~28ms]
        Total draft step: ~33ms

Target: [speculate handshake ~5ms] → [verify CudaGraph ~23ms] → [verify logic ~3ms]
        Total target step: ~65ms (target이 bottleneck, draft는 여유)
```

### MESA 2-Pass Step 타이밍 (실측)

```
MESA step 평균:
  irecv         =  0.1ms
  glue+select   =  5.5ms   (glue decode + draft token selection)
  draft_decode  = 18.8ms   (_decode_tree with draft_layout, N=B×5, K=4 steps)
  proxy_wait    =  0.0ms   (proxy 이미 도착 — overlap 성공!)
  proxy_select  =  0.9ms   (dedup + proxy token selection)
  proxy_decode  = 19.1ms   (_decode_tree with proxy_layout, N=B×10, K=4 steps)
  merge         =  0.1ms
  ─────────────────────
  Total         = 54.5ms   (vs baseline 33ms → +65%)
```

### Target 측 타이밍

```
Target step:
  graph_pre.replay()      = ~15ms  (layers 0-21)
  exit_logits + send      = ~0.5ms (norm + lm_head + NCCL send)
  graph_post.replay()     = ~8ms   (layers 22-31 + norm)
  compute_logits          = ~0.5ms
  verify logic            = ~3ms
  ─────────────────────
  Total verify            = ~27ms  (split CudaGraph overhead < 0.5ms)
  Total target step       = ~65ms  (handshake + verify, baseline과 동일)
```

---

## Overlap 분석

### 타이밍 다이어그램 (실측 기반)

```
시간(ms)  0    5    10   15   20   25   30   35   40   45   50   55   60   65
          |----|----|----|----|----|----|----|----|----|----|----|----|----|----|

TARGET:   [===== speculate handshake =====]
          [=graph_pre=15ms=][send][==graph_post==8ms==][logits][=verify=]

DRAFT:    [=service_spec_req=5ms=]
          [irecv]
          [==glue+select==5.5ms==]
          [=====draft_decode=====18.8ms=====]
                                              [proxy_wait=0ms]
                                              [sel=0.9ms]
                                              [====proxy_decode====19.1ms====]
                                                                              [merge]
```

**핵심 관측:**
1. **proxy_wait = 0.0ms** → irecv/send overlap 완벽. Target이 ~15ms에 send, draft의 irecv는 ~0ms에 걸림. Draft가 draft_decode 중간(~24ms)에 proxy 도착. draft_decode 완료(~30ms)까지 proxy가 이미 ready.

2. **Draft가 bottleneck** (54.5ms > target 65ms? No — target step = 65ms, draft step = 54.5ms. Draft는 target보다 빨리 끝남 → **draft는 다음 target 요청을 기다리며 idle!**)

3. **실제 throughput bottleneck은 target step (65ms)**. Draft가 54.5ms인건 문제가 아님 — target이 65ms로 더 느리기 때문.

### 그렇다면 왜 throughput이 하락했나?

Baseline에서 throughput을 결정하는 것은 `max(target_step, draft_step)`:
- Baseline: max(65ms, 33ms) = 65ms → **target bound**
- MESA: max(65ms, 54.5ms) = 65ms → **여전히 target bound**

**이론적으로 throughput 하락이 없어야 합니다!** 하지만 실측에서 하락이 있었습니다 (111 → 75 tok/s). 가능한 원인:

1. **첫 step 오버헤드**: 첫 step의 glue+select가 68.4ms (vs 이후 5.5ms). torch.compile recompilation.
2. **Step 간 동기화**: NCCL send/recv가 step 경계에서 추가 sync를 유발할 수 있음.
3. **CudaGraph replay 충돌**: draft가 3개 다른 CudaGraph (full for glue, draft for pass 1, proxy for pass 2)를 번갈아 replay하면서 GPU scheduler 비효율.
4. **Global cache clear**: 매 step마다 mask precompute cache가 MQ_LEN 변경으로 clear → 재계산 비용.

### Budget Split 분석

현재 설정: `draft_fan_out=1, proxy_fan_out=2, async_fan_out=3`

```
N_draft = B × (1 × (K+1)) = 1 × 5 = 5 nodes  → draft_decode 18.8ms
N_proxy = B × (2 × (K+1)) = 1 × 10 = 10 nodes → proxy_decode 19.1ms
N_full  = B × (3 × (K+1)) = 1 × 15 = 15 nodes → baseline decode ~28ms
```

**관측**: draft(5 nodes) + proxy(10 nodes) = 15 nodes이지만, 각각의 decode가 ~19ms. Full(15 nodes)은 ~28ms. **2-pass의 합(38ms) > 1-pass(28ms)!** CudaGraph replay당 고정 오버헤드(mask precompute, plan() 호출 등)가 ~9ms 추가.

---

## 개선 방향

### A. 즉시 개선 가능

1. **Mask cache를 graph_vars에 저장** (global → per-layout): 매 step의 cache.clear() 제거 → mask recompute 오버헤드 ~5ms 절약
2. **`_build_tree_batch` 리팩토링**: MESA 모드에서 full_layout tree_decode_args 구축 skip → ~2ms 절약
3. **`_select_proxy_sourced_tokens` vectorize**: 현재 B×K Python loop → torch vectorized로 ~0.5ms 절약

### B. 구조적 개선

4. **1-pass MESA** (alternative approach): full_layout 단일 decode를 유지하되, fork tokens를 proxy 기반으로 선택. 2-pass의 CudaGraph replay 오버헤드를 완전히 제거. Draft idle time은 있지만 decode 오버헤드 없음. (이전 v1 proxy token swap과 유사하지만, cache key 설계 개선 필요)

5. **Budget ratio 최적화**: `draft_fan_out=2, proxy_fan_out=1`로 변경하면 draft decode가 proxy보다 길어져 proxy 대기 시간이 0에 더 가까워질 수 있음. 하지만 proxy의 cache 커버리지 감소.

### C. 장기 최적화

6. **Persistent mask cache**: Layout별 mask를 graph_vars에 한 번 precompute → step마다 재계산 불필요
7. **Fused 2-pass CudaGraph**: Draft + proxy decode를 하나의 CudaGraph로 캡처 (N=N_draft+N_proxy로 패딩)

---

## 실험 결과 요약

### LayerSkip-Llama3-8B + Llama-3.2-1B (5 seqs, temp=0.6, output_len=128)

| 메트릭 | Baseline | MESA 2-pass | 차이 |
|--------|----------|-------------|------|
| Throughput | 111.16 tok/s | 75.22 tok/s | -32.3% |
| Accept Rate | 0.74 | **0.80** | **+8.1%** |
| Cache Hit | 0.77 | **0.78** | +1.3% |
| Tok/Step | 3.98 | **4.22** | **+6.0%** |
| Tok/Step (Miss) | 1.66 | **2.48** | **+49.4%** |

### LayerSkip-Llama2-7B + TinyLlama-1.1B (5 seqs, temp=0.6, output_len=128)

| 메트릭 | Baseline | MESA 2-pass | 차이 |
|--------|----------|-------------|------|
| Throughput | 80.74 tok/s | 61.71 tok/s | -23.6% |
| Accept Rate | 0.57 | **0.73** | **+28.1%** |
| Cache Hit | 0.56 | **0.79** | **+41.1%** |
| Tok/Step | 3.27 | **3.90** | **+19.3%** |

---

## 변경된 파일

| 파일 | 변경 |
|------|------|
| `ssd/config.py` | MESA params + validation |
| `ssd/models/llama3.py` | Split forward |
| `ssd/engine/model_runner.py` | Split CudaGraph + layout wrappers + layout-aware dispatch + tokenizer fallback |
| `ssd/engine/verifier.py` | Proxy fn + _compute_and_send_proxy |
| `ssd/engine/draft_runner.py` | TreeLayout + 2-pass decode + irecv + token selection + merge cache + timing |
| `ssd/engine/helpers/tree_layout.py` | **신규** TreeLayout dataclass |
| `ssd/engine/helpers/cudagraph_helpers.py` | mesa_verify + layout fi_tree_decode + cache invalidation + dtype fix |
| `ssd/layers/attention.py` | Layout-aware wrapper selection |
| `ssd/utils/context.py` | active_mq_len/active_wrappers |
| `ssd/utils/async_helpers/async_spec_helpers.py` | mesa_proxy token swap |
| `bench/bench.py` | --mesa args + --model_path/--draft_path |
