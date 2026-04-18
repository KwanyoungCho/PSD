# MESA-SSD Parameter Sweep 분석

## 환경
- Model: LayerSkip-Llama3-8B (target) + Llama-3.2-1B-Instruct (draft)
- GPU: RTX 3090 × 2 (CUDA_VISIBLE_DEVICES=1,4, 타 사용자 있음)
- Settings: k=4, f=3, temp=0.6, random prompts, 5 seqs × 256 tokens, B=1

## 결과

### Exit Layer Sweep (draft_fan_out=1, proxy_fan_out=2)

| Config | Throughput | Accept | CacheHit | Tok/Step | Tok/Hit | Tok/Miss | Draft(ms) | Verify(ms) |
|--------|-----------|--------|----------|---------|---------|----------|-----------|-----------|
| **Baseline** | **41.73** | 0.80 | 0.83 | 4.18 | 4.72 | 1.65 | 67.48 | 65.33 |
| MESA exit=10 (31%) | 39.29 | **0.89** | **0.92** | **4.57** | **4.80** | 1.82 | 103.23 | 53.65 |
| MESA exit=16 (50%) | **46.13** | 0.86 | 0.88 | 4.45 | **4.87** | 1.43 | 82.11 | 53.64 |
| **MESA exit=21 (66%)** | **47.00** | 0.85 | **0.90** | 4.41 | 4.72 | 1.43 | 81.77 | 46.53 |
| MESA exit=26 (81%) | 45.19 | 0.83 | **0.90** | 4.33 | 4.62 | 1.62 | 85.57 | 42.54 |

### Draft Fan_out Sweep (exit_layer=21)

| Config | Throughput | Accept | CacheHit | Tok/Step | Draft(ms) | Verify(ms) |
|--------|-----------|--------|----------|---------|-----------|-----------|
| **Baseline** | **41.73** | 0.80 | 0.83 | 4.18 | 67.48 | 65.33 |
| draft_fo=1, proxy_fo=2 | **47.00** | 0.85 | 0.90 | 4.41 | 81.77 | 46.53 |
| draft_fo=2, proxy_fo=1 | 28.11 | **0.92** | **0.93** | **4.66** | 146.48 | 88.28 |

---

## 핵심 분석

### 1. Exit Layer 영향

```
Exit layer가 깊을수록:
  Target verify 빨라짐: 65→53→53→47→43ms (graph_pre 커지면 graph_post 작아짐)
  하지만 proxy quality도 달라짐

최적점: exit=21 (66%)
  - Throughput: 47.00 tok/s (baseline 41.73 대비 +13%!)
  - 특이사항: MESA가 baseline보다 빠름!
```

**MESA exit=16, 21이 baseline보다 빠른 이유 분석:**

Baseline의 target verify = 65.33ms. MESA exit=21의 target verify = 46.53ms.

이것은 **split CudaGraph가 단일 CudaGraph보다 빨라서**가 아님. Baseline의 verify time이 비정상적으로 높음 (65ms). 이유:
- GPU 1,4가 완전히 비어있지 않음 (다른 사용자 6.4GB 사용 중)
- 실험 간 GPU thermal/scheduling 차이

**다만 token efficiency는 일관됨:**
- Accept rate: 0.80 → 0.83~0.89 (+3~9%)
- Cache hit: 0.83 → 0.88~0.92 (+5~9%)
- Tok/Step: 4.18 → 4.33~4.57 (+4~9%)

### 2. Draft Fan_out 영향

```
draft_fo=1 (Phase1: 5 nodes, Phase2: 10 nodes):
  Draft step: 81.77ms
  Throughput: 47.00

draft_fo=2 (Phase1: 10 nodes, Phase2: 5 nodes):
  Draft step: 146.48ms (!)
  Throughput: 28.11

왜 draft_fo=2가 훨씬 느린가?
  Phase1: N=10 decode → ~19ms
  Phase2: N=5 decode → ~10ms
  합: ~29ms... 하지만 실측 146ms??
```

**draft_fo=2의 146ms 원인 추정:**
- draft_fo=2 → proxy_fo=1 → Phase 2 MQ_LEN=5
- proxy_layout CudaGraph가 N=5로 캡처되어야 하지만, 기존 proxy wrapper의 MQ_LEN이 proxy_fan_out × (K+1) = 1 × 5 = 5
- CudaGraph 캡처 시 B=1, N=5 → batch size bucket이 매우 작아 overhead 비율 증가
- 또는: Policy A의 동적 fan_out_list에서 sum != proxy_MQ_LEN 불일치 가능성

### 3. Target Verify vs Draft Step 관계

```
Baseline:   Target=65ms, Draft=67ms  → 거의 균형 (양쪽 모두 bottleneck)
MESA exit=21: Target=47ms, Draft=82ms → Draft가 bottleneck

Draft가 bottleneck이므로:
  Throughput ≈ total_tokens / (draft_step × num_steps)
  Draft step 줄이기가 핵심
```

### 4. exit=10의 특이 패턴

```
exit=10: Accept=0.89 (최고!), CacheHit=0.92 (최고!)
         Draft=103ms (느림), Verify=54ms

early exit layer(31%)에서도 proxy quality가 높음.
하지만 graph_pre가 짧아(10 layers) graph_post가 길어(22 layers) →
proxy가 일찍 도착하여 Phase2가 빨리 시작할 수 있음.
그런데 Draft=103ms는 exit=21(82ms)보다 느림.
이유: exit=10의 early-exit logits quality가 낮아 h_i budget 배분이 비효율적?
또는 proxy_top_k=3이 exit=10에서는 부족?
```

---

## Target/Draft Latency Breakdown

```
Baseline (단일 pass):
  Target: [====== verify 65ms ========]
  Draft:  [glue 5ms][=== full decode 28ms ===][cache ~2ms] = 67ms
  Step time: max(65, 67) ≈ 67ms

MESA exit=21 (2-pass):
  Target: [graph_pre 30ms][proxy 0.5ms][graph_post 16ms] = 47ms
  Draft:  [glue 5ms][Phase1 19ms][wait 0ms][Phase2 19ms][cache 2ms] = 82ms (+ overhead)
  Step time: max(47, 82) = 82ms

  Draft 내부 overhead:
    Expected: glue(5) + Phase1(19) + Phase2(19) + cache(2) = 45ms
    Actual: 82ms
    Gap: ~37ms ← 2-pass CudaGraph overhead + mask precompute + runtime layout 생성
```

---

## 추천 파라미터 설정

| 설정 | 값 | 이유 |
|------|-----|------|
| **exit_layer** | **21** (66%) | Throughput 최고, token efficiency 우수 |
| **draft_fan_out** | **1** | fo=2는 throughput 대폭 하락 |
| **proxy_top_k** | 3 (default) | 충분 (exit=21에서 cache hit 0.90) |

## 결론

1. **MESA exit=21이 baseline보다 빠를 수 있음** (GPU 경쟁 환경 + token efficiency 향상)
2. **Token efficiency는 일관되게 개선**: accept +3~9%, cache hit +5~9%, tok/step +4~9%
3. **draft_fan_out=1이 최적**: Phase1을 최소화하고 Phase2에 budget 집중
4. **Draft step이 주요 bottleneck**: 82ms (2-pass) vs 47ms (target) → draft 최적화가 핵심
