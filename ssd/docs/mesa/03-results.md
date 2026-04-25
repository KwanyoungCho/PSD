# MESA-SSD 실험 결과 리포트

이 문서는 MESA-SSD 의 모든 실험 결과를 시간순 + 모델 크기 순으로 통합한다.
원본은 `MESA-RESULTS.md` (v1 초기 측정), `MESA-rev1-RESULTS.md` (Rev1
Policy A), `MESA-SWEEP-RESULTS.md` (parameter sweep, clean-GPU rerun) 셋이다.

34B / 70B 의 양자화 + MESA 최종 결과는 `quantization/03-final-report.md`
(quantization 카테고리) 와 각 실험 디렉토리 `tmp/final_exp*/REPORT.md` 에
별도로 정리되어 있다.

---

## Part 1. v1 초기 결과 — 8B / 7B 비교 (LayerSkip-Llama3 / Llama2)

### 환경

- Hardware: RTX 3090 × 2, k=4, f=3, B=1
- 5 seqs, temp=0.6, output_len=128

### 상세 타이밍 분석 (LayerSkip-Llama3-8B + Llama-3.2-1B)

#### Baseline SSD step

```
Draft:  [_service_spec_request ~5ms] → [_build_tree_batch ~5ms]
        → [_decode_tree(full, N=15) ~28ms]
        Total draft step: ~33ms

Target: [speculate handshake ~5ms] → [verify CudaGraph ~23ms]
        → [verify logic ~3ms]
        Total target step: ~65ms (target 이 bottleneck, draft 는 여유)
```

#### MESA 2-Pass step (실측)

```
MESA step 평균:
  irecv         =  0.1 ms
  glue+select   =  5.5 ms   (glue decode + draft token selection)
  draft_decode  = 18.8 ms   (_decode_tree with draft_layout, N=B×5, K=4)
  proxy_wait    =  0.0 ms   (proxy 이미 도착 — overlap 성공)
  proxy_select  =  0.9 ms   (dedup + proxy token selection)
  proxy_decode  = 19.1 ms   (_decode_tree with proxy_layout, N=B×10, K=4)
  merge         =  0.1 ms
  ─────────────────────
  Total         = 54.5 ms   (vs baseline 33 ms → +65%)
```

#### Target 측 타이밍

```
graph_pre.replay()      = ~15 ms  (layers 0-21)
exit_logits + send      = ~0.5 ms (norm + lm_head + NCCL send)
graph_post.replay()     = ~8 ms   (layers 22-31 + norm)
compute_logits          = ~0.5 ms
verify logic            = ~3 ms
─────────────────────
Total verify            = ~27 ms  (split CudaGraph overhead < 0.5 ms)
Total target step       = ~65 ms  (handshake + verify, baseline 과 동일)
```

### Overlap 분석

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

**핵심 관측**:

1. **proxy_wait = 0.0 ms** → irecv / send overlap 완벽. Target 이 ~15 ms
   에 send, draft 의 irecv 는 ~0 ms 에 걸림. Draft 가 draft_decode 중간
   (~24 ms) 에 proxy 도착. draft_decode 완료 (~30 ms) 까지 proxy 가 이미
   ready.
2. **Draft 가 bottleneck 처럼 보이지만 사실 아님** — target step = 65 ms,
   draft step = 54.5 ms. Draft 는 target 보다 빨리 끝남 → **draft 는 다음
   target 요청을 기다리며 idle**.
3. **실제 throughput bottleneck 은 target step (65 ms)**. Draft 가 54.5 ms
   인 건 문제가 아님 — target 이 65 ms 로 더 느리기 때문.

### 그렇다면 왜 throughput 이 하락했나?

Baseline 에서 throughput 을 결정하는 것은 `max(target_step, draft_step)`:

- Baseline: max(65, 33) = 65 ms → **target bound**
- MESA: max(65, 54.5) = 65 ms → **여전히 target bound**

**이론적으로 throughput 하락이 없어야 함**. 하지만 실측에서 하락이 있었음
(111 → 75 tok/s). 가능한 원인:

1. **첫 step 오버헤드** — 첫 step 의 glue+select 가 68.4 ms (vs 이후
   5.5 ms). torch.compile recompilation
2. **Step 간 동기화** — NCCL send / recv 가 step 경계에서 추가 sync 유발
3. **CudaGraph replay 충돌** — draft 가 3 개 다른 CudaGraph (full for glue,
   draft for pass 1, proxy for pass 2) 를 번갈아 replay 하면서 GPU
   scheduler 비효율
4. **Global cache clear** — 매 step 마다 mask precompute cache 가 MQ_LEN
   변경으로 clear → 재계산 비용

### Budget Split 분석

설정: `draft_fan_out=1, proxy_fan_out=2, async_fan_out=3`

```
N_draft = B × (1 × (K+1)) = 1 × 5 = 5 nodes  → draft_decode 18.8 ms
N_proxy = B × (2 × (K+1)) = 1 × 10 = 10 nodes → proxy_decode 19.1 ms
N_full  = B × (3 × (K+1)) = 1 × 15 = 15 nodes → baseline decode ~28 ms
```

**관측**: draft (5 nodes) + proxy (10 nodes) = 15 nodes 이지만, 각각의
decode 가 ~19 ms. Full (15 nodes) 은 ~28 ms. **2-pass 의 합 (38 ms) >
1-pass (28 ms)**. CudaGraph replay 당 고정 overhead (mask precompute,
plan() 호출 등) 가 ~9 ms 추가.

### v1 실험 결과 요약

#### LayerSkip-Llama3-8B + Llama-3.2-1B (5 seqs, temp=0.6, output_len=128)

| 메트릭 | Baseline | MESA 2-pass | 차이 |
|--------|----------|-------------|------|
| Throughput | 111.16 tok/s | 75.22 tok/s | -32.3% |
| Accept Rate | 0.74 | **0.80** | **+8.1%** |
| Cache Hit | 0.77 | **0.78** | +1.3% |
| Tok/Step | 3.98 | **4.22** | **+6.0%** |
| Tok/Step (Miss) | 1.66 | **2.48** | **+49.4%** |

#### LayerSkip-Llama2-7B + TinyLlama-1.1B (5 seqs, temp=0.6, output_len=128)

| 메트릭 | Baseline | MESA 2-pass | 차이 |
|--------|----------|-------------|------|
| Throughput | 80.74 tok/s | 61.71 tok/s | -23.6% |
| Accept Rate | 0.57 | **0.73** | **+28.1%** |
| Cache Hit | 0.56 | **0.79** | **+41.1%** |
| Tok/Step | 3.27 | **3.90** | **+19.3%** |

### v1 개선 방향 (당시 분석)

#### A. 즉시 개선 가능

1. **Mask cache 를 graph_vars 에 저장** (global → per-layout) — 매 step 의
   `cache.clear()` 제거 → mask recompute 오버헤드 ~5 ms 절약
2. **`_build_tree_batch` 리팩토링** — MESA 모드에서 full_layout
   tree_decode_args 구축 skip → ~2 ms 절약
3. **`_select_proxy_sourced_tokens` vectorize** — 현재 B×K Python loop →
   torch vectorized 로 ~0.5 ms 절약

#### B. 구조적 개선

4. **1-pass MESA** (alternative) — full_layout 단일 decode 유지하되, fork
   tokens 를 proxy 기반으로 선택. 2-pass CudaGraph replay 오버헤드 완전
   제거. Draft idle time 은 있지만 decode 오버헤드 없음
5. **Budget ratio 최적화** — `draft_fan_out=2, proxy_fan_out=1` 로 변경하면
   draft decode 가 proxy 보다 길어져 proxy 대기 시간이 0 에 더 가까워질
   수 있음. 하지만 proxy 의 cache 커버리지 감소

#### C. 장기 최적화

6. **Persistent mask cache** — Layout 별 mask 를 graph_vars 에 한 번
   precompute → step 마다 재계산 불필요
7. **Fused 2-pass CudaGraph** — Draft + proxy decode 를 하나의 CudaGraph
   로 캡처 (N=N_draft+N_proxy 로 패딩)

---

## Part 2. Rev1 Policy A 결과 (10 seqs × 256 tokens)

### Rev1 변경 사항

1. **Glue decode 분리**: `_glue_decode()` 추출 → ~2 ms / step 절약
2. **tolist() 최적화**: GPU sync B×K×3 회 → 3 회로 감소
3. **Target 측 `ĥ_i` + fan_out_list 계산**: Draft critical path 에서 제거
4. **Runtime TreeLayout**: 동적 fan_out_list 로 매 step 생성
5. **Policy A token selection**: `ĥ_i` 기반 position 별 proxy budget 배분
6. **Runtime layout 전달**: `Context.active_layout` → `run_model` →
   mask / plan 반영
7. **Cache key 수정**: `_merge_and_populate_cache` 에 runtime
   `proxy_layout` 전달

### 추가 Guard

- `assert jit_speculate` (MESA 사용 시 강제)
- `bench.py` 에서 `--mesa` → `jit_speculate=True` 자동 설정

### 환경

- GPU: RTX 3090 × 2 (비어있는 GPU)
- Settings: k=4, f=3, temp=0.6, random prompts, 10 seqs × 256 tokens =
  2560 total

### A. LayerSkip-Llama3-8B + Llama-3.2-1B

| 메트릭 | Baseline SSD | MESA v1 (고정) | MESA Rev1 (Policy A) |
|--------|-------------|----------------|---------------------|
| **Throughput** | 151.20 tok/s | 84.31 tok/s (-44%) | 84.00 tok/s (-44%) |
| **Accept Rate** | 0.87 | 0.83 | 0.79 |
| **Cache Hit** | 0.90 | 0.87 | 0.88 |
| **Tok/Step** | 4.47 | 4.31 | 4.15 |
| **Tok/Step (Hit)** | 4.72 | 4.57 | 4.42 |
| **Tok/Step (Miss)** | 2.14 | 2.56 | 2.19 |

### B. LayerSkip-Llama2-7B + TinyLlama-1.1B

| 메트릭 | Baseline SSD | MESA Rev1 (Policy A) |
|--------|-------------|---------------------|
| **Throughput** | 97.03 tok/s | 68.85 tok/s (-29%) |
| **Accept Rate** | 0.58 | 0.61 |
| **Cache Hit** | 0.61 | **0.80** (+31%) |
| **Tok/Step** | 3.31 | **3.42** (+3%) |
| **Tok/Step (Hit)** | 3.91 | 3.71 |
| **Tok/Step (Miss)** | 2.34 | 2.32 |

### Rev1 분석

#### Llama3-8B

Policy A 가 v1 (고정 fan_out) 보다 약간 떨어짐:

- Accept rate: 0.83 → 0.79 (-4.8%)
- Tok/Step: 4.31 → 4.15 (-3.7%)

**원인 추정**: `ĥ_i` 기반 budget 배분이 특정 position 에 과도하게 집중 →
다른 position coverage 감소. Llama3-8B 에서는 early-exit proxy 의 `ĥ_i`
예측이 실제 reject 패턴과 완벽히 일치하지 않을 수 있음.

#### Llama2-7B

Policy A 가 baseline 보다 cache hit 크게 개선:

- Cache hit: 0.61 → **0.80** (+31%)
- Accept rate: 0.58 → **0.61** (+5%)
- Tok/Step: 3.31 → **3.42** (+3%)

**분석**: LayerSkip-Llama2-7B 의 early-exit 이 reject 위치를 더 정확히
예측 → `ĥ_i` 기반 budget 배분 효과적.

#### Throughput

두 모델 모두 throughput 은 2-pass 구조적 비용으로 -29~44% 하락. Policy A
자체의 추가 overhead (runtime layout 생성 + `ĥ_i` 기반 token selection)
는 무시 가능 수준 (v1 의 84.31 vs Rev1 의 84.00).

### Rev1 구현 중 발견된 추가 이슈

- `_select_proxy_sourced_tokens` 에서 `mesa_proxy["accept_probs"].shape`
  참조 → `fan_out_list` 로 변경 후 제거 필요. B, K 를 `logits.shape` 과
  config 에서 직접 획득으로 수정.
- Runtime TreeLayout 의 `graph_key` 가 "fi_tree_decode_proxy" 인데,
  CudaGraph 선택은 정적 proxy graph 사용 (`MQ_LEN` 동일).
  `active_layout.graph_key` 로 정상 dispatch 확인.

---

## Part 3. Parameter Sweep (Clean-GPU Rerun, 300 seqs)

### 환경

- **Target**: LayerSkip-Llama3-8B (32 layers) [MESA / SSD baselines]
- **Target (EAGLE track)**: Llama-3.1-8B-Instruct
- **Draft (SSD/MESA)**: Llama-3.2-1B-Instruct
- **Draft (EAGLE)**: yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
- **GPUs**: 8× RTX 3090 (CUDA_VISIBLE_DEVICES isolated, TP=2 per run, 4
  runs in parallel across slots)
- **이전 run 은 GPU 1/4 의 다른 사용자 작업으로 contaminated 되었음**.
  Rerun 은 verified-empty GPU 에서 수행.
- **Prompts**: 300 random token sequences (input_len=128), temp=0.6, K=4,
  output_len=256, B=1
- **Speculation knobs**: `--async --spec --k 4`, baselines vary
  `--f ∈ {2,3,4,5}` to match MESA's total draft budget (`MQ_LEN = f·(K+1)`)

### Phase A — Baseline SSD (matched budget)

| Config | f (budget 5f) | Throughput | Accept | CacheHit | Tok/Step | Draft (ms) | Verify (ms) |
|--------|---------------|------------|--------|----------|----------|------------|-------------|
| baseline_f2 | 2 (MQ=10) | 139.57 | 0.81 | 0.82 | 4.23 | 24.52 | 22.96 |
| **baseline_f3** | **3 (MQ=15)** | **142.02** | **0.82** | 0.85 | **4.29** | 24.32 | 22.97 |
| baseline_f4 | 4 (MQ=20) | 133.78 | 0.82 | **0.86** | 4.28 | 25.60 | 22.93 |
| baseline_f5 | 5 (MQ=25) | 131.45 | 0.81 | **0.86** | 4.22 | 25.92 | 23.01 |

Baseline SSD 는 f=3 에서 peak (기존 default). 더 큰 tree 는 cache-hit 을
조금 올리지만 추가 tree-decode compute 가 이득을 상쇄. Accept rate 은
거의 flat — draft 가 이미 f=3 에서 saturation 가까움.

### Phase A — MESA (exit_layer=21 fixed, sweep f × draft_fan_out split)

총 budget = `f · (K+1)`. Split 표기 `(dfo, pfo)` = (Phase-1 per-position
branches, Phase-2 per-position branches), `dfo + pfo = f`.

| Config | f | Split (dfo, pfo) | Throughput | Accept | CacheHit | Tok/Step | Draft (ms) | Verify (ms) |
|--------|---|------------------|------------|--------|----------|----------|------------|-------------|
| mesa_f2_dfo1 | 2 | (1, 1) | 88.58 | 0.83 | 0.87 | 4.31 | 43.71 | 24.79 |
| **mesa_f3_dfo1** | **3** | **(1, 2)** | **87.54** | 0.81 | 0.88 | 4.24 | 43.40 | 24.75 |
| mesa_f3_dfo2 | 3 | (2, 1) | 85.29 | 0.79 | 0.87 | 4.15 | 43.74 | 24.64 |
| mesa_f4_dfo1 | 4 | (1, 3) | 85.29 | 0.79 | 0.88 | 4.18 | 43.93 | 24.72 |
| mesa_f4_dfo2 | 4 | (2, 2) | 85.47 | 0.79 | 0.88 | 4.18 | 43.84 | 24.77 |
| mesa_f4_dfo3 | 4 | (3, 1) | 87.10 | 0.82 | 0.88 | 4.26 | 44.02 | 24.65 |
| mesa_f5_dfo1 | 5 | (1, 4) | 82.91 | 0.80 | **0.90** | 4.21 | 45.17 | 24.89 |
| mesa_f5_dfo2 | 5 | (2, 3) | 86.76 | 0.81 | 0.89 | 4.26 | 43.99 | 24.73 |
| mesa_f5_dfo3 | 5 | (3, 2) | 86.37 | 0.81 | 0.89 | 4.24 | 44.11 | 24.75 |
| mesa_f5_dfo4 | 5 | (4, 1) | 82.19 | 0.80 | 0.88 | 4.18 | 45.93 | 25.58 |

모든 MESA 구성이 좁은 band (82–89 tok/s) 에 모임. **모든 MESA 구성이
baseline_f3 보다 35–42% 느림**, *심지어* 모든 MESA 구성이 더 높은
cache-hit rate 를 보임에도 그러함. Token-efficiency 이득은 실재하나 작음
(+0.02–0.05 CH, ≈-0.02 accept in a few). Wall-clock draft step 은 거의
*2 배* (43–46 ms vs baseline 24–26 ms). Target verify 도 ~2 ms 더 느림 →
split-CudaGraph (graph_pre + proxy + graph_post) 가 이 GPU 에서 target
시간 절약하지 못함.

### Phase B — exit_layer sweep (at f=3, dfo=1)

| Config | exit_layer (% of L=32) | Throughput | Accept | CacheHit | Tok/Step | Draft (ms) | Verify (ms) |
|--------|------------------------|------------|--------|----------|----------|------------|-------------|
| mesa_f3_dfo1_exit10 | 10 (31%) | 85.00 | 0.79 | 0.85 | 4.15 | 43.83 | 24.75 |
| **mesa_f3_dfo1_exit16** | **16 (50%)** | **88.41** | **0.81** | 0.87 | **4.26** | 43.18 | 24.71 |
| mesa_f3_dfo1 (=21) | 21 (66%) | 87.54 | 0.81 | 0.88 | 4.24 | 43.40 | 24.75 |
| mesa_f3_dfo1_exit26 | 26 (81%) | 86.54 | 0.80 | 0.89 | 4.21 | 43.64 | 24.72 |

Clean GPU 에서 exit-layer 효과는 이전 (noisy) sweep 보다 얕음. 4 점 모두
~3 tok/s 이내. Optimum 은 **exit=16 (50%)** 로 약간 이동 — 최고 accept
rate 와 tok/step. exit=21 은 최고 cache-hit rate 유지.

Split-timing 직관: 더 이른 exit → proxy 더 빨리 도착 → Phase-2 더 빨리
시작, 하지만 early-exit logits quality 약함, draft budgeting 도 덜
informative. 더 늦은 exit → proxy 강하지만 Phase-2 가 늦게 gating.

### Phase C — AR (no speculation) and EAGLE (different target)

| Config | Target | Throughput | Time (s) |
|--------|--------|-----------:|---------:|
| ar_layerskip | LayerSkip-Llama3-8B | 74.39 | 1032.38 |
| ar_llama31 | Llama-3.1-8B-Instruct | 74.01 | 1037.72 |
| eagle_f3_k4 | Llama-3.1-8B-Instruct + EAGLE-3 | **not run** | OOM at 23.4 GB/GPU even after `--max_model_len 2048` |

EAGLE 실패: `torch.compile` lowering 단계 OOM (`empty_strided_cuda((s77,
14336), torch.bfloat16)`). Rank 0 의 weight footprint (~22 GB) 가 target
이 TP-split 안 된 것을 시사 (`--eagle --gpus 2` 동시 사용 시). 재시도
전 dedicated reproducer 필요.

AR 두 8B target 은 noise 내 (74.01 vs 74.39 tok/s). LayerSkip-Llama3-8B
에서 Baseline SSD 가 AR 대비 ~1.9× (74 → 142), MESA 는 ~1.2× (74 → 87).
**MESA 는 여전히 AR 보다 빠르지만, baseline SSD 를 따라잡지 못함**.

### Comparison Summary

| Mode | Target | Best config | Throughput (tok/s) | vs AR (LayerSkip) | vs Baseline_f3 |
|------|--------|-------------|-------------------:|-----------------:|---------------:|
| AR | LayerSkip-Llama3-8B | — | 74.39 | 1.00× | 0.52× |
| AR | Llama-3.1-8B-Instruct | — | 74.01 | 0.99× | 0.52× |
| Baseline SSD | LayerSkip-Llama3-8B | f=3 | **142.02** | **1.91×** | 1.00× |
| MESA SSD | LayerSkip-Llama3-8B | f=3, dfo=1, exit=16 | 88.41 | 1.19× | **0.62×** |
| EAGLE | Llama-3.1-8B | — | *OOM (not run)* | — | — |

### Root-Cause Analysis

1. **Token efficiency 는 실재하지만 작다** — MESA 가 cache hit 을 +0.02–
   0.05, tok/step 을 좋은 config 에서 +0.1 까지 끌어올림. Proxy-driven
   selection 이 동작함을 확인. 다만 throughput 을 끌어올리기엔 부족.

2. **Draft step 이 bottleneck, 그리고 baseline 의 ~2 배** — Baseline 은
   24–26 ms, MESA 는 모든 config 에서 43–46 ms. 차이는 2-pass 구조가 지배:
   - Step 당 두 번의 CudaGraph replay (draft layout + proxy layout)
   - Step 당 두 번의 FlashInfer `wrapper.plan()` 호출 (draft + proxy
     layout)
   - Per-layout mask precompute
   - Glue compute + proxy-cache merge (Policy A 동적 fan_out_list) 가
     baseline single-pass glue 대비 ~2-4 ms 추가
   대략 ~37 ms 의 구조적 overhead 추정과 일치.

3. **Target verify 가 split CudaGraph 로 더 빨라지지 않음** — Baseline
   23 ms, MESA (graph_pre + proxy + graph_post) 24.7–25.6 ms. CudaGraph
   를 둘로 나누고 proxy tensor 로 handshake 하는 overhead 가 "Phase-2
   가 더 빨리 시작" 의 절약을 잡아먹음. Split CudaGraph 의 motivating
   benefit 이 이 모델 크기에서는 발현되지 않음.

4. **Sweep-knob 민감도 낮음** — exit=21 의 10-config f×split grid 에서
   MESA throughput 82.19–88.58 tok/s (±4%). Exit-layer sweep 도 더 flat
   (85.00–88.41). 공간이 대략 convex, optimum 이 (f=3, dfo=1, exit≈16–21)
   에 얕게. 어떤 knob choice 도 baseline 대비 -35% band 를 벗어나지 못함.

5. **이전의 "MESA beats baseline" 결과는 GPU contention artifact** — 공유
   GPU 에서 baseline target verify 가 65 ms 로 부풀려져 있었음. Clean GPU
   에서 23 ms 로 떨어지면 gap reverses.

### What this means for MESA as designed

이 모델 크기 (8 B, 32 layers) 와 hardware (RTX 3090, TP=2, B=1) 에서,
2-pass CudaGraph + dual-layout FlashInfer plan 의 구조적 overhead 가 proxy
가 제공하는 ~5–10% token-efficiency 이득을 상쇄. 이 verdict 를 뒤집으려면
아래 중 적어도 하나가 필요:

- **훨씬 저렴한 Phase-2** — runtime layout descriptor 로 keying 한 단일
  CudaGraph 로 collapse (recompile / re-plan per step 없이)
- **더 큰 target (70 B)** — proxy-driven Phase-1 → Phase-2 pipeline 이
  실제로 latency 를 hide 할 수 있음
- **Baseline 의 draft 가 saturating 에 가깝지 않은 regime** — proxy 의
  cache-hit 개선이 실제로 compounding

이 중 하나라도 성립할 때까지, baseline SSD at f=3 가 right default.

### Notes

- `mesa_f2_dfo1` 와 `mesa_f3_dfo2` 가 한 번 `DistStoreError` 로 실패했음
  (rapid slot reuse 시 multiprocessing-spawn race on port reuse). 같은
  hardware 에서 rerun 하여 위 표의 수치를 얻음. 알고리즘적 실패 아님.
- `bench/run_mesa_sweep.sh` 의 summary regex 도 "Total Throughput:" 를
  올바르게 capture 하도록 수정 (이전엔 "76800tok" = total token count 캡처).

---

## Part 4. 후속 — 34B / 70B Quantized 결과 위치

위의 8B / 7B / 32-layer 결과들은 RTX 3090 + TP=2 의 한계 안에서 측정.

이후 진행된 실험:

- **CodeLlama-34B** + Llama-3.2-1B draft (TP=4 target, TP=1 draft) — MESA
  이득이 실제로 발현되는 범위. 결과는
  `quantization/03-final-report.md` (target AWQ + draft AWQ 적용 시
  MESA 의 wall-time vs token-efficiency tradeoff 가 어떻게 바뀌는지) 에 정리.
- **layerskip-llama2-70B** + TinyLlama draft (TP=4 target, TP=1 draft) —
  AWQ 양자화 후 70B 가 single-node 에 적재되며 MESA 검증. `proxy_compute_send`
  관련 70B 특이 overhead (CUDA graph tail leak + TP gather sync) 분석은
  같은 final report 에 포함.

각 실험의 raw 결과 (`tmp/final_exp*/REPORT.md`) 는 해당 실험 디렉토리에
유지. Plot 들 (breakdown latency, timeline step) 도 동일 위치.
