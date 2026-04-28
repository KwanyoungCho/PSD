# MESA-SSD 실험 결과 리포트

이 문서는 MESA-SSD 의 모든 실험 결과를 시간순 + 모델 크기 순으로 통합한다.
원본은 `MESA-RESULTS.md` (v1 초기 측정), `MESA-rev1-RESULTS.md` (Rev1
Policy A), `MESA-SWEEP-RESULTS.md` (parameter sweep, clean-GPU rerun),
`MESA-PHASE2-HYBRID-{REPORT,FINAL-REPORT}.md` (Phase 2 hybrid 결과) 다섯
파일이다.

34B / 70B 의 양자화 + MESA 최종 결과는 `quantization/03-final-report.md`
(quantization 카테고리) 와 각 실험 디렉토리 `tmp/final_exp*/REPORT.md` 에
별도로 정리되어 있다.

> **읽는 순서**: Parts 1-3 = 초기 (8B/7B), Rev1, parameter sweep. Part 4 =
> 34B/70B quantized 결과 위치 표. Part 5 = **Phase 2 Hybrid 측정** (가장
> 최신, 현재 default 구현 기준). 70B 비교 실험은 `experiments/hybrid_vs_split_70b/`
> 에 reproducible 형태로 보존.

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

---

# Part 5. Phase 2 Hybrid 측정 결과

`MESA-PHASE2-HYBRID-REPORT.md` + `MESA-PHASE2-HYBRID-FINAL-REPORT.md` 의
실험 결과 + 추후 발견된 split 회귀 / 9D opt 결과를 통합.

## 5.1 환경 (공통)

- **Target**: `facebook/layerskip-llama2-70B` AWQ-calibrated W4A16 (TP=4)
- **Draft**: `TinyLlama-1.1B-Chat-v1.0` (TP=1)
- **Hardware**: RTX 3090 sm_86, 5 GPUs (4 target TP + 1 draft)
- **Prompts**: 200 (humaneval/alpaca/gsm8k/ultrafeedback × 50),
  output_len=256, temp=0.6, B=1, max_model_len=2048
- **Profiling**: `SSD_PROFILE_MESA=1` 로 zero-sync CUDA event 기반 per-phase
  ms/step 측정

비교 실험 위치: `experiments/hybrid_vs_split_70b/` (run.sh,
plot_3way_distinct.py, results/, results_pre_*).

## 5.2 8B 검증환경 — Step 9B 완료 시 (Phase 6)

8B target / 1B draft, 8 prompts × output_len=128:

| Mode | Total Throughput (tok/s) | vs split fallback |
|---|---:|---:|
| Split fallback (`SSD_FORCE_SPLIT_PHASE2=1`) | 86.94 | base |
| Eager hybrid (`SSD_FORCE_EAGER_HYBRID_PHASE2=1`) | 69.07 | -20.6% |
| **Native hybrid-default (CG long+short)** | **107.50** | **+23.6%** |

8/8 generation byte-identical to split fallback. Accept rate +11.3% (split
0.80 → hybrid 0.89). 8 CG family 모두 capture 확인:
`glue_long/short`, `phase1_long/short`, `phase2_hybrid_long/short`,
`verify_long/short`.

## 5.3 70B sweep — initial Phase 9B 결과 (dense draft)

50 prompts × output_len=128, dense TinyLlama draft:

| config | TPS | accept | tok/step | cache | P1 | P2 | draft_ms | verify_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| split_k5_dfo2_exit40 (baseline) | 54.10 | 0.56 | 3.81 | 0.68 | 0.56 | 0.12 | 57.39 | 45.74 |
| hybrid_k5_K1_2_K2_3_exit40 | 63.48 | 0.60 | 3.98 | 0.72 | 0.64 | 0.08 | 49.32 | 45.45 |
| **hybrid_k5_K1_3_K2_2_exit40** | **64.72** | **0.61** | **4.06** | **0.71** | **0.62** | **0.09** | **42.79** | **45.20** |
| hybrid_k6_K1_3_K2_3_exit40 | 64.29 | 0.58 | 4.50 | 0.70 | 0.60 | 0.10 | 55.84 | 46.72 |
| hybrid_k8_K1_4_K2_4_exit40 | 61.12 | 0.50 | 5.03 | 0.66 | 0.56 | 0.10 | 65.63 | 48.88 |

**최적**: K=5, K1=3, K2=2, exit_layer=40, dfo=2 (TPS 64.72).

분석:
- (K1, K2) 균형: K1 ≥ K2 가 일관되게 유리. K=5 K1=3,K2=2 (64.72) > K1=2,K2=3
  (63.48). K=8 K1=6,K2=2 (58.30) 는 accept 0.42 로 K1 너무 큰 경우 효과 떨어짐.
- 총 K depth: K=5 > K=8 (small target/draft 비율 — deep tree 추가 비용 ≥
  추가 accept 효과).
- exit_layer: 40 (= L/2) > 47 (= 7L/12) — earlier exit → target verify 빠름.
- **draft_ms 단축이 핵심**: hybrid 가 draft step 25% 단축 (cont+proxy 단일
  forward + CG capture). verify time 거의 동일. throughput 개선은 **per-step
  efficiency** 가 아니라 **draft step time 자체** 에서 옴.

## 5.4 70B head-to-head — both AWQ + 200 prompts × 256 (post-fix)

같은 prompt set / output_len 으로 (A) vs (B) vs (C) 직접 비교. Fix 적용 전후
구분 — Part 5.3 (sync fix) + Part 5.4 (per-depth label) 적용 후.

### 5.4.1 Top-line (post-sync-fix)

| | TPS | accept | cache | P1 | P2 | draft_ms | verify_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| (A) pre-hybrid + both AWQ | 66.71 | 0.52 | 0.78 | — | — | 42.29 | 47.20 |
| (B) current code + split fallback | 66.58 | 0.52 | 0.80 | 0.64 | 0.15 | 43.75 | 46.84 |
| (C) **current code + hybrid (K1=3,K2=2)** | **68.29** | 0.53 | 0.80 | 0.67 | 0.13 | **36.35** | 45.79 |

- **(C) vs (A)**: TPS **+2.4%**, draft_ms −14% (= -5.94 ms/step)
- **(C) vs (B)**: TPS **+2.6%**, draft_ms −17% (= -7.40 ms/step)
- accept rate flat (0.52~0.54): 개선은 per-step efficiency 가 아니라
  draft step time 자체에서 옴. final report §4.4 결론 재확인.

### 5.4.2 Sync fix 효과 (Part 5.3 적용 전 vs 후)

| | TPS regression (B vs A) | draft_ms regression |
|---|---:|---:|
| Sync fix 전 (dense draft) | **−4.4%** | +4.3% |
| Sync fix 후 (both AWQ) | **−0.2%** | +3.5% |

split path 의 phase1_build 회귀 +0.46 ms → −0.00 ms (사실상 닫힘).
hit_cache_respond +0.53 ms → +0.35 ms (대부분 닫힘). target_spec_wait 도
정상화 (cascade 효과 사라짐).

### 5.4.3 Per-phase breakdown (B → C, both AWQ)

draft side 의 build 와 forward 변화:

| label | (B) split | (C) hybrid | delta |
|---|---:|---:|---:|
| phase1_replay × 5 | 11.57 | — | -11.57 |
| phase2_replay × 5 | 12.12 | — | -12.12 |
| phase1_replay × 3 (hybrid) | — | 6.68 | +6.68 |
| phase2_hybrid_replay_long (× 1.74) | — | 4.78 | +4.78 |
| phase2_hybrid_replay_short (× 0.26) | — | 0.43 | +0.43 |
| phase2_hybrid_build | — | **3.36** | +3.36 |
| phase2_hybrid_prep (long+short) | — | 0.21 | +0.21 |
| tree_prep (× 3) | — | 0.55 | +0.55 |
| phase1_prep × 5 | 2.46 | — | -2.46 |
| phase2_prep × 5 | 2.35 | — | -2.35 |
| phase2_build | 1.79 | 1.34 | -0.45 |
| phase1_build | 0.15 | 0.15 | 0 |
| glue | 2.70 | 2.49 | -0.21 |
| hit_cache_respond | 2.87 | 2.55 | -0.32 |

**Net work change**: −12.5 ms 의 split forward + −5 ms 의 prep
사라짐 → +12 ms 의 hybrid forward + 4 ms hybrid build/prep 추가 =
**−1.9 ms 순감소**.

### 5.4.4 Idle 변화 (pipeline rebalancing 신호)

| | (B) split | (C) hybrid |
|---|---:|---:|
| draft_recv_cmd | 6.22 | 12.10 |
| proxy_wait | 6.90 | 12.32 |
| target_spec_wait | 14.43 (sync fix 전) / 4.29 (post-fix) | 4.02 |

draft compute 가 줄어 idle 이 +13 ms (recv_cmd + proxy_wait). target idle
은 거의 동일 (~4 ms) — pipeline 균형 변화. critical path 가 draft 에서 더
이상 안 됨을 보여줌.

### 5.4.5 Forward 횟수 비교

| | per-event | events/step | total/step |
|---|---:|---:|---:|
| split phase1_replay | 2.31 ms | 5 | 11.57 |
| split phase2_replay | 2.42 ms | 5 | 12.12 |
| **split forwards** | | **10** | **23.69** |
| hybrid phase1_replay | 2.97 ms | 3 | 8.90 |
| hybrid phase2_hybrid_replay (가중) | ~2.61 ms | 2 | 5.21 |
| **hybrid forwards** | | **5** | **14.11** |

forward 당 비용은 hybrid 가 약간 더 비쌈 (cont+proxy 통합으로 batch 가
더 큼) 하지만 **횟수가 절반** → 총 forward 시간 −9.58 ms.

이 절감이 `phase2_hybrid_build` 추가비용 (3.36 ms) 을 상쇄하고도 남음.

## 5.5 알려진 비효율 — `phase2_hybrid_build` (Part 5.5)

3.36 ms / step 이 1B draft 의 forward (2.97 ms) 보다 길다는 점이 비정상
신호. 분석된 원인 (Part 5.5):
1. `_build_hybrid_packed_mask_inplace` 가 K2 회 × ~30 kernel launch
   (~80% 차지)
2. 매 step `bit_weights`, `cont_idx_t`, `proxy_idx_t`, `kv_pos` 등 상수성
   tensor 재할당
3. bool mask `[total, L]` (~75 KB) → packed bytes (8x 작음) 변환 의 중간
   산출물 비용
4. K2 루프 3 개 (slot_maps, context_lens, kv_indptr) 별도

→ Phase 9D 최적화 (Part 5.6) 적용. 측정 결과는 9D run 완료 시 추가.

## 5.6 운영 가이드

### 환경 변수

| Env | 효과 |
|---|---|
| (none) | hybrid CG default — 권장 |
| `SSD_FORCE_EAGER_HYBRID_PHASE2=1` | eager hybrid fallback (CG 비교/디버그) |
| `SSD_FORCE_SPLIT_PHASE2=1` | split path fallback (legacy reference) |
| `SSD_HYBRID_PARITY=1` | parity harness on (overhead 큼 — 디버그 전용) |
| `SSD_HYBRID_CONT_ORACLE=1` | parity 안에서 cont oracle 동작 |
| `SSD_HYBRID_PROXY_ORACLE=1` | parity 안에서 proxy oracle 동작 |
| `SSD_HYBRID_SELF_CONSISTENCY=1` | parity 안에서 oracle/hybrid 자기-자기 비교 |
| `SSD_TRACE_BUCKET=1` | 매 step bucket (long/short) 로그 |

### 8 CG family 진단

```
[MESA] Captured draft glue_short CG (K_short+1=N)
[MESA hybrid] Captured 2-bucket verify CG ... long(K=...) + short(K=...)
About to capture FI cudagraphs for bs=[1] key=fi_tree_decode_phase1_long MQ_LEN=...
About to capture FI cudagraphs for bs=[1] key=fi_tree_decode_phase1_short MQ_LEN=...
[MESA hybrid] capturing phase2_hybrid CG bucket=long total_rows=...
[MESA hybrid] capturing phase2_hybrid CG bucket=short total_rows=...
```

위 6 라인 + draft 의 verify CG (K_long+1) = 8 family 가 모두 출력되어야
정상.

### Profiling 라벨 (post-9B+9C+9D)

draft side:
- `glue` / `draft_glue_replay` / `hit_cache_respond` / `merge_cache` —
  공통
- `phase1_build` / `phase1_replay × K1` — Phase 1
- `phase2_build` — Policy A token 선택 (split / hybrid 공통)
- `phase2_hybrid_build` — Phase 2 hybrid plan tensor 채우기 (hybrid only)
  - sub-labels: `phase2_hybrid_build_setup`, `_slots`, `_kv`, `_mask`
- `phase2_hybrid_prep_{long,short}` — KV plan + buffer copies (hybrid only,
  per-depth)
- `phase2_hybrid_replay_{long,short}` — graph.replay() (hybrid only, K2
  events/step)
- `phase2_hybrid_eager_{prep,replay}_{long,short}` — eager fallback
- `proxy_wait` — target proxy 받기 대기 (idle)
- `draft_recv_cmd` / `draft_send_response` — speculator 와 NCCL 송수신

target side:
- `verify_setup` / `verify_replay` / `verify_sample_accept` /
  `target_postprocess`
- `graph_pre` / `graph_post` (split verify CG, exit layer 분리)
- `target_spec_wait` — draft 결과 받기 대기 (idle)
- `proxy_compute_send` — early-exit 후 proxy 계산 + draft 로 송신
- `exit_logits` / `final_logits`

## 5.7 향후 작업

- **EAGLE × hybrid 통합**: `draft_acts` 가 hybrid forward 출력에서도 plumb
  필요.
- **dfo / pfo per-bucket 별도 sweep**: 현재 long/short 모두 같은 dfo, pfo
  사용. short bucket 에서 더 많은 draft 분기를 두는 변형 검토.
- **multi-batch (B > 1) 확장**: B = 1 invariant 해제 + per-seq plan 구조
  재설계 필요.
- **`phase2_hybrid_build` 추가 단축**: Phase 9D 후에도 mask 가 가장 큰 비용
  (~80%). bool mask 우회 + arithmetic 으로 packed bytes 직계산 또는
  Triton kernel 통합 고려.
