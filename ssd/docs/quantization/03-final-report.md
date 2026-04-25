# 양자화 최종 결과 리포트

이 문서는 양자화 작업의 최종 실험 결과를 통합한다. v1 (torchao INT4) 의
checkpoint 결과와 v2 (AWQ Marlin) 의 1B / 8B / 34B / 70B + draft AWQ
실험 결과를 모두 한 곳에 모았다.

---

## 1. v1 — torchao INT4 (legacy)

### 1.1 환경 / 결정

- 브랜치: `feature/int8-weight-only` (명칭은 history; 실제 채택 backend 는 INT4)
- 환경: torch 2.8.0+cu128, torchao 0.12.0, RTX 3090 (SM 86), `ssd` env 그대로

| 항목 | 초기 계획 | 최종 |
|---|---|---|
| Backend | `torchao.Int8WeightOnlyConfig` | `Int4WeightOnlyConfig(group_size=128)` (SM 86 tinygemm fast path) |
| fp16 모델 처리 | plain load | load-time bf16 upcast (Llama-2/CodeLlama) |
| MESA lm_head | quantize | bf16 유지 권장 (`--no_quant_lm_head`) |
| AWQ/SmoothQuant | 검토됨 | 철회 (fp16 overflow 는 outlier 문제 아님) |

### 1.2 Phase 결과

| Phase | 상태 | 핵심 결과 |
|---|---|---|
| 0. feasibility + graph-safety | ✅ | storage contract (A), CUDA graph OK, `inference_mode` OK, tying 방어 |
| 1. weight replacement contract | ✅ | `self.weight = dummy.weight` 재할당, forward 미변경 |
| 2. plain INT8 eager 통합 | ✅ | code OK, fp16 모델에선 overflow 발견 → bf16 upcast |
| 2.5 kernel path 최적화 | ✅ | INT8 → INT4 전환. INT4 tile_packed 가 SM 86 에서 dense 대비 0.25-1.25×, INT8 대비 2.7× 빠름 |
| 3. SSD graph path 확장 | ✅ | AR/spec/MESA graph 모두 INT4 호환, graph 는 eager 대비 2-4× |
| 4. MESA + lm_head ablation | ✅ | MESA 에서 lm_head bf16 유지 시 accept 0.41→0.38 (7% 손실) |
| 5. persistent artifact | ✅ | save/load AQT per rank — Llama-3-8B smoke test |
| 6. 34B 확장 | ✅ | CodeLlama-34B (fp16 → bf16 upcast) TP=4 INT4 동작, async spec 23.52 TP |

### 1.3 v1 실측 (async spec + sampling, temp=0.6, TP=2 or 4 + draft)

**Llama-3-8B (bf16 native)**

| config | TP | accept | 비고 |
|---|---|---|---|
| dense | 15.84 | 0.32 | baseline |
| INT8 wo | 14.25 | 0.30 | 90% of dense |
| **INT4 tile_packed** | **18.64** | **0.30** | **dense 대비 +18%** |
| MESA dense | 24.12 | 0.41 | baseline |
| MESA INT8 | 12.15 | 0.40 | |
| MESA INT4 + no_lm_head | 13.47 | 0.38 | accept 보존 |

**Llama-2-7B (fp16 → bf16 upcast)**

| config | TP | accept |
|---|---|---|
| INT8 wo | 15.31 | 0.44 |
| **INT4 tile_packed** | **41.85** | **0.35** |

**CodeLlama-34B (fp16 → bf16 upcast, TP=4 + draft, 5 GPU)** — 50 seq × 256
out × 4 dataset = 51200 tok:

| config | TP | accept | cache_hit | wall |
|---|---|---|---|---|
| pre-quant dense async spec | 68.45 | 0.44 | - | 747.98 s |
| **INT4 tile_packed async spec** | **75.28** | **0.44** | 0.66 | 680.09 s |

**INT4 가 dense 대비 +10% 빠르고 accept 완벽 동일.**

| config | 비고 |
|---|---|
| INT4 MESA (34B) | FAIL `QuantizedLinearNotImplementedError` — MESA-specific graph shape dispatch 이슈 |

**Persistent artifact (Llama-3-8B, TP=2)**:

| step | wall time | TP |
|---|---|---|
| 1. quantize + save | 63 s | 35.49 |
| 2. load (load_time mode) | 45 s | 40.71 |
| 3. load_only (persistent mode) | 44 s | 42.08 |

저장 overhead 없이 양자화 시간만큼 startup 단축. 70B 반복 실험에서 큰 이득
예상.

**Graph vs Eager (Llama-3-8B INT4)**:

| path | graph | eager |
|---|---|---|
| AR | 36.43 | 8.95 (4× slower) |
| spec | 14.51 | 8.01 (1.8×) |
| MESA | 30.62 | OOM-killed |

### 1.4 v1 미해결 / 후속

- **34B MESA INT4 dispatch 실패**: `QuantizedLinearNotImplementedError`.
  MESA verify graph 가 특정 shape 사용하는데 torchao tile_packed tinygemm
  dispatch 안 됨. async spec 은 동작. 우회: 34B MESA 에 INT8 (느리지만 OK)
- **70B**: 로컬 모델 없음 (~140GB 다운로드 필요). INT4 로 TP=2 per-rank
  8.75GB 예상 → 3 GPU 가능
- **INT4 accept 손실**: async spec 거의 0, MESA 7%. AWQ calibration 추가로
  더 개선 가능하나 현 단계 불필요
- **artifact CPU→GPU warning**: torchao 업스트림 이슈

---

## 2. v2 — AWQ W4A16 (Marlin)

### 2.1 Executive summary

Plan v2 의 9개 phase 모두 구현 + end-to-end 검증:

- AR decode, sync spec, async spec, CUDA-graph capture, TP=1, TP=2,
  **MESA split-verify** 모두 `layerskip-llama3-8B` Marlin W4A16 target 위에서
  정상 동작
- 8B TP=2 decode throughput: **74 tok/s dense → 147 tok/s AWQ (1.99×)**.
  KV cache block capacity 도 1.31× 증가 (398 → 519) — weight footprint 가
  ≈16 GB bf16 → ≈3.6 GB packed
- AWQ 환경 MESA accept rate + cache hit rate 가 dense MESA 와 같음 (accept
  0.43, cache-hit 0.67 on 8B smoke). plan §11 의 "default dense `lm_head`
  하 accept rate 급락 없음" 충족
- dense-matmul-on-dequantized-weight 기준 round-trip 수치 오차: fp16 ≈ 5×10⁻⁴,
  bf16 ≈ 4×10⁻³

계획 이탈 없음. 기존 torchao int4/int8 경로는 bf16 fallback 으로 tree 유지
(plan §12.3).

### 2.2 Backend 선택 (Phase 0)

**Chosen**: `sgl_kernel.gptq_marlin_gemm(b_q_type=scalar_types.uint4,
is_zp_float=False)`. AWQ 입력 텐서를 load time 에 `awq_marlin_repack` +
column-permutation helper (`ssd/quant/marlin_utils.py`, vLLM 포팅) 로 Marlin
layout 으로 repack.

Plan §5 모든 gate 통과 (RTX 3090 sm_86):

| Gate | 결과 |
|---|---|
| fp16 activation | ✅ |
| bf16 activation | ✅ |
| Decode-M (1, 4, 8) | ✅ |
| Verify-M (tree decode) | ✅ |
| Prefill-M (256, 1024) | ✅ |
| CUDA graph capture + replay | ✅ |
| GPU 에 quantized storage 유지 (dense materialization 없음) | ✅ |
| TP-local shard 모양 (qkv / gate_up / o_proj / down_proj) | ✅ |

torchao int4_wo_tile / int8_wo 는 unchanged, `model_runner.py` 의 fp16-runtime
gate 가 이제 `backend=awq_marlin` 을 예외 처리 (Marlin 이 fp16 native).

### 2.3 검증 결과 — 수치 정확성

`sandbox/awq_spike/01_tp_linear_roundtrip.py` — dense weight → RTN-quant →
Marlin matmul → `F.linear(x, dequantized_weight)` 비교:

| dtype | max rel err (decode-shapes) |
|---|---|
| fp16 | 5×10⁻⁴ |
| bf16 | 4×10⁻³ |

CUDA graph capture + replay 동일 수치 재현. dequantize-then-matmul 기준
0.1% 미만 → pure Marlin roundoff.

### 2.4 검증 결과 — End-to-end

**Llama-3.2-1B-Instruct, TP=1**:
> "The capital of France is Paris. Paris is the capital of France..."
> (AR decode, AWQ target)

**layerskip-llama3-8B, TP=1**:
> "The capital of France is Paris. The country is divided into 27 regions
> and 96 departments. The largest city in France is Paris, with a population
> of 2.2 million..."

**layerskip-llama3-8B, TP=2**:
> "...Paris, with a population of 2,229,621. The second largest city is
> Marseille, with a population of 852,..."
> (TP-shard 검증 — 첫 시도 GQA QKV-shard 버그로 노이즈; impl-issues 참조)

**Sync spec decode, TP=2, target AWQ + draft dense 1B**:
> Accept rate 0.42, tokens/verify-step 2.67, verify 12.85 ms

**MESA-SSD, target AWQ TP=2 + async dense 1B draft**:
> Accept rate 0.43, cache hit 0.67, tokens/step 2.72, verify 18 ms,
> split-verify CUDA graph 캡처 성공. 생성 텍스트 정상

### 2.5 Performance — Microbench

**Local TP-linear matmul, bf16, RTX 3090 sm_86 (μs/call)**:

| shape | dense | awq_marlin | speedup |
|---|---:|---:|---:|
| qkv_proj tp2 decode M=1 (K=4096, N=3072) | 38.3 | 32.4 | 1.18× |
| qkv_proj tp2 verify M=8 | 45.0 | 34.0 | 1.32× |
| gate_up tp2 decode M=1 (K=4096, N=14336) | 154.2 | 41.7 | **3.70×** |
| gate_up tp2 verify M=8 | 148.2 | 42.2 | **3.52×** |
| down_proj tp2 decode M=1 (K=7168, N=4096, row-parallel) | 75.5 | 32.4 | 2.33× |
| o_proj tp2 decode M=1 (K=2048, N=4096, row-parallel) | 25.2 | 33.8 | 0.75× |
| prefill qkv M=256 | 106.7 | 99.6 | 1.07× |
| prefill gate_up M=256 | 449.3 | 454.4 | 0.99× |

패턴: memory-bound decode matmul 이 클수록 (gate_up 지배) W4 이득 큼.
`o_proj M=1` 만 regression — 작은 shape 에서 bf16 이 이미 memory-bound +
Marlin launch overhead. Prefill 은 compute-bound 라 거의 동등.

### 2.6 Performance — End-to-end (8B TP=2)

**layerskip-llama3-8B TP=2, AR decode, 128 output tokens**:

| variant | prefill | decode | e2e | KV cache 블록 |
|---|---:|---:|---:|---:|
| dense bf16 | 9 tok/s | 74 tok/s | 55.3 tok/s | 398 |
| **awq_marlin** | **10 tok/s** | **147 tok/s** | **87.3 tok/s** | **519** |

Decode throughput +99% (**1.99×**). KV cache 31% 증가 — packed weights 가
HBM ≈12 GB freed.

### 2.7 MESA accept rate vs RTN quality

Plan §16.2 mitigation: "AWQ vs RTN MESA accept rate 측정 — 차이 미미하면
calibration pipeline 단순화 고려".

Phase 3b 임포터는 RTN 만 구현. RTN W4A16 MESA smoke (layerskip-llama3-8B):
accept 0.43, cache-hit 0.67. 기존 dense MESA baseline (`MESA-RESULTS.md`,
typical accept 0.40-0.50 at temp=0.6) 의 noise 범위 안. 직접
AWQ-calibrated vs RTN 비교는 target 모델의 외부 AutoAWQ 체크포인트가 없어
보류 — Phase 3a/3b 코드는 ingest 가능.

---

## 3. v2 — 34B 실험 (`final_exp2_quant`)

기존 `final_exp2/` 의 6 config 와 동일한 setup 으로 재실험. dense fp16 vs
AWQ W4A16 비교.

### 3.1 Setup

- **Target**: `facebook/layerskip-codellama-34B` — AWQ-calibrated (AutoAWQ,
  bs=32, 128 C4 samples, 17:30 calibration time on 4 GPU)
- **Draft**: `TinyLlama-1.1B-Chat-v1.0` (TP=1, dense)
- **Prompts**: 200 (humaneval/alpaca/gsm8k/ultrafeedback × 50)
- output_len=256, temp=0.6, B=1, max_model_len=2048
- 5 GPU (4 target + 1 draft) on RTX 3090

### 3.2 Throughput 결과

| Config | dense TP | AWQ TP | Speedup |
|--------|---------:|-------:|--------:|
| AR (TP=4) | 28.26 | **55.57** | **1.97×** |
| Baseline K=7 uniform | 67.99 | 75.50 | 1.11× |
| Baseline K=7 geo | 68.45 | 74.70 | 1.09× |
| MESA K=5 exit=24 | 60.73 | 60.19 | 0.99× |
| MESA K=5 exit=28 | 58.48 | 61.42 | 1.05× |
| MESA K=5 exit=32 | 58.22 | 61.07 | 1.05× |

### 3.3 Per-spec metrics

| Config | dense Accept / CacheHit / Tok/Step | AWQ Accept / CacheHit / Tok/Step |
|--------|:---:|:---:|
| Baseline K=7 uniform | 0.44 / 0.66 / 4.06 | 0.43 / 0.64 / 4.02 |
| Baseline K=7 geo | 0.44 / 0.66 / 4.07 | 0.42 / 0.66 / 3.96 |
| MESA K=5 exit=24 | 0.52 / 0.83 / 3.62 | 0.51 / 0.83 / 3.55 |
| MESA K=5 exit=28 | 0.52 / 0.85 / 3.60 | 0.52 / 0.85 / 3.62 |
| MESA K=5 exit=32 | 0.52 / 0.87 / 3.58 | 0.51 / 0.87 / 3.57 |

**Accept rate 가 dense 와 거의 동일** — AWQ quality 검증.

### 3.4 Per-phase breakdown

**Target side (compare_breakdown_dense_vs_quant.png)**:

| Phase | Dense (baseline_k7_uniform) | AWQ | Δ |
|---|---|---|---|
| verify_replay | 43.5 ms | **22.9 ms** | **-47%** ✅ |
| target_spec_wait | 10.2 ms | 22.3 ms | +12 ms (draft 기다리는 시간 증가) |
| sample_accept | 2.3 ms | 4.2 ms | +2 ms |
| **합계** | **56.0 ms** | **49.4 ms** | **-12%** |

**Draft side**: TinyLlama dense 그대로. 거의 변화 없음.

### 3.5 핵심 인사이트

- **verify_replay 가 절반으로 축소** — Marlin 이 target compute 에서 작동
- **target_spec_wait 가 +12 ms 증가** — target 이 너무 빨라져서 draft 가
  pipeline bottleneck
- 순 이득 target side 6-7 ms → throughput 1.1× 정도에 수렴
- spec/MESA 의 작은 speedup 은 **Amdahl 법칙** — target compute 만
  speedup 받고 draft + cross-process sync 는 그대로

---

## 4. v2 — 70B 실험 (`final_exp2_quant_70b`)

같은 6 config 를 `facebook/layerskip-llama2-70B` 에서 재현.

### 4.1 Setup

- **Target**: `layerskip-llama2-70B` — calibrated W4A16 (simplified AWQ,
  α=0.5, 128 C4 samples, group_size=128, zero-point)
  - SSD-native artifact: TP=4
- **Draft**: TinyLlama-1.1B (TP=1, dense)
- **Prompts**: 200, output_len=256, temp=0.6, B=1, max_model_len=2048
- 5 GPU (4 target TP + 1 draft) on RTX 3090

### 4.2 Calibration history (운영 노트)

Plan A — AutoAWQ — 70B 에서 두 번 실패:
- Run 1 (transformers 5.6): `caching_allocator_warmup` 이 14 GB 블록 reserve
  → `max_memory=18 GiB` cap 초과 → 로드 시점 OOM
- Run 2 (transformers 4.52): 모델 로드 완료, layer 51/80 에서 AWQ grid-search
  의 `assert torch.isnan(w).sum() == 0` 트립 — fp16 overflow on outlier-heavy
  channel (~2 시간 낭비)

Plan B (사용) — 자체 simplified AWQ in `ssd/quant/awq_calibrate.py`: 고정
α=0.5, 기하평균-normalized scale, q/k/v/gate/up 그룹의 preceding RMSNorm 으로
fold, 나머지 RTN W4A16. Grid search 없어 NaN corner case 없음. forward-pass
+ RTN-quantize 루프 ~25분 완료.

### 4.3 결과 (target AWQ + draft dense)

| Config | TP (tok/s) | Tokens/Step | Accept | CacheHit | DraftMs | VerifyMs |
|--------|:--:|:--:|:--:|:--:|:--:|:--:|
| AR (TP=4) | **32.87** | — | — | — | — | — |
| Baseline K=7 uniform | **69.94** | 4.06 | 0.44 | 0.67 | 47.6 | 44.4 |
| Baseline K=7 geo | **72.57** | 4.23 | 0.46 | 0.69 | 45.1 | 45.3 |
| MESA K=5 exit=40 | **61.02** | 3.70 | 0.54 | 0.80 | 53.9 | 45.8 |
| MESA K=5 exit=47 | **61.01** | 3.68 | 0.54 | 0.82 | 53.6 | 45.4 |
| MESA K=5 exit=53 | **58.85** | 3.61 | 0.52 | 0.84 | 54.7 | 44.7 |

### 4.4 MESA exit-layer 분석

70B 80 layer 의 exit fraction: 40 = 1/2, 47 ≈ 7/12, 53 = 2/3. 이는
`layerskip-codellama-34B` 의 24/28/32 (1/2, 7/12, 2/3) 와 동일 비율. 같은
qualitative 패턴:
- 일찍 exit (40 = 1/2L) → target 속도 최대 (verify_replay 짧음)
- 늦게 exit (53 = 2/3L) → proxy 품질 최대 → cache hit 최고 (0.84) 그러나
  TP 최저 (target 이 더 느린 draft 기다림)

### 4.5 vs 34B AWQ (같은 Marlin runtime, smaller target)

| Config | 34B AWQ TP | 70B AWQ TP | ratio (70B/34B) |
|---|:---:|:---:|:---:|
| AR | 55.57 | 32.87 | 0.59× |
| Baseline K=7 uniform | 75.50 | 69.94 | 0.93× |
| Baseline K=7 geo | 74.70 | 72.57 | 0.97× |
| MESA exit=24/40 (1/2L) | 60.19 | 61.02 | 1.01× |
| MESA exit=28/47 (7/12L) | 61.42 | 61.01 | 0.99× |
| MESA exit=32/53 (2/3L) | 61.07 | 58.85 | 0.96× |

**핵심 관찰**: AR 에서 70B 가 ~1.7× 느리지만 spec 에서 차이 ≤7% 로 좁아짐.
이게 **Amdahl boundary** — spec/MESA throughput 이 점점 dense draft
(TinyLlama-1.1B) 에 limited 되고 target 의 추가 parameter 가 spec_wait 에
흡수. Draft AWQ 도 함께 적용하면 (다음 §5) spec speedup 회복.

### 4.6 Per-phase breakdown (70B)

**Target side (mean ms/spec step, post-warmup)**:

| phase | baseline geo | MESA exit=40 | exit=47 | exit=53 |
|---|:---:|:---:|:---:|:---:|
| verify_replay | **39.6** | — | — | — |
| graph_pre + graph_post (split verify) | — | ≈40.3 | ≈40.3 | ≈42.5 |
| target_spec_wait | 9.5 | 11.6 | 11.7 | 13.5 |
| verify_sample_accept | 4.1 | 4.1 | 4.0 | 3.6 |
| **total** | **53.1** | **56.3** | **56.0** | **57.3** |

**Draft side (mean ms/spec step)**:

| phase | baseline geo | MESA exit=40 | exit=47 | exit=53 |
|---|:---:|:---:|:---:|:---:|
| tree_replay × K | **18.5** | — | — | — |
| phase1_replay × K | — | ≈16.8 | ≈16.8 | ≈16.8 |
| phase2_replay × K | — | ≈16.8 | ≈16.8 | ≈16.8 |
| hit_cache_respond | 8.3 | 4.1 | 3.7 | 3.4 |
| draft_recv_cmd / glue / etc. | 13.4 | 11.0 | 11.0 | 11.0 |
| proxy_wait | — | 0 | 0.05 | 1.7 |
| **total** | **40.2** | **48.0** | **47.8** | **48.9** |

MESA 의 draft chain (~48 ms) 이 baseline (~40 ms) 보다 길음 — 매 spec step
에 두 parallel replay (Phase 1 + Phase 2). 설계상 의도된 것.

### 4.7 70B `proxy_compute_send` 가 더 비싼 이유

34B AWQ 와 70B AWQ 비교 (same code, same V≈32k, same B=1, K=5):

| Phase | 34B AWQ | 70B AWQ | 비율 |
|---|---|---|---|
| `exit_logits` (lm_head + TP gather) | 0.33-1.43 ms | 1.17-1.35 ms | ~3× |
| `proxy_compute_send` | 0.72-0.80 ms | 1.98-2.25 ms | **~2.7×** |

**원인 분해**:

1. **CUDA 그래프 tail leak** (가장 큰 영향) — `graph_pre.replay()` 는 async,
   GPU 에 layer kernels 큐잉 후 즉시 return. `proxy_compute_send` 안의
   `.item()/.tolist()` (4번) 가 직전 GPU 작업 완료 대기. 70B 는 graph_pre 가
   더 많은 layer kernels 큐잉 (80 vs 48 → +67%) → tail 길어 → proxy 측정에
   흡수.
2. **lm_head 의 TP gather 비용** — `compute_logits` → `F.linear` +
   `dist.gather`. NCCL 동기 강제 → graph_pre 끝나기 대기. 70B 는 graph_pre
   더 길어 대기 시간 증가.
3. **HBM 메모리 압박** — GPU 0 점유: 34B AWQ TP=4 ≈17 GB, 70B AWQ TP=4 ≈22
   GB (한계 근접). caching allocator fragment 더 심해 kernel launch
   overhead +10-30%.
4. softmax/topk fp32 변환 비용은 동일. 모양 동일. 이 부분 ~150 us 두 모델
   동일.

**결론**: `proxy_compute_send` 는 순수 compute time 이 아니라 구조적으로
graph tail + TP gather 동기 대기를 흡수하는 측정 구간. 70B graph_pre 가
더 길고 GPU 0 메모리 압박이 커서 측정값 1.3-1.5 ms 증가. 실제 proxy
알고리즘 자체는 두 모델에서 거의 동일 (~0.5 ms compute + NCCL send 100 us).

---

## 5. v2 — Draft AWQ 비교 (`final_exp2_quant/draft_awq`)

target AWQ 가 spec/MESA 에서 거의 speedup 못 주는 (target 이 너무 빨라져서
draft 가 새 bottleneck) 문제를 풀기 위해 draft 도 W4A16 양자화. 같은 5
spec config 에서 비교.

### 5.1 Setup

- **Target**: 34B AWQ (`final_exp2_quant` 와 동일 artifact)
- **Draft**: TinyLlama-1.1B AutoAWQ-calibrated (~3 분 calibration)
- 5 GPU (4 target TP + 1 draft TP=1)

### 5.2 Throughput 결과

| Config | dense | target AWQ | +draft AWQ | 총 speedup |
|---|---:|---:|---:|---:|
| AR | 28.26 | **55.57** (1.97×) | — | 1.97× |
| baseline K=7 uniform | 67.99 | 75.50 | **102.05** | **1.50×** |
| baseline K=7 geo | 68.45 | 74.70 | **102.11** | **1.49×** |
| MESA exit=24 | 60.73 | 60.19 | **97.53** | **1.61×** |
| MESA exit=28 | 58.48 | 61.42 | **90.70** | **1.55×** |
| MESA exit=32 | 58.22 | 61.07 | **95.05** | **1.63×** |

### 5.3 Draft forward 시간 감소 (3-way 비교)

**Per-call draft forward**:

| 항목 | dense draft | target AWQ + dense draft | target+draft AWQ | dense → AWQ draft |
|---|---:|---:|---:|---:|
| `tree_replay` (baseline) | 4.64 ms | 4.65 ms | **3.56 ms** | **-23%** |
| `phase1_replay` (MESA) | 4.19 ms | 4.15 ms | **2.32 ms** | **-45%** |
| `phase2_replay` (MESA) | 4.19 ms | 4.18 ms | **2.41 ms** | **-42%** |
| `glue` (draft prefill/setup) | 3.97 ms | 3.94 ms | **2.07 ms** | **-48%** |

**Per spec-step total (draft 측 모든 phase 합)**:

| Config | dense | target AWQ + dense draft | target+draft AWQ | 감소 |
|---|---:|---:|---:|---:|
| baseline K=7 uniform | 28.54 ms | 22.19 ms | **13.10 ms** | **-54%** |
| baseline K=7 geo | 28.41 ms | 21.89 ms | **12.95 ms** | **-54%** |
| MESA exit=24 | 22.83 ms | 22.63 ms | **13.50 ms** | **-41%** |
| MESA exit=28 | 23.49 ms | 22.26 ms | **14.57 ms** | **-38%** |
| MESA exit=32 | 24.54 ms | 21.97 ms | **13.71 ms** | **-44%** |

### 5.4 핵심 관찰

1. **Per-call forward 는 MESA 가 가장 큰 이득 (-45%)** — 작은 tree (각 phase
   의 `MQ_LEN/2`) 에서 Marlin W4 압축 효과 극대화. baseline tree_replay 는
   더 큰 tree (MQ_LEN=24) 라 -23%
2. **Per-step total 은 baseline 이 -54% 로 더 큼** — baseline 은
   `tree_replay × 4` + `hit_cache_respond` + `draft_recv_cmd` 등이 모두 줄어듦.
   MESA 는 phase1+phase2 가 K=4 번씩 8 번 실행되지만 각 call 작아 비례
   이득
3. **target AWQ 만 켰을 때 (중간 컬럼) draft forward 는 거의 변화 없음** —
   당연 (draft 모델 자체 dense). per-step 6 ms 줄어든 건 cross-process sync
   (`hit_cache_respond`, `draft_recv_cmd`) 빨라진 부수효과
4. **Pipeline 관점**: dense 시 draft chain (28 ms) ≈ target verify (43 ms).
   target AWQ 로 verify 22 ms → target_spec_wait 12 ms 발생. **draft 도 AWQ
   로 13 ms 되니 target verify (22 ms) > draft chain (13 ms) → 이번엔 draft
   가 일찍 끝남** → critical path 가 다시 target 으로. 이게 throughput
   +35-62% 추가 이득의 직접 원인

### 5.5 비교 plot

- `final_exp2_quant/compare_throughput.png` — dense vs AWQ pair-bar
- `final_exp2_quant/compare_breakdown_dense_vs_quant.png` — per-phase
  dense vs AWQ stacked
- `final_exp2_quant/draft_awq/compare_breakdown.png` — target+draft AWQ
  의 per-phase contribution
- 각 config 별 `mesa_breakdown.png`, `mesa_timeline_step100.png`,
  `mesa_breakdown_over_time.png`

---

## 6. 운영 이슈 (실험 중 발견)

| 이슈 | 해결 |
|---|---|
| AutoAWQ + transformers 5.6 → `caching_allocator_warmup` OOM | transformers 4.52 다운그레이드 |
| AutoAWQ + 70B → layer 51 NaN assert | simplified AWQ (α=0.5) 로 fallback |
| 70B import 시 `--dtype float16` 사용했으나 70B 는 bf16 → 재import |  |
| 다중 calibration GPU 충돌 | 직렬화 + 명시적 `CUDA_VISIBLE_DEVICES` |
| Zombie multiprocessing workers | `pkill -f spawn_main` 후 재시작 |
| NCCL TCPStore zombie | port `fuser -k` |

총 작업 시간 (calibration + 실험 + plot): 약 7 시간 무인 실행.

---

## 7. Plan coverage

| Phase | 산출물 | 상태 |
|---|---|---|
| 0. 백엔드 feasibility 노트 | ✅ `02-impl-issues.md` |
| 1. Quant-state skeleton + module init 계약 | ✅ `state.py`, `init_context.py`, `linear.py` |
| 2. Runtime + local matmul adapter | ✅ `linear.py` + Marlin wrapper; fp16 rel-err 5e-4 |
| 3a. External AWQ thin adapter | ✅ `adapter.py` (합성 AutoAWQ 체크포인트로 검증) |
| 3b. SSD-native artifact pipeline | ✅ `importer.py` + `scripts/awq_import.py` |
| 4. Loader 통합 + config | ✅ `loader.py` + `config.py` + `bench.py` CLI |
| 5. E2E target-only 검증 | ✅ AR, sync-spec, CUDA graphs, TP=1, TP=2 |
| 6. MESA 검증 | ✅ async + MESA + AWQ on 8B TP=2 |
| 7. Perf benchmarks | ✅ micro + E2E |

**Draft AWQ 확장**: target/draft 둘 다 양자화하여 spec/MESA bottleneck 해결.
non-EAGLE Llama family + tp=1 draft 범위에서 검증.

---

## 8. Reproducibility quick-reference

```bash
# 0. 환경 (신규 설치 불필요 — ssd env 에 sgl-kernel + torchao 이미 포함)
source /home/chokwans99/PSD/ssd/env.sh

# 1. Llama 모델을 SSD-native W4A16 artifact 로 임포트 (RTN 경로)
python scripts/awq_import.py \
    --model /data2/chokwans99/models/layerskip-llama3-8B \
    --out   /tmp/awq_artifacts/layerskip8b_tp2 \
    --tp 2 --mode rtn --dtype bfloat16

# 2. AR decode smoke
CUDA_VISIBLE_DEVICES=0,1 python -O sandbox/awq_spike/04_tp2_8b_ar.py

# 3. MESA smoke (3 GPU)
CUDA_VISIBLE_DEVICES=0,1,2 python -O sandbox/awq_spike/07_mesa_awq.py

# 4. E2E perf (별도 프로세스 — 02-impl-issues.md [Phase 7] 참조)
CUDA_VISIBLE_DEVICES=0,1 python -O sandbox/awq_spike/08_perf_bench.py dense
CUDA_VISIBLE_DEVICES=0,1 python -O sandbox/awq_spike/08_perf_bench.py awq

# 5. CLI (bench.py)
python -O bench/bench.py --llama --size 8 --gpus 2 \
    --model_path /data2/chokwans99/models/layerskip-llama3-8B \
    --b 1 --temp 0 --numseqs 16 --output_len 128 --random \
    --quant_awq --quant_awq_artifact /tmp/awq_artifacts/layerskip8b_tp2

# 6. External AutoAWQ round-trip
python -O sandbox/awq_spike/09_fake_autoawq_roundtrip.py

# 7. Negative tests (모듈 누락 / 잘못된 zero_point / 잘못된 w_bit / 잘못된 group_size)
python sandbox/awq_spike/10_negative_checks.py

# 8. AutoAWQ calibration on a smaller / fitting model (별도 awq-quant env)
/home/chokwans99/anaconda3/envs/awq-quant/bin/python \
  scripts/awq_calibrate_autoawq.py \
    --model /data2/chokwans99/models/Llama-3.2-1B-Instruct \
    --out /data2/chokwans99/awq_calibrated/llama3p2_1b_autoawq \
    --dtype bfloat16 --max-calib-samples 128 --max-calib-seq-len 512

# 9. Custom (simplified) AWQ — 70B fallback path
python scripts/awq_calibrate.py \
    --model /data2/chokwans99/models/layerskip-llama2-70B \
    --out   /data2/chokwans99/awq_calibrated/layerskip_llama2_70b \
    --n-samples 128 --seq-len 512 --alpha 0.5 \
    --dtype float16 --device-map auto --max-gpu-memory 18GiB
```

---

## 9. Known limitations

- AutoAWQ 가 70B 에서 transformers 5.6 (warmup OOM) 와 4.52 (NaN in scale
  grid search) 둘 다 실패. simplified AWQ (α=0.5, no grid search, no clip)
  사용. 70B calibration *quality* 가 AutoAWQ 논문 보고치보다 약간 떨어질 수
  있음
- Draft 가 dense TinyLlama-1.1B → spec throughput 이 draft step time 에
  limited. `final_exp2_quant/draft_awq` 의 추가 양자화 결과로 spec speedup
  회복. 70B target 에 같은 조합 적용은 follow-up
- Eagle draft AWQ 미지원 (hard-fail at runner init)
- Draft `lm_head` / embeddings 양자화 미지원
- Draft TP > 1 미지원

## 10. Next steps (이 plan 범위 밖)

- **공개된 AutoAWQ 체크포인트 다운로드 후 ingest** (예:
  `hugging-quants/Meta-Llama-3-8B-Instruct-AWQ-INT4`), MESA 하 AWQ-calibrated
  vs RTN 비교. 합성 AutoAWQ roundtrip 으로 전체 flow 검증됐음 — 남은 건
  calibration-quality ablation 뿐. plan §16.2 mitigation
- **`lm_head` ablation** — 현재 dense 유지. plan §11.2 의 "quant lm_head 하
  accept rate" 측정 기준은 재검토 시 필요
- **Qwen3 family** — plan §10.2, Llama 안정화 후. `naming.py` 는 확장 가능한
  구조
- **Prefill 속도 parity** — `o_proj M=1` regression + prefill parity 는
  persistent workspace + kernel warm-start 로 Marlin launch overhead 숨길
  여지가 있음. 마이너 최적화

## 11. Out-of-scope (plan §17)

- bitsandbytes 통합 안 함
- scratch-Triton GEMM backend 안 함
- ~~Draft 는 dense 유지~~ — 후속 draft AWQ 확장에서 해결
- Embeddings 는 dense 유지
- `ssd/utils/quantize.py` 의 torchao 경로는 fallback 으로 유지 (cleanup PR
  에서 삭제 예정)
