# 양자화 통합 계획

이 문서는 SSD 엔진의 weight-only 양자화 통합 history 와 현재 (AWQ Marlin)
의 phase 별 설계를 한 곳에 모은 것이다. 두 단계의 계획을 거쳤다 —
v1 (torchao INT8/INT4) 은 SM 86 에서의 fp16 호환성 문제로 한계에 부딪혔고,
v2 에서 AWQ-style W4A16 (sgl-kernel Marlin) 으로 방향을 바꿨다.

---

## 1. v1 의 history — torchao INT8/INT4 (legacy)

### 1.1 v1 의 목표 (당시)

SSD 코드베이스에 **target-only INT8 weight-only** 지원을 추가. torchao 의
`Int8WeightOnlyConfig` / `Int4WeightOnlyConfig` 를 이용해 SSD custom TP
linear 의 local weight 를 AffineQuantizedTensor (AQT) 로 교체하는 방식.

### 1.2 v1 가 부딪힌 실제 문제

Phase 0 feasibility + Phase 2 eager 통합까지는 계획대로 완료되었으나,
초기 smoke test 에서 NaN/inf 크래시가 발생했다. 디버깅 결과:

- **현상 1**: sampling spec (temp>0) 에서 layer 1 MLP 출력의 극소수 위치
  (1.38M 중 8개) 에 `inf` → RMSNorm 에서 `0 * inf = NaN` → softmax →
  `multinomial` assert 크래시
- **현상 2**: greedy spec (temp=0) CodeLlama-34B accept rate `0.38 → 0.03`
- **재현**: `layerskip-llama2-7B` 와 `layerskip-codellama-34B` 모두

**초기 오진단**: "Llama outlier activation → AWQ/SmoothQuant 필요" — 사용자
지적대로 **weight-only 상황에 맞지 않는 해석**.

**실제 원인 (sub-op 추적 + AQT state 검사 결과)**:

torchao 가 선택한 두 backend (`Int4WeightOnlyConfig`, `Int8WeightOnlyConfig`)
는 공식 문서가 **bf16-activation workflow** 로 명시하는 backend 다.
fp16 activation 과는:
- `Int4`: scale/zero 를 bf16 으로 고정 생성 → fp16 activation 과 matmul
  kernel 레벨 dtype assert fail
- `Int8`: API 통과하지만 수치적으로 불안정 (원인: fp16 accum 가설 등 있으나
  미확정)

→ torchao 전체가 fp16 미지원은 아님. `GemliteUIntXWeightOnlyConfig` 등 다른
backend 는 fp16-native. 단 우리가 선택한 backend 는 fp16 runtime 신뢰 불가.

### 1.3 v1 채택 정책 (정정)

| checkpoint dtype | backend | 상태 |
|---|---|---|
| bf16 (Llama-3 family) | `int4_wo_tile` / `int8_wo` | **정상 지원** |
| fp16 (Llama-2, CodeLlama) | 위와 동일 | **미지원** — `ValueError` 기본, `target_quant_force_bf16_runtime=True` opt-in 으로 bf16 runtime 우회만 가능 |
| fp16 | `GemliteUIntXWeightOnly` / `Marlin` | 미통합 (별도 과제) |

### 1.4 v1 Phase 별 결과 요약

| Phase | 상태 | 핵심 결과 |
|---|---|---|
| 0. feasibility + graph-safety | ✅ | storage contract (A) — `self.weight = dummy.weight` 재할당, forward 미변경. CUDA graph + `inference_mode` 정상 |
| 1. weight replacement contract | ✅ | scale shape `(out_f,)` per-output-channel, block_size `(1, 128)` 확정. Column shard 안전, Row shard 도 local quantize 가 finer-grained |
| 2. plain INT8 eager 통합 | ✅ | code OK. fp16 모델에선 overflow 발견 → bf16 upcast 로 우회 |
| 2.5 kernel path 최적화 | ✅ | INT8 → INT4 전환. INT4 tile_packed 이 SM 86 에서 dense 대비 0.25-1.25×, INT8 대비 2.7× 빠름 |
| 3. SSD graph path 확장 | ✅ | AR/spec/MESA graph 모두 INT4 호환, graph 는 eager 대비 2-4× |
| 4. MESA + lm_head ablation | ✅ | MESA 에서 lm_head 양자화 시 accept 0.41 → 0.33 (20% 손실), bf16 유지 시 0.38 (7% 손실) |
| 5. persistent artifact | ✅ | save/load AQT per rank — Llama-3-8B smoke pass |
| 6. 34B 확장 | ✅ | CodeLlama-34B (fp16 → bf16 upcast) TP=4 INT4 동작, async spec TP=75.28 |

### 1.5 v1 실측 (async spec + sampling, temp=0.6)

**Llama-3-8B (bf16 native, TP=2 + draft):**

| config | TP | accept |
|---|---|---|
| dense | 15.84 | 0.32 |
| INT8 wo | 14.25 | 0.30 |
| **INT4 tile_packed** | **18.64** | **0.30** |
| MESA dense | 24.12 | 0.41 |
| MESA INT8 | 12.15 | 0.40 |
| MESA INT4 + lm_head off | 13.47 | 0.38 |

**CodeLlama-34B (fp16 → bf16 upcast, TP=4 + draft, 5 GPU, 51200 tok 동일 조건):**

| config | TP | accept | wall |
|---|---|---|---|
| pre-quant dense async spec | 68.45 | **0.44** | 747.98 s |
| **INT4 tile_packed async spec** | **75.28** | **0.44** | 680.09 s |

INT4 가 dense 대비 +10% 빠르고 accept 완벽 동일.

### 1.6 v1 미해결

- **34B MESA INT4 dispatch 실패** — `QuantizedLinearNotImplementedError`.
  MESA verify graph 의 특정 shape 에서 torchao tile_packed tinygemm 미지원.
  우회: 34B MESA 에 INT8 backend (느리지만 동작) 또는 torchao 업스트림 대기
- **70B** — 로컬 모델 없음 (~140 GB 다운로드 필요)
- **artifact CPU→GPU warning** — `TensorCoreTiledAQTTensorImpl` 변환 경고

### 1.7 v1 → v2 로 전환한 이유

torchao 경로는 **bf16-native 모델만 지원**, **SM 86 에서 INT8 fast kernel
부재**, **load 시 dense GPU weight 먼저 올린 후 교체** 등 한계가 명확했다.
계획 v2 는 다음 방향:

- **fp16/bf16 양쪽 native** 지원
- **Marlin W4A16 fast kernel** (SM 80/86/89 모두 fast path)
- **Offline AWQ artifact 직접 load** — dense GPU 단계 skip
- **TP linear local matmul 경계만** 변경, 나머지 (PagedAttention, CUDA
  graph, MESA orchestration) 그대로

---

## 2. v2 의 목표 — AWQ W4A16 (Marlin)

### 2.1 최종 목표

**fp16/bf16 호환 optimized weight-only backend** 를 SSD 에 통합:

- 큰 target 모델이 VRAM 에 들어가게
- 기존 SSD optimized path (TP / PagedAttention / CUDA graph / prefix cache)
  손대지 않고
- AR / spec verify / MESA target verify 모두 동작
- full-precision GPU weight 를 먼저 올린 뒤 양자화 하지 않음

Backend 방향: **AWQ-style W4A16** (torchao INT8 아님).

### 2.2 핵심 설계 결정

**유지할 것**:
- `ssd/layers/linear.py` 의 TP wrapper 와 semantics
- attention path / PagedAttention / FlashInfer wrappers
- KV cache layout, block-table handling
- Speculative decoding control flow
- MESA orchestration, split verify
- CUDA graph capture/replay
- `@torch.compile` norm/activation/rope path
- Prefix caching, scheduler

**바꿀 것**: heavy linear weight 의 storage 와 execution 만.

### 2.3 양자화 대상 / 비대상

**기본 양자화** (target heavy projections):
- `q_proj / k_proj / v_proj` → SSD packed `qkv_proj`
- `o_proj`
- `gate_proj / up_proj` → SSD packed `gate_up_proj`
- `down_proj`

**Dense 유지**:
- embeddings
- `lm_head` (per-step hot path, MESA exit-layer logits 민감)
- norms / rope / attention core / KV cache
- draft (v1 에서는 dense — v2 의 draft AWQ 확장에서 추가됨)

### 2.4 Backend 선택 — Marlin

**선택**: `sgl_kernel.gptq_marlin_gemm(b_q_type=scalar_types.uint4,
is_zp_float=False)`. AWQ-format 입력 텐서를 load time 에 Marlin
layout 으로 repack.

**Gate (모두 통과, RTX 3090 sm_86)**:

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

대안 검토:
- **bitsandbytes** — HF / `nn.Linear` 대체 패턴이라 SSD custom TP 와 충돌 → 제외
- **torchao 기존 경로** — fp16 미지원, INT8 SM 86 fast kernel 없음 → fallback 만 유지
- **AWQ + Marlin** — 정답. 아래 §3 부터 상세

### 2.5 Hard Constraints (모든 candidate 가 만족해야)

- runtime dtype: fp16 + bf16 둘 다 (또는 명확한 fallback 전략)
- shape: 작은 M (decode/verify/MESA) + 큰 M (prefill) 모두
- graph 호환: `inference_mode` + CUDA graph capture/replay + 현 TP wrapper
- storage: packed low-bit weight + scales + zero-point 를 GPU 에 유지

---

## 3. v2 최종 아키텍처

```
   external AutoAWQ hf dir                  dense HF checkpoint
           │                                       │
           ▼                                       ▼
   scripts/awq_import.py --mode autoawq    scripts/awq_import.py --mode rtn
           │                                       │
           └─────────────┬─────────────────────────┘
                         ▼
                 SSD-native artifact
            <prefix>.rank{r}.awq.pt  (per-rank, pickled)
                         │
                         ▼
     ssd.quant.loader.apply_ssd_awq_artifact(model, prefix, rank, tp)
                         │
                         ▼
          module.attach_quant_state(AwqQuantState)
                         │
                         ▼
          TP linear forward → awq_matmul → Marlin W4A16

   ← (Phase 3a) thin adapter: 디스크상의 SSD-native artifact 거치지 않고
      external AutoAWQ 를 live SSD model 에 직접 로드:
      ssd.quant.adapter.load_external_autoawq_into_model
```

### 3.1 통합 경계

**TP linear forward dispatch 만** 변경. 나머지 전부 (PagedAttention,
FlashInfer, KV cache, tree-verify mask, CUDA graph, MESA, prefix cache,
scheduler, draft process) 그대로.

### 3.2 Quant-mode 모듈 생성 (§6.3.1 option 2 = meta placeholder)

`quant_init_context()` 안에서 TP linear `__init__` 이 `self.weight` 를
`torch.device("meta")` 에 할당. dense weight 에 GPU 메모리를 소비하지 않음.
dense safetensors loader 는 meta param 을 silently skip 하고, AWQ loader 가
`module.attach_quant_state(state)` 로 placeholder 를 교체. Forward 는
`self.quant_state is not None` 여부로 분기.

### 3.3 Packed-module TP sharding

`shard_awq_column_parallel` 은 sub-part 인식: `qkv_proj` (GQA) 와
`gate_up_proj` (동일 크기 두 파트) 모두 각 sub-projection 을 `part_out //
tp_size` 단위로 자르고 per-rank 슬라이스를 concatenate. dense
`QKVParallelLinear.weight_loader` 규약과 정확히 일치 — q (32 heads) 와 k/v
(각 8 heads) 크기가 다른 GQA 모델에서 필수.

---

## 4. v2 Phase 별 계획

### Phase 0 — Backend Feasibility Spike

**Objective**: AWQ runtime 후보 backend 선택.

**Tasks**:
1. SSD-relevant local matrix shape 에서 측정:
   - decode-like 작은 M
   - verify-like 작은 M
   - prefill-like 큰 M
2. fp16 / bf16 runtime 지원 확인
3. graph capture safety
4. local shard shape 지원 확인

**Deliverable**: backend 선택 + 지원 dtype + 미지원 shape + graph 호환 결과.

**Hard gate**: 이 phase 닫히기 전엔 broad integration 시작 안 함.

### Phase 1 — Runtime Quant State Skeleton

**Objective**: in-memory quantized state 와 TP-module 변경 정의 — 외부 packed
AWQ weight 를 local TP module 에 attach 하여 실행 가능.

**Tasks**:
1. TP linear 의 quant-state ownership 정의
2. dense vs quant forward dispatch shape contract
3. Quant-mode 모듈 생성 contract (§6.3.1)
4. `weight_loader(param, loaded_weight[, shard_id])` 가 quant mode 에서
   어떻게 보존/적응되는지
5. 기존 TP class 확장 vs 가벼운 wrapper 선택

### Phase 2 — Runtime Quant State + Local Matmul Adapter

**Objective**: AWQ-backed local linear 실행을 model topology 변경 없이 추가.

**Tasks**:
1. TP linear modules 에 quantized state 추가
2. AWQ-backed matmul 의 local runtime 분기 추가
3. dense path 그대로
4. row/column/QKV/merged semantics 보존

### Phase 3a — External AWQ Checkpoint thin Adapter

**Objective**: external AWQ checkpoint 를 SSD-native artifact pipeline 거치지
않고 직접 load — 빠른 backend/runtime validation 용.

**Tasks**:
1. external AWQ checkpoint (HF safetensors + `quantize_config.json`) 읽기
2. HF module name → SSD module name runtime mapping
3. `qkv_proj` / `gate_up_proj` CPU repack
4. CPU TP shard
5. packed weight + scales + zero-points 를 runtime module 로 직접 load

### Phase 3b — SSD-Native Pre-Sharded Artifact Pipeline

**Objective**: production startup speed 위한 offline artifact pipeline.

**Tasks**:
1. external AWQ checkpoint input 수용
2. HF → SSD name 변환
3. `qkv_proj` / `gate_up_proj` repack
4. TP sharding
5. SSD-native per-rank artifact 저장
6. manifest / version metadata 작성

### Phase 4 — Loader Integration

**Objective**: SSD-native AWQ artifact 를 runtime 으로 직접 load.

**Tasks**:
1. artifact detection
2. artifact metadata validation
3. quant state 를 artifact 로부터 직접 instantiate
4. dense GPU weight materialization skip
5. quant disabled 시 dense loader 동작 그대로

### Phase 5 — End-to-End Target-Only Validation

**Required checks**: AR decode / verify / 1 speculative path / CUDA graph
capture+replay / prefix cache / TP gather correctness.

### Phase 6 — MESA Validation

**Required checks**:
1. split verify capture 동작
2. target MESA verify path correctness
3. `lm_head` dense default baseline
4. `lm_head` quant ablation (선택)
5. accept rate / throughput 비교

### Phase 7 — Performance and Startup Optimization

**Required benchmarks**:
1. decode-like microbench
2. prefill-like microbench
3. end-to-end AR
4. end-to-end spec
5. end-to-end MESA target path

**Compare against**: dense fp16/bf16, current torchao fallback (해당 시), AWQ.

---

## 5. v2 Config 설계 (§13)

### 5.1 구조

```python
@dataclass
class QuantConfig:
    role: str                              # "target" | "draft"
    enabled: bool = False
    artifact_path: str | None = None       # SSD-native artifact prefix
    quant_source: str = "ssd_artifact"     # "ssd_artifact" | "external_awq"
    external_quant_path: str | None = None
    group_size: int = 128
    expected_runtime_dtype: str = "float16"
```

**선언 정책**: 실제로 runtime 이 사용하는 필드만 유지. 초기 v2 plan 에는
`method`, `artifact_mode`, `runtime_backend`, `use_zero_point`,
`quantize_lm_head`, `quantize_embeddings` 가 선언됐으나 모두 dead surface
라 trim (review 반영).

### 5.2 Default policy

- `enabled=False`
- target-only when enabled (draft 도 별도 활성화 가능)
- `lm_head` / embeddings dense
- offline artifact 가 primary

### 5.3 Migration (flat → structured)

1. `QuantConfig` 도입
2. 기존 flat CLI/config 필드 (`target_quant_*`) 를 임시 compat shim 으로 유지
3. LLM/runner 경계에서 `QuantConfig` 파생
4. AWQ 안정화 후 legacy flat 필드 삭제 (cleanup PR 단계로 분리)

---

## 6. v2 의 draft AWQ 확장 (subsequent extension)

target-only AWQ 가 main 안정화 된 뒤 — async SSD 와 MESA 에서 dense draft 가
새 bottleneck 이 됐기 때문에 — draft 도 AWQ 로 양자화하는 확장을 추가.

### 6.1 설계 원칙

1. 기존 AWQ-Marlin runtime path 를 가능한 한 재사용
2. 현 SSD optimized 아키텍처 보존
3. async spec / MESA 에서 동작
4. target AWQ path 회귀 없음
5. 구현 localized, 유지보수 가능

### 6.2 Scope (v1)

- non-EAGLE draft only
- Llama family
- AWQ W4A16 only, backend `awq_marlin` only
- draft embeddings dense
- draft lm_head dense
- target path 동작 그대로
- SSD-native artifact 가 primary, external AutoAWQ 도 옵션

### 6.3 Non-goals (이번 pass)

- bitsandbytes / GPTQ / INT8 / scratch Triton kernel
- EAGLE draft 양자화
- draft embeddings / lm_head 양자화
- scheduler / verifier / MESA logic 재설계
- FlashInfer / PagedAttention / KV cache / CUDA graph 구조 재작성

### 6.4 핵심 설계 (role-aware)

target AWQ 와 draft AWQ 는 다음만 다름:
- config 출처
- artifact source
- model role validation
- runtime wiring

local quantized linear execution path 는 공유.

### 6.5 Phase 별 구현

- **Phase 0 (config)**: target_quant_* 와 독립적인 draft_quant_* 필드
  추가. `QuantConfig.role` 도입. clean shim — flat parse 1회 → 두
  structured config 산출. downstream 은 structured 만 읽음.
- **Phase 1 (runner)**: draft AWQ 활성화 시 draft model 도
  `quant_init_context()` 안에서 build. dense loader 는 dense 모듈에만 호출,
  attach AWQ quant state. target/draft 로직 명시적으로 분리. draft AWQ v1
  은 `tp_size==1` 가정 (현 SSD invariant).
- **Phase 2 (artifact)**: `model_role = "target" | "draft"` 메타데이터 추가.
  target/draft cross-load → load-time hard fail. schema version bump.
- **Phase 3 (loader)**: draft runner 가 동일 AWQ artifact loader 호출.
  completeness check 가 draft modules 에도 적용.
- **Phase 4 (external direct-load, optional)**: clean 하게 가능하면 draft
  direct-load 도 지원. 아니면 SSD-native artifact 만 supported, external 은
  experimental.
- **Phase 5 (graph validation)**: warmup with draft AWQ, draft decode graph,
  tree-decode graph capture+replay 검증.
- **Phase 6 (E2E spec/MESA)**: sync spec / async spec / MESA + draft AWQ.
- **Phase 7 (perf)**: dense vs AWQ draft step cost 비교. target_verify ms /
  draft_step ms / accept / cache_hit / tokens-per-step / e2e tok/s.

### 6.6 검증 결과

- non-EAGLE Llama 가족 정상 동작 확인
- 34B target AWQ + TinyLlama-1.1B draft AWQ 조합에서 spec/MESA 안정
- 자세한 수치는 `03-final-report.md` 참조

---

## 7. 위험과 mitigation

### 7.1 최상위 위험

1. 선택한 AWQ runtime 이 SSD decode-like 작은 M shape 를 잘 지원 못함
2. graph capture 호환성이 예상보다 약함
3. external AWQ artifact schema 가 SSD packed module 에 매핑하기 어색함
4. bf16 runtime 지원이 fp16 보다 약함
5. AWQ calibration 이 단순 RTN 대비 의미 있는 quality 개선을 못 줌 — 특히
   MESA accept rate 나 generation quality 가 별 차이 없으면 offline
   calibration pipeline 의 ROI 가 낮음

### 7.2 명시적 완화

1. Phase 0 가 약한 backend 를 일찍 reject
2. torchao path 를 임시 bf16 fallback 으로 유지
3. 초기에 `lm_head` 양자화 안 함
4. 초기에 draft 양자화 안 함 — 후속 단계에서 추가
5. Phase 5 초반에 MESA accept rate 를 AWQ vs RTN 으로 측정. 차이 미미하면
   calibration pipeline 단순화 고려

---

## 8. 비-목표 (Non-goals)

이 통합 작업의 명시적 범위 밖:

- 직접적인 bitsandbytes runtime 통합
- scratch Triton quant GEMM backend
- draft 양자화 — 이후 별도 확장 단계에서 추가
- embeddings 양자화
- 기본 `lm_head` 양자화
- day 1 generic multi-model-family 지원
- AWQ 검증 끝나기 전 torchao 코드 삭제

---

## 9. 최종 권고

v2 의 올바른 방향:

1. **SSD 엔진 아키텍처 그대로**
2. **현 torchao path 는 임시 fallback 으로만**
3. **AWQ-style offline artifact pipeline 추가**
4. **local TP linear boundary 에 optimized AWQ runtime 통합**
5. **Target-only first**
6. **Llama-family first**
7. **MESA validation 이 끝나야 작업 완료**

가장 좁은 plan 으로 실제 문제 (fp16/bf16 실용 지원, VRAM 절감, optimized
runtime, SSD 아키텍처 보존) 를 해결하는 길이다.
