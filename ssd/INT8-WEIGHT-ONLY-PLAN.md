# SSD Target-Only Weight-Only Quantization 지원 계획

> **Naming note (2026-04-21)**: 원래 "INT8 weight-only"로 시작했으나 Phase 2.5에서 SM 86 hardware kernel 제약 때문에 **INT4 tile_packed (`torchao.Int4WeightOnlyConfig`)를 기본 backend로 채택**. INT8 경로는 `--quant_int8` flag로 계속 호출 가능. 문서는 "weight-only quantization" 으로 일반화해 읽어야 한다.

## 1. 문서 목적 및 현재 상태

이 문서는 SSD 코드베이스에 `target-only INT8 weight-only` 지원을 추가하기 위한 **현실적인 구현 목표와 단계별 계획**을 정리한다.

### 1.1 이전 계획의 핵심 문제 (초기 draft 기준)

- 외부 포맷 호환, canonical format, importer, runtime loader를 한 번에 다 설계하려고 했다
- SSD 런타임이 실제로 어떤 quantized 연산 경로를 사용할지 확정되기 전에 저장 포맷부터 크게 설계했다
- 결과적으로 범위가 너무 넓고, 실제로 구현을 시작하기 전에 불확실성이 너무 많았다

### 1.2 Phase 0~2 진행 후 발견된 실제 blocker와 root-cause (2026-04-21)

Phase 0 feasibility spike + Phase 2 eager 통합까지는 계획대로 완료되었고, 초기 smoke test에서 실패 증상을 관찰했다. 심층 디버깅 결과 **진짜 원인은 "Llama outlier activation"이 아니라 fp16 dynamic range overflow**임이 밝혀졌다.

**초기 증상**:
- sampling spec (temp>0): layer 1 MLP 출력의 극소수 위치(1.38M 중 8개)에 `inf` 발생 → RMSNorm에서 `0 * inf = NaN` 전파 → softmax → `multinomial` assert 크래시
- greedy spec (temp=0), CodeLlama-34B: 크래시는 없지만 accept rate `0.38 → 0.03`
- `layerskip-llama2-7B` 및 `layerskip-codellama-34B` 모두 재현

**초기 오진단**: "Llama outlier activation" + "LLM.int8 / SmoothQuant / AWQ 필요"로 해석하려 했으나 — 사용자 지적대로 이건 **weight-only 상황에 맞지 않는 해석**.

**실제 root cause (sub-op 추적 + AQT state 검사 결과)**:

layer 1 내부를 sub-op 단위로 추적하니 dtype이 fp16이었고 값들이 다음과 같았다:
- `gate_up_proj` 출력 absmax = **33.78**
- `silu(gate) * up` (down_proj 입력) absmax = **997.5**
- `down_proj` 출력: 8 inf

수치 분석:
- 모델이 **fp16**으로 로드됨 (`layerskip-llama2-7B`, CodeLlama 등의 `torch_dtype="float16"` 기본값)
- **fp16 max = 65504** (bf16의 3.4e38 대비 극히 좁음)
- down_proj matmul: 입력 997.5 × weight(~1.29 abs max)를 5504 채널 누적. fp32 accumulator엔 `5504 × 997.5 × 1.29 ≈ 7.1M` 저장 가능하나, **최종 fp16 cast 시 65504 초과로 overflow → inf**
- 즉 activation outlier 문제가 아니라, Llama MLP 중간 활성화 자체가 fp16 표현 범위를 넘는 정상적인 계산 결과. 이건 quant 여부와 무관하게 fp16이면 취약한 지점 (dense fp16이 통과하는 건 tensor core 경로가 조금 다른 saturate 동작)

**AR 경로가 통과한 이유**:
- AR은 prefill을 fixed-length `(512, 4096)` padded로 수행 → padding이 outlier 자극 완화
- spec은 실제 prompt 길이 (`(338, 4096)`)로 prefill → 실제 content가 overflow 유도

**검증 (2026-04-21, `layerskip-llama3-8B` bf16 native)**:

동일 테스트 스위트 전부 통과 — dtype이 **원인이었음을 확정**.

| test | TP | accept | 결과 |
|---|---|---|---|
| AR dense | 24.84 | n/a | baseline |
| AR int8 | 9.48 | n/a | 동작 (cpp ext 없음 → 속도 손해) |
| async spec dense sampling | 15.84 | 0.32 | baseline |
| **async spec int8 sampling** | **14.25** | **0.30** | **Llama-2(fp16)에선 crash → Llama-3(bf16)에선 통과 ✓** |
| MESA dense sampling | 24.12 | 0.41 | baseline |
| **MESA int8 sampling** | **12.15** | **0.40** | **통과, accept 거의 보존 ✓** |

**원인 재정정 (2026-04-21 post-investigation)**: dense fp16 layer 1 absmax=1597 실측 → 65504의 1/40으로, overflow 아님. int4 no-upcast는 `ValueError: Expected zeros fp16, got bf16` (torchao API dtype assert), int8 no-upcast는 layer 1 `inf=22 finite_absmax=440` (수치 불안정). 즉 "fp16 overflow"가 아니라 **현재 선택한 torchao weight-only backend (`Int4WeightOnlyConfig` / `Int8WeightOnlyConfig`)가 공식 문서상 bf16-activation workflow**라 fp16 activation과 호환되지 않는 것이 실체. 이하 §4.1.2에서 상세.

### 1.3 채택한 정책 — bf16-native는 완성, fp16은 opt-in workaround 또는 미지원

**AWQ/SmoothQuant 전환은 철회**. 이번 작업 범위 안에서 해결할 수 있는 문제가 아니었다.

| checkpoint dtype | backend | 상태 |
|---|---|---|
| bf16 (Llama-3/3.1 계열) | `int4_wo_tile` / `int8_wo` | **정상 지원** (완성) |
| fp16 (Llama-2, CodeLlama) | `int4_wo_tile` / `int8_wo`, 기본 | **미지원**. `target_quant_enabled=True`면 `ValueError` at init |
| fp16 + `target_quant_force_bf16_runtime=True` | 위와 동일 | **bf16 runtime 우회**. 주의: "fp16 checkpoint가 bf16 런타임에서 돈다"는 것이고 "fp16 runtime 지원"이 아님 |
| fp16 + fp16-native WO backend (Gemlite / Marlin 등) | 미통합 | 별도 과제 |

**즉 "fp16 checkpoint를 fp16 runtime으로 유지"하는 지원은 현재 backend로는 달성 불가**. 이 요구사항은 fp16-native WO backend 통합이 끝나야 가능하며, 이번 계획의 범위 밖이다.

이번 계획의 현실적 목표 (정정):

> "torchao `Int4WeightOnlyConfig` / `Int8WeightOnlyConfig` 기반 target-only weight-only 양자화를 SSD에 통합한다. **bf16-native checkpoint**에서 MESA target verify까지 동작하게 한다. fp16 checkpoint는 현 backend로는 지원되지 않으며, 사용자가 명시적으로 bf16 runtime override를 선택할 때만 workaround로 허용한다."

---

## 2. 최종 목표 및 지원 범위 (현재 상태)

이번 작업의 최종 목표는 다음과 같다.

1. SSD에서 **target model만** weight-only 양자화 (INT4 또는 INT8)로 실행할 수 있어야 한다.
2. 구현은 **torchao의 기존 `Int4WeightOnlyConfig` / `Int8WeightOnlyConfig` runtime 경로**를 재사용한다.
3. SSD의 기존 모델 구조는 유지하고, `TP wrapper + scheduler + engine`은 최대한 건드리지 않는다.
4. SSD의 custom TP linear는 유지하되, local weight 상태가 "AQT (AffineQuantizedTensor)" 가 되도록 한다.
5. Llama 계열 target까지 확장 가능한 구조여야 하며, **1차 검증은 `layerskip-llama3-8B`(bf16 native)에서 수행**하고 성공 후 Llama-3.1-70B (bf16 native) 순으로 확장한다.
6. 최종적으로는 **MESA target verify 경로가 quantized target model 위에서 동작**해야 한다.

### 2.0 지원 범위 (2026-04-21 기준, 정직한 구분)

**완성된 지원**:
- bf16-native checkpoint (Llama-3, Llama-3.1 계열)
- torchao `int4_wo_tile`, `int8_wo` backend
- target-only 양자화 (draft 미건드림)
- MESA 포함 target path (async spec, MESA split verify)
- Artifact save/load with strict validation (schema v2, runtime dtype 포함)

**미완성/범위 밖**:
- **fp16 checkpoint를 fp16 runtime으로 양자화 지원** — 현재 선택한 torchao backend가 bf16-activation workflow라 근본적 미지원. `target_quant_force_bf16_runtime=True`로 bf16 runtime 우회는 가능하나 이는 "fp16 runtime 지원"과 다르다. 해결하려면 fp16-native WO backend (Gemlite/Marlin 등) 통합이 필요
- **Artifact를 이용한 startup/peak-memory 최적화** — 현재 load_model이 float weight를 전부 먼저 올린 뒤 artifact를 읽는다. artifact 포맷으로 직접 load하여 float weight load를 건너뛰는 경로는 구현되지 않음. 70B 반복 실험 효율 위해 future work

즉 이 문서의 최종 도착점은 단순한 "quantized target decode"가 아니다.

- normal target decode (AR)
- target verify (speculate)
- MESA split verify

가 모두 같은 INT8 path를 공유하며, **sampling (temp>0) 포함 전 경로에서 NaN/inf 없이 동작**하고, dense 대비 accept rate 가 의미있게 보존되는 상태가 최종 목표다.

### 2.1 모델 진행 순서

1. **`layerskip-llama3-8B` (TP=2)** — 1차 검증. 현재 환경(24 GB RTX 3090 × 8)에서 TP=2로 돌릴 수 있고 layerskip 변형이므로 MESA early-exit도 그대로 확인 가능.
2. **`layerskip-codellama-34B` (TP=4)** — 실제 MESA 실험용 타겟. TP=4 안정화가 주된 검증.
3. **`Llama-3.1-70B` (TP=4 또는 TP=2)** — 최종 stretch goal. 9 GPU → 5 GPU (TP=4 + draft) 또는 3 GPU (TP=2 + draft) 도달 목표.

1단계가 깨지면 2/3단계 진입하지 않는다. 각 단계는 별도의 correctness gate (§Phase 2) 를 모두 통과해야 다음으로 넘어간다.

---

## 3. 이번 계획의 범위

### 3.1 지원 범위

- **target-only**
- **Weight-only quantization** via torchao `Int4WeightOnlyConfig` (기본) 또는 `Int8WeightOnlyConfig`
- **bf16-native checkpoint 완전 지원** (Llama-3 / Llama-3.1 계열). fp16 checkpoint는 §1.3 정책대로 기본 거부, `target_quant_force_bf16_runtime=True` opt-in 시만 bf16 runtime 우회
- **Llama 계열** — 1차 `layerskip-llama3-8B` (bf16), 이후 Llama-3.1-70B (bf16). Llama-2 / CodeLlama (fp16)는 force 플래그로만 가능
- Qwen3는 이번 단계 범위 밖이지만 동일 구조이므로 이후 확장 용이
- **TP 환경 지원**
- **SSD custom linear 내부에서 tensor subclass weight로 local quantized 경로 실행**

### 3.2 이번 단계에서 양자화 대상

이번 단계에서 양자화 대상은 **target 내부의 linear weight 전체**로 본다.

구체적으로는:

- `QKVParallelLinear`
- `RowParallelLinear`
- `MergedColumnParallelLinear`
- `ParallelLMHead`

즉 Llama 기준으로는 다음이 포함된다.

- `self_attn.qkv_proj`
- `self_attn.o_proj`
- `mlp.gate_up_proj`
- `mlp.down_proj`
- `lm_head`

`ReplicatedLinear` (`ssd/ssd/layers/linear.py:42-65`)는 Llama 경로에서 사용되지 않으므로 이번 범위 미포함. hook 순회 시 `LinearBase` 전체를 잡지 말고 **명시적 type 매칭** (`isinstance(m, (QKVParallelLinear, RowParallelLinear, MergedColumnParallelLinear, ColumnParallelLinear, ParallelLMHead))`)으로 제외한다.

#### 3.2.1 `lm_head` 양자화는 flag로 분리

- flag 이름 예: `target_quant_lm_head: bool = True`
- 기본값은 on (양자화 대상에 포함)
- 하지만 MESA proxy 경로에서 early-exit hidden state에 같은 quantized `lm_head`를 적용하면 proxy quality가 final logits보다 더 민감하게 손상될 가능성이 있음
- 따라서 **Phase 4 (MESA 통합)에서 on/off ablation 필수**
- 필요시 MESA 실험 한정 off로 운영

#### 3.2.2 `tie_word_embeddings` 처리 (치명적 주의)

`ssd/ssd/models/llama3.py:333-334`:

```python
if config.tie_word_embeddings:
    self.lm_head.weight.data = self.model.embed_tokens.weight.data
```

즉 tied 모델에서는 `lm_head.weight`와 `embed_tokens.weight`가 **동일 텐서를 공유**한다.

이 상태에서 `lm_head.weight`의 storage를 INT8 quantized 상태로 바꾸면 `embed_tokens.weight`도 같이 바뀌는데, `VocabParallelEmbedding.forward`는 `F.embedding(x, self.weight)`를 호출한다. `F.embedding`은 INT8 weight에 대해 정상 동작하지 않는다 (index-based lookup이 dequant 경로를 타지 않음).

따라서 규칙:

- **load 직후 `hf_config.tie_word_embeddings`를 확인** — 이것이 runtime에서 신뢰할 수 있는 유일한 판정 기준
- **tied = True** 면 다음 중 하나를 강제:
  - (a) `lm_head.weight`를 **새 텐서로 untie**한 뒤 untie된 사본만 양자화 (권장)
  - (b) 또는 `target_quant_lm_head = False`로 자동 강제
- **tied = False** 면 그대로 lm_head 양자화 가능

이 분기는 **Phase 0 spike에서 확인**하고, **Phase 2 구현 시 model_runner의 quantize hook에 반드시 포함**한다.

모델별 tied 상태는 **코드 기준으로 판정**하며, 아래는 참고 예시일 뿐 문서의 보장이 아니다.

- Llama-3.2-1B: tied (일반적으로)
- Llama-3.1-8B / 70B, Llama-2 / CodeLlama, TinyLlama: 보통 not tied

실제 구현은 반드시 runtime의 `hf_config.tie_word_embeddings` 값을 사용한다. 70B 타겟이 현재 tied가 아니더라도 SSD가 범용 코드이므로 반드시 방어 분기를 둔다.

### 3.3 이번 단계에서 양자화하지 않는 것 / 피해야 할 것

다음은 **초기 구현 단계**에서 제외한다.

범위 제외:

- `draft model`
- `embedding` (`VocabParallelEmbedding`, `F.embedding` 호환성 때문에)
- `norm`
- attention / KV cache 자체

방법 제외 (의도적으로 피함):

- 여러 quant backend 동시 지원 (AWQ/SmoothQuant/GPTQ/bitsandbytes를 동시에 올리지 않는다)
- AWQ/SmoothQuant 도입 (Phase 2-AWQ 방향은 심층 디버그 결과 근본 원인이 fp16 overflow로 밝혀져 **철회**. W+A 양자화가 필요 없는 weight-only 상황에선 과도한 방법)
- GPTQ/bitsandbytes 직접 runtime 지원
- 외부 quantized model importer 범용화
- 직접 CUDA/Triton 커널 작성
- MESA를 초기 단계에서 같이 해결 (연산 경로는 공유하지만 검증은 Phase 4로 분리)
- 저장 포맷을 runtime contract보다 먼저 고정 (§4.5)

여기서 `embedding` 제외는 "편의상 부분 양자화"가 아니라, **이번 목표가 linear backend 통합**이기 때문이다.

이번 단계는:

- target 전체 모델 중
- **linear 경로 전체를 INT8 weight-only로 바꾸는 것**

이 목표다.

중요한 구분:

- **최종 목표**에는 MESA target verify 지원이 포함된다
- 하지만 **초기 구현 단계**에서는 MESA correctness / 성능 검증을 뒤로 미룬다

즉 MESA는 범위 밖이 아니라, **Phase 4로 미룬 최종 목표**다.

---

## 4. 핵심 결정

### 4.1 backend는 하나만 본다

이번 구현은 `torchao` 계열의 INT8 weight-only 경로를 기준으로 잡는다.

우선순위는 다음과 같다.

1. `torchao.quantization.Int8WeightOnlyConfig`
2. 필요시 `IntxWeightOnlyConfig(weight_dtype=torch.int8)` 검토

중요한 건, 이번 단계는 **backend를 여러 개 추상화하지 않는다**는 점이다.

이번 문서에서 backend라고 하면 우선 `torchao INT8 weight-only linear`를 뜻한다.

#### 4.1.1 tested version pin

torchao는 minor version마다 API가 변할 수 있다. backend 하나로 고정하는 전략이므로 버전 고정이 더 중요하다.

고정 조합 (브랜치 `feature/int8-weight-only` 기준):

- `torch`: 2.8.0+cu128
- `torchao`: **0.12.0** (0.17.0은 torch 2.8에서 cpp extension 호환 경고 발생 → 0.12로 downgrade)
- `cuda`: 12.8
- GPU: RTX 3090 (sm_86)

이 조합으로 Phase 0 feasibility spike와 이후 Phase 전체를 수행한다.

#### 4.1.2 fp16 checkpoint × quant backend 호환성 (정정 2026-04-21)

**이전 문서는 "fp16 overflow가 원인이라 bf16 upcast가 해결책"이라고 적었으나 이는 오진단이었다**. 실제 측정:

- dense fp16 Llama-2-7B async spec: layer 1 `finite_absmax=1597` (fp16 max 65504의 1/40) — overflow 아님
- int4 WO + fp16 no-upcast: `ValueError: Expected zeros fp16, got bf16` (torchao API dtype assert)
- int8 WO + fp16 no-upcast: layer 1 `inf=22 finite_absmax=440` (수치 불안정)

**실체**: 우리가 선택한 torchao weight-only backend (`Int4WeightOnlyConfig`, `Int8WeightOnlyConfig`)는 **공식 문서가 bf16 activation workflow로 명시한 경로**다 (https://docs.pytorch.org/ao/stable/workflows/inference.html). 이 backend는:

- Int4 tile_packed: scale/zero를 bf16으로 고정 생성. fp16 activation과 matmul 시 dtype assert fail
- Int8 WO: API는 통과하지만 fp16 runtime에서 수치적으로 불안정 (원인 세부는 미확정 — fp16 accum 가설 수준)

**torchao 전반이 fp16 미지원은 아님**. `GemliteUIntXWeightOnlyConfig`는 fp16 전용 backend로 문서화되어 있고, Marlin/exllamav2 류 kernel도 fp16을 지원함. 하지만 현 스택에 통합되어 있지 않다.

**정책 (2026-04-21)**:

| checkpoint dtype | 선택 backend | 동작 |
|---|---|---|
| bf16 (Llama-3/3.1) | int4_wo_tile / int8_wo | 정상 지원 |
| fp16 (Llama-2, CodeLlama) | int4_wo_tile / int8_wo, default | **`ValueError` at init** (unsupported combination surfaced loudly) |
| fp16 + `target_quant_force_bf16_runtime=True` | int4_wo_tile / int8_wo | **bf16 runtime 우회 (workaround)** — fp16 체크포인트를 bf16 runtime에서 돌림. "fp16 runtime 지원"과 **다르다**는 점 명시 |
| fp16 | Gemlite / Marlin / 기타 fp16-native WO | 미통합 (future work) |

**bf16 runtime 우회 경로 (`target_quant_force_bf16_runtime=True`)의 의미**:
- `config.hf_config.torch_dtype`을 fp16 → bf16으로 교체
- Helper 전부 (`FlashInfer` plan q/kv dtype, MESA `exit_hidden/residual/outputs`, draft 버퍼)가 bf16 사용
- activation / KV cache / graph buffer 모두 bf16
- 즉 **"weight만 quant이고 나머지는 fp16"이 아님** — runtime 전체를 bf16으로 바꾸는 선택
- Dense fp16 vs (fp16ckpt→bf16runtime + int4) 비교할 때 **dtype 변수가 섞임**. 공정 비교 필요하면 dense에도 bf16 강제 옵션을 둬야 함 (현재는 별도 flag 없음)

**구현**:

```python
# model_runner.py __init__ 초반
if target_quant_enabled and hf_config.torch_dtype == float16:
    if target_quant_force_bf16_runtime:
        # 사용자가 명시적 opt-in: bf16 runtime으로 override
        config.hf_config.torch_dtype = torch.bfloat16
    else:
        # 기본: loud failure — 사용자가 무슨 조합인지 알고 선택하게 함
        raise ValueError("fp16 checkpoint + torchao WO backend not supported; ...")
```

이전 자동 upcast 동작은 **"지원 범위를 속이는" 것**이었으며 제거되었다.

### 4.2 SSD 모델 구조는 유지한다

다음은 유지한다.

- `LlamaModel`
- `LlamaAttention`
- `LlamaMLP`
- scheduler / engine / sequence contract
- target/draft 프로세스 구조

즉 모델 구조를 `nn.Linear` 기반의 다른 모델로 바꾸지 않는다.

### 4.3 SSD custom TP linear는 유지하고 local weight 상태만 교체한다

다음 레이어는 그대로 유지한다.

- `ColumnParallelLinear`
- `RowParallelLinear`
- `QKVParallelLinear`
- `MergedColumnParallelLinear`
- `ParallelLMHead`

**기본 가정**: forward 코드는 가능한 변경하지 않는다. SSD의 모든 TP linear는 이미 `F.linear(x, self.weight, self.bias)` 형태로 동작한다 (`ssd/ssd/layers/linear.py:65, 98, 196` 및 `embed_head.py:88, 95, 111`). torchao `Int8WeightOnlyConfig`는 tensor subclass (`AffineQuantizedTensor`) 방식으로 동작하므로, `self.weight` 위치의 값이 quantized 상태이면 기존 `F.linear` 호출이 `__torch_dispatch__`를 통해 int8 matmul 커널로 라우팅될 것으로 기대한다.

**저장 계약: Phase 0 결과로 (A) 확정**. 다음과 같이 동작한다.

1. Quantize 시점에 `dummy = nn.Linear(in, out, bias=False).cuda().to(dtype)`를 만들고 float weight를 `dummy.weight`에 복사
2. `quantize_(dummy, Int8WeightOnlyConfig())` 적용 — 이 시점에 `dummy.weight`는 `nn.Parameter`를 유지한 채 내부 `data`가 `AffineQuantizedTensor`로 교체된다
3. SSD TP 모듈에 `self.weight = dummy.weight`로 재할당 — `nn.Parameter` 타입이므로 parameter slot 규칙과 충돌 없음
4. SSD의 기존 `F.linear(x, self.weight, self.bias)` 호출이 `__torch_dispatch__`를 통해 INT8 커널로 자동 라우팅

검증 결과 (`sandbox/int8_spike/` 참조):

- (A) assignment 경로 — toy, qkv_packed, gate_up_packed, o_proj_row, down_proj_row 전 크기에서 성공, dense 대비 cosine ≥ 0.9999
- CUDA graph capture + replay 10회 — numerical diff 0.0
- `@torch.inference_mode()` 하에서 정상 동작 — diff 0.0
- `@torch.compile` 모듈 공존 — 정상
- graph capture under `inference_mode` (SSD 실제 패턴) — diff 0.0

Forward 코드 변경은 불필요. `ParallelLMHead`의 gather 경로 등도 `F.linear(x, self.weight)` 호출 자체는 그대로이므로 수정 없이 동작할 것으로 기대 (Phase 2 회귀 테스트로 확인).

과거 초안의 (B) register_buffer, (C) 별도 attribute, (D) wrapper parameter 경로는 (A)로 해결되었으므로 사용하지 않는다.

### 4.4 처음부터 직접 커널을 작성하지 않는다

이번 계획은:

- custom CUDA kernel 작성
- custom Triton kernel 작성

을 목표로 하지 않는다.

이번 단계의 목적은 **기존 quantized linear 실행 경로를 SSD에 붙이는 것**이다.

커널 직접 작성은 이 단계가 실패하거나, 성능이 충분하지 않을 때 다음 선택지로 본다.

### 4.5 저장 포맷 설계는 runtime contract 이후로 미룬다

이전 계획에서 가장 과했던 부분이 여기다.

이번에는 먼저 아래를 확정한다.

- SSD custom linear가 실제로 어떤 quantized tensor subclass를 들고 있을지
- weight_loader / state_dict 호환이 어떻게 되는지
- local forward가 정확히 어떤 호출로 실행될지 (사실상 `F.linear(x, self.weight)` 그대로)

그 다음에야 저장 포맷을 결정한다.

즉 이번 순서는:

1. runtime contract 확정 (Phase 0~1)
2. local quantized weight 교체 통합 (Phase 2)
3. 그 후 checkpoint/export 포맷 확정 (Phase 5)

이다.

---

## 5. 이번 단계에서 해결해야 하는 기술 문제

### 5.1 torchao tensor subclass dispatch가 SSD `F.linear` 호출에서 동작하는가

핵심 질문:

> SSD의 TP linear는 이미 `F.linear(x, self.weight, bias)`를 호출한다. `self.weight`를 torchao `AffineQuantizedTensor`로 교체했을 때, 이 `F.linear` 호출이 `__torch_dispatch__`를 타고 INT8 커널로 정상 라우팅되는가?

확인 포인트:

- torchao `quantize_()`는 문서상 `nn.Linear` 모듈을 walker로 탐색해 변환한다. SSD custom linear는 `nn.Linear` 서브클래스가 아니므로 자동 탐색 대상이 아닐 가능성이 높다. 따라서 **수동으로 `self.weight`를 subclass로 교체**하는 경로가 유력 후보다 (단, Phase 0에서 최종 확정).
- 교체 이후 `F.linear(x, self.weight, bias)`가 dense 결과 대비 numerical sanity를 만족하는지
- packed weight (`QKVParallelLinear` q/k/v 합친 형태, `MergedColumnParallelLinear` gate/up 합친 형태)는 **local 관점에서 단순히 output dim이 더 큰 하나의 matrix**이다. per-output-channel 양자화는 output row별 독립이므로 packed 여부와 무관하게 동일한 방식이 적용된다. 별도 특수 처리 불필요.

### 5.2 CudaGraph / torch.compile / inference_mode와 양립 가능한가

SSD는 hot path에서 CudaGraph를 많이 쓴다. decode / verify / MESA verify 경로가 전부 graph이므로, quantized linear path가 graph 경로와 호환되지 않으면 이번 계획 전체가 흔들린다.

특히 torchao tensor subclass의 `__torch_dispatch__`는 Python 레벨에서 동작한다. 이것이 CUDA graph capture/replay와 호환되는지가 **가장 큰 기술 리스크**다.

또한 SSD 내부에는 tensor subclass dispatch와 공존해야 하는 다른 런타임 장치가 있다:

- `@torch.compile`이 적용된 모듈 (`ssd/ssd/layers/layernorm.py`, `activation.py`, `rotary_embedding.py`)
- `@torch.inference_mode()`로 감싼 전체 forward (`ssd/ssd/engine/model_runner.py:645` 등)

quantized linear가 같은 forward pass 안에서 compiled norm / rotary와 공존할 때, 그리고 inference_mode 아래에서 `__torch_dispatch__`가 정상 동작하는지는 Phase 0에서 확인해야 한다.

따라서 quantized linear path가:

- capture-safe인지
- replay-safe인지
- hidden dynamic allocation이 없는지 (예: dequant 중간 버퍼가 caching allocator를 타는지)
- 첫 호출 시 triton/tinygemm autotune이 capture 밖에서 끝나는지
- `@torch.compile` 모듈과 같은 forward 안에서 dispatch 충돌이 없는지
- `@torch.inference_mode()` 아래에서 subclass dispatch가 정상 동작하는지

를 확인해야 한다.

**이 리스크는 가장 크다. 따라서 Phase 0 종료 조건에 1차 gate로 포함한다** (§6 Phase 0 참조).

이 항목이 초기에 막히면 다음 순서로 대응한다.

1. 먼저 eager에서 correctness 확인
2. 그 다음 작은 isolated `F.linear` + quantized weight로 graph capture/replay
3. 그 다음 SSD decode graph
4. 그 다음 SSD verify graph
5. 마지막에 speculate / MESA verify 검토

### 5.3 TP wrapper와 결합 시 collective semantics를 유지하는가

`RowParallelLinear`는 local matmul 뒤 `all_reduce`가 필요하다.

`ColumnParallelLinear`는 shard output만 반환한다.

`ParallelLMHead`는 local logits 뒤 gather가 필요하다.

즉 local weight 상태만 바꾸더라도:

- local output shape
- bias 처리
- TP collective 시점

은 기존과 완전히 동일해야 한다.

§4.3 (A) 경로로 forward를 그대로 둘 수 있다면 이 계약은 기본적으로 보존되지만, (B)~(D) 경로를 택하게 되면 아주 작은 보정이 필요할 수 있으므로 **Phase 2 회귀 테스트로 반드시 검증**한다.

### 5.4 scale granularity가 TP shard 경계와 충돌하지 않는가

torchao `Int8WeightOnlyConfig`는 **per-channel (output channel, dim=0) 양자화**를 사용한다 (공식 문서 기준).

SSD TP shard 방향 대비:

- `ColumnParallelLinear` / `QKVParallelLinear` / `MergedColumnParallelLinear`: weight shape `[out/tp, in]`, tp_dim=0 (output)
  - shard 경계가 output 축 → scale 축과 동일 방향
  - 각 rank가 자기 output channel에 대한 독립 scale을 가짐
  - local quantize가 global quantize와 수치적으로 동등 ✓
- `RowParallelLinear`: weight shape `[out, in/tp]`, tp_dim=1 (input)
  - shard 경계가 input 축 → scale 축(output)과 직교
  - scale이 output별이므로 local matmul 뒤 `all_reduce`로 partial 합산해도 문제 없음
  - local vs global scale 비교:
    - global scale: `s[i] = max(|W[i, :]|)/127`
    - local scale (per rank): `s_r[i] = max(|W[i, shard_r]|)/127 ≤ s[i]`
    - 즉 local scale이 **항상 더 타이트**
  - 결과적으로 per-rank local quantize는 사실상 **output channel × TP group**의 2D finer-grained quantization과 동등하며, per-output-channel quantization보다 **수치 정밀도가 동등하거나 더 높다**
  - 즉 local quantize가 **유리**하거나 최소한 손해는 아니다

실무 원칙:

- **기본: per-rank local quantize** (자연스러움, 정밀도 손해 없음, 구현 간단)
- pre-quantized checkpoint를 **다른 TP size로 재사용**하려는 시나리오에서만 문제가 될 수 있는데, Phase 5 artifact에 `tp_size`를 명시적으로 저장하므로 이 시나리오는 차단됨 (§Phase 5)
- 만약 Phase 2 correctness gate에서 RowParallel로 인한 실용적 정확도 문제가 드러나면 그때 rank 0 global quantize + broadcast 전략 검토. 아닐 경우 추가 조치 불필요

### 5.5 weight tying 방어 확인

§3.2.2에서 설명한 `tie_word_embeddings = True` 케이스 방어가 구현되었는지를 Phase 0 spike에서 확인하고 Phase 2 hook에 반영해야 한다 (세부는 §3.2.2).

### 5.6 fp16 overflow 이슈 (2026-04-21 root-cause 확정)

초기 Phase 2에서 관찰된 inf → NaN → multinomial assert 크래시의 **실제 원인은 fp16 dynamic range overflow**였다.

- 현상: 실제 prompt prefill 시 layer 1 MLP 출력에서 극소수 위치(예: 338×4096 중 8 곳)에 `inf`
- 재현 모델: `layerskip-llama2-7B`, `layerskip-codellama-34B` — **둘 다 fp16 원본 모델**
- 모드 무관: `--eager`, `TORCHDYNAMO_DISABLE=1`, sync/async spec 전부 동일
- **root cause**: Llama MLP 중간 활성화 (`silu(gate) * up`)가 실제 prompt에서 absmax ~1000까지 도달. 그 값을 down_proj (input 5504 ch) 으로 넘겨 matmul하면 fp32 accumulator엔 담기지만, 최종 fp16 cast 시 65504 초과 → inf
- 초기 오진단: "Llama outlier + AWQ/SmoothQuant 필요"로 해석했으나, 이는 **weight+activation 양자화 상황의 개념**이고 pure weight-only + fp16 overflow 문제와는 다른 병증이었다
- **해결**: fp16 모델을 load-time에 bf16으로 upcast (§4.1.2). bf16은 range 3.4e38이라 여유 있음
- 검증: Llama-2-7B fp16 + upcast + int8 + async spec sampling → TP=15.31, accept=0.44 (crash 없이 완주)
- Llama-3 / Llama-3.1 계열은 원래 bf16이라 upcast 자체 불필요

---

## 6. 단계별 구현 계획

실제 작업 순서:

| Phase | 상태 | 요약 |
|---|---|---|
| 0 | ✅ DONE | backend feasibility + graph-safety + tying spike |
| 1 | ✅ DONE | 저장 계약 (A) 확정 (§4.3) |
| 2 | ✅ DONE | Plain INT8 integration + fp16→bf16 upcast. Llama-3-8B 및 Llama-2-7B(upcast) 전 시나리오 통과 |
| **2.5** | **✅ DONE** | **INT4 tile_packed (torchao TensorCoreTiledLayout) 전환. INT8 대비 2.7x 빠름, async spec int4가 dense보다도 18% 빠름. MESA는 lm_head dense 유지 권장 (현 default).** |
| 3 | ✅ DONE | graph path 확장 검증. graph는 eager 대비 2-4x, accept 동일, INT4 graph-safe |
| 4 | ✅ DONE | MESA + lm_head ablation. lm_head dense 유지 권장 → config default `target_quant_lm_head=False` |
| 5 | ✅ DONE | persistent artifact (save/load AQT per rank). 8B 기준 wall 63s→44s |
| 6 | ✅ DONE | **CodeLlama-34B TP=4 INT4 async spec: TP=23.52, accept=0.31, 9→5 GPU 축소 성공**. MESA 34B는 torchao dispatch 이슈로 async spec만 지원. 70B는 로컬 모델 없어 미테스트 |

원칙:

- 저장 포맷을 먼저 고정하지 않는다
- MESA를 초기 단계에 같이 붙이지 않는다
- custom kernel부터 만들지 않는다
- graph-safety 검증을 뒤로 미루지 않는다 (Phase 0 1차 gate 완료)
- Model progression: `layerskip-llama3-8B` → `CodeLlama-34B` → `Llama-3.1-70B`, 각 단계 gate 통과 후 다음 진입

### Phase 0. 사전 검증: backend feasibility + graph-safety + tying spike  **[COMPLETED 2026-04-20]**

> 결과: 전 항목 통과. 상세는 `INT8-IMPL-ISSUE.md` 및 `sandbox/int8_spike/`.
> - `quantize_` walker는 `nn.Linear`만 변환 (SSD custom은 수동 처리 필요)
> - 저장 계약 **(A)** 채택: `dummy.weight` → `self.weight` 재할당, forward 코드 변경 없음
> - CUDA graph capture + 10회 replay numerical diff = 0
> - `@torch.inference_mode()`, `@torch.compile` 공존 모두 diff 0
> - scale granularity 확인: Column/QKV/Merged 완전 동등, Row는 local이 global 대비 finer-grained (유리)
> - tying 방어: `quantize_()`가 attribute 교체 방식이라 원본 float 저장소 보존됨

#### 목표

`torchao INT8 weight-only`를 SSD custom TP linear에 붙일 수 있는지, 그 경로가 CUDA graph와 양립하는지, weight tying 방어가 성립하는지 **1차 gate로** 확인한다.

현재 공식 torchao 문서 기준 `quantize_()`는 `nn.Linear` 모듈을 탐색해 변환한다. SSD custom linear는 `nn.Linear`가 아니므로 이 자동 경로는 대상이 아닐 가능성이 높다. 따라서 이 spike는 "quantize_ 데모"가 아니라, **SSD가 쓰는 호출 형태에서 경로가 동작하고 graph-safe한지**를 검증한다.

#### 산출물

- 작은 isolated experiment (SSD 코드와 분리된 sandbox)
- 결론:
  - `F.linear` + tensor subclass weight 경로 feasibility (가능 / 불가능)
  - graph capture/replay feasibility
  - scale granularity와 shard 경계 호환 여부의 초기 감
  - weight tying 방어 경로 성립 여부
  - 막히는 경우 어느 layer / 어느 호출에서 막히는지 명확한 기록

#### 확인 항목

1. **quantize_ 적용 경로 확인**
   - `torchao.quantization.quantize_`가 실제로 `nn.Linear`만 변환하는지 1차 확인
   - SSD custom linear에는 그대로 먹지 않는다는 가정이 맞는지 확인

2. **quantized weight 교체 경로 확인 + 저장 계약 선택**
   - 작은 weight tensor를 INT8 weight-only로 quantize한 뒤 여러 저장 계약 후보를 검증한다 (§4.3 (A)~(D))
   - 구체 API는 Phase 0에서 확정하되, **internal API 직접 호출 (`to_affine_quantized_intx` 등)보다 더 안정적인 경로를 우선 검토**한다. 예:
     ```python
     # 안정 후보: dummy nn.Linear에 quantize_() 적용 후 weight만 뺀다
     dummy = nn.Linear(in_f, out_f, bias=False)
     dummy.weight = nn.Parameter(original.data)
     quantize_(dummy, Int8WeightOnlyConfig())
     module.weight = dummy.weight   # tensor subclass가 담긴 Parameter
     ```
     - 이 경로가 `nn.Module` parameter slot 규칙과 충돌하지 않는지 먼저 시도 (§4.3 (A))
     - 충돌 시 (B) buffer / (C) 별도 attr / (D) wrapper parameter 후보로 내려간다
   - **기존 `F.linear(x, self.weight, bias)` 호출이 `__torch_dispatch__`를 통해 int8 경로로 라우팅되는지 확인**
   - 결과 dtype / shape / numerical sanity (dense 결과와의 max abs diff, cosine 유사도)

3. **local shard 크기 테스트**
   - 작은 toy 크기뿐 아니라, SSD에서 실제 쓰이는 local shard 크기로도 같은 경로가 도는지 확인
   - 예: `[11008, 4096]`, `[4096 // TP, 4096]` 급 packed/shard 크기
   - packed weight (qkv 합친 형태, gate_up 합친 형태) 모사해서 확인 (단순히 output dim이 큰 matrix임)

4. **CUDA graph capture/replay 1차 검증** (**가장 큰 리스크**)
   - 2번 경로를 **warmup 후** (triton autotune 종료 보장)
   - `torch.cuda.graph()`로 capture
   - replay 여러 번 (최소 10회 이상)
   - 결과가 non-graph 경로와 동일한지 확인
   - hidden dynamic allocation, `__torch_dispatch__` Python path가 capture 내부에서 실패하지 않는지 확인

5. **scale granularity 확인**
   - 양자화 후 scale tensor의 shape과 축 확인 (per-channel dim=0 예상)
   - ColumnParallel 계열(tp_dim=0)과 shard 충돌 없음 확인
   - RowParallel 계열(tp_dim=1)에서 local quantize scale ≠ global quantize scale 차이를 **수치적으로 관찰만** 해 둠 (Phase 2 gate에서 정량 판정)

6. **weight tying 방어 경로 확인**
   - 두 텐서가 `.data`를 공유하는 미니 모델을 만든 뒤 한쪽만 양자화 시도
   - 의도대로 untie가 성립하는지 (또는 untie 없이 교체하면 embed 쪽이 망가지는지) 재현
   - model_runner hook에서 쓸 방어 pseudo-code 확정

7. **weight_loader 호환 1차 스케치**
   - `param.data.copy_()`, `param.narrow()`, `param.size(dim)` 호출이 tensor subclass 위에서 어떻게 동작하는지 확인
   - **실무 전략**: loader는 float weight 그대로 동작 → load 완료 후 hook에서 quantized state로 교체 → subclass(또는 §4.3 (B)~(D) 저장 상태)는 loader 이후 단계에만 등장
   - 이 순서로 호환성 확보 가능한지 Phase 0에서 검증

8. **`@torch.compile` / `@torch.inference_mode()` 공존 확인**
   - `ssd/ssd/layers/layernorm.py` 등의 `@torch.compile` 모듈과 같은 forward pass 안에서 subclass dispatch가 충돌하지 않는지
   - `ssd/ssd/engine/model_runner.py:645`의 `@torch.inference_mode()` 아래에서 `__torch_dispatch__`가 정상 동작하는지
   - 두 데코레이터 사용 패턴이 SSD 전체 hot path에 걸쳐 있으므로 graph capture 전에 반드시 1차 확인

#### 종료 조건

다음 모두가 분명해져야 한다.

- **feasibility**: 기존 `F.linear` 호출이 quantized state로 int8 경로를 탐 (또는 확실히 안 됨)
- **저장 계약 선택**: §4.3 (A)~(D) 중 어느 것으로 갈지 결정, 그 결정을 §4.3에 갱신
- **graph-safety 1차 통과**: isolated `F.linear` + quantized weight 조합이 capture/replay 가능하며 결과가 동일함 (또는 실패 원인 명확)
- **compile/inference_mode 공존**: 두 데코레이터 아래에서 subclass dispatch가 깨지지 않음 확인
- **scale/shard 관찰**: scale 축과 TP shard 방향 관계 확인, RowParallel에서 local이 global 대비 정밀도 동등 이상임을 경험적으로 재확인
- **tying 방어**: 방어 pseudo-code로 tied 모델에서 embed가 망가지지 않음 확인
- **loader 순서**: float load → 이후 quantize 순서가 기존 weight_loader와 충돌하지 않음 확인

#### 실패 시 대응

- `F.linear` 경로 자체가 안 된다면: 이번 계획 전체를 다시 세워야 한다
- graph-safety가 안 된다면: quantized target을 **eager-only 경로**로 먼저 운영하고 graph 지원은 별도 과제로 분리할지 판단 (SSD hot path 수정 범위가 커지므로 전체 가치 재평가)
- scale granularity가 RowParallel에서 실용적으로 깨진다면: Phase 2에서 (a)안(rank 0 global quantize + broadcast) 고려

즉 Phase 0은 필수 gate이며, 여기서 나온 결과에 따라 이후 Phase의 전제가 결정된다.

---

### Phase 1. Weight replacement contract 확정

#### 목표

Phase 0에서 확인된 tensor subclass 방식을 **SSD 모든 TP linear에 일관되게 적용**하기 위한 최소 계약을 확정한다.

§4.3에서 명시한 대로 **forward 코드는 건드리지 않는다**. 이 Phase의 범위는 weight 교체 시점과 계약이다.

#### 결정 항목

1. **교체 시점**
   - load_model 완료 후, warmup 이전 (구체 위치는 Phase 2 §3.5)
   - model_runner.py 안에 단일 hook 함수

2. **교체 대상 선정**
   - 순회 기준: `target_quant_enabled` + module type 매칭 (`ColumnParallelLinear`/`RowParallelLinear`/`QKVParallelLinear`/`MergedColumnParallelLinear`/`ParallelLMHead`)
   - `VocabParallelEmbedding`는 순회 대상 제외
   - `ParallelLMHead`는 `target_quant_lm_head`와 `tie_word_embeddings` 두 조건을 모두 확인

3. **교체 방식**
   - Phase 0에서 선택된 저장 계약 (§4.3 (A)~(D)) + 안정 API 경로 (dummy `nn.Linear` + `quantize_()` 등)에 맞춰 확정
   - `nn.Parameter` wrapping 여부는 Phase 0 결과에 따라 결정
   - `weight_loader` attribute는 이미 float 시점에 호출 완료되므로 이후 보존 불필요

4. **불변 조건 (기본 가정)**
   - forward 코드 변경은 최소화
   - TP collective 코드 변경 없음
   - bias dtype / shape 변경 없음
   - `ParallelLMHead` 등 forward 뒤 gather 로직이 있는 모듈에서 아주 작은 보정이 필요할 수 있음 (§4.3)

#### 종료 조건

- 위 4가지가 한 곳(예: `INT8-RUNTIME-CONTRACT.md` 또는 본 문서 §4.3 세부)에 고정 기록됨
- Phase 2 구현이 이 계약만 따르면 되는 상태

---

### Phase 2.5. Kernel path 최적화 — INT4 tile_packed 전환  **[DONE 2026-04-21]**

#### 배경

Phase 2 INT8 통합은 정확도/기능은 완벽했으나 throughput이 느림:
- AR int8: 9.48 TP (dense 24.84, 38%)
- async spec int8: 14.25 TP (dense 15.84, 90%)
- MESA int8: 12.15 TP (dense 24.12, 50%)

Microbench로 원인 확정: `torchao.Int8WeightOnlyConfig`는 SM 86에서 **fused INT8 kernel 없음** — "dequant + bf16 matmul" 경로라 decode/verify에서 3x 느려짐. torchao/torch 업그레이드해도 동일.

#### 해법

**`Int4WeightOnlyConfig(group_size=128)`** 전환. 기본 `TensorCoreTiledLayout`이 tinygemm 경로 활성화 → SM 86에서 실질적 fast kernel. 원래 `ssd` env (torch 2.8 + torchao 0.12) 그대로 사용 (업그레이드 불필요).

#### 실측 결과 (Llama-3-8B, TP=2 + draft)

| config | TP | accept |
|---|---|---|
| AR dense | 24.84 | - |
| AR INT8 | 9.48 | - |
| async spec dense | 15.84 | 0.32 |
| async spec INT8 | 14.25 | 0.30 |
| **async spec INT4** | **18.64** | **0.30** (dense보다 18% 빠름) |
| MESA dense | 24.12 | 0.41 |
| MESA INT8 | 12.15 | 0.40 |
| MESA INT4 (lm_head on) | 13.15 | 0.33 |
| **MESA INT4 (lm_head off)** | **13.47** | **0.38** |

Llama-2-7B (fp16→bf16 upcast) 재확인:

| config | TP | accept |
|---|---|---|
| async spec INT8 | 15.31 | 0.44 |
| **async spec INT4** | **41.85** | **0.35** (INT8 대비 2.7x) |

#### 핵심 관찰

1. **INT4가 INT8보다 모든 면에서 우월** (SM 86 기준): 속도 2.7x, 메모리 4x 절감, accept 보존
2. **MESA에선 `target_quant_lm_head=False` 권장**: lm_head bf16 유지 시 accept 0.33→0.38. early-exit logit 정밀도 이슈
3. **Prefill은 느려지지만 one-shot** — verify/decode 가중 평균에서 INT4 우세
4. accept 손실은 async spec에선 거의 0, MESA에선 dense 대비 ~7% (lm_head off 기준)

#### 구현 변경

- `ssd/config.py`: `target_quant_backend: str = "int4_wo_tile"` (기본값), `"int8_wo"` 남김
- `ssd/utils/quantize.py`: `_quantize_weight_to_int4_wo()` 추가, `apply_quantization_to_target(backend=...)` 확장 (이전 이름 `apply_int8_weight_only_to_target`)
- `bench/bench.py`: `--quant_int4` / `--quant_int8` flag 분리

#### 제약

- `input_dim` 이 `group_size=128`의 배수여야 함 (모든 Llama TP=2/4 shard 충족 ✓)
- fp16 모델은 여전히 load-time bf16 upcast 필요 (§4.1.2)

---

### Phase 2. Plain INT8 eager runtime 통합  **[DONE 2026-04-21]**

> 결과: **통합 코드 + fp16 upcast로 모든 검증 통과**.

구현 산출물:
- `ssd/ssd/config.py`: `target_quant_enabled`, `target_quant_lm_head`, `target_quant_backend`, `target_quant_mode`
- `ssd/ssd/utils/quantize.py`: `apply_quantization_to_target()` — Phase 0 계약 (A), tying 방어, `SSD_QUANT_SKIP` 환경변수 (`SSD_INT8_SKIP` legacy alias)
- `ssd/ssd/engine/model_runner.py`: `load_model` 직후 hook 삽입 (draft 제외) + **fp16→bf16 upcast 분기** (§4.1.2)
- `bench/bench.py`: `--quant_int4` / `--quant_int8`, `--quant_lm_head` (opt-in), `--quant_force_bf16_runtime`, `--quant_artifact`, `--quant_artifact_load_only`. `--no_quant_lm_head`는 deprecated no-op
- 디버그 모듈: `ssd/ssd/utils/int8_debug.py` (env `SSD_INT8_DEBUG=1`로 H1/H2/H3 진단)

#### 검증 결과 (2026-04-21)

**`layerskip-llama3-8B` (bf16 native, TP=2+draft)** :

| test | TP | accept |
|---|---|---|
| AR dense | 24.84 | n/a |
| AR int8 | 9.48 | n/a |
| async spec dense sampling | 15.84 | 0.32 |
| **async spec int8 sampling** | **14.25** | **0.30** ✓ |
| MESA dense sampling | 24.12 | 0.41 |
| **MESA int8 sampling** | **12.15** | **0.40** ✓ |

**`layerskip-llama2-7B` (fp16 원본 → bf16 upcast, TP=2+draft)**:

- async spec + int8 + sampling: **TP=15.31, accept=0.44** ✓

즉 fp16 overflow가 유일한 blocker였고, upcast로 완전 해소.

#### 실제로 수행한 sub-op debug 결과 (§5.6 근거)

- layer 1 sub-op 추적에서 `silu(gate)*up` absmax = **997.5**, down_proj 이후 8 inf 등장
- fp16 max = 65504로 accumulation overflow. outlier 채널 문제가 아니라 dynamic range 부족이 root cause
- bf16으로 올리면 absmax 997.5 × weight × 5504 ≈ 7.1M, 3.4e38에 비하면 여유

#### 종료 조건 (전부 충족)

1. Dense flag-off 회귀: Llama-2/Llama-3 둘 다 기존 throughput/accept 변동 없음 ✓
2. AR int8: Llama-3-8B, CodeLlama-34B 완주 ✓
3. Greedy spec + int8: Llama-3-8B 완주
4. Sampling spec + int8: Llama-2-7B (fp16+upcast), Llama-3-8B 둘 다 완주 ✓
5. MESA + int8 + sampling: Llama-3-8B 완주, accept=0.40 (dense 0.41 대비 보존) ✓
6. TP correctness 및 tying 방어: Phase 0 단계에서 확인됨

#### 34B 관련 주의

- `layerskip-codellama-34B` (fp16 원본) + bf16 upcast → load peak 17 GB/rank (TP=4). 24 GB RTX에서 다른 job 공존 시 OOM
- 해결: exclusive GPU 확보 또는 80 GB A100. 본 계획의 기능적 완성도에는 영향 없음

---

### Phase 3. Graph path 확장 검증

#### 목표

Phase 0에서 **isolated** `F.linear` + quantized weight 조합으로 graph-safety 1차 gate는 통과한 상태다. 이 단계는 그 검증을 **SSD 전체 graph 경로**에서 확장해 동작을 확인한다.

Phase 2에서 eager/async spec이 이미 정상 동작 확인되었으므로 graph 경로도 큰 이슈 없이 작동할 것이 예상되나, 다음을 명시적으로 측정한다.

#### 순서

1. normal decode graph (AR)
2. verify graph (speculate)
3. target-only speculate verify graph

이 단계에서는 MESA를 아직 직접 검증하지 않는다 (Phase 4).

#### 확인 항목

- SSD 실제 graph 크기/입력 프로파일에서 capture 시 예외 여부
- 실제 운영 batch shape에서 replay 시 예외 여부
- hidden dynamic allocation 여부 (caching allocator 확인 포함)
- multi-replay 시 output stability (동일 입력에 동일 출력)
- dense graph 경로와의 latency 비교 (memory 절감 대비 속도 regression 정도 기록)

#### 실패 시 대응

- 우선 target-only quantized eager path 유지
- Phase 0에서는 통과했는데 여기서 실패하는 경우 → SSD hot path 고유 요인(입력 shape 다양성, stream 구조, capture 순서 등) 원인 기록

#### 종료 조건

- target decode / verify path 전체에서 graph capture + replay 안정 동작
- dense graph 대비 output 일관성 (Phase 2 correctness gate와 동일 기준 재적용)
- `layerskip-llama3-8B` (TP=2)에서 완주

---

### Phase 4. MESA target verify 통합 및 기능 검증

#### 목표

이미 안정화된 target-only quantized path를 **MESA target verify 경로**에 연결하고, 최소한 기능적으로 동작하는지 확인한다.

이 단계의 범위는 명확하다.

- quantized **target** model이
- `run_mesa_verify_cudagraph(...)`
- early-exit logits
- proxy 계산

까지 포함한 MESA target 경로에서 동작하는지 확인하는 것이다.

draft는 여전히 dense로 둔다.

#### 왜 이 단계를 별도로 두는가

MESA는 target model을 공유하므로, 연산 경로 자체는 target quantized path를 따라갈 가능성이 높다.

하지만 다음은 별도 검증이 필요하다.

- split verify capture/replay 안정성
- early-exit logits 변화
- proxy quality 변화 (특히 quantized `lm_head` 사용 시)
- acceptance / cache hit 변화

#### 확인 항목

1. quantized target으로 `run_mesa_verify_cudagraph(...)` capture 가능 여부
2. replay 가능 여부
3. early-exit logits shape / dtype / numerical sanity
4. proxy tensor 생성 정상 여부
5. dense baseline 대비 acceptance / reject 분포 변화

#### `lm_head` 양자화 ablation (필수)

§3.2.1에서 도입한 `target_quant_lm_head` flag에 대해 **이 Phase에서 반드시 on/off 양쪽을 비교**한다.

비교 항목:

- lm_head on 상태의 MESA acceptance / cache hit / throughput
- lm_head off 상태의 MESA acceptance / cache hit / throughput
- dense baseline MESA와의 상대 차이

판단:

- lm_head off 대비 on에서 acceptance가 유의미하게 떨어지면, MESA 실험 한정 off 운영을 기본값으로 한다
- 차이가 없으면 on 유지 (메모리 이득 유지)

#### 종료 조건

- MESA target verify가 quantized target model 위에서 기능적으로 동작
- 심각한 graph failure 없이 replay 가능
- proxy 계산이 정상 수행
- `target_quant_lm_head` on/off ablation 수행 및 결과 기록

이 단계에서 throughput 최적화까지 닫을 필요는 없다.

---

### Phase 5. Persistent artifact 설계

#### 목표

Phase 2의 load-time quantization을 운영용으로 대체할 **저장 포맷**을 추가한다.

#### 왜 이 단계에서야 저장 포맷을 설계하는가

이 시점에는 이미 다음이 확정되어 있다.

- SSD runtime이 실제로 어떤 quantized state를 쓰는지
- 각 linear가 어떤 internal representation을 가지는지 (`AffineQuantizedTensor` 구조 등)
- backend가 어떤 object/tensor를 필요로 하는지

즉 저장 포맷을 억지로 먼저 설계하지 않아도 된다.

#### 추가 동기 — load-time quantization의 한계

Phase 2의 quantize hook이 **module별 순차 교체** 패턴을 자연스럽게 따른다면 peak memory는 크게 늘지 않는다:

```text
for module in model.modules() if isinstance(module, TARGET_TYPES):
    old = module.weight                       # bf16 local shard
    new = quantize(old.data)                  # int8 + scale/metadata
    module.weight = new                       # old 참조 해제 → 곧 free
```

이 방식이면 per-rank peak ≈ (로드된 전체 bf16) + (단일 모듈 임시 bf16 복사본 + 양자화 임시) 수준이다.

예: 70B TP=4 per-rank ≈ 35 GB bf16. 단일 모듈 최대 크기(예: `gate_up_proj` packed ≈ 수백 MB) 임시 추가 → peak ~35~36 GB. **80 GB A100**에서는 load-time만으로도 충분히 가능하다.

**따라서 Phase 5 artifact는 Phase 6의 "필수 전제"가 아니다**. 다만 다음 경우 artifact가 유리하거나 사실상 필요:

- 24 GB GPU 환경에서 load-time bf16 단계가 per-rank capacity를 넘을 때 (70B TP=8 per-rank ~17.5 GB → 여유는 있지만 다른 job 공존 시 빠듯)
- 반복 실험에서 매번 분 단위 quantize 시간을 피하고 싶을 때
- 재현 가능한 quantization state를 공유해야 할 때

요컨대 Phase 5는 **70B 실험의 권장 선행**이되 hard prerequisite은 아니다. 실제 장애 상황에 따라 Phase 6가 먼저 load-time으로 돌고 나서 Phase 5로 가도 된다.

#### 저장 포맷 원칙

이번에는 범용 canonical format이 아니라,

> SSD target-only torchao INT8 runtime에 필요한 최소 저장 포맷

으로 제한한다. 단 내용은 **backend-specific reconstructable state**로 본다. torchao tensor subclass를 그대로 복원하려면 단순 int8 + scale + zp 만으로는 부족할 수 있고, 다음이 포함될 수 있다:

- model family
- tp_size (pre-quantized checkpoint를 다른 TP size로 재사용 차단, §5.4 참조)
- backend id + version
- per-module `AffineQuantizedTensor` state (int8 weight, scale, zero-point, **layout/packing metadata, mapping type, block size / granularity, dtype-specific state** 포함 가능)

구체 필드는 Phase 0 결과로 확정된 저장 계약 (§4.3 (A)~(D))에 따라 결정한다. 지금 단계에서 필드를 고정하지 않는다.

#### 종료 조건 (현재 상태)

- load-time quantization 없이 artifact만으로 target-only **실행** 가능 ✓
- artifact에 `schema_version`, `backend`, `tp_size/tp_rank`, `quantize_lm_head`, `model_id`, `effective_runtime_dtype`, `original_checkpoint_dtype`, torch/torchao 버전 명시되어 mismatched 재사용 차단 ✓
- 현재 artifact는 **raw torchao AQT pickle 기반**이라 동일 환경 재사용용 로컬 캐시에 가깝고 장기 호환 checkpoint 포맷은 아님

#### 미완성 / future work

**startup 최적화**는 구현되지 않음. 현재 `model_runner.py`는 artifact 존재 여부와 무관하게 `load_model()`을 **먼저** 실행해 float weight 전부를 GPU로 로드한 뒤 artifact를 읽어 weight를 교체한다. 즉 persistent mode여도:

- float checkpoint load time (70B ~ 분 단위)
- float shard peak memory (70B TP=4 per-rank ~ 35 GB bf16)

를 한 번 치른다. "artifact 지원"은 됐지만 "artifact로 빠른/가벼운 startup"은 아직 안 됨. `loader.py` 재구조화가 필요한 비-trivial 작업이라 70B 반복 실험에서 필요해지면 착수 (TODO가 `model_runner.py` 주석에 있음).

---

### Phase 6. 대형 모델 확장 (70B 위주)

#### 목표

Phase 3~5 8B 완료 후, bf16-native 대형 모델로 확장 검증.

#### 진행 순서

1. **Llama-3.1-70B (bf16 native, TP=4 또는 TP=2)**
   - Bf16 native이므로 force flag 없이 quant 바로 가능
   - Phase 5 artifact 경로가 있으면 load 시간 절약에 유리 (단 현재 구현은 float load를 건너뛰지 않음)
   - Gate: Phase 2 종료 조건 + MESA 동작 + memory footprint 측정
2. **CodeLlama-34B (fp16 원본, TP=4) — 선택적**
   - fp16 checkpoint이므로 `target_quant_force_bf16_runtime=True` 필요 (bf16 runtime 우회)
   - 주의: bf16 override 후 load peak memory ~17 GB/rank. 24 GB RTX 환경에선 exclusive GPU 또는 80 GB A100 필요
   - "fp16 runtime 유지"가 요구사항이면 이 경로는 미완성 (§1.3 참조)

#### 확인 항목

- 메모리 절감 효과 (per-GPU)
- load 시간 (fp16 upcast 포함 시간)
- target decode / verify latency
- graph path 안정성
- MESA target verify 안정성 (Phase 4 결과 재검)

#### 성공 기준 (최소)

- 기존 dense target보다 적은 memory 로 load 가능
- dense 대비 correctness gate 통과
- MESA target verify 기능 동작

#### Stretch goal (검증 대상)

- 70B target TP=4 + draft TP=1 조합이 24 GB GPU 5대 (총 120 GB)에 안정적으로 올라감 (**원래 9 GPU → 5 GPU 축소**)
  - per-GPU 17.5 GB weight + α(KV/graph/overhead)가 24 GB에 실제로 들어가는지 실측
- 24 GB 환경에 안 들어가면 80 GB A100에서 9→N GPU 축소로 대체 검증
- INT4 양자화는 별도 후속 과제 (scope out)

---

## 7. 구현 전에 확정해야 하는 체크리스트

Phase 0 종료 시점에 아래 항목에 답이 나와야 Phase 1에 진입한다.

### Backend / dispatch  **[Phase 0 completed]**

- `F.linear(x, self.weight, bias)`가 quantized weight 상태로 int8 경로를 타는가 ✓
- §4.3 저장 계약 (A) 확정됨 ✓
- 안정 API 경로 (예: dummy `nn.Linear` + `quantize_()`) 사용이 가능한가 ✓
- local shard / packed 크기에서도 동일한가 ✓
- dtype / shape / numerical sanity가 dense와 일치하는가 ✓

### Runtime 공존

- `@torch.compile` 적용 모듈과 같은 forward에서 subclass dispatch가 충돌하지 않는가
- `@torch.inference_mode()` 아래에서 subclass dispatch가 정상 동작하는가

### Version pin

- tested `torch` / `torchao` 버전 조합이 §4.1.1에 고정 기재되었는가

### Weight tying

- `hf_config.tie_word_embeddings` 분기가 Phase 2 hook에 구현되었는가
- tied 모델에서 embed 경로가 망가지지 않는가 (Phase 0 재현)

### Loader / state

- float weight_loader 완료 후 quantize hook 순서가 확립되었는가
- Phase 5에서 persistent artifact로 넘어갈 때 어떤 state를 저장할지 방향이 있는가

### Graph (가장 큰 리스크)

- isolated `F.linear` + quantized weight 조합이 capture/replay 가능한가 (**Phase 0 gate**)
- SSD decode / verify / speculate graph 전체 경로로 확장되는가 (Phase 3)

### Scale granularity

- Column/QKV/Merged 계열은 shard 충돌 없음이 확인되었는가
- RowParallel은 local quantize가 global 대비 정밀도 동등 이상임이 경험적으로 확인되는가 (Phase 2 correctness gate 통과 여부)

### lm_head

- `target_quant_lm_head` flag가 config에 추가되었는가
- Phase 4에서 on/off ablation이 계획되어 있는가

### Correctness gate (Phase 2)

- dense flag-off 회귀 통과 ✓ (Llama-3-8B, Llama-2-7B)
- AR int8 완주 ✓
- Sampling spec int8 NaN/inf 없이 완주 ✓ (Llama-2-7B upcast, Llama-3-8B)
- Accept rate 보존: Llama-3-8B 0.30 (dense 0.32), MESA 0.40 (dense 0.41), Llama-2-7B upcast 0.44
- 1차 검증 모델: `layerskip-llama3-8B` (TP=2)

### TP

- row/column/lm_head collective semantics가 기존과 동일하게 유지되는가 (Phase 0에서 확인 ✓)

### fp16 overflow mitigation (§5.6 대응)

- bf16 native 모델 (Llama-3 계열)은 어떤 override도 없이 정상 지원 ✓
- fp16 원본 모델 (Llama-2, CodeLlama)은 **기본적으로 `ValueError`로 거부**. `target_quant_force_bf16_runtime=True` opt-in 시에만 bf16 runtime 우회 경로로 동작 (이 경우 "fp16 runtime"이 아님을 인정하는 workaround)
- fp16 runtime 지원 자체는 현 backend로 불가. Gemlite/Marlin 등 fp16-native WO backend 통합이 future work
