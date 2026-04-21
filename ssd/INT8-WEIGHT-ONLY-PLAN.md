# SSD Target-Only INT8 Weight-Only 지원 계획

## 1. 문서 목적

이 문서는 SSD 코드베이스에 `target-only INT8 weight-only` 지원을 추가하기 위한 **현실적인 구현 목표와 단계별 계획**을 정리한다.

이전 계획의 핵심 문제는 다음과 같았다.

- 외부 포맷 호환, canonical format, importer, runtime loader를 한 번에 다 설계하려고 했다
- SSD 런타임이 실제로 어떤 quantized 연산 경로를 사용할지 확정되기 전에 저장 포맷부터 크게 설계했다
- 결과적으로 범위가 너무 넓고, 실제로 구현을 시작하기 전에 불확실성이 너무 많았다

이번 계획은 그 반대로 간다.

- **지원 backend를 하나로 고정**
- **target model만 지원**
- **SSD의 custom TP linear가 사용하는 local weight 상태를 quantized 표현으로 교체**하는 방식 (저장 계약 후보는 §4.3, 최종 확정은 Phase 0에서)
- **MESA는 최종 목표에 포함하되, 초기 구현 범위에서는 분리**
- **draft, 외부 포맷 호환성은 나중으로 미룸**

이 방향이 현실적인 이유:

- SSD는 일반 PyTorch 모델이 아니다. 모델 클래스·loader·TP·graph capture가 전부 SSD 안에 묶여 있고, HF quantized model object를 그대로 받는 구조가 아니다.
- 따라서 backend는 하나로 고정하고, SSD custom TP linear 안에서 weight만 교체하는 방식이 scheduler/sequence contract와 MESA/speculate 프로토콜을 건드리지 않고 projection linear에 바로 접근할 수 있는 가장 짧은 경로다.

즉 이번 계획의 목표는:

> "torchao 계열의 기존 INT8 weight-only linear 실행 경로를 SSD custom TP linear 안에 붙여서, target model이 SSD 내부에서 실제 quantized forward를 수행하도록 만들고, 최종적으로는 그 경로가 MESA target verify에서도 동작하도록 만드는 것"

이다.

---

## 2. 최종 목표

이번 작업의 최종 목표는 다음과 같다.

1. SSD에서 **target model만** INT8 weight-only로 실행할 수 있어야 한다.
2. 구현은 **PyTorch 계열의 기존 quantized linear 실행 경로**를 재사용하는 방식으로 한다.
3. SSD의 기존 모델 구조는 유지하고, `TP wrapper + scheduler + engine`은 최대한 건드리지 않는다.
4. SSD의 custom TP linear는 유지하되, 내부의 local matmul만 quantized backend를 사용하게 한다.
5. 최종적으로 `Llama 계열 target model`, 특히 `70B target / TP4`까지 확장 가능한 구조여야 한다.
6. 최종적으로는 **MESA target verify 경로가 quantized target model 위에서 동작**해야 한다.

즉 이 문서의 최종 도착점은 단순한 "quantized target decode"가 아니다.

- normal target decode
- target verify
- speculate target verify
- MESA split verify

가 모두 같은 target quantized path를 공유하는 상태가 최종 목표다.

---

## 3. 이번 계획의 범위

### 3.1 지원 범위

- **target-only**
- **INT8 weight-only**
- **Llama 계열부터 시작**
- Qwen3는 이번 단계 범위 밖이지만 Llama와 동일한 `nn.Linear`→`F.linear(x, self.weight)` 구조이므로 이후 확장이 용이하다 (`ssd/ssd/models/qwen3.py`, `model_runner.py`에서 이미 지원되는 모델)
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

- 여러 quant backend 동시 지원 (AWQ/GPTQ/bitsandbytes를 동시에 지원)
- AWQ/GPTQ/bitsandbytes 직접 runtime 지원
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

---

## 6. 단계별 구현 계획

실제 작업 순서는 아래 Phase를 순서대로 따른다. 특히:

- 저장 포맷을 먼저 고정하지 않는다
- MESA를 초기 단계에 같이 붙이지 않는다
- custom kernel부터 만들지 않는다
- graph-safety 검증을 뒤로 미루지 않는다 (Phase 0 1차 gate)

### Phase 0. 사전 검증: backend feasibility + graph-safety + tying spike

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

### Phase 2. Eager runtime 통합

#### 목표

Phase 1에서 확정된 weight replacement contract를 SSD 런타임에 정식 추가하고, **eager 경로로 먼저 동작 확인**한다. Graph 경로는 다음 Phase.

#### 변경 대상 파일

- `ssd/ssd/config.py`
- `ssd/ssd/engine/model_runner.py`
- (필요시) `ssd/ssd/layers/linear.py`, `ssd/ssd/layers/embed_head.py` — 단, Phase 1 계약상 forward 코드 변경 없음. 이 파일 수정은 subclass 호환을 위한 minimal change만 허용

#### 세부 계획

##### 2.1 config

추가 플래그:

- `target_quant_enabled: bool = False`
- `target_quant_backend: str = "torchao_int8_wo"`
- `target_quant_lm_head: bool = True`
- `target_quant_mode: str = "load_time"`  # 개발용. Phase 5에서 `"persistent"` 추가

##### 2.2 loader 정책 (변경 최소)

- 기존 float loader는 **그대로 둔다**
- `param.data.copy_()` 기반 `weight_loader`는 float 시점에만 호출됨을 전제 (§5.1, Phase 0 검증 7번 참조)
- subclass 교체는 loader 완료 이후 별도 hook에서 수행

##### 2.3 model_runner.py quantize hook 삽입 위치

`ssd/ssd/engine/model_runner.py:294` 기준 정확한 위치:

```python
load_model(self.model, config.model, ...)   # 기존 float weight 로드 완료
# ← 여기에 quantize hook 삽입
#   - if config.target_quant_enabled and not self.is_draft:
#   - _apply_target_int8_weight_only(self.model, config, hf_config)
#     · tie_word_embeddings 방어 (§3.2.2)
#     · lm_head 양자화 on/off 결정
#     · module 순회하면서 weight 교체
self.warmup_model()                          # 이 시점부터 quantized forward
self.allocate_kv_cache()
# CudaGraph capture (이미 quantized path; Phase 3에서 graph-safety 확장 검증)
```

draft는 그대로 둔다 (`self.is_draft` 분기로 skip).

##### 2.4 weight tying 방어 구현

hook 내부에서:

```text
if hf_config.tie_word_embeddings:
    if config.target_quant_lm_head:
        # 자동 untie: lm_head weight를 새 텐서로 clone한 뒤 교체
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.clone())
        # 이제 embed_tokens와 별개 저장소 → 안전하게 양자화 가능
    else:
        pass   # lm_head 양자화 off면 tied 상태 유지 OK
```

정확한 API는 Phase 0 검증 결과에 따라 확정.

#### 종료 조건

- **Eager**에서 target-only quantized decode forward 동작
- TP correctness: dense와 동일 input에 대해 각 TP rank 결과가 맞물려 최종 logits가 일치
- **작은 Llama에서 정량적 correctness gate 통과** (Phase 2 시작 시 하나 이상으로 고정, **가능하면 perplexity + token-level 둘 다** 병행 권장):
  - **held-out dataset perplexity 변화율** ≤ 1% (dense 대비) — **가장 안정적인 업계 표준 메트릭, 최우선 권장**
  - **First divergence index** ≥ 128 (32개 prompt × 256 token greedy decode에서, 256 token 중 128번째 이전엔 token이 완전히 일치) — 단, argmax flip에 취약하므로 단독 사용 시 모델/프롬프트 편차 주의
  - 또는 **First-token top-1 match rate** ≥ 98%
  - 또는 logits KL divergence (per-token mean) ≤ 0.05 또는 cosine similarity ≥ 0.999
- **RowParallel 포함 모델**에서 위 gate 통과 (local quantize 수용 가능성 판정, §5.4)
- tied 모델 (예: Llama-3.2-1B)에서도 tying 방어 동작 확인
- 미통과 시 (가) 구현 버그 의심 → Phase 3 진입 금지, 또는 (나) RowParallel만 깨졌다면 §5.4 (a)안 도입 검토

---

### Phase 3. Graph path 확장 검증

#### 목표

Phase 0에서 **isolated** `F.linear` + quantized weight 조합으로 graph-safety 1차 gate는 통과한 상태다. 이 단계는 그 검증을 **SSD 전체 graph 경로로 확장**해서 동작을 확인한다.

즉 Phase 3은 "첫 graph 체크"가 아니라, Phase 0에서 확인된 저수준 feasibility가 SSD의 실제 decode / verify / speculate 경로에서도 유지되는지 확장 검증하는 단계다.

#### 순서

1. normal decode graph
2. verify graph
3. target-only speculate verify graph

이 단계에서는 MESA를 아직 직접 검증하지 않는다 (Phase 4).

#### 확인 항목

- SSD의 실제 graph 크기/입력 프로파일에서 capture 시 예외 여부
- 실제 운영 batch shape에서 replay 시 예외 여부
- hidden dynamic allocation 여부 (caching allocator 확인 포함)
- multi-replay 시 output stability (동일 입력에 동일 출력)
- dense graph 경로와의 latency 비교 (memory 절감 대비 속도 regression 정도 기록)

#### 실패 시 대응

graph-safe 하지 않다면:

- 우선 target-only quantized eager path 유지
- 원인 분석 후 graph 별도 대응
- Phase 0에서는 통과했는데 여기서 실패하는 경우 → SSD hot path 고유 요인(입력 shape 다양성, stream 구조, capture 순서 등) 중 어느 것이 원인인지 기록

#### 종료 조건

- target decode / verify path 전체에서 graph capture + replay 안정 동작
- dense graph 대비 output 일관성 (Phase 2 correctness gate와 동일 기준 재적용)

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

#### 종료 조건

- load-time quantization 없이 artifact만으로 target-only 실행 가능
- artifact에 `tp_size` + backend 버전이 명시되어 mismatched 재사용이 차단됨
- Phase 6 70B 실험에서 권장 선행 (필수 아님)

---

### Phase 6. 70B / TP4 확장

#### 목표

실제 목표인 `70B target / TP4`로 확장한다.

#### 전제 및 전략

- Phase 2 quantize hook의 module별 순차 교체 패턴 (§Phase 5)이 기본 동작
- load-time 경로로 먼저 시도 가능. 다만 24 GB 환경에서 peak이 넘치거나 반복 실험 효율이 떨어지면 Phase 5 artifact로 전환
- 즉 이 Phase는 load-time → 필요시 artifact 전환의 유연한 경로를 가진다

#### 확인 항목

- 메모리 절감 효과 (per-GPU)
- startup/load 시간
- target decode latency
- target verify latency
- graph path 안정성
- MESA target verify 안정성 (Phase 4 결과 재검)

#### 성공 기준 (최소)

- 기존 dense target보다 적은 메모리로 target load 가능
- TP 구성 (4 또는 8)에서 target이 안정 동작하고 dense 대비 correctness gate 통과 (§Phase 2)
- MESA target verify 기능 동작 유지

#### Stretch goal (가설, 검증 대상)

아래는 이번 계획의 동기로 깔려 있는 목표이지만, 실제 달성 여부는 이번 Phase에서 검증한다. 현 시점에는 **가설**이다.

- 70B target TP=4 + draft TP=1 조합이 24 GB GPU 5대 (총 120 GB)에 안정적으로 올라감
  - 양자화된 weight state + scale/metadata + KV cache + graph pools/workspaces + draft 공존까지 전부 합산한 실제 memory footprint는 아직 검증 전
  - per-GPU 17.5 GB weight + α(KV/graph/overhead)가 24 GB에 실제로 들어가는지 Phase 6에서 측정
- 기존 9 GPU가 필요했던 70B + async + MESA 실험이 **5 GPU 또는 더 적은 수로 축소됨**
  - 24 GB 환경에서 들어가지 않는다면 80 GB GPU 환경에서의 9→N GPU 축소가 대체 목표

stretch goal은 실패해도 이번 계획의 설계 실패가 아니라 후속 과제 (추가 양자화, draft co-location 등)로 이어진다.

---

## 7. 구현 전에 확정해야 하는 체크리스트

Phase 0 종료 시점에 아래 항목에 답이 나와야 Phase 1에 진입한다.

### Backend / dispatch

- `F.linear(x, self.weight, bias)`가 quantized weight 상태로 int8 경로를 타는가
- §4.3 (A)~(D) 중 어느 저장 계약으로 갈지 결정되었는가
- 안정 API 경로 (예: dummy `nn.Linear` + `quantize_()`) 사용이 가능한가
- local shard / packed 크기에서도 동일한가
- dtype / shape / numerical sanity가 dense와 일치하는가

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

### Correctness gate

- Phase 2 correctness 기준이 고정되었는가 (perplexity 변화율 ≤ 1% 권장, 추가로 divergence index / top-1 match / KL/cosine 병행 가능)

### TP

- row/column/lm_head collective semantics가 기존과 동일하게 유지되는가 (기본적으로 forward 코드 미변경으로 보존되지만 Phase 2 회귀 테스트로 확인)
