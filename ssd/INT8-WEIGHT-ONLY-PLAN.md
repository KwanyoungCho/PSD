# Llama2-70B INT8 지원 재설계 계획

## 1. 목표

이 계획의 최종 목표는 다음 두 가지를 동시에 만족하는 것이다.

1. 이 코드베이스에서 `Llama2-70B` 타겟 모델을 `weight-only INT8`로 실행할 수 있어야 한다.
2. 나중에 Hugging Face에 올라온 기존 양자화 모델도, 가능한 한 큰 구조 변경 없이 **import/convert 후 바로 사용할 수 있어야 한다.**

즉 이번 계획의 핵심은 단순히 "우리 코드 안에서 INT8를 만든다"가 아니다.

핵심은 아래 두 층을 분리하는 것이다.

- **실행 포맷(runtime format)**: 우리 엔진이 직접 읽고 실행하는 내부 표준 포맷
- **입력 포맷(import format)**: Hugging Face float 모델, Hugging Face 양자화 모델, 향후 외부 포맷

이렇게 해야:

- 엔진 내부는 단순해지고
- 외부 호환성은 importer로 점진적으로 확장할 수 있다

## 2. 왜 기존 계획을 갈아엎어야 하는가

기존 계획은 크게 두 가지 전제를 갖고 있었다.

1. 먼저 우리 엔진 안에서 weight-only INT8를 구현한다
2. 필요하면 나중에 오프라인 아티팩트를 만든다

하지만 네 실제 요구는 다르다.

- 운영 관점에서는 당연히 **사전 양자화된 아티팩트**가 필요하다
- 그리고 더 중요한 건 **나중에 Hugging Face INT8 모델을 구해도 사용할 수 있어야 한다**

즉 계획의 중심은 더 이상

- load-time quantization

이 아니라,

- **canonical runtime quantized format**
- **offline quantization/import pipeline**

이어야 한다.

이 요구를 반영하지 않으면, 나중에 외부 양자화 모델을 얻어도 결국 또 포맷 문제로 다시 설계를 뜯어야 한다.

## 3. 최종 방향

이제 계획의 메인 방향은 다음과 같다.

1. 우리 엔진이 직접 읽는 **canonical quantized runtime format**을 정의한다
2. 오프라인 importer/quantizer를 만들어서 외부 모델을 이 포맷으로 변환한다
3. 엔진은 이 canonical format만 읽는다
4. v1에서는 `HF float -> canonical INT8`를 먼저 지원한다
5. v2부터 `HF quantized -> canonical INT8` importer를 추가한다

즉 실행 엔진은 포맷을 하나만 알면 되고, 외부 호환성은 importer가 담당한다.

이게 가장 확장 가능하고, 가장 깔끔하며, 가장 유지보수 가능한 구조다.

## 4. 핵심 설계 원칙

### 4.1 엔진은 외부 포맷을 직접 읽지 않는다

엔진은 아래 둘 중 하나만 읽는다.

- 기존 float model path
- 우리 canonical quantized runtime format path

엔진이 직접 아래 포맷을 다 지원하도록 만들지 않는다.

- bitsandbytes runtime model
- torchao quantized tensor format
- compressed-tensors / llm-compressor format
- GPTQ/AWQ 포맷
- 기타 Hugging Face quantized checkpoint 포맷

이걸 엔진에 직접 넣으면 loader, linear, TP, graph가 전부 외부 포맷에 오염된다.

### 4.2 외부 포맷 호환성은 importer가 담당한다

외부 모델을 가져와서 우리 포맷으로 변환하는 계층을 둔다.

즉 구조는 다음과 같다.

```text
Hugging Face float model ─┐
Hugging Face INT8 model ──┼─> importer / converter ─> canonical SSD INT8 format ─> runtime engine
기타 외부 quant model ────┘
```

이 구조가 필요한 이유:

- 엔진은 간단해짐
- 외부 포맷 추가 지원이 쉬워짐
- 디버깅 경계가 명확해짐

### 4.3 v1은 target-only INT8

첫 구현에서는 target만 INT8로 간다.

- target: INT8 weight-only
- draft: dense
- embeddings: dense
- LM head: dense
- norm: dense

이유:

- projection matrix가 메모리 대부분을 차지한다
- target 메모리 절감 효과를 먼저 확인해야 한다
- 외부 포맷 importer 문제와 draft까지 동시에 풀면 범위가 너무 커진다

### 4.4 운영 경로는 오프라인 아티팩트 중심

이번 재설계에서는 load-time quantization을 메인 경로로 두지 않는다.

운영 메인 경로는:

1. 외부 모델 준비
2. 오프라인 importer/quantizer 실행
3. canonical INT8 아티팩트 생성
4. 엔진이 canonical INT8 아티팩트 로드

load-time quantization은 필요하면 **디버그용 fallback** 정도로만 둔다.

### 4.5 graph-mode only

최종 기능은 여전히 `graph-mode only`다.

이유:

- 현재 엔진의 성능 경로가 CudaGraph 중심
- MESA도 이미 eager를 production 경로로 보지 않음

다만 개발 초기에는 아주 작은 isolated unit test 수준에서만 eager 검증을 허용한다.

## 5. 목표 사용 시나리오

### 시나리오 A: HF float 모델에서 시작

예:

- `meta-llama/Llama-2-70b-hf`

흐름:

1. float checkpoint를 입력으로 준다
2. importer가 TP4 기준 local shard를 계산한다
3. weight-only INT8로 양자화한다
4. canonical INT8 포맷으로 저장한다
5. 엔진이 그 포맷을 읽는다

### 시나리오 B: 나중에 HF INT8 모델을 얻음

예:

- torchao 기반 모델
- quanto 기반 모델
- 향후 HF 양자화 모델

흐름:

1. importer가 해당 포맷을 감지한다
2. 해당 포맷에서 quantized weight와 metadata를 읽는다
3. 필요하면 우리 canonical format으로 재정렬 / 재패킹한다
4. canonical INT8 포맷으로 저장한다
5. 엔진이 그 포맷을 읽는다

즉 "HF 양자화 모델을 바로 사용"의 정확한 의미는:

- runtime이 외부 포맷을 직접 실행한다

가 아니라,

- **별도 수작업 없이 importer가 canonical format으로 바꿔준 뒤 엔진이 바로 실행한다**

이다.

이 정도가 현실적으로 가장 맞는 해석이다.

## 6. Canonical Runtime Format

### 6.1 기본 구조

권장 디렉토리 구조:

```text
Llama2-70B-INT8-WO-TP4/
  manifest.json
  rank_0/
    model.safetensors
  rank_1/
    model.safetensors
  rank_2/
    model.safetensors
  rank_3/
    model.safetensors
```

각 `rank_i/model.safetensors`에는 현재 rank가 필요로 하는 **로컬 shard의 quantized weight**만 저장한다.

즉 runtime 시점에는 더 이상 full weight를 읽지 않는다.

### 6.2 저장할 tensor

기본적으로 각 quantized linear module마다:

- `module_name.qweight`
- `module_name.scales`
- optional `module_name.bias`

를 저장한다.

예:

```text
model.layers.0.self_attn.qkv_proj.qweight
model.layers.0.self_attn.qkv_proj.scales
model.layers.0.self_attn.o_proj.qweight
model.layers.0.self_attn.o_proj.scales
model.layers.0.mlp.gate_up_proj.qweight
model.layers.0.mlp.gate_up_proj.scales
model.layers.0.mlp.down_proj.qweight
model.layers.0.mlp.down_proj.scales
```

### 6.3 Manifest

`manifest.json`에는 최소 아래 필드가 필요하다.

```json
{
  "format": "ssd_int8_wo_v1",
  "model_family": "llama",
  "source_model": "meta-llama/Llama-2-70b-hf",
  "tp_size": 4,
  "scheme": "per_channel_symmetric",
  "scale_dtype": "fp16",
  "quant_method": "int8_wo",
  "target_only": true,
  "skip_embed": true,
  "skip_lm_head": true,
  "created_by": "ssd/scripts/import_quantized_model.py",
  "source_format": "hf_float"
}
```

추가로 들어가면 좋은 필드:

- `hf_config_hash`
- `weight_map_hash`
- `transform_recipe`
- `source_quant_backend`
- `version`

### 6.4 왜 rank별 저장이 필요한가

이 repo는 TP shard를 runtime 전에 이미 알아야 한다.

즉 canonical format은 "모델 전체의 quantized full tensor"보다,

- **실행할 TP 크기에 맞는 rank-local shard**

를 저장하는 편이 더 맞다.

이 방식의 장점:

- runtime loader가 단순함
- 메모리 낭비가 적음
- startup time이 짧음

단점:

- `tp4`용 아티팩트와 `tp8`용 아티팩트는 별개다

하지만 현재 요구는 `tp4 target + 1 draft`이므로 이 tradeoff를 받아들이는 것이 맞다.

## 7. 지원할 입력 포맷과 importer 전략

### 7.1 v1에서 지원할 입력 포맷

v1에서는 아래 하나만 필수 지원한다.

- **Hugging Face float checkpoint**
  - `.safetensors`
  - 표준 Llama weight naming

즉 v1의 importer는 사실상:

- `HF float -> canonical INT8`

변환기다.

### 7.2 v2에서 지원할 입력 포맷

v2부터 다음을 고려한다.

- torchao 기반 HF quantized model
- Hugging Face Quanto 계열 모델
- 향후 표준화된 compressed-tensors 계열 모델

이때 중요한 건 "바로 실행"이 아니라, 아래 구조를 유지하는 것이다.

- 외부 quantized model -> importer -> canonical runtime format

### 7.3 지원 우선순위가 낮은 포맷

다음은 우선순위를 낮춘다.

- bitsandbytes runtime-dependent 모델
- vLLM 전용 quantized artifact
- GPTQ/AWQ 전용 packed 포맷

이유:

- 우리 엔진 계약과 차이가 크다
- custom TP와 직접 연결하기 어렵다
- import/convert 계층을 별도로 많이 만들어야 한다

즉 "당장 HF INT8 모델을 나중에 쉽게 쓰고 싶다"는 요구는 중요하지만,

- 가장 먼저 맞춰야 할 대상은 float HF
- 그 다음 호환성이 높은 quantized HF

순으로 가는 게 맞다.

## 8. 외부 기존 repo를 어디까지 활용할 수 있는가

### 8.1 `torchao`

`torchao`는 가장 중요한 참고 대상이다.

공식 문서 기준:

- `Int8WeightOnlyConfig` 존재
- `quantize_()`로 `nn.Linear` 기반 모델에 int8 weight-only 적용 가능

출처:

- https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Int8WeightOnlyConfig.html
- https://docs.pytorch.org/ao/stable/workflows/inference.html

하지만 이 repo에 그대로 적용되지는 않는다.

이유:

- 우리는 custom TP linear를 사용한다
- loader도 custom이다
- graph path도 custom이다

따라서 `torchao`는 다음 용도로 쓴다.

- quantization semantics 참고
- 수치 기준 참고
- 향후 backend 개선 시 reference 사용

즉 **직접 runtime dependency라기보다, canonical format과 kernel 설계의 기준**으로 본다.

### 8.2 `bitsandbytes`

`bitsandbytes`는 HF에서 가장 쉬운 8bit 옵션이지만, 우리 메인 경로로는 적합하지 않다.

공식 문서 기준:

- `BitsAndBytesConfig(load_in_8bit=True)`로 `torch.nn.Linear` 기반 모델을 양자화 로드한다

출처:

- https://huggingface.co/docs/transformers/en/quantization/bitsandbytes

문제:

- runtime replacement 중심
- custom TP linear와 잘 안 맞음
- 순수한 "canonical weight-only INT8 storage format"으로 보기 어려움

따라서 `bitsandbytes`는 importer 우선순위도 낮다.

### 8.3 `llm-compressor` / `vLLM` 계열

`llm-compressor`는 오프라인 양자화 workflow 참고용으로는 좋다.

출처:

- https://github.com/vllm-project/llm-compressor

활용 가능 영역:

- calibration workflow 참고
- offline quantized artifact 설계 참고
- quantization recipe 아이디어 참고

하지만 직접 runtime 호환 대상으로 보지는 않는다.

이유:

- vLLM 친화 포맷 기준
- 우리 loader 계약과 다름
- 우리 TP/runtime graph 구조와 직접 맞지 않음

### 8.4 최종 판단

외부 repo 활용 원칙은 아래처럼 정리한다.

- `torchao`: 핵심 reference
- `bitsandbytes`: 비교 대상
- `llm-compressor`: 오프라인 workflow 참고용

즉 실제 구현 경로는:

- **우리 canonical runtime format**
- **우리 importer**
- **우리 custom TP-aware linear runtime**

이 세 축으로 간다.

## 9. 런타임 아키텍처

### 9.1 엔진이 알아야 하는 포맷

엔진은 아래 두 경로만 인식한다.

1. 기존 float 경로
2. canonical INT8 경로

즉 `Config.model`이 가리키는 path가:

- 일반 HF float checkpoint인지
- canonical INT8 artifact directory인지

만 판단하면 된다.

### 9.2 엔진이 외부 포맷을 모르게 해야 하는 이유

엔진이 외부 포맷까지 직접 알게 되면 아래가 다 오염된다.

- `loader.py`
- `linear.py`
- `model_runner.py`
- graph capture 경로

이건 유지보수상 최악이다.

그래서 importer는 런타임 밖으로 분리하는 것이 맞다.

## 10. Config 변경

`ssd/ssd/config.py`에 아래 필드를 추가한다.

```python
# Quantization
quant_method: str | None = None
quant_target_only: bool = True
quant_group_size: int | None = None
quant_scale_dtype: str = "fp16"
quant_skip_lm_head: bool = True
quant_skip_embed: bool = True
quant_source_format: str | None = None
quant_runtime_format: str | None = None
```

권장 값:

- `quant_method="int8_wo"`
- `quant_runtime_format="ssd_int8_wo_v1"`

검증 규칙:

- `quant_method == "int8_wo"`면 target family는 `llama`
- v1은 `quant_target_only=True`
- production에서는 `enforce_eager=False`

## 11. 실제 코드 구조

### 11.1 새 패키지

추가:

- `ssd/ssd/quantization/__init__.py`
- `ssd/ssd/quantization/int8_weight_only.py`
- `ssd/ssd/quantization/runtime_format.py`
- `ssd/ssd/quantization/importers/__init__.py`
- `ssd/ssd/quantization/importers/hf_float.py`
- 나중에:
  - `ssd/ssd/quantization/importers/hf_torchao.py`
  - `ssd/ssd/quantization/importers/hf_quanto.py`

### 11.2 역할 분리

`int8_weight_only.py`

- quant primitive
- qweight/scales 생성
- quantized linear helper

`runtime_format.py`

- manifest schema
- canonical key naming
- rank shard save/load helper

`importers/hf_float.py`

- HF float checkpoint -> canonical INT8 artifact 변환

### 11.3 새 스크립트

추가:

- `ssd/scripts/import_quantized_model.py`

이 스크립트가 실제 변환 진입점이다.

예:

```bash
python ssd/scripts/import_quantized_model.py \
  --source /path/to/Llama-2-70b-hf \
  --source-format hf_float \
  --output /path/to/Llama2-70B-INT8-WO-TP4 \
  --tp-size 4 \
  --quant-method int8_wo
```

나중에는:

```bash
python ssd/scripts/import_quantized_model.py \
  --source /path/to/hf-int8-model \
  --source-format hf_torchao \
  --output /path/to/Llama2-70B-INT8-WO-TP4 \
  --tp-size 4 \
  --quant-method int8_wo
```

같은 형태를 지원한다.

## 12. Quantization Primitive

`ssd/ssd/quantization/int8_weight_only.py`에는 최소 아래를 둔다.

```python
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class Int8WeightOnlyState:
    qweight: torch.Tensor
    scales: torch.Tensor
    bias: torch.Tensor | None = None
    scheme: str = "per_channel_symmetric"


def quantize_weight_per_channel_int8(weight, scale_dtype=torch.float16):
    max_abs = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = (max_abs / 127.0).squeeze(1).to(scale_dtype)
    q = torch.round(weight / scales.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scales.contiguous()


def dequantize_weight_per_channel_int8(qweight, scales, out_dtype):
    return qweight.to(out_dtype) * scales.to(out_dtype).unsqueeze(1)


def int8_weight_only_linear(x, qweight, scales, bias=None):
    # v1 correctness-first
    w = dequantize_weight_per_channel_int8(qweight, scales, x.dtype)
    return F.linear(x, w, bias)
```

중요:

- v1은 correctness-first
- 즉 실제 int8 kernel이 아니라 dequantize+F.linear로 시작해도 된다
- canonical format과 loader/runtime 계약을 먼저 안정화하는 게 우선이다

## 13. Runtime Format 코드 초안

`ssd/ssd/quantization/runtime_format.py`

```python
from dataclasses import dataclass, asdict
import json
import os
from safetensors.torch import save_file, load_file


@dataclass
class QuantManifest:
    format: str
    model_family: str
    source_model: str
    source_format: str
    tp_size: int
    quant_method: str
    scheme: str
    scale_dtype: str
    target_only: bool
    skip_embed: bool
    skip_lm_head: bool


def save_manifest(manifest: QuantManifest, out_dir: str):
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(asdict(manifest), f, indent=2)


def load_manifest(model_dir: str) -> QuantManifest:
    with open(os.path.join(model_dir, "manifest.json")) as f:
        data = json.load(f)
    return QuantManifest(**data)


def save_rank_state(rank_dir: str, state_dict: dict):
    os.makedirs(rank_dir, exist_ok=True)
    save_file(state_dict, os.path.join(rank_dir, "model.safetensors"))


def load_rank_state(rank_dir: str) -> dict:
    return load_file(os.path.join(rank_dir, "model.safetensors"))
```

## 14. Importer 설계

### 14.1 공통 인터페이스

모든 importer는 아래 계약을 따른다.

```python
class BaseImporter:
    def inspect(self, source_path: str) -> dict: ...
    def export(
        self,
        source_path: str,
        out_dir: str,
        tp_size: int,
        quant_method: str,
    ) -> None: ...
```

### 14.2 HF float importer

`ssd/ssd/quantization/importers/hf_float.py`

역할:

1. source가 HF float checkpoint인지 확인
2. layer 이름을 우리 runtime canonical key로 정규화
3. TP4 기준 local shard 계산
4. shard별 INT8 양자화
5. rank별 safetensors 저장
6. manifest 저장

중요:

- 이 importer가 v1의 핵심 기능이다

### 14.3 HF quantized importer

v2부터 추가한다.

예:

- `hf_torchao.py`
- `hf_quanto.py`

역할:

1. 해당 포맷의 quantized tensor를 읽는다
2. 우리 canonical runtime format으로 매핑한다
3. 필요하면 scale/packing을 다시 정규화한다

중요:

- v2도 엔진은 안 바뀌어야 한다
- importer만 늘어나야 한다

## 15. Linear 런타임 구현

### 15.1 공통 상태

`ssd/ssd/layers/linear.py`의 `LinearBase`에 아래를 추가한다.

```python
self.quant_method = None
self.qweight = None
self.scales = None
self._weight_loaded = False
```

공통 헬퍼:

```python
def set_quantized_weight(self, qweight, scales):
    self.qweight = qweight
    self.scales = scales
    self.quant_method = "int8_wo"
    self._weight_loaded = True
    if hasattr(self, "weight") and self.weight is not None:
        self.register_parameter("weight", None)


def has_quant_weight(self):
    return self.quant_method == "int8_wo" and self.qweight is not None
```

### 15.2 forward 변경

dense path:

```python
return F.linear(x, self.weight, self.bias)
```

quant path:

```python
if self.has_quant_weight():
    return int8_weight_only_linear(x, self.qweight, self.scales, self.bias)
return F.linear(x, self.weight, self.bias)
```

`RowParallelLinear`은 all-reduce는 그대로 유지한다.

### 15.3 packed linear

`QKVParallelLinear`, `MergedColumnParallelLinear`는 다음 두 방식 중 하나다.

1. importer가 최종 packed local weight를 그대로 만들어 저장
2. runtime load 후 packed float를 조립하고 그 뒤 quantize

이번 재설계에서는 **1번이 더 맞다.**

즉 importer가 이미 최종 local packed qweight를 만들어 저장한다.

이렇게 하면 runtime loader가 훨씬 단순해진다.

## 16. Loader 재설계

### 16.1 기존 loader의 역할 축소

`ssd/ssd/utils/loader.py`는 이제 두 모드만 처리한다.

1. float model load
2. canonical quantized artifact load

즉 외부 포맷별 parsing은 loader가 하지 않는다.

### 16.2 canonical quantized load

흐름:

1. `manifest.json` 확인
2. 현재 rank에 해당하는 `rank_i/model.safetensors` 로드
3. `module_name.qweight`, `module_name.scales`를 각 모듈에 주입

이때는 기존 `weight_loader()` 대신 전용 quant loader를 추가하는 것이 낫다.

예:

```python
def load_quantized_model(model, model_dir, rank):
    manifest = load_manifest(model_dir)
    state = load_rank_state(os.path.join(model_dir, f"rank_{rank}"))
    for name, module in model.named_modules():
        qk = f"{name}.qweight"
        sk = f"{name}.scales"
        if qk in state and sk in state:
            module.set_quantized_weight(
                state[qk].to("cuda"),
                state[sk].to("cuda"),
            )
```

즉 float loader와 quant loader는 명시적으로 분리하는 것이 맞다.

## 17. ModelRunner 변경

### 17.1 model path가 canonical quantized artifact인지 판별

`Config.model`이 가리키는 path에 `manifest.json`이 있고 `format == "ssd_int8_wo_v1"`이면 quantized runtime artifact로 본다.

### 17.2 target만 양자화

`ModelRunner.setup_and_warmup_model_and_cudagraphs()`에서:

- target이면 quantized runtime artifact 로드 허용
- draft이면 dense model만 허용

즉:

```python
effective_quant_method = None if self.is_draft else self.config.quant_method
```

같은 정책을 유지한다.

## 18. 새 스크립트

### 18.1 메인 importer 스크립트

추가:

- `ssd/scripts/import_quantized_model.py`

CLI:

```bash
python ssd/scripts/import_quantized_model.py \
  --source /path/to/model \
  --source-format hf_float \
  --output /path/to/output \
  --tp-size 4 \
  --quant-method int8_wo
```

나중에:

```bash
python ssd/scripts/import_quantized_model.py \
  --source /path/to/hf-int8-model \
  --source-format hf_torchao \
  --output /path/to/output \
  --tp-size 4 \
  --quant-method int8_wo
```

도 가능해야 한다.

## 19. 단계별 구현 순서

### Phase 0: canonical format 확정

먼저 확정해야 할 것:

- manifest schema
- rank shard directory layout
- key naming 규칙
- 어떤 모듈을 저장할지

이게 먼저 고정돼야 importer와 runtime이 따로 개발돼도 맞물린다.

### Phase 1: HF float importer 구현

v1의 실제 1순위 구현이다.

산출물:

- `importers/hf_float.py`
- `scripts/import_quantized_model.py`
- canonical INT8 artifact 생성 가능

### Phase 2: canonical runtime loader 구현

산출물:

- quantized artifact를 읽는 loader
- `LinearBase.set_quantized_weight()`
- quantized forward path

### Phase 3: 작은 모델 end-to-end

작은 Llama 모델로:

- importer 실행
- artifact 생성
- runtime load
- decode
- verify

를 검증한다.

### Phase 4: CudaGraph 검증

이 단계는 hard gate다.

- target decode capture
- target verify capture

가 깨지지 않아야 한다.

### Phase 5: 70B TP4

목표 실행 구성:

- `num_gpus=5`
- target `tp4`
- draft `1`

### Phase 6: async + MESA 재검증

마지막에:

- async speculate
- MESA

를 다시 검증한다.

## 20. 검증 매트릭스

### Unit-level

- quantize/dequantize roundtrip
- dense vs quantized linear
- runtime format save/load
- manifest validation

### Importer-level

- HF float -> canonical INT8
- rank별 shard shape 검증
- key naming 검증

### Runtime-level

- canonical INT8 artifact 로드
- decode
- verify
- async speculate
- MESA

## 21. 최종 판단

이제 계획의 메인 축은 아래 세 가지다.

1. **canonical runtime quantized format**
2. **offline importer / converter**
3. **custom TP-aware quantized runtime**

즉 "엔진이 직접 외부 양자화 포맷을 다 읽는 구조"로 가지 않는다.

그 대신:

- v1: `HF float -> canonical INT8`
- v2: `HF quantized -> canonical INT8`

로 단계적으로 확장한다.

이 방향이

- 네가 원하는 "나중에 HF INT8 모델을 얻어도 바로 쓸 수 있는 구조"
- 현재 엔진의 custom TP / graph 구조

를 동시에 만족시키는 가장 현실적인 방법이다.
