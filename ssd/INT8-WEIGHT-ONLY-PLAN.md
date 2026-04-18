# Llama2-70B Weight-Only INT8 지원 계획

## 1. 목표

이 코드베이스에서 Llama2-70B 타겟 모델을 `weight-only INT8` 양자화로 실행할 수 있도록 지원한다.

주요 배포 목표는 다음과 같다.

- `draft_async=True`
- 타겟은 `tp4`
- draft는 `1` GPU
- `num_gpus=5`로 실행

이 경로는 현재 async topology 계약을 바꾸지 않고 `70B target + async draft`를 가능하게 하는 가장 짧은 방법이다.

중요한 함의는 다음과 같다.

- v1에서는 `8 GPUs -> target uses 7-way TP` 문제를 해결할 필요가 없다.
- 전체 `5` GPU만 사용하면 된다. `4`장은 target TP, `1`장은 draft에 사용한다.
- 남는 GPU는 사용하지 않아도 된다.

## 2. V1 범위에서 제외하는 것

첫 구현에서는 아래 항목을 명시적으로 범위 밖으로 둔다.

- v1에서 임의의 pre-quantized checkpoint format 지원 안 함
- `bitsandbytes` 통합 안 함
- OpenVINO 통합 안 함
- eager-mode production 지원 안 함
- v1에서 draft model 양자화 안 함
- embedding, LM head, norm 양자화 안 함
- async 모드에서 8 GPU를 모두 활용하려는 시도 안 함
- 첫 단계에서 non-Llama target model 지원 안 함

## 3. 왜 이 접근이 맞는가

이 repo는 일반적인 Hugging Face 모델 실행 스택이 아니다.

다음과 같은 커스텀 구조를 갖고 있다.

- custom model classes
- custom tensor-parallel linear layers
- custom weight loading
- custom CudaGraph capture 경로
- custom speculative decoding 및 MESA 경로

즉 "기존 8-bit 라이브러리를 그대로 쓰면 된다"는 가정은 현실적이지 않다.

가장 신뢰할 수 있는 접근은 다음과 같다.

1. 현재 실행 모델은 유지한다
2. linear layer에 대해 repo-native weight-only INT8 경로를 추가한다
3. target weight는 TP sharding 이후에 양자화한다
4. draft는 우선 dense로 유지한다

이 방식이 현재 엔진 구조를 보존하면서 변경 범위를 최소화한다.

## 4. 핵심 설계 결정

### 4.1 양자화 방식

타겟 linear layer에 대해 **weight-only INT8**를 사용한다.

- weights: `int8`
- activations: 현재 모델 dtype에 맞춘 `bf16` 또는 `fp16`
- scales: float로 저장되는 per-output-channel 또는 group-wise

초기 권장 방식:

- 시작은 **per-output-channel symmetric INT8**
- 메모리/성능이 더 필요할 때만 group-wise로 확장

이유:

- loader와 shard 로직이 가장 단순함
- 수치 디버깅이 가장 쉬움
- TP 통합이 가장 쉬움

### 4.2 backend 전략

`bitsandbytes`를 메인 통합 대상으로 삼지 않는다.

이유:

- 이 repo의 hot path는 표준 `nn.Linear`를 사용하지 않는다
- TP와 loader 동작이 커스텀이다
- 엔진이 CudaGraph 동작에 강하게 의존한다

권장 전략은 다음과 같다.

- repo-native quantized linear path를 만든다
- `torchao`의 semantics와 kernel 방향을 참고한다
- full integration 전에 graph-safety를 검증한다

### 4.3 checkpoint 전략

v1은 **기존 float checkpoint를 load-time quantization**하는 방식으로 간다.

즉 다음 순서다.

1. 기존 `.safetensors`에서 정상적인 Llama2 weight를 읽는다
2. 현재와 동일하게 TP sharding을 적용한다
3. 각 local shard를 INT8로 양자화한다
4. 양자화 후 float shard는 버린다

v1에서 커스텀 INT8 serialized checkpoint format부터 시작하지 않는다.

이유:

- 구현 범위가 훨씬 작다
- 현재 float 모델과 correctness 비교가 쉽다
- backend가 검증되기 전에 checkpoint tooling에 시간을 쓰지 않아도 된다

### 4.4 v1에서 지원할 모듈

양자화 대상:

- `QKVParallelLinear`
- `MergedColumnParallelLinear`
- `RowParallelLinear`
- 일반 projection으로 쓰이는 `ReplicatedLinear`

dense 유지:

- embeddings
- LM head
- layernorm / RMSNorm
- sampler 및 verification 유틸리티

이유:

- projection matrix가 메모리 비중이 가장 크다
- embedding / LM head는 통합 난이도가 더 높다
- 이 구성이 correctness 디버깅에 유리하다

### 4.5 Eager mode 정책

production 지원은 계속 **graph-mode only**로 둔다.

이건 현재 설계 방향과 일치한다.

- MESA는 이미 `enforce_eager=False`를 요구한다
- tree decode와 verify hot path는 graph-first다

개발 정책은 다음과 같다.

- bring-up 초기에 작은 isolated eager-only debug utility를 쓰는 것은 허용한다
- 하지만 엔진 레벨 기능은 graph-mode-only로 간주한다

## 5. 현재 코드에서 중요한 제약

### 5.1 Linear layer는 dense float weight를 가정한다

현재 linear layer는:

- float `nn.Parameter` weight를 할당하고
- shard를 직접 그 weight에 로드하며
- `F.linear(x, self.weight, ...)`를 호출한다

관련 파일:

- [ssd/ssd/layers/linear.py](/home/chokwans99/PSD/ssd/ssd/layers/linear.py:12)

즉 양자화는 loader만 손봐서 끝나는 문제가 아니다.

새로운 parameter contract와 새로운 forward path가 필요하다.

### 5.2 Loader는 float tensor를 가정한다

현재 loader는:

- `safetensors` 또는 `pytorch_model*.bin`을 읽고
- tensor를 바로 parameter에 복사한다

관련 파일:

- [ssd/ssd/utils/loader.py](/home/chokwans99/PSD/ssd/ssd/utils/loader.py:186)

INT8 지원을 위해 loader는:

- shard 이후 선택적으로 quantize하고
- float parameter 대신 quantized buffer를 채울 수 있어야 한다

### 5.3 TP sharding이 weight loading에 이미 내장돼 있다

현재 row/column/QKV loader는 `narrow()` / `chunk()`를 사용해 rank-local shard를 만든다.

관련 파일:

- [ssd/ssd/layers/linear.py](/home/chokwans99/PSD/ssd/ssd/layers/linear.py:90)

이건 v1에 오히려 유리하다.

즉 full tensor를 먼저 양자화하는 대신, **local sharding 이후에 quantize**하면 된다.

### 5.4 엔진 topology

현재 `draft_async=True`에서는 엔진이 다음 구조를 사용한다.

- target TP는 `num_gpus - 1`
- 마지막 GPU 하나는 draft rank

관련 파일:

- [ssd/ssd/engine/llm_engine.py](/home/chokwans99/PSD/ssd/ssd/engine/llm_engine.py:62)

v1에서는 이 구조를 그대로 받아들인다.

즉 다음처럼 실행한다.

- `num_gpus=5`
- target TP size = `4`
- draft rank = `4`

첫 milestone에서는 topology refactor가 필요 없다.

## 6. 제안하는 아키텍처

### 6.1 새로운 quantization 패키지

다음 패키지를 추가한다.

- `ssd/ssd/quantization/__init__.py`
- `ssd/ssd/quantization/int8_weight_only.py`
- 이후 필요 시: `ssd/ssd/quantization/kernels.py`

책임은 다음과 같다.

- local dense weight shard를 INT8로 양자화
- quantized buffer와 scale 보관
- custom linear layer가 사용할 matmul/linear 인터페이스 제공

### 6.2 Quantized linear state

각 quantized linear module은 아래를 가져야 한다.

- `qweight`: `torch.int8`
- `scales`: float tensor
- optional `bias`: 원래 bias dtype
- optional metadata: `group_size`, `axis`, `scheme`

양자화가 끝나면 dense `weight`는 메모리에 남아 있지 않아야 한다.

### 6.3 Linear forward 계약

기존의 직접 `F.linear(...)` 호출을 backend dispatch로 바꾼다.

- dense path
- int8 weight-only path

권장 추상화:

- `LinearBase.forward_impl(x)`
- `DenseLinearMethod`
- `Int8WeightOnlyLinearMethod`

핵심 목표는 quantization과 TP semantics를 분리하는 것이다.

TP semantics는 기존 module class 안에 그대로 유지하는 편이 맞다.

### 6.4 Shard-then-quantize 규칙

항상 다음 순서를 따른다.

1. checkpoint에서 full tensor를 읽는다
2. 현재 rank의 local shard를 지금과 동일하게 계산한다
3. local shard를 quantize한다
4. local float shard는 버린다

v1에서는 packed tensor를 먼저 양자화한 뒤 shard하는 방식은 쓰지 않는다.

그 방식은 TP 처리까지 불필요하게 복잡하게 만든다.

### 6.5 CudaGraph 정책

양자화 지원은 아래 조건을 만족할 때만 받아들인다.

- isolated eager test에서 correctness가 맞아야 한다
- hot path에서 graph capture / replay가 안정적이어야 한다

첫 번째 graph 검증 대상은 speculative tree decode가 아니라, 일반 target decode와 verify다.

## 7. Config 변경 사항

[ssd/ssd/config.py](/home/chokwans99/PSD/ssd/ssd/config.py:7)에 다음 항목을 추가한다.

- `quant_method: str | None = None`
- `quant_target_only: bool = True`
- `quant_group_size: int | None = None`
- `quant_scale_dtype: str = "fp16"`
- `quant_skip_lm_head: bool = True`
- `quant_skip_embed: bool = True`

v1에서 허용하는 값은 다음처럼 단순화한다.

- `None`
- `"int8_wo"`

검증 규칙:

- `quant_method == "int8_wo"`는 target model family가 `llama`일 때만 지원
- `quant_method == "int8_wo"`는 v1에서 target에만 적용
- `draft_async=True`와 quantization은 함께 지원
- `enforce_eager=True`는 quantized run의 production 경로로 지원하지 않음

## 8. 구현 단계

### Phase 0: Feasibility 및 메모리 예산 확인

산출물:

- Llama2-70B INT8 target을 TP4로 올렸을 때 메모리가 들어갈 수 있는지 계산한 짧은 노트 또는 스크립트

확인 항목:

- INT8 이후 target projection weight 메모리 추정
- bf16로 남는 dense 모듈 메모리 추정
- 원하는 context length 기준 KV cache 예산 추정

종료 조건:

- target INT8 TP4가 메모리상 충분히 plausible하다는 결론이 나와야 함

### Phase 1: Quantization Primitive Bring-Up

생성 파일:

- `ssd/ssd/quantization/int8_weight_only.py`

구현 내용:

- `quantize_weight_per_channel_int8(weight) -> (qweight, scales)`
- `dequantize_weight_per_channel_int8(qweight, scales) -> weight`
- 최소 기능의 `int8_weight_only_linear(x, qweight, scales, bias=None)`

중요:

- 첫 버전은 correctness 중심 backend여도 괜찮다
- 하지만 원래 float weight를 계속 들고 있는 방식이어서는 안 된다

테스트:

- random tensor에 대해 dense vs quantized linear 출력 비교
- max error 및 relative error 측정

종료 조건:

- standalone linear layer 하나가 CUDA에서 수치적으로 정상 동작해야 함

### Phase 2: Custom Linear Module과 통합

수정 파일:

- [ssd/ssd/layers/linear.py](/home/chokwans99/PSD/ssd/ssd/layers/linear.py:12)

작업 내용:

- `LinearBase`를 quantization-aware state를 갖도록 확장
- quantized forward path 추가
- 기존 TP 동작은 유지

적용 대상:

- `ReplicatedLinear`
- `ColumnParallelLinear`
- `MergedColumnParallelLinear`
- `QKVParallelLinear`
- `RowParallelLinear`

요구사항:

- quantization 비활성화 시 기존 constructor/동작이 그대로여야 함
- TP shard 수학은 dense path와 동일해야 함

종료 조건:

- 작은 Llama 모델이 loader를 건드리지 않고도 quantized projection layer를 instantiate할 수 있어야 함

### Phase 3: Loader 통합

수정 파일:

- [ssd/ssd/utils/loader.py](/home/chokwans99/PSD/ssd/ssd/utils/loader.py:206)

작업 내용:

- target model이 quantized loading을 원하는지 감지
- linear module이 quantized인 경우:
  - 현재와 동일하게 local shard 계산
  - local shard를 즉시 양자화
  - `qweight/scales`를 module에 기록
- 그 외 모듈은 기존 dense loading 유지

중요:

- v1에서 checkpoint format은 바꾸지 않음
- draft용 별도 loader pipeline은 만들지 않음

종료 조건:

- 표준 float safetensors로부터 target model을 로드하되, linear weight는 quantized state로 메모리에 올라가야 함

### Phase 4: ModelRunner 및 Warmup 안전화

수정 파일:

- [ssd/ssd/engine/model_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/model_runner.py:247)

작업 내용:

- model construction 시 quant config를 target module에 전달
- draft model은 dense 유지
- warmup이 quantized target path로 정상 수행되도록 보장

검증:

- normal target decode
- verify path
- 처음에는 MESA 비활성 상태로 검증

종료 조건:

- quantized linear module을 가진 target model이 한 번의 decode step을 정상 수행해야 함

### Phase 5: CudaGraph 검증

목표:

- quantized target decode와 verify가 graph capture-safe인지 확인

검증 항목:

- target `decode` capture
- target `verify` capture
- 필요하면 첫 검증에서는 MESA를 꺼도 됨

capture 실패 시:

- 실패 원인이 아래 중 어디인지 분리해야 함
  - quantized linear forward path
  - 임시 allocation
  - unsupported op

이 단계는 hard gate다.

선택한 backend가 graph capture와 근본적으로 맞지 않으면, full integration으로 더 깊이 들어가기 전에 먼저 해결해야 한다.

### Phase 6: Small-Model End-to-End 검증

먼저 작은 Llama target으로 검증한다.

권장 순서:

- Llama2-7B 또는 유사한 작은 Llama checkpoint

검증 항목:

- decode correctness
- speculative verify correctness
- cache 동작이 구조적으로 깨지지 않는지
- dense 대비 throughput이 너무 나빠지지 않는지

측정 메트릭:

- tokens/s
- peak memory
- decode latency
- speculate 활성 시 acceptance 통계

종료 조건:

- target quantization이 적용된 작은 모델의 end-to-end run이 안정적으로 돌아야 함

### Phase 7: Llama2-70B Target on TP4

필요하면 target-only부터, 이후 async draft까지 확장한다.

주요 배포 설정:

- `num_gpus=5`
- `draft_async=True`
- target은 rank `0..3`
- draft는 rank `4`

검증 항목:

- target load 성공
- target warmup 성공
- target graph capture 성공
- target GPU당 peak memory가 허용 범위인지

종료 조건:

- Llama2-70B target이 TP4 + INT8 weight-only로 decode run을 끝낼 수 있어야 함

### Phase 8: Async Draft + MESA 호환성 검증

target quantization이 안정화된 뒤:

- async speculate를 다시 검증
- 이후 MESA 경로도 다시 검증

이 단계를 마지막에 두는 이유:

- target quantization은 target logits를 바꾼다
- verify 결과도 바뀔 수 있다
- MESA proxy도 같이 바뀐다

검증 항목:

- MESA 없이 async speculate
- MESA verify
- MESA draft/proxy tree path
- cache hit 및 acceptance 통계

종료 조건:

- async나 MESA 경로에서 correctness 붕괴나 큰 불안정성이 없어야 함

## 9. 파일 단위 작업 계획

### 새로 추가할 파일

- `ssd/ssd/quantization/__init__.py`
- `ssd/ssd/quantization/int8_weight_only.py`

### 수정할 기존 파일

- [ssd/ssd/config.py](/home/chokwans99/PSD/ssd/ssd/config.py:7)
- [ssd/ssd/layers/linear.py](/home/chokwans99/PSD/ssd/ssd/layers/linear.py:12)
- [ssd/ssd/utils/loader.py](/home/chokwans99/PSD/ssd/ssd/utils/loader.py:206)
- [ssd/ssd/engine/model_runner.py](/home/chokwans99/PSD/ssd/ssd/engine/model_runner.py:247)

나중에 필요할 수 있는 파일:

- [ssd/ssd/layers/embed_head.py](/home/chokwans99/PSD/ssd/ssd/layers/embed_head.py:9)
- [ssd/ssd/engine/helpers/cudagraph_helpers.py](/home/chokwans99/PSD/ssd/ssd/engine/helpers/cudagraph_helpers.py:799)

## 10. 권장 검증 매트릭스

### Unit-level

- random weight에 대한 quantize/dequantize roundtrip
- dense vs quantized linear output 비교
- row/column/QKV 케이스의 TP shard load equivalence

### Module-level

- Llama attention block 하나
- Llama MLP block 하나
- decoder layer 하나

### Engine-level

- target decode only
- target verify only
- speculate without MESA
- MESA off
- MESA on

### Scale-up

- small Llama
- 가능하면 medium Llama
- 마지막으로 Llama2-70B TP4

## 11. 성능 기대치

기대 효과:

- projection weight 메모리가 대략 절반 수준으로 감소
- target 메모리가 줄어들어 TP4 구성이 현실화될 가능성이 높음

예상 비용:

- kernel path가 최적화되지 않으면 throughput 저하 가능
- load-time quantization 때문에 startup time 증가
- graph capture 호환성 때문에 backend 선택 제약이 생길 수 있음

중요한 기준:

- v1의 성공 기준은 **최고 throughput**이 아니라
- **메모리 현실성 + 기능적 correctness**다

## 12. 리스크 목록

### Risk A: Graph capture 비호환

영향:

- isolated test에서는 되지만 CudaGraph capture에서 깨질 수 있음

대응:

- 깊은 엔진 통합 전에 graph safety를 먼저 검증

### Risk B: Loader startup이 너무 느림

영향:

- 70B shard를 startup 시점에 quantize하면 로딩이 오래 걸릴 수 있음

대응:

- v1에서는 허용
- backend가 검증된 이후 local quantized shard cache를 추가 검토

### Risk C: 정확도 변화가 speculate/MESA에 영향

영향:

- acceptance ratio 변화
- recovery distribution 변화
- 간접적으로 cache hit behavior 변화

대응:

- target-only quantization을 별도 serving mode로 취급
- dense baseline과 acceptance 및 throughput 비교

### Risk D: LM head가 그대로 커서 메모리 절감이 부족함

영향:

- 기대보다 메모리 감소 폭이 작을 수 있음

대응:

- v1에서는 수용
- target TP4가 여전히 안 들어갈 때만 다시 검토

### Risk E: 8 GPU를 모두 쓰는 구조는 여전히 지원되지 않음

영향:

- 물리적으로 8 GPU가 있어도 v1 배포 구성은 5 GPU만 사용

대응:

- v1에서는 수용
- "total GPU count와 target TP size를 분리하는 작업"은 별도 후속 프로젝트로 분리

## 13. 중단 조건

아래 중 하나라도 발생하면 설계를 멈추고 다시 봐야 한다.

- quantized linear path가 graph-capture-safe하게 만들 수 없음
- weight-only INT8 이후에도 target TP4가 메모리에 안 들어감
- dense 대비 speculate/MESA 품질 저하가 지나치게 큼
- kernel path가 너무 느려서 throughput이 실용적이지 않음

이 경우 다음 fallback 옵션은:

1. target dense decode/verify만 양자화하고 speculate는 끈다
2. 다른 backend 전략으로 이동한다
3. 양자화보다 엔진 topology 변경을 먼저 검토한다

## 14. 실제 권장 구현 순서

실제로는 다음 순서로 구현하는 것이 좋다.

1. config flag 추가
2. standalone INT8 weight-only primitive 구현
3. quantized linear module 통합
4. loader에 shard-then-quantize 추가
5. 작은 Llama에서 target decode 검증
6. CudaGraph capture 검증
7. target verify 검증
8. 70B target TP4 실행
9. async speculate 재검증
10. MESA 재검증

## 15. 최종 권장안

아래부터 시작하지 않는다.

- `bitsandbytes` 통합
- pre-quantized checkpoint 지원
- draft quantization
- eager-mode feature parity
- full 8-GPU topology refactor

아래부터 시작한다.

- target-only repo-native weight-only INT8
- TP sharding 이후 load-time quantization
- graph-mode target execution
- `num_gpus=5` 기반의 `tp4 target + 1 draft` 배포 목표

이 경로가 현재 엔진 설계와 가장 잘 맞고, 구현 범위도 가장 작고, 성공 가능성도 가장 높다.
