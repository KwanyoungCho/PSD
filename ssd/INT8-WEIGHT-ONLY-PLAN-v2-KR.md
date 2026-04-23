# AWQ 통합 계획 v2

> 파일 이름은 레거시입니다. 이 문서는 더 이상 "INT8 우선" 계획이 아닙니다.
> 새로운 목표는 SSD의 현재 추론 아키텍처에 맞고 실제로 필요한
> 체크포인트/런타임 조합을 지원하는 **AWQ 스타일의 최적화된 weight-only 백엔드**입니다.

## 1. 목표

### 1.1 최종 목표

SSD에 **fp16/bf16 호환 최적화 weight-only 백엔드**를 통합하여:

- 대형 타겟 모델이 VRAM에 적재 가능하고,
- 기존 SSD 최적화 경로가 그대로 유지되며,
- 자기회귀 디코드, 추론적 검증, MESA 타겟 검증이 계속 동작하고,
- 전체 정밀도 가중치를 GPU에 먼저 올린 뒤 양자화하는 방식을 사용하지 **않는다**.

이 계획의 백엔드 방향은 torchao INT8이 아닌 **AWQ 스타일 W4A16**이다.

### 1.2 방향 전환 이유

현재 torchao 경로는 일부 bf16 네이티브 모델에 대한 폴백으로 유용하지만,
이 저장소의 장기적 답이 아니다:

- 현재 선택된 torchao WO 백엔드는 fp16 런타임에 잘 맞지 않고,
- INT8 경로는 우리 하드웨어에서 최적화된 빠른 경로가 아니며,
- 현재 구현은 여전히 dense GPU 가중치를 먼저 적재한 뒤 교체하므로 시작/피크 메모리 절약이 불완전하고,
- SSD에서 "최적화된 추론"은 기존 TP / PagedAttention / CUDA graph / prefix-cache 아키텍처를
  그대로 유지하면서 로컬 linear 백엔드만 교체하는 것에 의존한다.

### 1.3 핵심 설계 선택

우리는 다음을 하지 **않을** 것이다:

- SSD 스케줄링 재작성,
- PagedAttention / KV 캐시 / attention 커널 재작성,
- 엔진을 Hugging Face 런타임으로 교체,
- 첫 단계부터 양자화 GEMM 커널을 처음부터 작성.

우리는 다음을 **할** 것이다:

- SSD의 현재 최적화된 엔진 유지,
- **오프라인 AWQ 아티팩트 파이프라인** 추가,
- **SSD 로컬 TP linear 레이어를 위한 AWQ 런타임 어댑터** 추가,
- 양자화 범위를 좁고 명시적으로 유지,
- **로컬 linear matmul** 경계에서만 통합.


## 2. 변경되지 않는 부분

하드 블로커가 발견되지 않는 한 SSD의 다음 부분은 보존한다:

- `ssd/ssd/layers/linear.py`의 텐서 병렬 래퍼 및 의미론
- attention 경로 및 PagedAttention / FlashInfer 래퍼
- KV 캐시 레이아웃 및 블록 테이블 처리
- 추론적 디코딩 제어 흐름
- MESA 오케스트레이션 및 분할 검증 구조
- CUDA graph 캡처/리플레이 구조
- `@torch.compile` norm / activation / rope 경로
- prefix 캐싱 및 스케줄러 동작

이것이 전체 계획에서 가장 중요한 아키텍처 제약 조건이다.


## 3. 변경되는 부분

**무거운 linear 가중치의 저장 및 실행**만 변경된다.

### 3.1 기본 양자화 대상

기본적으로 대형 타겟 측 프로젝션 가중치만 양자화한다:

- `q_proj / k_proj / v_proj` -> SSD 패킹 모듈 `qkv_proj`
- `o_proj`
- `gate_proj / up_proj` -> SSD 패킹 모듈 `gate_up_proj`
- `down_proj`

### 3.2 기본 dense 유지 대상

이후 명시적으로 활성화하지 않는 한 다음은 dense로 유지한다:

- 임베딩
- `lm_head`
- norm
- rope
- attention 코어
- KV 캐시
- 드래프트 모델

### 3.3 `lm_head`가 기본 dense인 이유

SSD에서 `lm_head`는 첫 양자화 대상으로 부적절하다:

- 매 스텝마다 호출되는 핫 경로이고,
- `ParallelLMHead`에 이미 TP gather/cat 오버헤드가 있으며,
- MESA는 exit-layer logit을 사용하므로 `lm_head` 품질에 더 민감하다.

따라서 기본 정책은:

- **타겟 linear 프로젝션: 양자화**
- **타겟 `lm_head`: dense**
- **드래프트: dense**


## 4. 왜 AWQ인가, bitsandbytes나 현재 torchao가 아닌 이유

### 4.1 왜 bitsandbytes가 아닌가

bitsandbytes는 Hugging Face / `nn.Linear` 교체 흐름에서 가장 자연스러우므로
SSD의 주 백엔드로는 맞지 않다.

SSD는 그렇게 구축되어 있지 않다:

- 커스텀 TP linear 모듈을 사용하고,
- QKV / gate-up에 대한 패킹 로더 규칙이 있고,
- graph 중심 실행이며,
- 커스텀 러너 오케스트레이션이 있다.

bitsandbytes는 SSD 아키텍처에 맞추기보다 싸워야 하는 상황을 만들 것이다.

### 4.2 왜 현재 torchao를 주 경로로 유지하지 않는가

torchao는 일부 bf16 네이티브 케이스에 대한 임시 폴백으로 유용하지만,
더 이상 주 방향이 아니다:

- 현재 선택된 torchao WO 경로는 원하는 fp16 런타임 지원을 제공하지 않고,
- 현재 INT8은 원하는 최적화 경로가 아니며,
- 구현 모델은 편리하지만 최종 목표에 충분하지 않다.

### 4.3 왜 AWQ가 더 나은 방향인가

AWQ 스타일 백엔드는 일반적으로 다음과 같아서 더 잘 맞는다:

- 오프라인 양자화 우선,
- 저비트 weight-only,
- 백엔드에 따라 fp16/fp8/bf16 추론에 activation 친화적,
- 일반적인 역양자화 `F.linear`가 아닌 최적화된 추론 커널과 연계.

SSD에서 중요한 점은 AWQ 알고리즘 단독이 아니다. 다음의 쌍이다:

1. **오프라인 AWQ 아티팩트**
2. **로컬 linear matmul을 위한 최적화된 런타임 백엔드**

### 4.4 AWQ 캘리브레이션 vs AWQ 런타임 백엔드

이 두 개념은 분리 가능하며 혼동해서는 안 된다:

- **AWQ 캘리브레이션 알고리즘**: activation 인식 중요도 가중치를 사용하여
  채널별 스케일링 팩터를 결정하는 오프라인 절차. 양자화된 체크포인트(아티팩트)를
  생성한다. 이것은 *품질* 결정이다.
- **AWQ 런타임 백엔드**: 추론 시간에 패킹된 저비트 가중치 × fp16/bf16
  activation matmul을 실행하는 최적화된 커널 경로. Marlin, AutoAWQ GEMM,
  ExLlamaV2 등이 예시이다. 이것은 *성능* 결정이다.

캘리브레이션 알고리즘과 런타임 백엔드는 독립적으로 선택된다:

- Marlin 커널은 AWQ 캘리브레이션된 아티팩트와 GPTQ 캘리브레이션된 아티팩트 모두 실행 가능하고,
- 다른 캘리브레이션 방법도 동일한 런타임 백엔드와 호환되는 아티팩트를 생성할 수 있다.

이 문서 전체에서:

- "AWQ 아티팩트"는 **캘리브레이션 출력**(양자화된 가중치 + 스케일 +
  제로포인트를 특정 패킹 포맷으로)을 지칭한다.
- "AWQ 런타임" 또는 "AWQ 백엔드"는 추론에 사용되는 **커널 경로**를 지칭한다.
- 구분이 중요한 경우 명시적으로 표기한다.


## 5. 새 백엔드의 하드 제약 조건

AWQ 런타임 후보는 통합 전에 다음 게이트를 충족해야 한다.

### 5.1 런타임 dtype 지원

선택된 백엔드는 실제로 관심 있는 런타임 dtype을 지원해야 한다:

- fp16 activation 런타임
- bf16 activation 런타임, 또는 명확히 범위가 정해진 폴백 전략

후보가 fp16만 잘 지원하는 경우:

- fp16 체크포인트에 먼저 사용할 수 있지만,
- bf16 네이티브 모델은 기존 torchao 폴백을 계속 사용하거나
  두 번째 백엔드를 기다려야 한다.

이것은 중간 상태로 허용되지만, 반드시 명시적이어야 한다.

### 5.2 Shape 지원

백엔드는 SSD 관련 영역을 커버해야 한다:

- **디코드 / 검증 / MESA**: 매우 작은 M, GEMV 유사 또는 작은 GEMM 영역
- **프리필**: 더 큰 M GEMM 영역

백엔드가 큰 GEMM에서만 벤치마크가 좋고 디코드 크기 matmul에서 무너지면,
SSD의 주 양자화 백엔드로 허용할 수 없다.

### 5.3 Graph 호환성

백엔드는 다음과 호환되어야 한다:

- `torch.inference_mode()`
- 기존 CUDA graph 캡처/리플레이
- 현재 TP 래퍼

graph 호환성이 실패하면, 백엔드를 거부하거나 수정될 때까지
eager-only 디버깅으로 범위를 한정한다.

### 5.4 저장 계약

백엔드는 GPU에 양자화된 저장소를 유지할 수 있어야 한다:

- 패킹된 저비트 가중치
- 스케일
- 제로포인트 / 메타데이터 (필요한 경우)

전체 fp16 가중치를 GPU에 확장하고 그대로 유지하는 폴백은 허용할 수 없다.


## 6. 최종 아키텍처

### 6.1 상위 수준 구조

최종 구조는 다음과 같아야 한다:

1. **오프라인 AWQ 생산자**
   - 외부 도구 또는 임포트 스크립트
   - 원본 체크포인트에서 양자화된 아티팩트 생성

2. **SSD AWQ 임포터**
   - 외부 AWQ 아티팩트를 읽음
   - SSD 친화적인 랭크별 아티팩트로 변환
   - 패킹 모듈(`qkv_proj`, `gate_up_proj`) 해결
   - TP 랭크별 사전 샤딩

3. **SSD 런타임 어댑터**
   - 랭크별 양자화 가중치를 직접 로드
   - 기존 SSD TP 래퍼가 dense `F.linear` 대신 AWQ 런타임 연산 호출

4. **SSD 엔진**
   - attention / 캐시 / graph 오케스트레이션 변경 없음

### 6.2 통합 지점

통합 지점은 **오직** 로컬 linear 실행 경계이다.

즉:

- 기존 TP 래퍼 클래스 유지,
- 모델 정의 대부분 변경 없음,
- 스케줄러 / 러너 / attention 구조 변경 없음,
- linear forward 내에서 가중치 종류 / 양자화 백엔드에 따라 분기.

가능하면 완전히 새로운 두 번째 모델 스택을 만들지 **않는다**.

### 6.3 선호 모듈 전략

완전히 새로운 모델 모듈의 병렬 계층 구조를 구축하는 것부터 시작하지 **않는다**.

선호 접근법:

- `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`,
  `MergedColumnParallelLinear` 유지,
- 양자화 상태를 추가하거나 이들이 소유하는 작은 헬퍼 객체 추가,
- forward 디스패치:
  - dense 경로 -> 기존 `F.linear`
  - AWQ 경로 -> 백엔드별 로컬 양자화 matmul

이렇게 하면 다음이 안정적으로 유지된다:

- TP 의미론,
- 로더 시그니처,
- 모델 코드,
- graph 호출 지점

### 6.3.1 양자화 모드 모듈 인스턴스화 계약

이것은 구현 시작 전에 명시적으로 정의되어야 한다.

현재 SSD 모델 구성은 기본적으로 양자화 친화적이지 않다:

- `ModelRunner`는 모델 구성 전에 `torch.set_default_device("cuda")`를 설정하고,
- TP linear 모듈은 `__init__` 중에 dense `nn.Parameter(torch.empty(...))` 가중치를 할당하며,
- 따라서 "모델을 먼저 구축한 뒤 양자화 아티팩트를 로드"하면 로더가 dense 체크포인트
  텐서를 복사하지 않더라도 구성 시점에 dense GPU 가중치 저장소를 할당하게 된다.

AWQ 모드에서 이것은 허용할 수 없다. 양자화 모드는 다음 계약 중 하나를 사용해야 한다:

1. **양자화 인식 TP 모듈 init**
   - 양자화 모드에서 TP linear 모듈은 dense GPU `weight`를 할당하지 **않음**
   - 대신 양자화 상태 플레이스홀더 / 메타데이터 홀더만 할당

2. **Meta/CPU 플레이스홀더 init**
   - TP linear 모듈이 `meta` 또는 CPU에 플레이스홀더 저장소를 할당
   - 로더/런타임이 나중에 실제 양자화 상태를 부착

3. **구성 직후 즉시 교체**
   - dense `weight`가 통제된 플레이스홀더 형태로만 일시적으로 존재하고,
   - 실제 GPU 상주 비용이 발생하기 전에 양자화 상태로 교체

선호 방향은 (1) 또는 (2)이다. 이 계획은 양자화 모드에서
**일반 모델 구성의 일부로 dense GPU 가중치 텐서를 생성해서는 안 된다**고 가정해야 한다.

**CUDA graph 상호작용 참고**: Python 수준 forward 분기는 graph 캡처 시점에 한 번 평가되고
이후 모든 리플레이에서 고정된다. 양자화 모드에서 캡처된 모델은 항상 양자화 분기를 리플레이하고,
dense 모드에서 캡처된 모델은 항상 dense 분기를 리플레이한다. 양자화 연산 자체가
graph 안전하기만 하면 문제없다. Phase 0에서 선택된 백엔드에 대해 이를 검증해야 한다.

### 6.4 새 모듈 클래스가 허용되는 경우

런타임 백엔드가 더 깔끔한 분리를 강제하는 경우, 다음과 같은 경량 래퍼를 사용한다:

- `ColumnParallelAWQLinear`
- `RowParallelAWQLinear`

단, 구현 명확성을 위해 필요한 경우에만.

그 경우에도:

- forward 시그니처는 현재 모듈과 일치해야 하고,
- TP 의미론은 변경 없이 유지되어야 하며,
- 교체는 모듈 구성 / 로드 시점에만 발생해야 한다.


## 7. 오프라인 아티팩트 전략

### 7.1 AWQ는 오프라인 우선 계획

현재 로드 시점 torchao 접근법과 달리, 이 계획은 양자화가
정상 SSD 서빙 시작 **전에** 수행된다고 가정한다.

이것은 의도적이다.

주 런타임은 다음을 로드해야 한다:

- 양자화된 랭크별 아티팩트,
- 전체 dense 가중치가 아님.

### 7.2 외부 아티팩트 vs SSD 로컬 아티팩트

가능하면 런타임이 원시 외부 AWQ 체크포인트 형식에 직접 의존하지 **않아야** 한다.

선호 흐름:

1. 원본 HF 체크포인트
2. 외부 AWQ 양자화
3. AWQ 체크포인트 / 파일
4. **SSD 임포터**
5. SSD 네이티브 랭크별 아티팩트
6. 런타임 로드

이렇게 하면 다음을 통제할 수 있다:

- TP 샤딩,
- 패킹 모듈 명명,
- 시작 속도,
- 버전 검증,
- 런타임에 필요한 정확한 메타데이터.

### 7.3 SSD 네이티브 아티팩트 내용

SSD 네이티브 AWQ 아티팩트는 TP 랭크별로 다음을 저장해야 한다:

- 양자화 방법: `awq_int4`
- 비트
- 그룹 크기
- 제로포인트 플래그
- 연산 dtype 기대값
- 소스 모델 ID / 리비전
- TP 크기 / 랭크
- 백엔드 종류
- 모듈 이름 목록
- 모듈별:
  - 패킹된 양자화 가중치
  - 스케일
  - 제로포인트 (사용 시)
  - 직접 런타임 로드에 필요한 백엔드별 레이아웃 메타데이터

### 7.4 아티팩트 명명 및 버전 관리

필수 메타데이터:

- `artifact_version`
- `quant_scheme`
- `backend`
- `model_id`
- `tp_size`
- `tp_rank`
- `group_size`
- `use_zero_point`
- `expected_runtime_dtype`
- `quantize_lm_head`
- `quantize_embeddings`

이것은 best-effort 캐시가 아닌 엄격한 런타임 계약으로 취급해야 한다.


## 8. 로더 계획

### 8.1 주요 규칙

GPU에 전체 dense 가중치를 먼저 적재하지 **않는다**.

새 로더 흐름은:

1. AWQ SSD 아티팩트 감지
2. 모델 모듈 구축
3. 타겟 디바이스에 양자화 저장소 할당
4. 패킹된 가중치/스케일/메타데이터를 직접 로드
5. dense GPU 가중치 로드를 완전히 건너뜀

이것은 추가적인 런타임 규칙을 함축한다:

- 양자화 모드는 현재의 "dense GPU 파라미터를 먼저 생성한 뒤
  나중에 덮어쓰는" 동작에 의존할 수 없다.

로더 작업만으로는 충분하지 않다; §6.3.1의 모듈 구성 계약도
함께 구현되어야 한다.

### 8.2 Dense 체크포인트 로드가 유지되는 경우:

- 비양자화 경로
- 드래프트 경로
- 개발 중 미지원 모델 패밀리
- 폴백 디버깅

### 8.3 임포트 시점 CPU 작업

원샷 임포터 경로가 필요한 경우:

- CPU에서 외부 AWQ 체크포인트를 읽고,
- CPU에서 SSD 명명 및 TP 레이아웃으로 리패킹하고,
- SSD 네이티브 아티팩트를 기록하며,
- dense GPU 적재가 절대 필요하지 않다.


## 9. TP 및 패킹 모듈 매핑

### 9.1 현재 SSD 의미론 보존 필수

현재 SSD는 다음을 사용한다:

- column-parallel 샤드 규칙
- row-parallel 샤드 규칙
- 패킹된 QKV 로딩
- 패킹된 gate/up 로딩

이러한 의미론은 변경 없이 유지되어야 한다.

### 9.2 필수 매핑 작업

Llama 패밀리 모델의 경우:

- `q_proj`, `k_proj`, `v_proj` -> `qkv_proj`
- `gate_proj`, `up_proj` -> `gate_up_proj`

이 매핑은 이미 dense SSD에 존재하며, 재발명하지 않고 재사용해야 한다.

### 9.3 AWQ 임포터 책임

임포터는 외부 AWQ 텐서가 SSD 패킹 모듈 저장소에 어떻게 매핑되는지
정확히 정의해야 한다. 이것은 계획에서 구현 복잡도가 가장 높은 섹션이다.

#### 9.3.1 패킹 모듈의 concat 순서

SSD는 여러 HF 프로젝션을 단일 모듈로 패킹한다. 임포터는
외부 AWQ 텐서를 **출력 차원(dim=0)** 방향으로 정확히 다음 순서로 concat해야 한다:

- `qkv_proj`: `q_proj` → `k_proj` → `v_proj`
- `gate_up_proj`: `gate_proj` → `up_proj`

관련된 모든 메타데이터(스케일, 제로포인트)도 같은 차원에서 동일한
순서로 concat되어야 한다.

#### 9.3.2 Concat 후 샤드 vs 샤드 후 concat

선호 순서: **먼저 concat, 그 다음 TP 샤드**.

근거:

- 외부 AWQ 아티팩트는 프로젝션별 텐서를 저장하고(사전 패킹이 아님),
- 먼저 concat하면 전체 SSD 패킹 텐서가 생성되고,
- 그 다음 TP 샤드 규칙을 적용하면 올바른 랭크별 슬라이스를 얻는다.

대안(각 프로젝션을 먼저 샤딩한 뒤 concat)도 유효하지만 검증이 더 어렵고,
q/k/v의 크기가 다른 QKV에서 오류가 발생하기 쉽다.

#### 9.3.3 모듈 유형별 TP 샤딩 규칙

- **ColumnParallelLinear** (`qkv_proj`, `gate_up_proj`):
  **출력 차원(dim=0)**에서 샤딩. 그룹은 입력 차원이므로 이 샤딩에 의해
  그룹 경계가 영향받지 않는다.

- **RowParallelLinear** (`o_proj`, `down_proj`):
  **입력 차원(dim=1)**에서 샤딩. 그룹도 입력 차원이므로 임포터는
  `shard_size % group_size == 0`을 검증해야 한다. 실패하면
  임포터는 해당 구성을 거부해야 한다.

#### 9.3.4 그룹 경계 정렬 검증

모든 RowParallelLinear 모듈에 대해 임포터는 다음을 assert해야 한다:

```
input_size_per_partition = input_size // tp_size
assert input_size_per_partition % group_size == 0, \
    f"RowParallel shard size {input_size_per_partition} not divisible by group_size {group_size}"
```

group_size=128인 표준 Llama 패밀리 모델의 경우, 모든 알려진 구성
(8B, 34B, 70B at TP=1/2/4/8)에서 이것이 성립한다. 하지만 assert는 반드시 있어야 한다.

#### 9.3.5 스케일 및 제로포인트 텐서 샤딩

스케일과 제로포인트 텐서는 `[out_features, num_groups]` shape을 가지며,
`num_groups = in_features // group_size`이다.

- **ColumnParallel 샤드**: dim=0(출력)에서 스케일을 슬라이스, dim=1은 그대로 유지.
- **RowParallel 샤드**: dim=0(출력)은 그대로 유지, dim=1(그룹)에서 스케일을
  슬라이스. 그룹이 샤딩되는 입력 채널에 대응하기 때문.

#### 9.3.6 통합 전 필수 단위 테스트

런타임 통합 진행 전:

1. 왕복 테스트: concat → 샤드 → 로드 → shape이 예상 랭크별 크기와 일치하는지 검증
2. 수치 테스트: 샤딩된 양자화 matmul == 비샤딩 양자화 matmul (예상 허용 오차 내)
3. 엣지 케이스 테스트: group_size가 shard 크기를 나누지 못할 때 assert 실패


## 10. 모델 범위

### 10.1 Phase-1 모델 패밀리

첫 구현 대상:

- Llama 패밀리만

이유:

- 패킹 모듈 매핑이 이미 명확하고,
- 현재 SSD 양자화/디버그 작업이 이미 Llama 패밀리 중심이며,
- MESA 타겟 경로도 현재 거기서 가장 중요하다.

### 10.2 초기 범위 밖

- Qwen3
- EAGLE 드래프트 양자화
- 양자화된 임베딩
- 양자화된 `lm_head`
- 범용 다중 모델 패밀리 임포터

Qwen3는 Llama 패밀리 통합이 안정화된 후 추가할 수 있다.

### 10.3 향후 선택적 범위를 위한 가중치 타이 참고

Llama 패밀리와 Qwen3 모두 HF config에서 `tie_word_embeddings=True`일 때
`lm_head.weight`를 `embed_tokens.weight`에 타이한다.

이것은 첫 AWQ 통합에 영향을 미치지 않는다:

- 임베딩은 dense로 유지하고,
- `lm_head`는 dense로 유지하며,
- 양자화는 무거운 내부 프로젝션만 대상으로 한다.

`lm_head` 양자화를 나중에 재검토할 경우, 구현은 다음 중 하나를
명시적으로 선택해야 한다:

- 양자화 전에 `lm_head` 타이 해제,
- 임베딩과 `lm_head` 모두 dense 유지,
- 또는 타이된 양자화 임베딩/logit 경로를 의도적으로 지원.


## 11. MESA 정책

### 11.1 최종 목표에 MESA 포함

양자화된 타겟 경로가 다음에서도 동작하지 않으면 이 계획은 완료되지 않는다:

- 일반 자기회귀 디코드
- 검증 경로
- MESA 타겟 검증 경로

### 11.2 MESA별 정책

데이터가 달리 입증하지 않는 한 이 정책을 유지한다:

- 타겟 무거운 linear 레이어: 양자화
- `lm_head`: 기본 dense

이것이 MESA 프록시 품질과 수락률을 위한 가장 안전한 첫 정책이다.

### 11.3 MESA 검증은 선택사항이 아님

AR과 일반 검증이 통과하더라도, 다음이 측정될 때까지 계획은 완료되지 않는다:

- 분할 MESA 검증 캡처,
- MESA 타겟 경로 정확성,
- dense `lm_head` vs 양자화된 `lm_head`의 수락률 영향


## 12. 백엔드 선택 게이트

구현 시작 전에 구체적인 런타임 백엔드 스타일을 선택해야 한다.

### 12.1 후보 클래스

일반적인 `F.linear` 폴백이 아닌 **AWQ 스타일 최적화 런타임**을 평가하고 있다.

후보 방향의 예:

- AWQ + Marlin 스타일 백엔드
- AWQ + 우리 dtype/shape을 지원하는 다른 최적화 런타임 백엔드

### 12.2 선택 기준

다음 조건이 모두 충족될 때만 후보를 수용한다:

1. SSD 관련 작은 M 디코드 유사 shape에서 동작
2. SSD 프리필 유사 더 큰 GEMM shape에서 동작
3. GPU에 양자화 저장소 유지
4. TP 로컬 matmul 통합과 호환
5. CUDA graph 캡처와 호환되거나 그렇게 되기 위한 명확한 계획이 있음
6. 최소 fp16 런타임 지원; bf16 지원 강력히 선호

### 12.3 후보가 통과하지 못한 경우

AWQ 런타임 후보가 Phase 0 게이트를 통과하지 못하면, 즉시 커스텀
Triton 커널 구현으로 전환하지 **않는다**.

대신:

1. 현재 torchao 경로를 임시 bf16 폴백으로 유지,
2. 백엔드 검색을 더 좁히고,
3. 집중적인 백엔드 평가 문서 후에만 커스텀 커널 작업을 고려.

커스텀 커널 작업은 기본 계획이 아닌 최후의 수단이다.


## 13. Config 설계

### 13.1 새 config 형태

불리언을 흩뿌리지 않고 구조화된 config을 사용한다.

권장 형태:

```python
@dataclass
class QuantConfig:
    enabled: bool = False
    method: str = "none"          # "none" | "awq_int4"
    target: bool = True
    draft: bool = False
    quantize_lm_head: bool = False
    quantize_embeddings: bool = False
    artifact_path: str | None = None
    artifact_mode: str = "load_only"   # "load_only" | "import_then_load"
    runtime_backend: str = "auto"      # Phase 0에서 선택된 구체적 백엔드
    quant_source: str = "ssd_artifact" # "ssd_artifact" | "external_awq"
    external_quant_path: str | None = None
    group_size: int = 128
    use_zero_point: bool = True
```

### 13.2 기본 정책

초기 릴리스 기본 정책:

- `enabled=False`
- `method="none"`
- 활성화 시 타겟만
- `quantize_lm_head=False`
- `quantize_embeddings=False`
- 오프라인 아티팩트 필요

### 13.3 현재 flat config으로부터의 마이그레이션

현재 SSD 코드는 다음과 같은 flat 양자화 관련 필드를 사용한다:

- `target_quant_enabled`
- `target_quant_backend`
- `target_quant_lm_head`
- `target_quant_mode`
- `target_quant_artifact_prefix`

새 구조화된 config은 전체 engine/bench 스택의 flag-day 재작성을
요구해서는 안 된다.

마이그레이션 규칙:

1. `QuantConfig` 도입
2. 기존 flat CLI/config 필드를 임시 호환 shim으로 유지
3. LLM/러너 경계에서 레거시 필드로부터 `QuantConfig`을 파생
4. AWQ 경로가 안정화된 후에만 레거시 flat 필드를 제거


## 14. 정확한 구현 Phase

## Phase 0. 백엔드 실현 가능성 스파이크

### 목표

구체적인 AWQ 런타임 방향을 선택한다.

### 작업

1. SSD 관련 로컬 행렬 shape에서 후보 백엔드를 측정:
   - 디코드 유사 작은 M
   - 검증 유사 작은 M
   - 프리필 유사 더 큰 M
2. fp16 런타임 지원 확인
3. bf16 런타임 지원 확인
4. graph 캡처 안전성 확인
5. 로컬 샤드 shape이 지원되는지 확인

### 산출물

다음을 포함하는 짧은 기술 노트:

- 선택된 백엔드
- 지원되는 런타임 dtype
- 알려진 미지원 shape
- graph 호환성 결과
- SSD가 torchao를 bf16 폴백으로 유지해야 하는지 여부

### 하드 게이트

이 phase가 종료되기 전에 광범위한 통합을 시작하지 않는다.

### 선택적 탐색 확인

Phase 0 동안 낮은 우선순위의 부수적 조사로, AWQ 캘리브레이션된 가중치를
현재 torchao INT4 런타임 경로를 통해 로드할 수 있는지 확인한다. 이것은
torchao의 내부 AQT/tile-packed 레이아웃 계약을 맞춰야 하므로 비자명하며
상당한 어댑터 작업 없이는 실현 불가능할 수 있다. 이것을 주요 단순화 경로로
취급하지 **않는다** — 알면 좋은 가설일 뿐이다.


## Phase 1. 런타임 양자화 상태 스켈레톤

### 목표

인메모리 양자화 상태와 외부 패킹된 AWQ 가중치를 로컬 TP 모듈에 부착하고
실행하는 데 필요한 최소한의 TP 모듈 변경을 정의한다.

이 phase는 실제 로더 통합 **이전에** 존재한다. thin adapter 경로와
SSD 네이티브 아티팩트 경로 모두 패킹된 가중치를 위한 런타임 목적지가 필요하기 때문이다.

### 작업

1. TP linear 모듈의 양자화 상태 소유권 정의
2. dense vs 양자화 forward 디스패치 shape 계약 정의
3. §6.3.1의 양자화 모드 모듈 구성 계약 정의
4. 기존 `weight_loader(param, loaded_weight[, shard_id])`
   계약이 양자화 모드에서 어떻게 보존 또는 적응되는지 정의
5. 기존 TP 클래스를 제자리에서 확장할 수 있는지 또는 경량 래퍼가 필요한지 결정

### 성공 기준

- 패킹된 AWQ 가중치, 스케일, 제로포인트, 백엔드 메타데이터에 대한
  구체적인 인메모리 표현이 존재
- TP 모듈이 dense GPU 가중치 상주 없이 양자화 모드로 존재 가능
- 현재 로더의 패킹/QKV/병합 로더 호출 규약이 양자화 모드에서도
  명확한 등가물을 가짐
- 이 계약이 종료되기 전에 로더나 임포터 작업을 시작하지 않음


## Phase 2. 런타임 양자화 상태 + 로컬 Matmul 어댑터

### 목표

모델 토폴로지를 변경하지 않고 AWQ 기반 로컬 linear 실행의 런타임 지원을 추가한다.

### 작업

1. TP linear 모듈이 양자화 상태를 보유하도록 확장
2. AWQ 기반 matmul을 위한 로컬 런타임 분기 추가
3. 기존 dense 경로 그대로 유지
4. row/column/QKV/merged 의미론 보존

### 관련 예상 파일

- `ssd/ssd/layers/linear.py`
- `lm_head`에 대한 선택적 후속 작업이 있을 경우 `ssd/ssd/layers/embed_head.py`

### 성공 기준

- 단위 테스트: 로컬 양자화 경로가 예상 오차 내에서 dense 참조와 일치
- TP 의미론 여전히 정확
- 양자화 모드에서 dense GPU 가중치 저장소 불필요


## Phase 3a. 외부 AWQ 체크포인트용 Thin 어댑터

### 목표

전체 SSD 네이티브 아티팩트 파이프라인을 먼저 구축하지 않고 외부 AWQ 체크포인트를
SSD에 직접 로드하여 빠른 백엔드/런타임 검증을 가능하게 한다.

### 작업

1. 외부 AWQ 체크포인트 읽기 (HF safetensors + `quantize_config.json`)
2. HF 모듈 이름을 런타임에 SSD 모듈 이름으로 매핑
3. CPU에서 `qkv_proj`와 `gate_up_proj` 리패킹
4. CPU에서 TP 샤딩 규칙 적용
5. 패킹된 가중치 + 스케일 + 제로포인트를 런타임 모듈에 직접 로드

### 관련 예상 파일

- `ssd/ssd/utils/` 또는 `ssd/ssd/quant/` 아래의 새 thin 로더/어댑터
- `ssd/ssd/utils/loader.py` 또는 `ssd/ssd/engine/model_runner.py` 수정

### 성공 기준

- 외부 AWQ 체크포인트가 크래시 없이 SSD 런타임에 로드
- SSD 네이티브 아티팩트 파일 불필요
- Phase 3b를 기다리지 않고 엔드투엔드 런타임 검증 가능


## Phase 3b. SSD 네이티브 사전 샤딩 아티팩트 파이프라인

### 목표

프로덕션 시작 속도를 위한 오프라인 아티팩트 파이프라인을 구축한다.

### 작업

1. 외부 AWQ 체크포인트를 입력으로 받기
2. HF 모듈 이름을 SSD 모듈 이름으로 변환
3. `qkv_proj`와 `gate_up_proj` 리패킹
4. TP 샤딩 규칙 적용
5. SSD 네이티브 랭크별 아티팩트 저장
6. 매니페스트/버전 메타데이터 작성

### 관련 예상 파일

- `ssd/scripts/` 아래의 새 임포터 스크립트
- `ssd/ssd/utils/` 또는 `ssd/ssd/quant/` 아래의 새 헬퍼 모듈

### 성공 기준

- 임포터가 전적으로 CPU에서 실행
- 랭크별 아티팩트 파일 생성
- SSD 런타임 부팅 없이 아티팩트 검사 가능
- 시작 시간이 Phase 3a thin 어댑터 경로보다 실질적으로 빠름


## Phase 4. 로더 통합

### 목표

SSD 네이티브 AWQ 아티팩트를 런타임에 직접 로드한다.

### 작업

1. 아티팩트 감지 추가
2. 아티팩트 메타데이터 검증 추가
3. 아티팩트에서 양자화 상태를 직접 인스턴스화
4. dense GPU 가중치 적재를 건너뜀
5. 양자화 비활성화 시 dense 로더 동작을 그대로 유지

### 관련 예상 파일

- `ssd/ssd/utils/loader.py`
- `ssd/ssd/engine/model_runner.py`
- `ssd/ssd/config.py`
- `ssd/bench/bench.py`

### 성공 기준

- 양자화된 타겟이 dense 가중치를 GPU에 먼저 배치하지 않고 로드
- 드래프트는 여전히 dense 경로로 로드
- 시작 VRAM이 실질적으로 감소


## Phase 5. 엔드투엔드 타겟 전용 검증

### 목표

양자화된 타겟 경로가 실제 엔진에서 동작하는지 확인한다.

### 필수 검사

1. AR 디코드
2. 검증 경로
3. 추론적 경로 하나
4. CUDA graph 캡처/리플레이
5. prefix 캐시 경로
6. TP gather/all_reduce 정확성

### 성공 기준

- 크래시 없음
- 올바른 shape/dtype
- 짧은 프롬프트에서 안정적 생성
- 필수 핫 경로에서 graph 캡처 성공


## Phase 6. MESA 검증

### 목표

MESA 하에서 양자화된 타겟을 검증한다.

### 필수 검사

1. 분할 검증 캡처가 여전히 동작
2. 타겟 MESA 검증 경로 정확성
3. `lm_head` dense 기본 기준선
4. 이후 활성화 시 선택적 `lm_head` 양자화 ablation
5. 수락률 / 처리량 비교

### 성공 기준

- MESA 경로가 올바르게 실행
- 기본 dense `lm_head` 하에서 심각한 수락률 붕괴 없음


## Phase 7. 성능 및 시작 최적화

### 목표

백엔드가 실제로 사용할 가치가 있는지 확인한다.

### 필수 벤치마크

1. 디코드 유사 마이크로벤치
2. 프리필 유사 마이크로벤치
3. 엔드투엔드 AR
4. 엔드투엔드 추론적
5. 엔드투엔드 MESA 타겟 경로

### 비교 대상

- dense fp16/bf16 기준선
- 해당되는 경우 현재 torchao 폴백
- AWQ 경로

### 필수 분석

성능이 나쁜 경우 먼저 확인:

- 잘못된 백엔드 선택
- 숨겨진 dense 적재
- 추가 복사
- 잘못된 샤드 패킹
- 나쁜 작은 M 동작
- eager로의 graph 폴백


## 15. 변경 예상 파일

예상 주요 파일:

- `ssd/ssd/config.py`
- `ssd/bench/bench.py`
- `ssd/ssd/utils/loader.py`
- `ssd/ssd/layers/linear.py`
- `ssd/ssd/engine/model_runner.py`

예상 신규 파일:

- AWQ 아티팩트 임포터 스크립트
- AWQ 런타임 헬퍼 모듈
- 검증 / 스모크 / 마이크로벤치 스크립트

가능한 추후 파일:

- `lm_head` 양자화를 재검토할 경우 `ssd/ssd/layers/embed_head.py`


## 16. 리스크

### 16.1 최고 리스크

1. 선택된 AWQ 런타임이 SSD 디코드 유사 작은 M shape을 잘 지원하지 않음
2. graph 캡처 호환성이 예상보다 약함
3. 외부 AWQ 아티팩트 스키마가 SSD 패킹 모듈에 매핑하기 어려움
4. bf16 런타임 지원이 fp16 런타임 지원보다 약함
5. AWQ 캘리브레이션이 SSD 사용 사례에서 더 단순한 round-to-nearest 양자화 대비
   유의미한 품질 개선을 내지 못함 — 특히, MESA 수락률과 생성 품질이
   naïve W4A16 양자화가 생산하는 것보다 실질적으로 낫지 않다면,
   오프라인 캘리브레이션 파이프라인의 복잡도 대비 ROI가 낮음

### 16.2 명시적 완화

1. Phase 0에서 약한 백엔드를 조기에 거부해야 함
2. 필요시 현재 torchao 경로를 임시 bf16 폴백으로 유지
3. `lm_head`를 초기에 양자화하지 않음
4. 드래프트를 초기에 양자화하지 않음
5. Phase 5에서 AWQ vs round-to-nearest의 MESA 수락률을 조기에 측정;
   차이가 미미하면 캘리브레이션 파이프라인 단순화를 고려


## 17. 비목표

다음은 첫 AWQ 통합의 범위에서 의도적으로 제외한다:

- 직접적인 bitsandbytes 런타임 통합
- 처음부터 만드는 Triton 양자화 GEMM 백엔드
- 드래프트 양자화
- 양자화된 임베딩
- 기본 양자화 `lm_head`
- 첫날부터의 범용 다중 모델 패밀리 지원
- AWQ가 입증되기 전에 현재 torchao 코드 삭제


## 18. 최종 권고

올바른 v2 방향은:

1. **SSD 엔진 아키텍처를 그대로 유지**
2. **현재 torchao 경로는 임시 폴백으로만 유지**
3. **AWQ 스타일 오프라인 아티팩트 파이프라인 추가**
4. **로컬 TP linear 경계에서 최적화된 AWQ 런타임 통합**
5. **타겟 전용 우선**
6. **Llama 패밀리 우선**
7. **작업 완료 선언 전에 MESA 검증 필수**

이것은 실제 문제를 해결하는 가장 좁은 계획이다:

- fp16/bf16 실용적 지원,
- VRAM 절감,
- 최적화된 런타임 경로,
- SSD 아키텍처 보존.
