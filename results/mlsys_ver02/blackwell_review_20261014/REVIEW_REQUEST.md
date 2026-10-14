# MLSys ver0.1과 ver0.2 Blackwell 검증 수정 요청

작성일: 2026-10-14. 전달 대상: MLSys 통합 브랜치의 엔진·테스트·실험 실행기 작업자.

**현재 70B 검증은 완료되지 않았다.** B=1 시스템 비교와 일부 연산 검사는 끝났지만, B=2 초기화의 FlashInfer workspace 부족으로 대기열이 중단됐다. 또한 이 서버의 검증 실행기가 디버그 출력을 켜고 기존 fanout 설정을 바꾼 사실이 확인됐다. 따라서 이번 약 56 TPS를 기존 약 65 TPS 대비 ver0.2의 성능 회귀로 사용하면 안 된다.

이 문서는 확인된 문제, 수정 요청, 재검증 기준을 모은 인수인계다. **이 문서 작성으로 runtime을 수정하거나 GPU 실행을 재개하지 않았다.** 원시 결과와 기존 논문 수치는 변경하지 않았다.

## 1 검토 기준과 현재 상태

| 항목 | 기준 |
|---|---|
| 비교 기준 | `MLSys-ver0.1` 엔진 `f12d0f1da6a2e4a26e9cb11ddcca54c189736d19` |
| 검토 대상 | `MLSys-ver0.2` 엔진 `76c98f946820179a121b5326d2eb8a2c4e3e3545` |
| 별도 검증 브랜치 | `MLSys-ver0.2-test` |
| 주 평가 모델 | LayerSkip Llama-2-70B BF16 + TinyLlama 1.1B, target TP2 |
| 주 평가 조건 | B=1, k1/k2=8/4, exit56, 출력1024, 전체 문맥2048, T=0.7, seed42 |
| 추가 검사 | B>1 항목만 B2/B8. 새 모델 조합은 제시된 파라미터의 호환성 검사 |
| 서버 GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120, GPU 0·1·2 |
| 설치 환경 | PyTorch 2.8.0, FlashInfer 0.5.2, sgl-kernel 0.3.17.post1, Transformers 4.57.1, driver 580.178.04 |
| 진행 상태 | 31개 작업 중 13개 결과 기록 완료. 최초 실패 진단 보존 1개를 포함하므로 모두 성공한 본 실험 13개라는 뜻이 아님 |
| 중단 작업 | `12_native_70b_b2_t07/attempt1`, `status.state=needs_attention` |
| 미완료 | native 모델·batch·greedy 검사 13개와 전체 데이터 알고리즘 비교 5개. 실패한 B2 재시도 포함 |

완료 표시는 원시 산출물 해시와 함께 확인했다. 원본 회귀 검사는 26개 묶음 중 23개가 통과했고, 실패한 3개는 원본 로그를 보존한 별도 진단으로 다뤘다. 작은 연산 검사 통과를 70B 전체 실행 통과로 대체하지 않는다.

아래 코드 경로는 각 검토 브랜치의 저장소 루트 기준이다. 링크는 **현재 Blackwell 서버의 증거 위치**이며, 다른 서버에서는 해당 로그와 문서를 함께 전달받아야 한다. 이 서버의 절대 경로가 든 launcher/manifest를 그대로 production에 복사하지 않는다.

## 2 수정 우선순위와 발생 범위

| ID | 우선순위 | 수정 또는 확인할 사항 | 발생 범위와 판정 |
|---|---|---|---|
| R1 | P0 | Assertion과 매 토큰 디버그 출력을 분리 | 출력 구조는 ver0.1부터 존재. 이번 검증 실행기에서 활성화되어 측정 조건 오염 |
| R2 | P0 | 원래 fanout과 모든 파생 config 복원 | 기존 제약과 논문 설정의 불일치를 검증 adapter가 3→4로 바꾸며 회피. ver0.2 알고리즘의 자동 변경이 아님 |
| R3 | P1 | Draft Config 생성 시 exit layer를 target 기준으로 검사 | ver0.1과 ver0.2 양쪽에 남은 조건 검사 문제 |
| R4 | P1 | Blackwell attention backend 및 cached-prefill 사전 plan 계약 검사 | ver0.1의 FlashInfer 대응을 유지해야 함. 기존 테스트가 SGL을 강제해 우회 경로를 무시 |
| R5 | P0 | 70B B>1 FlashInfer workspace 크기와 시작 전 검사 | 새 B>1 target 경로의 128 MiB가 이번 shape에서 부족. B1 override와 별개 |
| R6 | P1 | Raw 점수 오차와 전송 양자화 오차를 분리한 회귀 검사 | ver0.2 dense/flat 비교. 전체 tolerance를 느슨하게 만드는 해결 금지 |
| R7 | P1 | 최초 rank 실패 시 자기 작업의 다른 rank를 빠르게 종료 | B2 target 실패 뒤 draft가 약 600초 추가 대기. 최초 원인과 NCCL 후속 오류 구분 |
| R8 | P1 | 중단 상태와 검증 범위를 보고서에 정확히 표시 | 이 서버 `report.py`가 실패한 작업도 ‘실행 중’으로 표시. 공통 엔진 버그와 별개 |
| R9 | P1 | 모델별 tokenizer 의존성·dtype·실제 생성 검증 | 설치 준비 및 token-ID 검사만으로 모델 지원 완료라고 판정하지 않음 |
| R10 | P1 | 새 기본 정책과 기존 비교군을 명시적으로 분리 | 최신 원격 기본값에는 calibration tree와 full-vocabulary 정규화가 포함됨. 이 서버의 q-path/topm 방향과 다름 |

P0는 성능 비교 또는 남은 실험의 재개를 막는 항목이다. 모든 항목이 새 ver0.2 회귀라는 뜻은 아니다.

## 3 성능 측정과 기존 기준 복원

### R1 디버그 출력과 안전 검사를 분리

기존 실행 명령은 `python -B -O`였다. 이번에는 assertion을 유지하기 위해 `-O`를 뺐지만, 그 결과 `__debug__`도 참이 됐다.

관련 코드:

- `ssd/ssd/engine/step.py`: `SpecDecodeStep`의 GPU tensor 출력, `.tolist()`, 토큰별 문자열 변환, `flush=True` 출력.
- `ssd/ssd/engine/verifier.py`: recovery 토큰 문자열 출력과 평균 suffix 길이 출력.
- 이 서버 `results/blackwell_ver02_validation/common.py`: 실행 명령 구성.

이번 28턴 latency 실행은 측정된 검증 단계 3,475개였으며, 로그에는 warmup을 포함한 speculation 출력이 3,899회 있었다. GPU 값을 읽는 동기화와 CPU 문자열 변환·파일 출력은 inference 경로에 포함될 수 있다. 실제 시간 영향은 별도 대조 없이 확정하지 않는다.

수정 요청:

1. 상세 토큰 출력은 명시적인 opt-in 설정으로 분리한다. 예를 들어 `SSD_DEBUG_TOKENS=0`을 기본으로 하고, **문자열 변환과 tensor 읽기 자체가 조건문 안에서만** 실행되게 한다. 이름은 제안이며 현재 구현된 옵션이 아니다.
2. 필수 config·shape·문맥 경계 검사는 `if ...: raise ValueError/RuntimeError`처럼 `python -O`에서도 유지되게 한다. 출력 제거를 위해 모든 assertion을 무조건 끄는 방식으로 문제를 숨기지 않는다.
3. B1 latency/TPS는 토큰 출력과 진단 profile을 끈 상태로 다시 측정한다. correctness 실행과 performance 실행의 설정을 구분해 기록한다.

완료 기준: 기본 실행에 매 단계 tensor/토큰 출력이 없고, 상세 출력 ON/OFF에서 같은 제어된 입력과 난수 상태의 출력·AL·hit가 보존되며, 잘못된 설정은 둘 다 거부한다.

### R2 Fanout을 바꾸지 않고 기존 실행 조건을 재현

저장된 target rank Config를 대조한 차이:

| 항목 | 기존 `17_full_latest` | 이번 `04_latency_v01_r1` |
|---|---:|---:|
| `async_fan_out` | 3 | 4 |
| `duet_draft_fan_out` | 3 | 3 |
| `duet_proxy_fan_out` | 0 | 1 |
| `fan_out_list` | 3이 13개 | 4가 13개 |
| `fan_out_list_miss` | 3이 13개 | 4가 13개 |
| target `MQ_LEN` | 39 | 52 |
| 명시 P2 budget / root count | 15 / 10 | 15 / 10 |

기존 `Config.__post_init__`에는 `0 < duet_draft_fan_out < async_fan_out` assertion이 있다. 과거 `-O` 실행에서는 이 검사가 생략됐다. 이번 검증 adapter가 이를 통과시키려고 `async_fan_out=max(current, DS+1)`로 바꾼 것이 차이의 직접 원인이다. **이는 이 서버의 검증 구성 오류이며, 새 최적화 코드가 스스로 fanout을 바꾼 것이 아니다.**

명시 P2 budget이 별도로 존재하므로, `duet_proxy_fan_out=0`만 보고 기존 P2가 없었다고 해석하면 안 된다. 반대로 모든 실행 모드에서 이 조합이 안전하다고 단정해서도 안 된다.

수정 요청:

1. 위 fanout 자동 증가 adapter를 일반 성능 비교에서 제거한다.
2. 명시 split P2 budget을 사용하는 경우와 기존 fanout 차이로 P2를 배분하는 경우를 구분해 config 계약을 검토한다. 합법적인 split 구성이라면 그 모드에 맞는 검사를 작성한다. 해당 assertion을 모든 모드에서 삭제하지 않는다.
3. 근거 있는 조건부 검사 또는 명시적인 legacy 재현 adapter로 기존 값을 보존한다. 기존 구성을 보존할 수 없다면 ‘같은 설정 재현’이 아니라 별도 설정 변경으로 보고한다.
4. 사용자 인자뿐 아니라 rank별 최종 Config와 graph shape를 비교한다. `fan_out_list`, `MQ_LEN`, split budget, root 수, verify cap, 온도, dtype 등을 자동 검증한다.

완료 기준: 원래 파생 config로 기준 실행을 재현한 뒤 **같은 조건으로** v0.1/v0.2만 바꿔 비교한다. 현재 AL 하락을 fanout 변경 한 가지 때문이라고 확정할 증거는 아직 없다.

### R3 Exit layer는 target의 층 수로 검사

`DraftRunner.create_draft_config()`가 `dataclasses.replace`로 draft 설정을 만들 때 `hf_config`가 TinyLlama 22층으로 바뀐다. 그런데 DUET의 exit layer 56은 target 80층에 대한 설정이다. 이를 draft의 층 수와 비교해 assertion이 실패했다. 양 버전에서 같은 문제가 확인됐고, 기존 `-O` 실행에서는 드러나지 않았다.

수정 요청:

- Target 역할과 draft 역할의 모델 config를 구분하고 target exit 검사는 target의 층 수를 기준으로 한다.
- 이 서버의 ContextVar 기반 우회는 검증용이다. 이를 설계 검토 없이 production에 그대로 붙이지 않는다.
- 잘못된 exit 값은 계속 거부한다. 이 구현의 0-based 계약에서 target80/exit56은 통과하고 exit80은 거부해야 한다.

CPU 확인에서는 두 버전 모두 위 두 조건을 통과했다. 본 수정은 엔진 정식 반영이 아니라 검증 adapter에만 존재한다. [검증 코드와 CPU 검사](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/test_validation.py)

### 현재 TPS의 사용 범위

| 실행 | Decode TPS | AL | Hit | ms/step |
|---|---:|---:|---:|---:|
| 이전 full `17_full_latest`, 후보 T=0.7 | 65.068 | 4.386 | 77.461% | 67.223 |
| 위 raw에서 이번과 같은 24문항/28턴만 추출 | 68.134 | 4.603 | 79.861% | 67.366 |
| 이번 v0.1 latency 2회 평균 | 56.480 | 3.894 | 71.523% | 68.663 |
| 이번 v0.2 시스템 latency 2회 평균 | 56.539 | 3.894 | 71.523% | 68.604 |

같은 문항 추출은 재실행이 아니다. 전체 실행과 subset 실행은 앞선 입력·warmup·난수 소비가 달라 생성 경로가 동일하지 않다. 이번 v0.1/v0.2의 28턴 출력과 검증 횟수·AL·hit는 서로 일치했지만, 이전 full에서 추출한 출력과는 일치하지 않았다.

이번 약 56 TPS를 논문 수치로 승격하거나 ‘ver0.2 때문에 약 14% 느려졌다’고 보고하지 않는다. AL 하락과 step 시간 증가를 분리해야 하며, 디버그 출력만으로 전체 차이를 설명해서도 안 된다.

## 4 Blackwell과 FlashInfer 대응

### R4 Ver0.1부터 있던 backend 전환과 plan 계약을 유지

**이 서버의 SM120과 설치된 sgl-kernel 조합에서는 Hopper attention 커널을 그대로 실행할 수 없다.** 원본 cached-prefill 검사에서 다음 오류가 발생했다.

```text
CUDA error (.../hopper/flash_fwd_launch_template.h:166):
no kernel image is available for execution on the device
```

이것은 이번 로그에서 확인한 커널 호환성 문제다. 모든 Blackwell 제품·모든 SGL 버전이 지원 불가라는 뜻은 아니며, 과거 서버의 GPU 초기화 장애와도 구분한다.

Ver0.1 `f12d0f1d`에 이미 있는 대응:

- `ssd/ssd/layers/fi_attn.py`: `SSD_ATTN_BACKEND=auto`가 이 서버에서 FlashInfer를 선택한다. 명시 선택값은 `flashinfer`다.
- `ssd/ssd/layers/attention.py`: 일반 prefill/decode/verify의 FlashInfer 경로.
- `ssd/ssd/engine/model_runner.py`: layer 실행 전 필요한 paged/ragged/graph wrapper를 계획한다.
- Tree 전용 wrapper는 일반 dense attention 선택과 별도로 FlashInfer를 사용한다. `auto` 한 값으로 모든 workspace가 같은 크기가 되는 것은 아니다.

이번 실패한 `ssd/tests/test_cached_prefill.py`는 `SSD_ATTN_BACKEND=sgl`을 강제했다. 이 테스트 파일과 `fi_attn.py`는 f12d0f1d 이후 해당 검토 범위에서 변경되지 않았다. 따라서 새 ver0.2에서 처음 생긴 문제로 분류하지 않는다.

테스트에서 `sgl`만 `auto`로 바꾸는 것도 충분하지 않았다. Fixture가 `ModelRunner`를 거치지 않고 `Attention`을 직접 호출해 FlashInfer wrapper가 아직 plan되지 않았기 때문이다. 실제 runner와 같은 `plan_paged(...)`를 추가한 별도 검사에서는 기존 수치 허용 오차를 유지한 채 통과했다.

수정 요청:

1. 지원되는 backend별로 테스트를 분리하고, SM120에서는 실제 serving의 `auto/flashinfer` 경로를 반드시 실행한다. 지원되지 않는 SGL 전용 테스트는 명시적으로 구분하고 skip을 pass 개수로 위장하지 않는다.
2. Cached-prefill fixture에 실제 runner의 plan 계약을 구현한다. 비연속 prefix page, cached/uncached 혼합, FP16/BF16 결과를 dense 기준과 대조한다.
3. Backend 및 plan 정보는 시작 로그에 남긴다. 실제 선택된 backend, GPU capability, FlashInfer/torch/sgl 버전, dtype, head 수, TP, query/page shape가 필요하다.
4. Plan을 각 transformer layer 안으로 옮기지 않는다. 기존 모델 호출 또는 graph replay 경계의 계획·재사용 규칙을 보존한다.
5. 환경 이식에서는 기존 FlashInfer adapter를 누락하지 않는다. SM120에서 SGL을 강제하거나 `TORCH_CUDA_ARCH_LIST`만 바꿔 이미 빌드된 미지원 커널이 실행될 것으로 가정하지 않는다.

이 서버가 사용한 기본 환경은 `SSD_ATTN_BACKEND=auto`, `SSD_CUDA_ARCH=12.0`, `TORCH_CUDA_ARCH_LIST=12.0`이다. 다른 GPU 서버에 이 아키텍처 값을 무조건 복사하지 않는다.

증거: [원본 backend 실패](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/checks/test_cached_prefill.process.log), [plan을 포함한 별도 진단](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/checks_adapted/cached_prefill.log), [진단 구현](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/server_checks.py).

### R5 Workspace는 경로와 실제 shape별로 점검

| 경로 | 제어 위치 | 이번 확인 |
|---|---|---|
| 일반 FlashInfer attention | `fi_attn.py` 내부 workspace | 기존 320 MiB. 아래 tree workspace와 별개 |
| B1 tree verify graph | `SSD_TREE_VERIFY_WORKSPACE_MB` | 기존 서버 검증에서는 224 MiB 명시 사용 |
| Draft tree executor | `SSD_TREE_EXEC_WORKSPACE_MB` | 원본 검사 64 MiB에서 약82.25 MiB 요구로 실패. 128 MiB 별도 진단 통과 |
| P1 draft executor | `SSD_P1_TREE_EXEC_WORKSPACE_MB` | 이번 검증 128 MiB 명시 사용 |
| 새 B>1 target/glue forward | `SSD_BATCH_TREE_WORKSPACE_MB` | 기본128 MiB. 실제70B B2 target prewarm에서188 MiB 요구로 실패 |

**기존 `SSD_TREE_VERIFY_WORKSPACE_MB=224`를 주어도 새 B>1 workspace는 증가하지 않는다.** 이번 중단의 직접 원인이다.

```text
batch_prefill_tmp_v with size 197132288
but only 134217728 bytes available in AlignedAllocator
Increase the workspace buffer size.
```

관련 코드: `ssd/ssd/engine/helpers/batch_tree_forward.py`의 `workspace()`와 `capture()`, `batch_tree_warmup.py`의 `prewarm_target()`.

수정 요청:

1. 우선 새 변수로 **256 MiB 등을 별도 재시도 후보**로 지정할 수 있다. 현재 실패 지점의188 MiB보다 크다는 뜻일 뿐, B2 전체 shape나 B8까지 충분하다고 검증된 값은 아니다.
2. 사용 모델·TP·dtype·B·query width·page bucket의 최대 요구량을 시작 전에 확인한다. 설정값은 양수인지 검사하고 부족하면 해당 shape와 요구/가용 크기를 함께 출력한다.
3. 필요하다면 시작 단계에서 한정된 재할당·재계획을 구현하되, 이미 capture된 주소를 바꾸거나 매 decode step마다 재할당하지 않는다. CUDA graph의 고정 주소 계약을 지킨다.
4. B1의 기존 graph별 wrapper/plan 상태 격리를 유지한다. Ver0.1에는 서로 다른 page shape capture가 앞선 graph 상태를 오염시키지 않도록 분리한 코드가 있다. 이를 메모리 절약 목적으로 근거 없이 공유시키지 않는다.
5. 새 B>1 경로의 공유 workspace도 shape를 번갈아 replay했을 때 이전 plan/출력/KV가 보존되는지 검사한다. **현재 공유로 인한 corruption을 관측했다는 뜻은 아니며**, workspace 수정 시 필요한 회귀 조건이다.

성공 기준: 실제70B에서 B1/B2/B8의 선언된 모든 warmup shape와 생성이 통과하고, eager/captured logits·KV·다른 요청의 prefix 격리가 보존되어야 한다. GPU 전체 VRAM을 더 예약하거나 GPU reset을 하는 것은 이 오류의 해결책이 아니다.

증거: [B2 최초 실패](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/runs/12_native_70b_b2_t07/attempt1/run.log), [별도 executor 검사](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/checks_adapted/batched_tree_executor.log).

## 5 수치 검사와 실행 관리

### R6 양자화 전후의 오차를 구분

원본 `test_flat_tree_runtime`은 packed score의 200개 항목 중 1개에서 최대 약1.6624e-6 차이로 실패했다. 전송 점수는 **16-bit 로그 양자화**이므로 FP16 반올림이라고 부르지 않는다.

별도 진단에서 확인한 범위:

- 양자화 전 raw score 최대 절대 차이 약2.98e-8. 원래의 엄격한 raw 허용 오차를 유지했다.
- 확인한 packed 차이는 로그 양자화 한 칸 수준이었다.
- 검사한 양수 후보의 위치·token ID가 일치했고, 고정 난수의 검증 출력 비교240회가 일치했다.

수정 요청:

1. Raw score는 기존 tolerance로 대조한다.
2. Packed score는 로그 bin 경계의 명시적인 계약으로 검사한다. Zero-mass/유효 후보 mask는 별도 검사하고, 0의 로그로 검사 자체가 무의미해지지 않게 한다.
3. 후보 순서, 동률 경계, 형제 순서, 수락/복구 출력, RNG 상태를 각각 대조한다.
4. 더 넓은 입력의 후보 순서까지 달라지면 ‘허용 가능한 양자화 차이’로 자동 통과시키지 말고 영향부터 확인한다.

현재 결과는 모든 입력의 bitwise 동일성 증명이 아니다. 단순히 `atol`을 키워 원본 실패를 없애는 수정은 요청하지 않는다. [원본 실패](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/checks/test_flat_tree_runtime.suite.log), [분리 진단](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/checks_adapted/flat_tree_runtime.json).

### R7 최초 실패를 빠르게 전파하고 자기 자원만 정리

이번에는 target prewarm의 workspace 오류가 먼저 발생했고, draft rank는 수신을 기다리다가 약600초 후 NCCL timeout을 출력했다. 이 timeout을 첫 원인으로 오인하지 않는다.

- 작업 supervisor는 최초 worker/rank의 실패를 감지해 같은 작업의 남은 rank를 종료하고 첫 traceback을 보존한다.
- 정상 종료와 오류 종료 모두 자신이 만든 process group/SHM만 정리한다. 다른 사용자 GPU 프로세스, `/dev/shm` 전체, 다른 semaphore를 건드리지 않는다.
- Ver0.1의 기존 전역 cleanup은 이 서버 adapter에서 격리했고 ver0.2에는 소유 자원 중심의 변경이 들어 있다. 이미 개선된 부분을 옛 전역 cleanup으로 되돌리지 않는다.
- 원시 attempt를 덮어쓰지 않고 재시도는 새 디렉터리에 기록한다. OOM/잘못된 config를 무한 재시도하지 않는다.

### R8 보고서 상태와 검증 범위 수정

이 서버 `results/blackwell_ver02_validation/report.py`는 완료되지 않은 작업의 이름이 `state.job`과 같으면 무조건 ‘실행 중’으로 표시한다. 그래서 `needs_attention`인 B2가 표에서는 실행 중으로 남았다.

수정 요청:

- `running`, `waiting_for_assigned_gpus`, `stopped`, `drained_after_job`, `needs_attention`, `complete`를 구분한다. PID와 마지막 heartbeat도 확인할 수 있게 한다.
- ‘진단 실행 완료’, ‘정확성 통과’, ‘실모델 생성 통과’, ‘성능 검증 완료’를 구분한다. 첫 실패 진단의 process exit0/status complete만 보고 정확성 통과로 세지 않는다.
- 전체 완료는 모든 필수 작업의 산출물·분모·hash가 검증됐을 때만 표시한다.
- 현재 빈 전체 데이터 표와 미완료 모델 항목을 성공으로 해석하지 않는다.
- 기존 무디코딩 종료의 엄격한 분류, `None` 처리, 유효 분모 공개, MT 문항 평균 규칙은 유지한다. 이 문제를 해결하려고 보호 종료를 없애거나 문맥 상한을 올리지 않는다.

이전 [무디코딩 종료 인수인계](../PSD-MLSys-ver0.1-test/results/blackwell_validation/VALIDATION_HANDOVER.md)는 계속 유효하다. 다만 그 문서의 ‘남은 실험’은 작성 당시 상태이며 현재 실행 상태의 근거는 각 캠페인의 status다.

## 6 모델 지원과 정책 기본값

### R9 추가 모델은 실제 생성까지 확인

Llama2-7B/AMD135M, Llama3-8B/Qwama0.5B의 모델 파일과 token-to-ID 동일성은 확인했다. 전자는32,000개, 후자는128,256개 vocabulary mapping이다. **이 서버의 native 생성 검사는 아직 완료되지 않았다.**

7B snapshot의 tokenizer는 SentencePiece 변환이 필요했지만 공용 venv에 의존성이 없었다. 이 서버에서는 Python3.11용 `sentencepiece==0.2.0`을 검증 전용 `deps311`에 설치했다. 공용 venv는 변경하지 않았으며 이 경로도 새 모델 검사에만 적용했다.

작업자는 tokenizer 의존성과 Python ABI를 재현 가능한 환경 명세에 포함하고, 실제 dtype와 head dimension, target/draft vocabulary 크기 및 모든 token ID의 대응을 검사해야 한다. 과거70B/TinyLlama용 dtype 강제 adapter를 AMD/Qwama에 무조건 적용하지 않는다.

완료 기준은 두 모델 각각의 T0/T0.7·B1/B2 생성, 요청별 KV/출력·문맥 경계 검사, 모델별 T0 AR 기준선과의 비교다. T0 차이가 있으면 첫 분기와 logits/수치 경로부터 조사한다. Tokenizer 검사만으로 정확성·최적 파라미터·성능 우위를 선언하지 않는다.

### R10 현재 서버 정책과 원격 기본값을 구분

| 정책 | 최신 원격 기본값 | 현재 서버에서 논의한 방향 |
|---|---|---|
| 위치 | 0.75 h + 0.25 hbar | 동일 혼합 |
| Correction 토큰 | e(1-q) | 동일 |
| 위치별 토큰 정규화 | 전체 어휘 `full` | 기존 상위 후보 집합 `topm` |
| 후속 tree 점수/확장 | 역사적 calibration 기반 `reach_gain_frontier` | Calibration 없는 q-path 기준, 확장 방식은 별도 비교 |

이는 구현 오류 목록이 아니라 **평가 조건의 차이**다. Calibration 없는 정책으로 검토한다고 하면서 원격 기본값을 그대로 실행하지 않는다. 반대로 원격 구현을 검증하는 release 비교군에 임의로 q-path를 넣고 ‘원격 그대로’라고 부르지 않는다.

모든 실행에 실제 resolved policy, fallback 여부, calibration 파일 hash, 정규화, 후보 온도, P1 allocation과 physical forward width를 기록한다. 같은 성능 비교에서 root와 tree를 함께 바꿨다면 개별 효과로 해석하지 않는다. 이 서버의 전체 데이터 2×2 비교와 별도 local q-path 비교는 아직 남아 있다.

## 7 현재까지 인정할 수 있는 개선

| 항목 | 이 서버의 관측 | 주장 가능한 범위 |
|---|---|---|
| B1 시스템 변경 | 같은 디버그 조건에서 v0.1/v0.2 출력·AL·hit 일치. TPS 차이 약+0.10% | 해당 표본의 동작 일치. Production 속도 개선 증거로는 부족 |
| 중간 proposal padding 복사 제거 | 여러 고정 shape의 준비 시간 중앙값 약0.022–0.026ms 감소 | 작은 준비 구간의 실질적 감소. 전체 TPS 개선과 구분 |
| Dense 대비 확률 연산 압축 | B8, vocab128256 고정 입력에서 준비·gather·proxy·수락 합계 중앙값 약16.5–29.4% 감소 | Target forward 미포함. 70B 전체 성능 향상률로 사용 불가 |
| 회귀 검사 | 원본23/26 묶음 통과, 나머지3개 별도 서버 진단 통과 | 원본 실패 보존 및 위 조건·허용 오차 명시 필요 |
| B>1와 추가 모델의 전체 동작 | 미완료 | 통합 성공 또는 성능 우위 주장 불가 |

이번 B1 비교에서 큰 차이가 없는 것은 주요 최적화가 B>1 경로에 집중되어 있다는 코드 구조와도 맞는다. 다만 디버그 출력 제거 및 기준 config 복원 전에는 작은 개선이나 회귀를 정량 결론으로 확정하지 않는다.

## 8 수정 후 재검증 순서

1. **CPU 계약 검사**: target exit 범위, 명시 split budget의 합법/불법 구성, 파생 config 고정, `-O` 유무와 무관한 안전 검사, 무디코딩 및 report 상태.
2. **SM120 소규모 검사**: backend 선택, cached-prefill plan, 실제 shape별 workspace, eager/captured 출력·KV, shape 교대 replay, raw/packed 수치 검사.
3. **기존70B B1 기준 재현**: 디버그 출력 OFF, 원래 파생 config, candidate T0.7 등 원시 실행과 같은 조건. 먼저 작은 입력에서 경로를 확인하고 필요하면 같은 full 순서로 재현한다. 한 번의 subset TPS를65에 억지로 맞추지 않는다.
4. **B1 시스템 A/B**: 같은 모델·입력·warmup·온도·seed·정책·파생 config로 v0.1↔v0.2를 정순/역순 반복. 출력·AL·hit 변화가 있다면 latency 개선과 분리해 조사한다.
5. **70B B2/B8 및 모델 기능 검사**: workspace 준비 통과 후 시작. T0 AR 비교 및 B>1 요청별 격리 확인. 작은 모델이 통과했다고70B B>1을 완료 처리하지 않는다.
6. **전체 데이터 알고리즘 비교**: 480문항/560턴, 문항 평균 decode-only TPS, correction 포함 AL, P1/P2 hit와 조건부 AL, 지표별 유효 분모. Calibration/default와 q-path 대안을 구분한다.

소스 변경 전에 기존 manifest와 실패 attempt를 보존하고, 변경한 파일만 명시적으로 새 hash에 반영한다. 이전 결과를 새 소스에서 수행한 것처럼 표시하지 않는다. 오류가 해결되기 전 기존 논문 결과를 이번56 TPS로 교체하지 않는다.

## 9 작업자에게 바로 전달할 요약

> Blackwell 서버의 ver0.2 검증은 아직 미완료입니다. 먼저70B B2의 `SSD_BATCH_TREE_WORKSPACE_MB` 기본128 MiB가188 MiB 요구에 못 미쳐 초기화 실패한 부분을 보완해야 합니다. 기존 B1 verify workspace 설정과 다른 변수입니다. Ver0.1부터 SM120에서는 설치된 SGL Hopper 커널 대신 FlashInfer 경로와 runner의 사전 plan이 필요했으며, cached-prefill 테스트가 SGL을 강제해 다시 실패했습니다. 이 계약을 유지한 backend별 테스트가 필요합니다.
>
> 성능 측정 쪽에서는 assertion을 켜면서 `__debug__` 토큰 출력까지 켜졌고, 검증 adapter가 fanout을3→4로 바꾸어 MQ_LEN도39→52가 됐습니다. 두 부분을 먼저 바로잡아 기존70B B1 기준을 재현해야 합니다. Exit56을 draft22층과 비교하는 Config 문제는 ver0.1/0.2 공통이므로 target80층 기준으로 검증하십시오. Packed 점수의 한 양자화 bin 차이는 raw score·후보 ID·검증 출력과 분리해 검사하고, 원본 tolerance를 일괄 완화하지 마십시오.
>
> 현재56 TPS를 ver0.2의 성능 회귀로 사용하지 않습니다. 추가 모델과 B>1 및 full 알고리즘 결과도 아직 미확정입니다. 상세 실패 로그와 수정 후 통과 조건은 이 인수인계의 R1–R10을 참고하십시오. 다른 사용자의 프로세스나 시스템 GPU 설정은 변경하지 마십시오.

## 10 증거 위치

- [현재 검증 보고서](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/REPORT.md) 및 [실제 상태](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/status.json). 보고서 작업 표의 ‘실행 중’ 오류는 R8 참조.
- [기존 ver0.1 완료 보고서](../PSD-MLSys-ver0.1-test/results/blackwell_validation/REPORT.md).
- [기존 약65 TPS 실행 명령](../PSD-MLSys-ver0.1-test/results/blackwell_validation/runs/17_full_latest/attempt1/command.json) 및 [실제 Config](../PSD-MLSys-ver0.1-test/results/blackwell_validation/runs/17_full_latest/attempt1/model_rank0.json).
- [이번 약56 TPS 실행 명령](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/runs/04_latency_v01_r1/attempt1/command.json) 및 [실제 Config](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/runs/04_latency_v01_r1/attempt1/model_rank0.json).
- [원본 회귀 검사](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/checks/CHECKS.json), [별도 진단 결과](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/checks_adapted/GATE.json).
- [중간 복사 제거 측정](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/runs/10_component_payload_staging/attempt1/component.json), [확률 연산 압축 측정](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/runs/11_component_dense_vs_flat/attempt1/component.json).
- [고정 소스 manifest](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/manifest.json), [계획](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/plan.json), [집계와 분모](../PSD-MLSys-ver0.2-test/results/blackwell_ver02_validation/summary.json).
