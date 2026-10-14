# MLSys ver1.0 통합과 검증

기준은 `MLSys-ver0.2@1b93104b`다. 별도 `MLSys-ver1.0` 브랜치에 현재 서버의 확정된 알고리즘과 필요한 실행·분석 수정을 통합한다. 기존 연구 checkout과 과거 실험 원본은 바꾸지 않는다. 이미 들어온 기능은 중복 이식하지 않는다.

다른 서버에서 실행을 이어가는 작업자는 [기본 설정과 인수인계 지침](MLSYS_VER1_HANDOVER.md)부터 확인한다.

## 적용한 알고리즘

| 항목 | ver1.0 동작 | 출처와 판단 |
|---|---|---|
| DS 시작 후보 | 위치별 균등 선정, 위치까지 draft 경로 점수 × 후보 토큰의 draft 확률 | ver0.2에 이미 구현됨. 위치별 후보 수와 forward 폭은 별개 |
| PS 시작 후보 | DS 중복을 제외하고 위치·토큰 점수로 적응 배분 | ver0.2에 이미 구현됨 |
| 토큰 점수 | correction 후보는 e(1−q), bonus는 e | 이미 적용됨. target의 실제 correction sampling은 바꾸지 않음 |
| 위치 점수 | 0.75 h_hat + 0.25 h_bar | 이미 적용됨. 수락률을 섞는 것이 아니라 최종 종료 위치 질량을 섞음 |
| 후보 온도 | 요청의 target 온도로 e, 실제 draft 온도로 q 계산 | 이미 적용됨. T=0은 greedy 검증과 별개의 soft 후보 순위 사용 |
| 정규화 | **`topm` 기본**, `full`은 명시적 비교 옵션 | 사용자 결정에 따라 현재 서버의 top-M을 기본으로 채택 |
| 후속 tree 점수 | 시작 후보 점수 × 원래 draft 경로 확률 | calibration 없는 q-path. 추정 acceptance 표를 읽지 않음 |
| 확장 범위 | `last` 또는 `frontier`, phase별 override 가능 | 별도 tree 확정 브랜치에서 이식 |
| 자식 수 | 최대 C=3, 생성 공간이 제한되면 더 적게 배정 | C3는 출발점이지 70B 최적값으로 검증된 값은 아님 |
| 검증할 subtree | hit된 후보의 생성 순서 prefix, 최대 M개 | ver0.2의 ordered-WOR 검증 계약 유지. score rerank 없음 |

`duet_root_overlap_mix`는 **h_bar의 비율**이다. 0.25는 PPT에서 h_hat 앞에 붙인 lambda=0.75와 같은 혼합이다. top-M의 M은 `duet_proxy_top_k`가 결정하고, Config가 wire 용량 때문에 올렸으면 올라간 실제 값이 사용된다. 고정 상수 28로 하드코딩하지 않는다.

사용자 결정에 따라 **Config·CLI·70B preset·chain/dense/flat 후보 함수의 기본 정규화는 모두 `topm`**이다. ver0.2의 전체 어휘 정규화를 재현하려면 `duet_root_normalization='full'`을 명시한다. 서로 다른 정규화의 결과를 같은 조건으로 합치지 않는다. top-M은 위치마다 남긴 후보 점수 합으로 나누는 방식이며, 실제 target verification 분포를 top-M으로 자르는 설정이 아니다.

## 새로 이식한 코드

| 항목 | 구현 | 확인할 점 |
|---|---|---|
| Native tree 범위 선택 | Config, draft runner, CPU rollout, arena, TreeExpansionPolicy | DS·PS, eager/captured/batched 모두 같은 해석. legacy calibrated 환경은 명시적 opt-in 없으면 오류 |
| Native root 설정 | `root_policy_config.py`, Config/CLI와 runtime manifest | residual/proxy/complement, full/topm, 위치 혼합 비율을 eager·dense·flat graph에 동일 전달 |
| Scalar 온도 graph 안전성 | `root_policy.probabilities` | Python scalar를 device fill로 생성. 캡처 중 pageable CPU→GPU 복사 방지. 실제 q는 공통 함수 유지 |
| BF16 실험 설정 | Config/CLI의 `dtype`, `ssd.presets.duet_70b_paper_kwargs` | target/draft 역할 재구성에도 dtype 보존. 일반 API는 미지정 시 기존 checkpoint dtype 유지 |
| 일반 Llama2 보조 버퍼 | `utils/checkpoint_rope.py`, dense loader | unscaled Llama의 inv_freq만 설정과 정확히 일치하거나 FP16 왕복 일치할 때 생략. 학습 가중치 불변 |
| B1 Blackwell tree workspace | `utils/workspace.py`, model runner, capture helper | 실제 70B에서 남아 있던 96MiB 부족을 확인. SM120의 넓은 tree graph 기본 256MiB, eager도 동일 크기 이상으로 확보하고 capture 전에만 할당 |
| 오프라인 시간 분석 | `bench/duet_profile_analysis.py` | sparse tree 라벨로 요청 번호를 추정하지 않고 request anchor로 정렬. 250ms 이상 표본도 임의 제거하지 않음 |

root 설정은 검증 후보를 준비하는 점수에만 적용된다. 실제 proposal q, 수락 판단, residual 보정 분포와 RNG 계약은 변경하지 않았다. 새 계산은 초기화·graph 준비 또는 원래의 점수 계산에 연결했으며 추론 step에 파일 기록·모델 hook·추가 동기화를 넣지 않았다.

## 이미 반영된 내용을 확인한 항목

다음은 최신 ver0.2를 유지한다. 과거 리뷰의 수정 요구가 현재도 미해결이라는 뜻으로 재기재하지 않는다.

- Blackwell에서 지원되는 FlashInfer attention 선택, cached-prefill 검사와 workspace 오류 진단.
- B>1 target/glue workspace 256MiB, draft P1/P2 workspace 128MiB. 충분성은 실제 batch·폭에 따라 다르다.

**B1 tree 검증 workspace는 이 목록에 포함되지 않았다.** 첫 native 70B 검사에서 96MiB 공간에 197,132,288bytes를 요구하여 초기화가 실패했고, 이번 통합에서 별도로 수정했다. 과거 서버 launcher의 `SSD_TREE_VERIFY_WORKSPACE_MB=224`가 가리던 기본값 문제다. 새 기본값도 모든 모델의 충분성을 보장하는 상한은 아니며, 부족하면 shape·예산을 포함한 오류로 중단하고 초기화 전에 명시적으로 조정한다.

나머지 기존 수정은 다음과 같다.

- 토큰 디버그 기본 비활성화, Python `-O`와 무관한 주요 설정 검사.
- 명시 P2 예산일 때 F=DS 허용. 3을 몰래 4로 늘리던 서버 adapter는 사용하지 않는다.
- draft 역할의 config 재구성에서도 target의 80-layer exit 범위와 runtime dtype 보존.
- dense/flat 후보 순위의 reduction layout 일치. 새 top-M 경로에서도 같은 처리를 유지한다.
- 생성된 worker만 종료하는 초기화 실패 처리와 worker 사망 감시. 다른 프로세스의 semaphore를 지우지 않는다.
- 출력·문맥 한도, 실제 반환 토큰 집계, 명시적인 context guard 종료와 undefined 지표 구분.
- `mlsys_request_reporting.py`의 문항 내 turn 평균, 지표별 유효 분모, 공통 decode 문항 비교. P1/P2 AL의 recovery 포함 지표는 이 보고 경로를 사용한다. 이름이 `accepted_lens_phase*_hit`인 과거 raw 필드는 recovery 제외 의미를 유지하므로 혼동하지 않는다.
- target/draft token ID 호환 검사, Qwen2/Qwama·Qwen3와 head 차원 처리, 배치별 live topology·온도 입력.

모델 family 지원은 모든 checkpoint와 RoPE 변형의 정확성 보장이 아니다. 기존 Llama3 scaled-RoPE 경로에는 scaling 무시 코드가 남아 있어 별도 검증 없이 해당 모델의 HF 동등성을 주장하지 않는다. 이번 일반 Llama2 로더는 scaled RoPE 버퍼를 명시적으로 거부한다.

## 일반 실행에 넣지 않은 항목

| 보존만 하는 항목 | 이유 |
|---|---|
| calibration acceptance와 gain 정책 | 사용자가 q-path로 확정. 과거 재현은 명시적 legacy opt-in만 허용 |
| h_bar 단독을 기본으로 강제하는 옛 launcher | 최신 결정은 혼합이며 일부 과거 보고서의 제목·기본값은 당시 조건 |
| e−lambda q, 상황별 보정·threshold 탐색 | 최종 채택되지 않은 연구 정책 |
| tokenizer 교체, layer hidden-state 수집, proxy oracle replay | 통제 실험·진단 전용. native 모델 지원이나 성능 측정과 다름 |
| 과거 `LD_PRELOAD` GPU 장애 우회와 NVML 장치 가림 | 서버 상태 의존적 우회. 정상 runtime에 적용하지 않음 |
| 자동 반복 실행·GPU 점유·PID/lock 파일 | 서버 운영 상태이며 배포 코드가 아님 |
| 이전 G14/M12 score pruning, calibration 최적 파라미터 강제 | 현재 generation prefix 및 q-path 결정과 충돌 |
| GB 단위 원시 데이터·가중치 | 원본을 보존하고 출처를 기록. 코드를 이전한다고 과거 결과가 새 버전의 결과가 되지는 않음 |

현재 서버 자료의 경로와 SHA256은 [SOURCE_INVENTORY.json](results/ver10_integration_20261015/SOURCE_INVENTORY.json)에 남긴다. 현재 서버의 보고서는 연구 시점별로 조건이 다르므로 원문을 새로운 기본 설명으로 무작정 덮어쓰지 않는다.

## 70B 논문 조건의 시작 설정

```python
from ssd import LLM, SamplingParams
from ssd.presets import duet_70b_paper_kwargs

kwargs = duet_70b_paper_kwargs(draft_path)  # last, topm 기본
llm = LLM(target_path, **kwargs)
params = SamplingParams(temperature=0.7, max_new_tokens=1024)
```

- BF16, B=1, target TP2와 독립 draft GPU, exit56, K1/K2=8/4, 문맥2048.
- DS·PS 모두 tree, DS 위치당3개, DS forward scale1.5, PS root cap10, PS budget15, proxy top-M 입력28.
- G/M은 DS12/12·PS8/8, C=3, last 기본 또는 frontier 선택, 확장 threshold0.
- **과거 C2·threshold 설정의 65 TPS 결과와 동일 설정은 아니다.** 이 preset은 새 알고리즘을 명시적으로 실행하는 시작점이며, 기존 full dataset 성능을 승계하지 않는다.
- `gpu_memory_utilization`은 현재 엔진에서 모델 로딩 후 남은 메모리에 대한 KV 비율이다. GPU 전체 VRAM의 사용 비율과 다르며, 다른 사용자의 메모리를 확보하거나 GPU를 reset하는 코드는 없다.

## 검증 상태와 결과 위치

[통합 검증 보고서](results/ver10_integration_20261015/REPORT.md)에 CPU/GPU 회귀, 70B native 모델 실행, 원본과의 greedy 출력 일치를 분리해 기록한다. 짧은 모델 검사는 기능 검사이며 AL·cache hit·TPS 개선의 증거로 사용하지 않는다. 최종 성능 표는 고정 소스에서 같은 입력·출력 한도로 별도 측정해야 한다.

최종 top-M 기본값의 회귀는 **340개 통과·지원 불가 검사1개 제외**, native LayerSkip70B·B1 기본 실행은 네 입력 모두 정상 종료했다. 기본값 전환 전 원본70B·LayerSkip70B의 8개 실행과 logits 진단도 보존했다. strict greedy AR 비교는 **9/12 일치**이고, LayerSkip 최초 불일치에서는 각 실행이 자기 target argmax를 선택했으며 작은 logit 차이로 동점이 발생했다. 따라서 기능 통합 완료와 AR 출력 전체 일치 보장을 구분한다. 상세 수치·시점·한계는 위 보고서에 있다.

원래 연구 브랜치와 이전 ver0.1/ver0.2 검증 checkout은 그대로다. 이 통합에서 원격 push나 기존 논문 결과 표 교체는 하지 않는다.
