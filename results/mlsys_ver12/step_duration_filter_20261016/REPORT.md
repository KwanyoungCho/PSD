# 전체 decode step의 2배 초과 시간 제외

2026-10-16, `MLSys-ver1.2`. 사용자 지정 기준을 향후 성능 집계의 기본값으로 반영했다.
모델·GPU 성능 실험은 사용자의 요청에 따라 실행하지 않았다.

## 제외 규칙

- 기존 출력 상한·clip 및 부분 batch 제외 후, 같은 generate/request group·실제 batch·P1/P2/기타 hit/miss 구성의 전체 step 시간 중앙값을 구한다.
- **전체 step 시간 > 중앙값 × 2**인 step을 한 번만 제외한다. 정확히 2배인 step과 빠른 step은 유지한다. 재귀적으로 다시 자르지 않는다.
- 예: 중앙값이 10ms이면 20ms는 유지하고, 20ms 초과는 제외한다.
- 안정적인 기준을 위해 같은 구성의 표본이 5개 이상일 때 적용한다. 5개 미만인 구성은 유지한다. 이 값과 배수는 내부 고정값이며 새 실행 옵션은 추가하지 않았다.
- B=1은 P1 hit/P2 hit/miss를 구분한다. B>1은 각 종류의 요청 **개수 조합**을 구분하며, 요청 순서는 비교 기준에 영향을 주지 않는다. 정상적인 miss 지연을 hit 기준으로 자르지 않는다.
- AR은 같은 실제 batch와 request group으로 비교한다. DUET·SSD·SD·AR에 같은 규칙을 쓴다.
- `SSD_PROFILE_DUET`, `SSD_PROFILE`, `SSD_PROFILE_DRAFT` 중 하나라도 1인 진단 실행에서는 이 시간 기준 제외를 적용하지 않는다. 성능 실행은 profile OFF로 진행한다.

## 집계와 원시 보존

- 제외 step의 **시간과 실제 출력 토큰을 함께** 성능 TPS의 분모·분자에서 뺀다. 같은 step의 AL·hit event도 함께 제외한다.
- 생성 결과, 원시 step 시간, 원시 decode 토큰·시간 누계는 수정하지 않는다. 추론·sampling·tree·KV·스케줄링 알고리즘은 변경하지 않았다.
- CUDA event, 구간별 profile, graph capture 검사나 GPU 동기화를 추가하지 않는다. 종료 후 이미 수집한 CPU step 기록으로 계산한다.
- 콘솔에 `Excluded long decode steps: N`, benchmark JSON에 `summary.excluded_long_steps`, W&B에 `metrics_excluded_long_steps`를 기록한다. 사용자에게는 제외 개수로 보고한다.
- 상세 재현 정보는 `duration_filter`의 중앙값·한계·제외 index에 남긴다. 제외 전 집계는 native `before_duration_filter`와 benchmark `summary_unfiltered`에 보존한다.
- 기존 raw 누계와 request별 end-to-end TPS는 전체 처리 기록이다. 새로 선택한 decode-step 성능 모집단과 구분한다. W&B의 주요 step 시간·AL·hit histogram도 선택된 step과 맞추고 원시 값은 `raw_*`로 남긴다.
- A/B adapter는 자신의 소스 옆 집계 모듈을 사용한다. `PAPER_CHECKOUT`의 옛 엔진을 실행하더라도 양 arm에 동일한 현행 규칙을 적용하며, 사용한 모듈 SHA-256을 결과에 기록한다.

집계 정책은 native `full_batch_without_cap_clip_or_long_step_v2`, benchmark
`complete_step_duration_filter_v1`이다. 이전 보고서·원시 결과는 재작성하지 않았다.
긴 시간만으로 원인을 특정하지 않으므로, 이번 변경은 CPU/GPU overhead 개선 결과가 아니라
사용자가 지정한 성능 모집단의 정의다. 논문에는 제외 규칙과 제외 개수를 함께 명시한다.

## 변경 파일과 확인

- `ssd/ssd/utils/performance.py`: 공통 선택 규칙과 원시/선택 집계.
- `ssd/ssd/engine/llm_engine.py`: 콘솔의 선택 모집단 일치 및 제외 개수.
- `ssd/bench/bench.py`: W&B의 같은 모집단 및 제외 개수.
- `ssd/bench/mlsys_measurement.py`: profile 없이 실행하는 A/B 공통 후처리.
- `ssd/bench/mlsys_tree_results.py`: 선택된 step을 사용하는 source 집계·bootstrap.
- `ssd/tests/test_long_step_filter.py`, `test_step_performance_metrics.py`, `test_mlsys_measurement.py`: 회귀 검사.

CPU 검사에서는 정확히 2배인 경계, 긴 꼬리만 제외, 정상 miss 유지, B>1 구성,
request group 분리, 희소 표본, profile ON, AR, 잘못된 시간 거부,
토큰·시간·AL·hit/index 일치, 원시 보존, 옛 baseline과 같은 집계,
콘솔·W&B 개수를 확인한다. 무디코딩 context guard의 기존 검증도 함께 검사한다.
최종 결과와 소스 해시는 `CHECKS.json`, 로그는 `checks/final_cpu_tests.log`에 남긴다.
최종 CPU 회귀 검사는 **31개 통과, 실패·오류·skip 0개**다. `git diff --check`도 통과했다.

이후 모델 실험을 실행할 때 실제 제외 개수를 보고한다. 이번에 새 모델 실험이나
기존 성능 결과의 재집계는 하지 않았다.
