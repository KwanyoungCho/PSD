# 28 — K1/K2 비동기 latency calibration

## 목적

Early-exit layer, P1 fanout schedule, P2 root/forward 폭을 먼저 고정한 뒤
K1과 K2를 비동기 파이프라인의 두 도착 시각에 맞춘다. 단순 forward
시간 합이나 `proxy_wait` 하나만 최소화하지 않는다. K를 너무 크게 해서
상대편이 먼저 끝난 경우도 signed gap으로 보존한다.

## 직접 측정하는 두 gap

### K1

```text
K1 gap = draft GPU에서 proxy가 도착할 것으로 추정한 시각
         - draft P1 완료 시각
```

- 양수: P1이 너무 빨리 끝나 proxy를 기다림 → K1을 늘릴 여지
- 음수: proxy는 이미 준비됐는데 P1이 계속 실행 중 → K1이 너무 큼
- 0 부근: P1 마지막 forward와 proxy 도착이 겹침

Target `proxy_compute_send.end`만 사용하면 NCCL 전달 시간이 빠진다.
분석기는 실제 `proxy_wait`가 발생한 step에서
`draft.proxy_wait.end - target.proxy_compute_send.end`를 측정해 전달
지연을 추정하고 모든 step에 적용한다.

### K2

```text
K2 gap = target이 다음 request를 보낼 준비가 된 시각
         - draft가 P2 cache 생성을 끝낸 시각
```

구현상 `draft.merge_cache(step s).end`와
`target.target_send_request(step s+1).start`를 비교한다.

- 양수: draft가 먼저 끝나 다음 target request를 기다림 → K2를 늘릴 여지
- 음수: target이 다음 request를 먼저 보냄 → draft가 늦어 target wait 노출
- 0 부근: 다음 step 경계에서 두 파이프라인이 동시에 준비됨

## 실행

다음 스크립트는 전체 K1×K2 격자를 돌리지 않는다.

1. 현재 K1/K2 한 번 측정
2. round당 실측 시간으로 local 균형점 예측
3. 예측 K1과 이웃 두 값, 현재 기준값 측정
4. 선택 K1에서 예측 K2와 이웃 두 값 측정

기본 최대 7개 짧은 profile이며 중복된 현재 arm은 재실행하지 않는다.
선형 예측이 한 round 이상 빗나가도 현재 기준과의 비교가 사라지지 않는다.

```bash
cd ssd
OUT=$PWD/experiments/proxy_async_overlap/tree_sweep/kbalance_exit56 \
CALIB_EXIT_LAYER=56 \
CALIB_P1_FANOUT=2 \
CALIB_P1_FANOUT_LIST_TEMPLATE=2,2,2,2,2,2,1,1,1,1 \
CALIB_P2_ROOTS=10 CALIB_P2_BUDGET=10 \
CALIB_BASE_K1=9 CALIB_BASE_K2=4 \
CALIB_PROXY_THRESHOLD=0.01 CALIB_CONF_THRESHOLD=0.03 \
bash tools/duet_calibration/calibrate_k_balance.sh
```

중간에 서버나 rank가 종료된 경우 같은 `OUT`에 `RESUME=1`을 주면
`EXIT:0`과 profile 파일이 모두 있는 arm만 재사용하고 미완료 arm부터
계속한다.

모델 경로, temperature, output 길이도 threshold collector와 동일하게
환경 변수로 바꿀 수 있다. 기본 profile 표본은 `2 × --all` prompt,
output 128이며 첫 10 step을 버리고 최소 30개 정렬 step을 요구한다.

선형 예측 범위가 불안하면 장시간 sweep 대신 명시적인 작은 후보만 준다.

```bash
CALIB_K1_VALUES="8 9 10 11" \
CALIB_K2_VALUES="3 4 5" \
bash tools/duet_calibration/calibrate_k_balance.sh
```

출력:

- `baseline.txt/json`: 현재 pair와 local 예측
- `stage1_k1.txt/json`: K1 signed-gap 비교
- `final.txt/json`: 선택 K1에서 K2 비교
- `k_balance.env`: `DUET_K1`, `DUET_K2` 추천값

각 arm은 별도 process group으로 실행한다. 한 rank가 CUDA error로
죽어도 그 arm의 남은 target/draft child를 종료해 GPU orphan을 남기지
않는다.

## 선택 규칙

주 기준은 step별 `|signed gap|`의 중앙값이다. 측정 오차 범위
`0.25 ms` 안의 동률이면 더 큰 K를 선택해 proposal 품질을 보존한다.
보고서에는 다음을 함께 남긴다.

- signed gap p10/p50/p90
- absolute gap p50/p90
- draft가 기다린 평균 시간과 target이 기다린 평균 시간
- P1/P2 round당 실측 시간
- P1/P2 hit, accepted length, tok/step, TPS

Profiler가 켜진 calibration TPS 자체는 최종 성능값이 아니다. 추천 pair를
고른 후 profiler OFF 짧은 실제 엔진 비교 한 번으로 AL/hit/TPS를 확인한다.

## Threshold calibration과의 순서

1. Early-exit layer, P1 fanout, P2 root/폭을 고정
2. 이 도구로 K1/K2 latency 균형 선택
3. 선택된 K1/K2에서 27번 도구로 proxy/confidence threshold 재calibration
4. 최종 profiler-OFF A/B 한 번

Threshold는 CUDA graph shape와 forward 수를 바꾸지 않아 latency 균형에
미치는 영향이 작다. 반대로 K2를 바꾸면 확장 가능한 depth와 confidence
사후 label이 바뀌므로 threshold calibration은 K1/K2 선택 뒤에 해야 한다.

## 기존 profile sanity

`champion_chain_tree_20260807_v3/profile_tree`의 과거 coverage profile을
새 분석기로 읽으면 다음과 같다.

- K1=9: signed gap p50 `+4.14 ms`, P1 round `2.82 ms` → local 예측 K1=10
- K2=4: signed gap p50 `-0.50 ms`, absolute p50 `0.94 ms` → K2=4 유지

이는 분석기 sanity일 뿐 현재 EAGLE+threshold 설정의 최종 판정은 아니다.
Early-exit/fanout/root가 같은 경우 K1=10/K2=4가 첫 후보라는 뜻이며,
실행기가 만드는 세 점 local profile로 확정한다.

## 최신 EAGLE 실측 결과 (2026-08-07)

고정 설정은 exit layer 56, P1 fanout template
`2,2,2,2,2,2,1,1,1,1`, P2 roots/width 10, `Nv=8`, threshold
`.01/.03`이다. eslab17에서 profiler ON 짧은 local calibration을 수행했다.

### K1

| K1/K2 | K1 signed p50 | `|gap|` p50 | 판정 |
|---|---:|---:|---|
| 9/4 | +5.04 ms | **5.04 ms** | draft가 먼저 끝남 |
| 10/4 | -6.26 ms | 6.31 ms | proxy가 먼저 준비됨 |
| 11/4 | -9.75 ms | 9.75 ms | 과대 |
| 12/4 | -13.18 ms | 13.18 ms | 과대 |

정수 K1 한 round가 0을 건너뛰므로 완전한 0은 없다. 가까운 쪽은 K1=9다.
남는 약 5 ms는 K1을 10으로 늘려 6.3 ms의 반대편 지연을 만드는 것보다
작다. 향후 이 slack을 사용하려면 K1 자체가 아니라 마지막 P1 폭/작업을
조정하는 별도 품질 실험이 필요하다.

### K2

| K1/K2 | K2 signed p50 | `|gap|` p50 | 판정 |
|---|---:|---:|---|
| 9/3 | +3.98 ms | 4.06 ms | draft가 먼저 끝남 |
| 9/4 | **-0.04 ms** | **0.82 ms** | 거의 정확히 일치 |
| 9/5 | -4.13 ms | 4.13 ms | target이 먼저 끝남 |

따라서 이 설정의 latency 균형 추천은 기존과 같은 **K1=9, K2=4**다.
원 로그와 보고서는
`experiments/proxy_async_overlap/tree_sweep/kbalance_eagle_20260807/`에
보존했다. 이 런의 profiler TPS는 최종 성능 비교에 사용하지 않는다.
