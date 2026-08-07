# DUET calibration tools

이 디렉터리는 모델·early-exit layer·P1 fanout·P2 root 수를 정한 뒤,
DUET 실행 설정을 짧은 실측으로 보정하는 도구를 모아 둔다.

## 어떤 도구가 chain/tree에 적용되는가

| 도구 | chain | tree | 하는 일 |
|---|---|---|---|
| `calibrate_k_balance.sh` | 가능 | 가능 | K1/K2 종료 시각이 target과 draft 사이에서 최대한 겹치는 지점 탐색 |
| `collect_tree_thresholds.sh` | 해당 없음 | 가능 | 실제 사후 결과로 proxy/confidence 가지치기 threshold 계산 |
| `verify_tps.sh` | 가능 | 가능 | profiler를 끈 실제 TPS·hit·accepted length 최종 비교 |
| `analyze_k_balance.py` | 가능 | 가능 | 이미 수집된 timeline 재분석 |
| `analyze_thresholds.py` | 해당 없음 | 가능 | 이미 수집된 tree calibration trace 재분석 |
| `summarize_tps.py` | 가능 | 가능 | `verify_tps.sh` 결과 재요약 |

Threshold는 tree에서만 의미가 있다. Chain은 한 root마다 한 경로만 연장하므로
동적으로 보존하거나 제거할 형제가 없다. 반면 K1/K2는 두 방식 모두 target과
draft의 도착시각을 맞추는 값이므로 공통으로 보정할 수 있다. 단, tree의 P2
실행시간과 chain의 P2 실행시간이 다르므로 **각 모드에서 따로 실행해야 한다.**

## 1. K1/K2 latency 균형 찾기

프로젝트의 `ssd` 디렉터리에서 실행한다.

Tree:

```bash
cd /home/chokwans99/PSD/ssd
OUT=$PWD/experiments/proxy_async_overlap/tree_sweep/kbalance_tree \
CALIB_MODE=tree \
CALIB_BASE_K1=9 CALIB_BASE_K2=4 \
CALIB_P2_ROOTS=10 CALIB_P2_BUDGET=10 \
CALIB_PROXY_THRESHOLD=0.01 CALIB_CONF_THRESHOLD=0.03 \
bash tools/duet_calibration/calibrate_k_balance.sh
```

Chain:

```bash
cd /home/chokwans99/PSD/ssd
OUT=$PWD/experiments/proxy_async_overlap/tree_sweep/kbalance_chain \
CALIB_MODE=chain \
CALIB_BASE_K1=9 CALIB_BASE_K2=4 \
bash tools/duet_calibration/calibrate_k_balance.sh
```

출력의 `k_balance.env`에 추천 `DUET_K1`, `DUET_K2`가 기록된다. 이 선택은
GPU profiler가 켜진 timeline에서 **서로 기다리는 시간을 최소화**한 값이다.
TPS 최댓값을 보장하는 값이 아니므로 마지막에 반드시 profiler-OFF 검증을 한다.

기본 동작은 현재 값 한 번, 선형 예측 주변 세 점, 선택된 K1에서 K2 주변 세
점을 보는 작은 local 탐색이다. 전체 grid sweep은 하지 않는다. 후보를 직접
제한하려면 다음처럼 지정한다.

```bash
CALIB_K1_VALUES="8 9 10" CALIB_K2_VALUES="3 4 5" \
bash tools/duet_calibration/calibrate_k_balance.sh
```

중단된 실행은 동일한 `OUT`에 `RESUME=1`을 주면 성공한 arm만 재사용한다.
각 arm은 독립 process group으로 실행되어 rank 하나가 죽어도 남은 GPU
프로세스를 정리한다.

## 2. Tree threshold 계산

이 단계는 `CALIB_MODE=tree`에서만 사용한다. Threshold를 0으로 둔 수집
실험에서 각 root가 실제 cache hit에 기여했는지, 각 자식 확장이 실제 accepted
path를 더 만들었는지를 기록해 고정 threshold를 계산한다.

```bash
cd /home/chokwans99/PSD/ssd
OUT=$PWD/experiments/proxy_async_overlap/tree_sweep/threshold_tree \
CALIB_K1=9 CALIB_DEPTH=4 \
CALIB_ROOTS=10 CALIB_P2_BUDGET=10 \
CALIB_SEEDS="42 123" CALIB_RISK_PROFILE=balanced \
bash tools/duet_calibration/collect_tree_thresholds.sh
```

결과 `threshold.env`에는 `TREE_PROXY_THRESHOLD`와
`TREE_CONF_THRESHOLD`가 기록된다. 다음 실행에서는 각각
`CALIB_PROXY_THRESHOLD`, `CALIB_CONF_THRESHOLD`로 넘긴다. 기본
`balanced`는 단 한 번의 희귀 성공 때문에 모든 낮은 점수 노드를 보존하지
않도록 전체 기여량과 95% 상한을 함께 사용한다. 새로운 모델, temperature,
early-exit layer, K2, root 수에서는 다시 수집한다.

## 3. 실제 TPS 최종 확인

`verify_tps.sh`는 profiler를 완전히 끄고 같은 seed에서 후보를 교대로 실행한다.
예를 들어 tree의 9/3과 9/4를 비교하려면:

```bash
cd /home/chokwans99/PSD/ssd
OUT=$PWD/experiments/proxy_async_overlap/tree_sweep/tps_tree_9x \
CALIB_MODE=tree \
CALIB_K_CANDIDATES="9:3 9:4" \
CALIB_SEEDS="42 123 2024" \
CALIB_NUMSEQS=10 CALIB_OUTLEN=384 \
CALIB_PROXY_THRESHOLD=0.01 CALIB_CONF_THRESHOLD=0.03 \
bash tools/duet_calibration/verify_tps.sh
```

Chain도 모드만 바꾼다.

```bash
OUT=$PWD/experiments/proxy_async_overlap/tree_sweep/tps_chain_9x \
CALIB_MODE=chain CALIB_K_CANDIDATES="9:3 9:4" \
CALIB_SEEDS="42 123 2024" \
bash tools/duet_calibration/verify_tps.sh
```

`summary.txt`에는 후보별 평균 TPS, tok/step, P1/P2 accepted length와 같은
seed의 paired TPS 차이가 출력된다. 빠른 확인은 기본 `10×384`로 충분하지만,
논문/최종 champion 수치는 `CALIB_NUMSEQS=50 CALIB_OUTLEN=512`처럼 표본을
늘리고 최소 3 seed를 사용한다.

## 권장 전체 순서

1. 모델, temperature, early-exit layer, P1 fanout, P2 roots/width를 고정한다.
2. `calibrate_k_balance.sh`를 chain과 tree에 각각 실행한다.
3. Tree는 선택된 K1/K2에서 `collect_tree_thresholds.sh`를 실행한다.
4. 추천값과 바로 옆 후보를 `verify_tps.sh`로 profiler-OFF 비교한다.
5. TPS뿐 아니라 hit, P1/P2 accepted length, tok/step이 함께 유지되는 후보만 쓴다.

도구의 모든 모델·artifact·dataset 경로는 `MODEL_PATH`, `TARGET_AWQ`,
`DRAFT_PATH`, `DRAFT_AWQ`, `SSD_DATASET_DIR`, `PY` 환경 변수로 덮어쓸 수
있다. 다른 서버에서는 경로만 바꾸고 같은 명령을 재사용하면 된다.

### P2 executor warmup 주의

실행기는 도달 가능한 attention page 크기를 시작할 때 모두 준비하는 것이
기본값이다(`SSD_TREE_EXEC_WARMUP=all`). 각 page 크기는 독립 FlashInfer
workspace와 graph 내부 상태를 가지며, 유효하고 유한한 합성 입력으로 capture
된다. 따라서 첫 요청이 새로운 page 경계를 만날 때 capture/compile 비용을
지불하지 않는다. 특정 bucket만 재현할 때에만
`SSD_TREE_EXEC_WARMUP=3`처럼 목록을 제한하고, 성능 측정에서는 기본 `all`을
유지한다.
