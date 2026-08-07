# 27 — P2 동적 트리 사후 calibration과 정적 threshold

## 결론

EAGLE식 전역 확장에서 다음 두 값을 정적 기본값으로 사용한다.

- `duet_tree_proxy_threshold = 0.01`
- `duet_tree_conf_threshold = 0.03`

두 값은 토큰이나 cache root를 삭제하는 기준이 아니다. 모든 P2 root는
첫 forward를 그대로 받고, 이미 뽑은 자식도 검증 가능한 leaf로 남는다.
threshold보다 작은 경우에만 그 root/자식 **아래의 다음 draft forward를
막는다**. 따라서 정확 verifier가 보는 proposal을 사후에 다른 토큰으로
바꾸지 않으며, CUDA graph의 `4 x W` 실행 모양과 CPU 왕복 수도 변하지
않는다.

이 기본값은 현재 champion 조합(target/draft 모델, temperature와 tree
예산)에 대해 calibration한 값이다. serving 중에는 정적으로 유지하지만,
모델이나 sampling temperature를 바꾸면 같은 사후 절차로 다시 calibration
해야 한다. 모든 조합에 보편적인 확률 상수로 간주하지 않는다.

## 사후 라벨

Proxy는 E0 champion trace의 24,861 step, retained root 248,610개를
사용했다. 양성은 해당 root의 정확한 `(terminal, recovery token)` cache
key가 다음 step에서 실제 P2 hit된 경우뿐이다.

Confidence는 seed 42/123, eslab18/17에서 따로 수집했다. 합계 P2-hit
tree 2,012개, node 15,591개이며, 마지막 depth를 뺀 실제 확장 후보는
13,680개다. 각 node에 다음 라벨을 분리했다.

- `attempted`: verifier가 실제로 이 형제까지 검사함
- `accepted/rejected`: 검사 후 수락/거절됨
- `on_path`: 실제 수락 경로에 포함됨
- `expansion_useful`: 이 node 아래 자식까지 추가로 수락됨

나중 형제가 앞 형제 수락 때문에 검사되지 않은 경우는 reject로 세지
않았다. Threshold는 node 자체가 아니라 그 아래 확장만 막으므로 최종
판정에는 `expansion_useful`을 사용했다. 마지막 depth leaf를 후보 수에
넣어 threshold를 인위적으로 안전하게 보이게 하지 않았다.

## Calibration 결과

### Proxy

| P 미만 | root 슬롯 비중 | 슬롯별 실제 hit | hit 기여 | 95% hit 상한 |
|---:|---:|---:|---:|---:|
| 0.003 | 22.89% | 0.090% | 0.816% | 0.118% |
| 0.010 | 41.46% | 0.237% | 3.906% | 0.268% |
| 0.030 | 68.27% | 0.570% | 15.479% | 0.607% |

0.003은 거의 무손실인 안전선이다. 그러나 0.01 미만 root도 슬롯의
41.5%를 차지하면서 hit 기여는 3.9%뿐이다. Root 자체는 계속 보존하고
깊은 확장만 막는 실제 구현은 이 표의 3.9%를 cache miss로 잃지 않는다.
0.03은 실제 hit 기여 15.5%를 건드리므로 채택하지 않았다.

### Confidence

| q 미만 | 확장 후보 비중 | 실제 유용 확장 수 | 전체 유용 확장 기여 | 후보별 유용률 95% 상한 |
|---:|---:|---:|---:|---:|
| 0.010 | 29.07% | 5 | 0.457% | 0.294% |
| 0.030 | 37.25% | 14 | 1.279% | 0.461% |
| 0.050 | 42.40% | 22 | 2.009% | 0.574% |
| 0.100 | 49.59% | 48 | 4.384% | 0.937% |

0.01은 안전선이다. 하지만 0.03 미만 후보는 전체의 37.3%인데 실제
추가 수락에 기여한 비중은 1.28%뿐이다. 0.05부터 유용 확장 손실이 더
빠르게 늘어나므로 0.03을 성능 후보로 정했다.

중요하게, `q < 0.03` token 자체를 삭제하지 않는다. 낮은 q token도
실제로 수락되는 경우가 있으므로 leaf로 검증하고, 통계적으로 거의
도움이 없었던 그 아래 확장만 중단한다.

## 엔진 A/B

Blind sweep 대신 세 조건만 같은 길이로 비교했다.

- 기준: `0 / 0`
- 안전선: `0.003 / 0.01`
- 사후 효율선: `0.01 / 0.03`

각 서버에서 10 x 4 sequence, output 384, profiler/trace OFF로 실행했다.

| seed / 서버 | threshold | TPS | tok/step | P1 hit | P2 hit | P1 AL | P2 AL | target ms | verify ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 / eslab18 | 0 / 0 | 62.93 | 3.79 | .565 | .243 | 3.78 | 1.40 | 62.26 | 52.45 |
| 42 / eslab18 | .003 / .01 | 65.28 | 3.91 | .571 | .237 | 3.77 | 1.86 | 61.85 | 52.21 |
| 42 / eslab18 | **.01 / .03** | **70.69** | **4.24** | .595 | .249 | 4.19 | **1.84** | 62.07 | 52.79 |
| 123 / eslab17 | 0 / 0 | 74.44 | 4.13 | .609 | .230 | 4.12 | 1.43 | 57.45 | 50.27 |
| 123 / eslab17 | .003 / .01 | 69.74 | 3.89 | .570 | .247 | 3.76 | 1.81 | 57.68 | 49.83 |
| 123 / eslab17 | **.01 / .03** | **72.27** | **4.03** | .577 | .242 | 3.93 | **1.88** | 57.67 | 49.93 |

두 seed 단순 평균에서 기준 대비 효율선은 다음과 같다.

- P2 AL: 1.415 -> **1.860** (+0.445)
- tok/step: 3.960 -> **4.135** (+4.4%)
- TPS: 68.685 -> **71.480** (+4.1%)
- P1 hit: .587 -> .586 (동등)
- P2 hit: .237 -> .246 (+0.9%p)
- target full step: 59.855 -> 59.870 ms (동등)
- target verify: 51.360 -> 51.360 ms (동등)
- draft step: 57.105 -> 57.090 ms (동등)

즉 이 threshold의 이득은 latency를 줄였다고 가장한 것이 아니라, 같은
고정 draft 실행 시간 안에서 약한 가지의 연속 확장을 막아 P2 tree의
실제 수락 길이를 높인 결과다. Seed별 P1 수치는 독립 target sampling
trajectory 때문에 흔들렸지만, 두 seed 평균은 기준과 동등하고 P2 AL
방향은 두 서버 모두 일치했다.

## 구현 계약

CLI/config:

- `--duet_tree_proxy_threshold`
- `--duet_tree_conf_threshold`

두 값은 CPU 참조 rollout, eager GPU arena, production P2 CUDA graph
실행기에 동일하게 배선했다. Round 0에는 적용하지 않고 round 1 이후
부모 선택 eligibility에만 적용한다. 값 0/0은 기존 EAGLE 정책을 정확히
재현한다.

재현 산출물:

- `tools/duet_calibration/analyze_thresholds.py`
- `experiments/proxy_async_overlap/tree_sweep/threshold_calibration_20260807/`
- `experiments/proxy_async_overlap/tree_sweep/threshold_gate_s42_*_20260807/`
- `experiments/proxy_async_overlap/tree_sweep/threshold_gate_s123_*_20260807/`

## 다른 설정에서 다시 calibration하는 방법

### 한 서버에서 수집부터 추천까지

다음 스크립트는 threshold를 `0/0`으로 강제한 EAGLE tree를 실행하면서
proxy root의 다음-step 실제 hit와 confidence node 아래의 실제 추가 수락을
함께 기록한다. 진단 trace가 켜지므로 이 런의 TPS는 성능 수치로 사용하지
않는다.

```bash
cd ssd
OUT=$PWD/experiments/proxy_async_overlap/tree_sweep/calib_new_config \
CALIB_SEEDS="42 123" \
CALIB_TEMP=0.7 \
bash tools/duet_calibration/collect_tree_thresholds.sh
```

모델이나 tree 형상이 다르면 환경 변수로 경로와 형상을 바꿀 수 있다.
스크립트 뒤에 추가한 bench 인자는 기본 인자를 덮어쓴다. 단,
`tree_policy=eagle`과 threshold `0/0`은 unbiased calibration을 위해
스크립트가 마지막에 다시 강제한다.

```bash
MODEL_PATH=/path/to/target \
TARGET_AWQ=/path/to/target_artifact \
DRAFT_PATH=/path/to/draft \
DRAFT_AWQ=/path/to/draft_artifact \
CALIB_K1=9 CALIB_DEPTH=4 CALIB_ROOTS=10 CALIB_P2_BUDGET=10 \
CALIB_NV=8 CALIB_C_TENSOR=3 \
CALIB_NUMSEQS=10 CALIB_OUTLEN=384 \
bash tools/duet_calibration/collect_tree_thresholds.sh \
  --duet_exit_layer 48 --temp 0.8
```

`CALIB_K1`을 바꾸면 길이가 `K1+1`인
`CALIB_P1_FANOUT_LIST`도 함께 지정한다.

출력은 다음 세 파일이다.

- `calibration.txt`: 사람이 읽는 tail 표와 safe/balanced 추천
- `calibration.json`: 표본 수, 판정 기준, 모든 후보 threshold의 원수치
- `threshold.env`: 최종 짧은 A/B에 바로 주입할 정적 값

표본이 기본 하한(proxy slot 10,000, 실제 proxy hit 100, 확장 후보
1,000, 실제 유용 확장 100)에 못 미치거나 추천값을 찾지 못하면
`threshold.env`를 쓰지 않고 실패한다. Smoke trace가 우연히 좋아 보인다는
이유로 운영값이 되는 것을 막기 위한 규약이다.

### 여러 서버의 trace 합치기

17번과 18번에서 seed를 나눠 수집한 뒤 분석기 입력에 두 결과를 함께
주면 된다. 수집 런을 다시 돌릴 필요는 없다.

```bash
python tools/duet_calibration/analyze_thresholds.py \
  --e0-dir /result18/seed_42/e0 /result17/seed_123/e0 \
  --confidence /result18/seed_42/confidence.jsonl \
               /result17/seed_123/confidence.jsonl \
  --depth-cap 4 --risk-profile balanced \
  --json-out combined.json --config-out threshold.env --strict
```

`safe`는 거의 무손실 하한이고 `balanced`는 낮은 가치의 확장을 실제로
줄이는 기본 성능 후보다. 현재 데이터에서 각각 `.003/.01`과
`.01/.03`을 재현한다. 후보 확률 격자가 맞지 않는 모델은
`--proxy-thresholds`와 `--confidence-thresholds`에 쉼표 목록을 줄 수
있다.

```bash
--proxy-thresholds 0.001,0.003,0.01,0.02,0.03 \
--confidence-thresholds 0.005,0.01,0.02,0.03,0.05
```

### 재calibration이 필요한 변경

다음 중 하나가 바뀌면 이전 threshold를 그대로 복사하지 않는다.

- target 또는 draft 모델/quantization
- target/draft temperature나 sampling 규칙
- root 수, forward 폭, 깊이, `Nv`, 자식 sampling 수
- proxy score나 confidence score의 정의
- 실제 serving prompt 분포가 크게 다른 dataset

Calibration은 “tail 확장이 사후에 얼마나 유용했는가”를 정하므로
latency 최적점을 직접 측정하지 않는다. 생성된 `balanced` 값과 `0/0`을
짧은 실제 엔진 A/B로 한 번 비교하고, AL/hit/TPS가 모두 악화되지 않을
때 정적 기본값으로 채택한다. 여러 후보를 장시간 sweep할 필요는 없다.
