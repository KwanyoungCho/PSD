# 로컬 DS/PS 입력 폭과 시간 예산 진단

2026-10-15 사용자 지정 출발점에 대한 진단이다. 최적 파라미터를 확정하거나 전체
Spec-Bench 품질을 비교하는 실험이 아니다. 현재 진행/완료 여부는 `PLAN.json`과
원시 실행 폴더의 `STATUS*.json`에서 확인한다. 최종 결과는 `REPORT.md`,
완료 후 검증 기록은 `CHECKS.json`에 작성한다.

## 고정 조건

- 로컬 orion, RTX4090. 7B target 한 장 + 68M draft 한 장.
  BF16 13B target은 두 장에 분산하고 draft에는 별도 한 장을 사용한다.
- LayerSkip-Llama2-7B/13B + JackFram/llama-68m만 측정한다.
- B=1, BF16, T=0.7, target 요청 seed=9101(warmup9100), K1=K2=3, C=3, G=M=40.
  기존 sampler의 import-time `SSD_SEED=0`을 유지하므로 draft executor의 전용 RNG 초기 seed는0이다.
  모든 arm은 동일한 import 순서와 seed를 사용한다.
- Exit layer는 0-based 21/26. DS 위치당 root 3개, PS 전체 root 9개.
- 기존 last-depth tree 확장, draft 경로 점수, complement root, topm 정규화,
  위치 overlap 혼합 0.25. 알고리즘을 이번 진단에서 변경하지 않는다.
- 여섯 Spec-Bench 하위 과제에서 사전 선택한 12턴, 입력 상한512/출력 상한192,
  warmup3턴, 같은 입력 목록을2회 반복. 반복은 독립 문항 수로 세지 않는다.
- W1/W2=9/9,18/9,27/9,9/18,9/27. 별도9/9 계측 OFF 실행으로 출력과 계측 비용을 확인한다.

## 중요한 구분

DS 첫 forward는 context bucket×3행을 처리한다. 이후 두 forward만 W1 상한의
영향을 받으며 실제 폭은 min(bucket roots,W1)이다. 실제 root 수와 패딩된 행 수를
별도로 기록한다. PS의 root 수는9개를 유지하지만 매 forward 입력 폭을9/18/27로 바꾼다.

한 root의 최대 자손 수는3+9+27=39이므로 M=40은 이 실험에서 생성 제한에 걸리지 않는다.
다만 M은 실행기의 버퍼/그래프 크기에도 영향을 준다. Target tree query bucket은41행이고
DS tree-hit root bucket은123행이다. 따라서 M=40을 '비용 없는 무제한'으로 해석하면 안 된다.

현재 step이 P1 hit/P2 hit/miss인지를 기준으로 구간 시간을 나눈다. 해당 step의 DS/PS는
다음 outcome에 대비한 계산이다. 생성 root 전체의 노드 통계와 실제 hit되어 검증한 tree의
노드 통계는 서로 다른 모집단이다. Root 자체는 노드 수에서 제외하며 query 수에는1을 더한다.

## 계측 방법과 해석

`ssd/bench/mlsys_phase_probe.py`는 benchmark 전용이다. CUDA graph 안의 매 model forward와
LM head 앞뒤에 GPU globaltimer marker를 넣는다. Root별 생성 노드 수도 GPU 버퍼에 저장하고
요청 사이에서만 CPU로 가져온다. 매 step의 D2H 동기화나 모델/샘플링 정책 변경을 추가하지 않는다.
Marker kernel 비용이 포함되므로 계측 OFF 대조군과 함께 보고한다.

- **forward 1회**: transformer+LM head/FP32 변환. 모델 본체와 head도 별도 기록.
- **graph/K**: 샘플링·후보 선택까지 포함하는 기존 값. 위의 실제 forward와 혼동하지 않는다.
- **DS 가용 계산창**: `측정한 DS forward 합 + (proxy send 종료 − DS 준비 종료)`.
- **PS 가용 계산창**: `측정한 PS forward 합 + (다음 target 요청 시작 − draft 수신 준비 완료)`.
- 두 계산창은 이미 사용한 forward를 포함하고 준비·선택·정리 비용을 제외한다.
  다른 GPU 간 시간 정렬 오차와 통신/overlap 의존성이 있으므로, 이를 나눈 횟수를 그대로
  안전한 K로 보장하지 않는다. 음수 여유도0으로 자르지 않는다.

세로 흰 점선은 forward 횟수를 나타내는 균등 분할 표시다. 실제 각 forward의 시작/끝을
host 시간축에 정렬한 선은 아니다. 개별 forward의 실제 GPU 시간은 표와 CSV에 기록한다.
기존 profile의 graph/K와 새 forward 계측을 혼합하지 않는다.

Warmup, 요청별 처음10step, 마지막step, 출력 cap/clip, capture와 Triton first-use가 포함된 step은 제외한다.
빠른 step만 골라내지 않는다. 그래프와 표는 동일한 중간 구간을 사용한다. 상태별 표본 수,
평균과 분위수, 문맥 길이, 실제 폭을 함께 확인한다. 이번 TPS/AL은 진단 구간의 보조 통계다.

13B 첫 control에서는 rank1의 `_pack_verify_inputs_kernel` 특수화가 실행 중 처음 필요해져
큰 대기를 만들었다. 해당 CPU 컴파일 구간을 식별해 제외한 것이며, 엔진의 모든 최초
컴파일을 사전 준비하도록 수정한 것은 아니다. 준비 비용을 포함한 cold-start/전체 요청
지연을 비교할 때는 이 steady 표를 그대로 사용할 수 없다.

## 재현 및 원시 자료

사용자의 추가 지시에 따라7B는4·5에서 DS 폭 비교,6·7에서 PS 폭 비교를 병렬 실행한다.
PS 쌍에도9/9 기준(`psbase`)을 별도로 두며, PS 폭 효과를4·5의 기준값으로 나누지 않는다.
두 worker는 같은 NUMA의 서로 다른 CPU 코어를 쓴다.13B는 target2장+draft1장이 필요하여
두7B worker가 끝난 다음4·5·6에서 실행한다. 계획과 GPU 배치는`PLAN.json`에 있다.
`STATUS_ds.json`, `STATUS_ps.json`, `STATUS_13b.json`은각 worker의 상태다.

```bash
cd /home/chokwans99/PSD-MLSys-ver0.1
/home/chokwans99/PSD/ssd/.venv/bin/python -B results/mlsys_ver12/local_phase_latency_20261015/run_campaign.py
/home/chokwans99/PSD/ssd/.venv/bin/python -B results/mlsys_ver12/local_phase_latency_20261015/analyze.py
```

위 기본 명령은 순차 재현용이다. 이번 병렬 실행 명령은 각각
`--models 7b --arms control base ds2 ds3 --devices 4,5 --worker ds --port 29741`,
`--models 7b --arms psbase ps2 ps3 --devices 6,7 --cpuset 24-31,56-63 --worker ps --port 29743`이다.
13B는`--models 13b --worker 13b`로 실행한다.

7B base의 worker 인계 중 telemetry에 약36초 공백이 있었다. 이를 감추지 않고,
13B 종료 후 같은4·5에서 `--models 7b --arms base_repeat --worker repeat`를 추가 실행한다.
별도 반복은 원래 baseline을 대체하거나 더 좋은 수치를 고르는 용도로 쓰지 않는다.
GPU clock 평균은 프로세스 초기화/컴파일까지 포함한 값이며 steady 구간 clock 평균이 아니다.

완료 후 재집계·그림 생성·완료 검사는 다음 순서다. CPU 분석을 GPU 실험과 동시에 할 때는
벤치마크에 할당한 CPU 코어를 피한다. Portable timeline 폴더는 원시 자료 없이 다른 서버에서도 다시 그릴 수 있다.

```bash
python -B results/mlsys_ver12/local_phase_latency_20261015/analyze.py
python -B results/mlsys_ver12/local_phase_latency_20261015/audit.py
python -B results/mlsys_ver12/local_phase_latency_20261015/render.py
python -B results/mlsys_ver12/local_phase_latency_20261015/finalize.py
python ssd/tools/duet_timeline/build_model_timelines.py \
  --manifest results/mlsys_ver12/local_phase_latency_20261015/timelines/SOURCES.json \
  --output /tmp/local-phase-timelines
```

`SUMMARY.json`/`SAMPLES.json.gz`에는 모든 arm의 실제 입력 폭, 문맥, phase별 forward,
노드 분포를 보존한다. `AUDIT.json`에는 shape별 시간, 준비 구간 제외, 계측 ON/OFF 비교와
환경 확인이 있다. `RUN_PROVENANCE.json`에는 실제 명령·환경·소스 해시를 모은다.
Core inference runtime은 변경하지 않고 benchmark 계측 adapter와 분석 도구만 추가했다.

실행기는 로컬 hostname을 검사하며 다른 작업의 GPU를 종료하지 않는다. 원시 자료는
`/data/chokwans99/duet-local-phase-20261015`에 보존한다. 각 arm의 COMMAND, run,
native profile, phase probe, GPU telemetry를 함께 확인한다. `pilot*`는 준비 진단용이며
본 비교 수치에 포함하지 않는다. 모델 변환 검증은 `MODEL_PREPARATION.json`, target과
draft tokenizer 일치는 `TARGET_MODELS.json`에 기록한다.
