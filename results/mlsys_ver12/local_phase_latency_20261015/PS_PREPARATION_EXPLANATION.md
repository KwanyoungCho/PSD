# 13B P1 hit 대표 그림의 PS 준비 지연

검토 대상은 `timelines/llama2_13b_68m/images/01_p1_hit.png`의 첫 step23이다.
UID341_t0, epoch9, 현재 검증 자손9개, 확정 prefix56토큰이다. 기존 원시 자료를
읽어 분석했으며 inference/runtime과 그림은 변경하지 않았다.

## 실제 시간

그림과 같은 origin 기준으로 proxy 수신 완료22.649ms, PS graph 시작31.910ms,
PS graph 종료33.614ms, cache 완성34.067ms, 다음 명령 수신 준비34.601ms다.
Target transformer 종료는29.549ms, postprocess 종료31.606ms,
다음 target 요청 시작31.968ms다. 따라서 PS 준비가 끝날 무렵 target step도 끝나며,
draft 전체 수신 준비는 다음 target 요청보다2.633ms 늦다.

| PS graph 이전 구간 | 이 step (ms) | P1 hit741step 평균 (ms) |
|---|---:|---:|
| Proxy unpack / phase dispatch gap | 0.833 | 0.505 |
| 후보 선택·DS 중복 제거·위치별 grouping/layout·decode args (`phase2_build`) | 3.848 | 1.754 |
| KV slot·context metadata·상태 점검을 포함한 dispatch gap | 2.425 | 0.933 |
| Root 토큰·점수·RoPE·glue 가시성 준비 (`p2_prepare` A) | 1.001 | 0.422 |
| A→B 사이 dispatch gap | 0.232 | 0.092 |
| 고정 graph input·round별 slot/context·page buffer 복사 (`p2_prepare` B) | 0.880 | 0.410 |
| Graph launch gap | 0.042 | 0.018 |
| **PS forward 시작 전 합** | **9.261** | **4.133** |
| 세 forward+sampling/selection을 포함한 graph | 1.704 | 1.636 |

Table의 gap은 독립적인 세부 작업 계측이 없는 elapsed 구간이다. Code상 포함되는
작업을 표시했으며 모든 시간을 특정 동기화 하나의 비용으로 해석하지 않는다.

## 코드상 이유와 확인 범위

- `phase2_build`는 GPU 후보 검사·dedup·sort/grouping, 위치당 후보 수를 CPU list로
  읽는 `fan_out_tensor.tolist()`, 위치/RoPE 등 args 생성을 포함한다.
  이번 실행은 `python -B`여서 selector의 `__debug__` underfill 검사 `.item()`도 켜져 있다.
- 그 뒤 `_run_p2_tree_step`에서 stable metadata 복사, KV slot/context 계산,
  `_p2_step_state`의 유효성 판정 결과 readback이 있다. 이것이 첫 `p2_prepare` 전
  dispatch gap 안에 들어간다.
- 두 `p2_prepare`는 모두 model forward 밖에 있으며, root 가시성 준비와 고정 CUDA
  graph 입력 buffer 채우기를 나누어 기록한다. 작은 GPU 연산·복사·CPU launch가 많다.
- 이 step에서는 해당 준비 구간들이 평균보다 약2배 길다. PS graph는 평균보다 약4%만
  길고, target `graph_post`는6.918ms다. Forward가 급격히 느려진 모양은 아니다.
- 같은 UID의 두 번째 반복 step23(epoch21)에서는 검증 자손9개, prefix59토큰으로,
  PS graph 전 준비4.231ms, graph1.631ms, PS 여유+2.913ms다. Prefix와 생성 경로는
  완전히 동일한 tensor A/B가 아니므로 반복을 고립된 CPU 비용 실험으로 해석하지 않는다.
- 선택 step은 기존 준비 영향 제외 규칙을 통과했다. 최초 compile/capture 시간으로
  분류된 step이 아니다. 할당 GPU의 외부 process도 당시 telemetry에 관측되지 않았다.
- 기존 trace는 OS scheduling, 작은 allocation, 각 CUDA API launch, readback 대기를
  따로 기록하지 않았다. 특정 단계의 CPU/launch 경로가 느려진 관측은 가능하지만,
  그 내부의 정확한 원인을 어느 하나로 확정할 증거는 없다.

## 대표 그림을 해석할 때

그림은 **현재 target step 길이의 중앙값**으로 골랐다. PS 준비 시간의 중앙값을
고른 그림이 아니다. PS가 늦은 영향은 다음 step으로 전달되므로 현재 step 길이가
평범해도 현재 PS 준비가 느린 사례가 선택될 수 있다.

13B P1 hit 전체는 PS 여유 평균+3.760ms, deadline 초과10/741step(1.35%)이다.
선택 그림은 이10개에 포함된다. 이 한 그림을 평균적인 PS 준비 latency로 사용하면 안 된다.
이번 구성에도 평균4.133ms의 PS graph 전 준비 비용이 있으므로 준비 경로의 overhead는
분명 존재하지만, 그림의9.261ms가 매 step 반복된다고 볼 수는 없다.

수치/분위수/개별 step: `PS_PREPARATION_DIAGNOSIS.json`.
Trace 출처는 원래 portable window `timelines/data/dataset1_source1.json`과
`/data/chokwans99/duet-local-phase-20261015/13b_base/`다.

## 주변 step 확인: 한 step만의 현상이 아니다

같은 epoch9에서 PS graph 전 준비 시간을 보면 다음과 같다.

| Step | 현재 검증 자손 수 | 확정 prefix 토큰 수 | 준비 (ms) | PS graph (ms) |
|---|---:|---:|---:|---:|
| 22 | 33 | 52 | 3.978 | 1.630 |
| 23 | 9 | 56 | 9.261 | 1.704 |
| 24 | 27 | 58 | 8.157 | 1.705 |
| 25 | 3 | 62 | 7.667 | 1.688 |
| 26 | 12 | 64 | 8.097 | 1.719 |
| 27 | 12 | 68 | 8.202 | 1.718 |
| 28 | 15 | 70 | 12.282 | 1.825 |
| 29 | 3 | 73 | 5.063 | 1.640 |

따라서 "step23 하나만 느렸다"는 표현은 정확하지 않다. 이 요청 중 일부 연속 구간에서
CPU가 준비하고 GPU 작업을 발행하는 경로가 함께 느려졌다. 전체1589개 eligible step 중
준비가7ms 이상인 것은19개(1.20%)다. Tree 자손 수가 적은 step25도 느리며, prefix가
더 긴 step29는 다시 빨라진다. 입력 크기의 단조로운 증가로 이 현상을 설명할 수 없다.

발생 가능한 원인은 다음과 같지만, 이 중 하나를 실제 원인으로 확정하지 않는다.

- CPU scheduling: target/draft worker, 다른 CPU thread와 작업이 동시에 실행되면서
  해당 worker가 잠시 실행되지 못하면 CUDA graph 밖의 Python/CUDA API 준비 시간이 늘어난다.
- GPU 결과 readback: `tolist()`/`.item()`/`.cpu()`는 작은 데이터 자체의 이동뿐 아니라
  그 결과를 생산하는 앞선 GPU 작업의 완료까지 기다린다. CUDA/NCCL queue 지연을
  뒤의 readback 구간에서 관측할 수 있다.
- 작은 allocation/dispatch와 계측: 많은 작은 Torch 연산과 CUDA event 생성에도 CPU/API
  비용이 있으며, 이전 trace는 이를 개별 호출별로 나누어 기록하지 않았다.

할당 GPU에서 외부 model process가 없었다는 telemetry는 CPU 경쟁이 없었음을 증명하지
않는다. GPU 사용량 polling도 개별 CPU context switch나 CUDA API 대기를 기록하지 않는다.
직접 원인 판별에는 OS scheduling/CUDA API/NCCL을 함께 추적하는 별도 계측이 필요하다.

## `fan_out_tensor.tolist()` 제거 가능성

`_update_phase2_layout_inplace`는 이미 위치 map `fan_idx_hit`을 GPU에 만든다. B1 full
tree executor에서는 이 map으로 root 위치와 glue ancestry mask를 GPU에서 만들고,
고정 graph input buffer를 갱신하는 방식으로 CPU fan-out 목록을 생략할 수 있다.

현재 CPU 목록의 소비자는 `_p2tree_rollout`의 seed position/NumPy mask 생성과,
`_build_tree_decode_args_for_layout`의 legacy metadata 첫 fan-out 값, eager/chain attention
준비 경로다. 따라서 한 줄만 삭제하면 계약이 깨진다. Full tree 경로의 소비자를 GPU map
기반으로 바꾸고 CPU/eager fallback에서 실제 필요할 때만 목록을 만들어야 한다.

또한 `tolist()` 한 곳을 제거해도 selector underfill `.item()` 검사와 `_p2_step_state`의
유효성 결과 readback은 남는다. 이번9.261ms 전부를 `tolist()` 비용으로 볼 수 없으며,
이 문서는 코드 사용처 검토이지 제거 구현의 성능 검증 결과가 아니다.
