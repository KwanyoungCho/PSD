# 최종 비교 수치

같은 GPU 0,1에서 순차 실행한 B8/T0.7/480 first turns/input512/output64 결과. 첫 번째 cell에는 미리 생성되지 않은 graph의 capture가 포함된다. 두 번째 cell은 같은 engine을 다시 사용한다. 독립 반복의 평균/신뢰구간이 아니다.

| 모델 | 경로 | 첫 pass TPS | 두 번째 pass TPS | 두 번째 AL | 두 번째 cache hit | Target verify 중앙값 ms | Query 사용률 |
|---|---|---:|---:|---:|---:|---:|---:|
| llama2 | tree_eager_helpers | 262.49 | 294.33 | 2.0144 | 77.10% | 27.03 | 65.9% |
| llama2 | tree_optimized | 320.82 | 423.43 | 2.0062 | 76.77% | 24.77 | 65.2% |
| llama2 | chain | 592.08 | 538.72 | 1.8749 | 78.83% | 19.81 | — |
| llama2 | tree_packed | 179.73 | 277.57 | 2.0421 | 77.51% | 28.93 | 93.1% |
| llama3 | tree_eager_helpers | 199.41 | 221.36 | 2.3854 | 72.44% | 34.89 | 65.6% |
| llama3 | tree_optimized | 186.43 | 233.81 | 2.3923 | 72.46% | 36.36 | 65.6% |
| llama3 | chain | 358.60 | 362.86 | 2.3561 | 73.53% | 23.08 | — |
| llama3 | tree_packed | 145.14 | 224.40 | 2.3743 | 72.15% | 38.89 | 93.2% |

Query 사용률 = 실노드+각 root query 수 / 실제 model query 수. Chain의 물리 query 수는 이 field로 기록되지 않으므로 해당 비율은 tree끼리만 비교한다.

## 세 target seed cell의 AL 평균 (모델별 전체 입력 실험)

| 모델 | T | Tree AL | Chain AL | 차이 | Tree hit | Chain hit |
|---|---:|---:|---:|---:|---:|---:|
| llama2 | 0.0 | 2.3737 | 2.2407 | +5.94% | 84.68% | 87.91% |
| llama2 | 0.7 | 2.0006 | 1.9055 | +4.99% | 77.31% | 79.26% |
| llama3 | 0.0 | 2.8707 | 2.7507 | +4.36% | 80.28% | 84.08% |
| llama3 | 0.7 | 2.3929 | 2.3559 | +1.57% | 72.48% | 73.68% |

각 seed의 AL을 산술평균했다. Target RNG는 재설정하지만 draft RNG는 계속 진행한다. Full 비교 일부는 서로 다른 GPU pair에서 동시 실행했으므로 TPS 주장은 위 순차 비교를 사용한다.

## Packed tree greedy (B8, 480 first turns)

| 모델 | 첫 T0 pass TPS | 두 번째 T0 pass TPS | 두 번째 AL |
|---|---:|---:|---:|
| llama2 | 357.93 | 320.60 | 2.3766 |
| llama3 | 260.13 | 295.23 | 2.8761 |

같은 engine에서 T0.7 측정 후 실행했다. T0 전용 acceptance graph의 첫 capture는 새로 발생할 수 있다.

## Llama3 GPU 배치 비교 (B8/T0.7)

| 경로 | GPU0·1 NODE 두 번째 TPS | GPU4·5 PIX 첫 TPS | GPU4·5 PIX 두 번째 TPS | PIX 두 번째 AL |
|---|---:|---:|---:|---:|
| tree_optimized | 233.81 | 169.34 | 225.97 | 2.3923 |
| chain | 362.86 | 270.96 | 282.42 | 2.3561 |

코드와 CLI는 같지만 GPU pair/NUMA 위치 및 실행 시점이 다르다. 운영 배치의 영향으로 해석하며 PCIe 한 요소만의 인과 효과라고 주장하지 않는다.
