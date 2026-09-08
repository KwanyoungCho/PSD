# P1 rerank scheduling optimization

트리 생성/선택 파라미터는 모두 동일하다: `N1=14`, `M1=12`, `N2=M2=8`, `K1=8`, `K2=4`, seed42. 변경점은 P1 rerank의 실행 위치와 구현뿐이다.

- Legacy: 다음 P1 cache hit의 응답 경로에서 CPU rerank/compaction
- Optimized: P1 생성 직후 모든 root를 단일 fused CUDA kernel로 precompute; hit에서는 선택된 wire row만 readback/validate
- 3회 profiler-on + 2회 profiler-off, 각 7 prompts/436 steps
- 출력 hash 일치: **35/35 prompts**

## Profiler-on latency (3 paired runs)

| Metric (ms) | Legacy | Optimized | Paired delta (opt−legacy) |
|---|---:|---:|---:|
| P1 rerank on hit | 1.781 ± 0.523 | 0.268 ± 0.043 | -1.513 ± 0.564 |
| P1 cache response | 2.730 ± 0.812 | 0.979 ± 0.188 | -1.751 ± 0.992 |
| Fused rerank precompute | — | 0.046 ± 0.000 | — |
| P1 conditional target wait | 5.115 ± 1.068 | 3.177 ± 0.334 | -1.938 ± 1.383 |
| P1 conditional full target step | 71.523 ± 2.026 | 68.912 ± 0.829 | -2.611 ± 2.786 |
| All target wait | 6.542 ± 0.951 | 5.179 ± 0.309 | -1.363 ± 1.239 |
| All full target step | 71.040 ± 1.781 | 69.051 ± 0.586 | -1.989 ± 2.288 |
| Raw target step | 73.163 ± 1.997 | 71.017 ± 0.539 | -2.146 ± 2.509 |
| Raw outside target verify | 9.865 ± 1.144 | 8.380 ± 0.276 | -1.485 ± 1.420 |

## Profiler-off sanity check (2 paired runs)

| Metric (ms) | Legacy runs | Optimized runs | Paired delta |
|---|---:|---:|---:|
| Raw target step | 72.418, 71.497 | 71.302, 72.082 | -0.266 ± 1.202 |
| Raw target verify | 63.729, 62.908 | 62.842, 63.352 | -0.222 ± 0.941 |
| Raw outside verify | 8.689, 8.590 | 8.461, 8.730 | -0.044 ± 0.261 |

## Decision

- **채택 가능**: 직접 비용은 P1 hit rerank `−1.513 ± 0.564 ms`, P1 cache response `−1.751 ± 0.992 ms`로 반복 감소했다.
- precompute 비용은 step당 `0.046 ± 0.000 ms`라 P1 여유를 실질적으로 소모하지 않는다.
- profiler-on P1 조건부 wait는 `−1.938 ± 1.383 ms`였으나 profile-off 전체 step은 `−0.266 ± 1.202 ms`로 잡음보다 작았다. 따라서 전체 TPS 개선을 주장할 근거는 아니며, 직접 P1 latency 경로 최적화로 해석한다.
- AL/hit/트리 구조는 바뀌지 않았다. 모든 paired run의 prompt별 output hash와 verification-step 수가 같았다.

## Reproduction

```bash
# profiler-on, three paired runs
REPEATS=3 ./run_p1_rerank_precompute_ab.sh

# profiler-off sanity check
RESULT_TAG=rerank_precompute_fused_profile_off_ab \
  PROFILE_DUET_FLAG=0 PROFILE_DUET_DETAIL_FLAG=0 REPEATS=2 \
  ./run_p1_rerank_precompute_ab.sh
```

`SSD_P1_RERANK_PRECOMPUTE=0`은 legacy fallback, 기본값 `1`은 optimized path이다.
