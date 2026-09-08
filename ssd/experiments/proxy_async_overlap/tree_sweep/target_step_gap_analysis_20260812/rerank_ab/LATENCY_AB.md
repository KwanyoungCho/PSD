# P1 tree rerank latency A/B

동일한 7개 입력과 seed42를 사용한 latency-only 반복 진단이다. 두 설정 모두 `M1=12`이므로 target이 검증하는 최대 노드 수는 같다. `N1=12`는 `N1=14 → M1=12` hit-time rerank만 제거한다.

프로파일에는 runner의 warm-up 2회도 들어 있으므로 raw JSONL의 실제 verification-step 수와 맞춘 마지막 N개 span만 집계했다.

## Conclusion

- `14→12` subtree 재선택/compaction 비용은 P1 hit당 **0.914 ± 0.250 ms**로 반복 재현됐다.
- 이를 제거하면 P1 hit의 draft/spec wait는 **0.959 ± 0.762 ms**, P1 full step은 **1.225 ± 0.714 ms** 감소했다.
- 조작하지 않은 P2 full step은 `-0.236 ± 0.616 ms`, miss는 `+0.010 ± 0.900 ms` 흔들렸다. 따라서 overall `-0.531 ± 0.863 ms`는 방향만 참고할 수 있고 유의한 결론이 아니다.
- `N1=12,M1=12`는 P1 latency 약 1 ms를 줄이는 유효 후보지만, chain 대비 전체 4–6 ms 차이의 전부는 아니다. tree 공통 metadata/topology 준비와 더 넓은 target verify가 여전히 남는다.

두 arm은 같은 prompt/seed지만 N1 변경으로 생성 경로와 step 수가 달라졌다(`14/12`: 436, `12/12`: 470 steps/run). 따라서 직접 원인 판정은 전체 평균이 아니라 동일 코드 구간 span과 P1 조건부 latency를 사용한다.

## Overall latency

| Metric | N1=14, M1=12 | N1=12, M1=12 | Paired delta (12−14) |
|---|---:|---:|---:|
| Raw full target step | 71.962 ± 0.475 | 71.431 ± 0.404 | -0.531 ± 0.863 ms |
| Raw target verify | 62.383 ± 0.089 | 61.862 ± 0.094 | -0.521 ± 0.070 ms |
| Raw outside verify | 9.579 ± 0.387 | 9.570 ± 0.460 | -0.010 ± 0.841 ms |
| Profile full step | 70.046 ± 0.403 | 69.665 ± 0.362 | -0.381 ± 0.755 ms |
| Profile draft/spec wait | 5.677 ± 0.306 | 6.028 ± 0.481 | 0.351 ± 0.784 ms |
| Profile response→verify gap | 2.650 ± 0.024 | 2.423 ± 0.072 | -0.227 ± 0.057 ms |
| Profile verify | 61.280 ± 0.077 | 60.788 ± 0.069 | -0.491 ± 0.056 ms |

## Status-specific critical path

### P1 hit

| Segment | N1=14, M1=12 | N1=12, M1=12 | Difference |
|---|---:|---:|---:|
| Full profile step | 70.585 ± 0.468 | 69.360 ± 0.269 | -1.225 ± 0.714 ms |
| Draft/spec response wait | 4.089 ± 0.345 | 3.130 ± 0.431 | -0.959 ± 0.762 ms |
| Response→verify gap | 3.227 ± 0.034 | 3.175 ± 0.100 | -0.053 ± 0.080 ms |
| Target verify profile | 62.810 ± 0.108 | 62.601 ± 0.081 | -0.209 ± 0.060 ms |
| Post-verify | 0.458 ± 0.002 | 0.455 ± 0.012 | -0.003 ± 0.014 ms |

### P2 hit

| Segment | N1=14, M1=12 | N1=12, M1=12 | Difference |
|---|---:|---:|---:|
| Full profile step | 66.975 ± 0.298 | 66.739 ± 0.328 | -0.236 ± 0.616 ms |
| Draft/spec response wait | 3.135 ± 0.223 | 3.136 ± 0.412 | 0.001 ± 0.635 ms |
| Response→verify gap | 2.756 ± 0.019 | 2.763 ± 0.081 | 0.007 ± 0.065 ms |
| Target verify profile | 60.619 ± 0.061 | 60.389 ± 0.072 | -0.230 ± 0.052 ms |
| Post-verify | 0.465 ± 0.005 | 0.451 ± 0.015 | -0.014 ± 0.013 ms |

### Miss

| Segment | N1=14, M1=12 | N1=12, M1=12 | Difference |
|---|---:|---:|---:|
| Full profile step | 72.017 ± 0.373 | 72.027 ± 0.527 | 0.010 ± 0.900 ms |
| Draft/spec response wait | 12.166 ± 0.332 | 12.087 ± 0.599 | -0.079 ± 0.931 ms |
| Response→verify gap | 1.158 ± 0.010 | 1.121 ± 0.030 | -0.036 ± 0.021 ms |
| Target verify profile | 58.326 ± 0.043 | 58.452 ± 0.063 | 0.126 ± 0.074 ms |
| Post-verify | 0.367 ± 0.007 | 0.367 ± 0.009 | -0.001 ± 0.010 ms |

## Detailed tree spans

| Span | N1=14, M1=12 | N1=12, M1=12 | Difference |
|---|---:|---:|---:|
| P1 cache-hit response | 2.029 ± 0.259 | 1.138 ± 0.294 | -0.891 ± 0.545 ms |
| P1 hit rerank | 1.290 ± 0.158 | 0.376 ± 0.094 | -0.914 ± 0.250 ms |
| P1 parent-q gather | 0.089 ± 0.012 | 0.092 ± 0.024 | 0.004 ± 0.035 ms |
| P2 cache-hit response (control) | 1.083 ± 0.145 | 1.098 ± 0.274 | 0.014 ± 0.418 ms |
| Tree KV restore | 0.447 ± 0.042 | 0.418 ± 0.080 | -0.029 ± 0.121 ms |
| Target tree wire parse/validate | 0.673 ± 0.002 | 0.667 ± 0.027 | -0.006 ± 0.027 ms |
| Target topology prepare | 1.002 ± 0.015 | 0.985 ± 0.040 | -0.017 ± 0.036 ms |
| Target parent-q select | 0.142 ± 0.003 | 0.138 ± 0.005 | -0.005 ± 0.005 ms |
| Target verify setup | 1.126 ± 0.005 | 1.108 ± 0.039 | -0.018 ± 0.035 ms |

## Interpretation rule

- 이번 결과는 rerank span, P1 spec wait, P1 full step이 함께 줄어 `14→12` on-hit rerank가 verify 바깥 지연의 직접 원인임을 확인했다.
- 다음 진단은 모든 tree arm에 공통으로 남는 CPU topology parse/validation과 작은 H2D topology copy를 대상으로 한다.
- 이 표의 AL/hit mix는 결론 근거가 아니다. 정식 AL/TPS 비교는 원인 확인 후 full Spec-Bench에서 별도로 수행한다.

Per-run machine-readable values: `run_summary.csv`
