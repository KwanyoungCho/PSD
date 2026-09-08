# ref_trace_k8_k4_e56_nativefix

> Diagnostic run: profiling and topology tracing were enabled; its TPS is not a performance claim.

## Outcome

| Questions (turns) | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL |
|---:|---:|---:|---:|---:|---:|---:|
| 60 (70) | 4.091 | 0.769 | 0.566 | 4.767 | 0.204 | 3.254 |

## Tree opportunity

| Phase | Hit trees | Accepted nodes/tree | Reaches max depth | Alternative-sibling tree rate | Branch-assisted accepted share |
|---|---:|---:|---:|---:|---:|
| P1 | 1937 | 3.924 | 25.761% | 25.968% | 6.657% |
| P2 | 678 | 2.161 | 28.466% | 20.796% | 17.679% |

## Overlap tails

Positive signed gap means the draft finished before its deadline.

| Phase | aligned steps | late rate | signed p01 | signed p05 | signed p50 | overrun p95 | overrun p99 | max overrun |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 | 3289 | 27.334% | -9.946 | -0.852 | 0.656 | 0.852 | 9.946 | 195.694 |
| K2 | 3222 | 8.411% | -9.922 | -0.704 | 2.456 | 0.704 | 9.922 | 204.304 |

## Served-root pressure

| Phase | serves | rank p50 | rank p90 | rank p95 | max | last-two-rank rate |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 1937 | 15.0 | 36.0 | 36.0 | 38 | 1.497% |
| P2 | 678 | 2.0 | 8.0 | 9.0 | 9 | 11.947% |

P1 hit context counts: `{"0": 210, "1": 174, "2": 189, "3": 124, "4": 250, "5": 116, "6": 56, "7": 99, "8": 141, "9": 46, "10": 38, "11": 32, "12": 462}`
