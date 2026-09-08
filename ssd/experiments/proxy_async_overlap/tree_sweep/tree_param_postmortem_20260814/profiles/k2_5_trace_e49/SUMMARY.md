# k2_5_trace_e49

> Diagnostic run: profiling and topology tracing were enabled; its TPS is not a performance claim.

## Outcome

| Questions (turns) | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL |
|---:|---:|---:|---:|---:|---:|---:|
| 60 (70) | 3.820 | 0.653 | 0.503 | 4.839 | 0.149 | 3.360 |

## Tree opportunity

| Phase | Hit trees | Accepted nodes/tree | Reaches max depth | Alternative-sibling tree rate | Branch-assisted accepted share |
|---|---:|---:|---:|---:|---:|
| P1 | 1018 | 3.753 | 20.530% | 25.639% | 6.857% |
| P2 | 350 | 2.126 | 18.571% | 16.857% | 15.323% |

## Overlap tails

Positive signed gap means the draft finished before its deadline.

| Phase | aligned steps | late rate | signed p01 | signed p05 | signed p50 | overrun p95 | overrun p99 | max overrun |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 | 1866 | 57.556% | -15.659 | -12.021 | -4.425 | 12.021 | 15.659 | 268.656 |
| K2 | 1804 | 44.900% | -16.426 | -8.848 | 0.304 | 8.848 | 16.426 | 257.989 |

## Served-root pressure

| Phase | serves | rank p50 | rank p90 | rank p95 | max | boundary-tail rate |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 1018 | 15.0 | 36.0 | 36.0 | 38 | 13.556% |
| P2 | 350 | 2.0 | 8.0 | 9.0 | 9 | 16.286% |

Boundary tail is P1's last local token rank and P2's last two configured root ranks.

P1 hit context counts: `{"0": 145, "1": 88, "2": 97, "3": 83, "4": 75, "5": 121, "6": 47, "7": 29, "8": 68, "9": 21, "10": 27, "11": 29, "12": 188}`

P1 local root-rank counts: `{"0": 664, "1": 216, "2": 138}`
