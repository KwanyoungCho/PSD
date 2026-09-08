# postfull_winner_trace_n10m10

> Diagnostic run: profiling and topology tracing were enabled; its TPS is not a performance claim.

## Outcome

| Questions (turns) | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL |
|---:|---:|---:|---:|---:|---:|---:|
| 48 (56) | 3.685 | 0.644 | 0.505 | 4.593 | 0.139 | 3.262 |

## Tree opportunity

| Phase | Hit trees | Accepted nodes/tree | Reaches max depth | Alternative-sibling tree rate | Branch-assisted accepted share |
|---|---:|---:|---:|---:|---:|
| P1 | 1448 | 3.564 | 20.994% | 27.279% | 7.731% |
| P2 | 456 | 2.215 | 19.079% | 21.272% | 21.386% |

## Overlap tails

Positive signed gap means the draft finished before its deadline.

| Phase | aligned steps | late rate | signed p01 | signed p05 | signed p50 | overrun p95 | overrun p99 | max overrun |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 | 2644 | 72.012% | -16.853 | -7.547 | -5.295 | 7.547 | 16.853 | 203.784 |
| K2 | 2592 | 67.014% | -18.048 | -5.336 | -1.011 | 5.336 | 18.048 | 212.100 |

## Served-root pressure

| Phase | serves | rank p50 | rank p90 | rank p95 | max | boundary-tail rate |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 1448 | 15.0 | 36.0 | 36.0 | 38 | 13.191% |
| P2 | 456 | 2.0 | 8.0 | 9.0 | 9 | 12.061% |

Boundary tail is P1's last local token rank and P2's last two configured root ranks.

P1 hit context counts: `{"0": 235, "1": 132, "2": 156, "3": 76, "4": 94, "5": 156, "6": 36, "7": 41, "8": 64, "9": 53, "10": 94, "11": 28, "12": 283}`

P1 local root-rank counts: `{"0": 960, "1": 297, "2": 191}`
