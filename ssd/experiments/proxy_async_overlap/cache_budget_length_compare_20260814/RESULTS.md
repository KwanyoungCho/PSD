# Cache-budget length comparison

## Scope

The existing paper figure uses input length `K=10`, hence 11 cache positions
(`0..10`). This experiment repeats the same cache-hit study with:

- `K=9`: input length 9, 10 cache positions
- `K=8`: input length 8, 9 cache positions (the common paper chain length)

All runs use the same fixed 35-request Spec-Bench subset, output cap 1,024,
temperature 0.7, actual sampler seed 1, exit layer 56, and `proxy_top_k=90`.
The cache-root total is matched exactly at every point: `(K + 1) * budget`.
Hit rates are weighted by the number of verification steps.

DUET and Only-Proxy use the latest DUET engine. Geo and Uniform use the
dedicated SSD engine. Every K=8 and K=9 cell passed strict validation for all
35 UIDs, sampler seed, position count, and cache-root total (28/28 cells per
K).

## Results

| Method | K (positions) | B=2 | B=3 | B=4 | B=5 | B=6 | B=7 | B=8 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DUET | 10 (11) | 75.49 | 80.58 | 80.31 | 82.49 | 82.64 | 84.99 | 81.18 | 81.10 |
| DUET | 9 (10) | 79.00 | 81.17 | 80.79 | 83.10 | 82.81 | 79.44 | 84.69 | 81.57 |
| DUET | 8 (9) | 77.84 | 79.65 | 82.19 | 85.79 | 82.75 | 82.32 | 82.25 | 81.83 |
| Only-Proxy | 10 (11) | 70.84 | 74.12 | 76.22 | 75.65 | 76.55 | 76.57 | 75.21 | 75.02 |
| Only-Proxy | 9 (10) | 78.03 | 69.73 | 80.01 | 73.37 | 77.42 | 77.71 | 74.06 | 75.76 |
| Only-Proxy | 8 (9) | 72.95 | 74.84 | 76.89 | 74.02 | 81.20 | 77.49 | 78.04 | 76.49 |
| Geo | 10 (11) | 62.73 | 69.11 | 68.73 | 75.10 | 79.00 | 79.30 | 82.93 | 73.84 |
| Geo | 9 (10) | 62.79 | 66.06 | 74.13 | 75.54 | 80.08 | 79.38 | 84.48 | 74.64 |
| Geo | 8 (9) | 60.56 | 70.96 | 72.17 | 79.79 | 80.46 | 81.81 | 80.20 | 75.14 |
| Uniform | 10 (11) | 58.71 | 66.59 | 67.38 | 76.78 | 79.49 | 74.83 | 79.29 | 71.87 |
| Uniform | 9 (10) | 63.32 | 70.83 | 70.58 | 74.88 | 76.69 | 80.57 | 79.03 | 73.70 |
| Uniform | 8 (9) | 63.42 | 67.07 | 75.28 | 74.51 | 74.76 | 78.90 | 82.20 | 73.73 |

Values are cache hit rate (%). The K=10 rows above use the canonical actual
seed-1 result, not the pointwise seed-0/seed-1 selection used in the current
paper figure.

## Comparison with K=10

| Method | K=9 mean delta | K=8 mean delta |
|---|---:|---:|
| DUET | +0.47 pp | +0.73 pp |
| Only-Proxy | +0.74 pp | +1.47 pp |
| Geo | +0.80 pp | +1.29 pp |
| Uniform | +1.83 pp | +1.87 pp |

Reducing the number of positions does not materially weaken the mean hit
rate. K=8 has the highest seven-budget mean for every method in this seed.
However, the pointwise curves remain noisy: K=8 DUET peaks at B=5 and then
drops, while Only-Proxy has a large B=6 jump. K=9 Only-Proxy is even more
irregular. Therefore K=8 is configuration-consistent with the paper and has
good mean performance, but it is not automatically a cleaner replacement for
the current selected-seed figure.

## Figures

- `cache_hit_vs_budget_k9`: four methods at K=9
- `cache_hit_vs_budget_k8`: four methods at K=8
- `cache_hit_vs_budget_k10_k9_k8`: side-by-side actual-seed-1 comparison
- `cache_hit_vs_budget_by_method_length`: each method compared across K

The existing paper figure is intentionally not overwritten.

## Execution note

The initial K=9 Uniform B=6 process failed before producing a sample because a
Python multiprocessing semaphore could not be restored. Its empty partial
file and stale process tree were discarded, and B=6--8 were restarted on a
fresh process group and port. The restarted cells completed 35/35 requests and
passed the same strict validation as all other cells.
