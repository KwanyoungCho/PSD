# Cache-budget input-length comparison

This experiment keeps the original fixed 35-request dataset, output cap 1,024,
temperature 0.7, actual sampler seed 1, exit layer 56, and exact average budget
2--8. It changes only the common chain length:

- K=9: 10 cache positions, total roots = `10 * budget`
- K=8: 9 cache positions, total roots = `9 * budget`

DUET uses one Draft-source root per position and assigns the remainder to the
Proxy-source. Only-Proxy assigns the full budget to Proxy-source. Geo uses
normalized `0.82^position` weights with largest-remainder rounding; Uniform
uses an equal integer count per position. Every method therefore has the exact
same total root count at each `(K, budget)` cell.

DUET/Only-Proxy use the latest DUET engine. Geo/Uniform use the dedicated SSD
engine. All cells use fixed `proxy_top_k=90` where applicable.

```bash
GPU_ORDER=0,3,5 bash run_sweep.sh
```

The paper figure is not replaced automatically. Results are generated under
`results/k{8,9}_seed1/`; K=8/K=9 single figures and fair actual-seed-1
K=10/9/8 comparison figures are written to this directory.
