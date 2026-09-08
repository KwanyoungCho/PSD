# P1 full-backbone AL/TPS gate

Date: 2026-08-11

## Verdict

P1 tree generation/verification is structurally correct. The failing policy was the
allocation: an EAGLE2-style global frontier optimized one current proposal tree, while
DUET P1 must prepare a forest for a future cache hit whose target score is not available
yet. In addition, the former `backbone` path reserved only 27 continuation lanes for up
to 39 roots, so it did not actually guarantee a backbone.

The fixed full-root backbone with `K1=8, K2=4, C=2, N1=14, M1=12` improves both P1 AL
and TPS over the same-K1 P2-tree-only control on the 21-request, 1024-output gate.

## Main same-K1 result (Spec-Bench smoke21, seed 42, output 1024)

| arm | K1 | P1 allocation | C | P1 AL | tok/step | target step ms | verify ms | decode TPS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| P2-tree-only / P1 chain | 8 | chain | - | 4.023 | 4.392 | 67.092 | 61.637 | 66.204 |
| P1+P2 tree | 8 | global dynamic | 3 | 3.894 | 4.215 | 71.104 | 62.830 | 60.039 |
| P1+P2 tree | 8 | full backbone | 3 | 4.041 | 4.209 | 71.494 | 62.856 | 59.673 |
| **P1+P2 tree** | **8** | **full backbone** | **2** | **4.934** | **4.939** | **71.799** | **63.253** | **69.673** |

Relative to the control, the selected cell changes P1 AL by +22.6%, tokens/step by
+12.5%, target step by +4.71 ms, and decode TPS by +5.2%. Cache hit is reported only as
a control; tree topology is evaluated by conditional P1 AL.

## Why K1=9 is not the selected claim

K1=9/C2 full-backbone reached 69.155 TPS and P1 AL 4.919, but the same-K1 P2-only arm
reached 76.814 TPS and P1 AL 5.023. The higher absolute K1=9 chain quality is therefore
not a P1-tree benefit. K1=8 is the clean same-K topology win.

## Second-seed shape gate (Spec-Bench tiny7, seed 123, output 1024)

| arm | K1 | C | N1/M1 | P1 AL | decode TPS |
|---|---:|---:|---:|---:|---:|
| P2-tree-only / P1 chain | 8 | - | - | 4.004 | 62.744 |
| budget-limited backbone (old 39→27 width) | 8 | 3 | 14/12 | 3.424 | 52.755 |
| full backbone | 8 | 3 | 14/12 | 4.659 | 68.698 |
| **full backbone** | **8** | **2** | **14/12** | **4.806** | **68.885** |
| full backbone | 8 | 3 | 14/10 | 3.523 | 54.746 |
| full backbone | 8 | 3 | 12/12 | 4.403 | 64.783 |
| full backbone | 9 | 2 | 14/12 | 5.523 | 75.046 |
| full backbone | 10 | 2 | 14/12 | 4.283 | 59.758 |

The seed123 failure of the budget-limited backbone is direct evidence that root-score
ranking cannot replace a true per-root depth guarantee. N14/M12 retains useful search
surplus before the lossless subtree rerank; lowering either cap loses AL.

## Timing gate

The K1=8 selected profile reports:

- P1 ready to proxy arrival gap: +8.60 ms median (positive = P1 finishes first)
- draft cache ready to target next request gap: +1.09 ms median (positive = draft finishes first)
- P1 graph time per round: 3.74 ms median
- P2 graph time per round: 3.55 ms median

Thus P1 finishes before early-exit proxy arrival and the draft finishes before the target
needs its next response. The profiler is diagnostic only and is excluded from TPS.

## Correctness and implementation changes

- C=1/R=W tree forest equals ordinary speculative decoding under identical randomness.
- Every full-backbone root now gets a continuation lane in every round.
- Scheduler lookahead mirrors the executor's full-root compact cell count.
- When a compute-budgeted policy has R>W, likely root tips are prioritized instead of an
  arbitrary root-id prefix; R=W keeps exact lane stability.
- Multi-`generate()` profile analysis now keys events by `(request epoch, step id)` instead
  of overwriting repeated step ids.
- CUDA/CPU regression suite: 182 tests, 166 pass and 16 model-path skips.

## Reproduction

Harness: `../run_p1_tree_tps_gate_20260811.sh`

Selected environment/arguments:

```bash
DATA=/home/eslab/chokwans99/baseline/data/specbench_smoke.jsonl \
SEED=42 OUTLEN=1024 K1=8 K2=4 N1=14 P1_VERIFY=12 C_TENSOR=2 \
ARMS=p1_backbone bash ../run_p1_tree_tps_gate_20260811.sh
```

The harness fixes `proxy_top_k=28` across arms, uses P1 fanout/rpp 3, P2 budget 15,
P2 N/M 8, output 1024, context 4096 with draft RoPE extension, and disables profiling.

Results are stochastic at temperature 0.7. The next paper-grade step is the same-K1 paired
full Spec-Bench run, not treating these 21/7-request values as final paper numbers.
