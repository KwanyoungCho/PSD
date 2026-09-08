# Full Spec-Bench P1+P2 tree result

Date: 2026-08-11

## Configuration

- Dataset: full Spec-Bench, 560 requests
- Target/draft: LayerSkip Llama-2-70B / TinyLlama-1.1B
- Sampling: temperature 0.7, top-p 1.0, seed 42
- Maximum output/context: 1024 / 4096, draft RoPE extension enabled
- P1: tree on, full-root `backbone`, K1=8, C=2, N1/M1=14/12,
  roots per position=3, thresholds=0/0
- P2: tree on, K2=4, budget=15, roots=10, N2/M2=8/8,
  thresholds=0.01/0.01
- Shared proxy top-k: 28

## Overall result

| Requests | Verify steps | P1 AL | P2 AL | Tokens/step | Cache hit | Decode TPS | Wall TPS | Target step | Target verify |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 560 | 74,712 | 4.497 | 1.842 | 4.504 | 0.828 | 64.197 | 62.531 | 71.03 ms | 62.95 ms |

P1 hit rate was 0.633 and P2 hit rate was 0.195. AL is conditional on a hit;
cache hit is recorded as an execution control and is not used to judge tree topology.
The run generated 335,783 completion tokens. End-to-end execution, including model
load and graph capture, took about 91.5 minutes.

## Per-subtask result

| Subtask | Requests | P1 AL | P2 AL | Tokens/step | Cache hit | Decode TPS |
|---|---:|---:|---:|---:|---:|---:|
| math_reasoning | 80 | 4.341 | 2.074 | 4.490 | 0.856 | 64.138 |
| mt_bench | 160 | 4.305 | 2.139 | 4.472 | 0.832 | 63.849 |
| qa | 80 | 4.618 | 2.302 | 4.803 | 0.839 | 68.460 |
| rag | 80 | 4.659 | 2.156 | 4.691 | 0.790 | 66.161 |
| summarization | 80 | 4.472 | 0.793 | 3.752 | 0.856 | 53.851 |
| translation | 80 | 4.723 | 2.086 | 4.683 | 0.792 | 66.680 |

Summarization is the clear throughput/overall-AL weak subtask. Its P1 conditional AL
is not low, but P1 hit is only 0.497 and P2 conditional AL is 0.793. This is separate
from the P1 tree topology result and should be inspected as a root/proxy workload case.

## Integrity checks

- 560 JSONL rows and 560 unique UIDs
- Expected group counts: mt_bench 160; every other subtask 80
- Zero rows with an error field
- No traceback, CUDA error, rotary OOB, or assertion in the log
- Process exit code 0
- Every row records the selected tree configuration and seed 42

This is the requested P1+P2-tree arm only. A full same-seed K1=8 P2-tree-only/P1-chain
run is still required before claiming a full-dataset improvement caused by P1 tree.
