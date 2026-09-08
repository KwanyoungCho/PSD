# Priority 1: target GPU topology/mask preparation

Date: 2026-08-12

## Outcome

Target의 dynamic-tree 준비를 GPU persistent buffer 경로로 옮겼다. 알고리즘,
tree width, target verify row 수, proxy 후보 선택은 바꾸지 않았다.

- Proxy topology: CPU tensor 5개 생성 + 작은 H2D copy 5회를 Triton GPU pack
  1회로 대체했다.
- Target verify: Python depth/ancestor, NumPy mask pack, input/RoPE/slot copy를
  persistent CUDA-graph buffer에 직접 쓰는 fused Triton kernel 1회로 대체했다.
- Rank 0은 이미 GPU에 있는 tree wire를 재사용한다. TP follower는 SHM wire를
  persistent device wire에 H2D 1회만 수행한다.
- CPU wire parse/validation은 TP rank의 coherent failure를 위해 유지했다.
- `SSD_TREE_TOPOLOGY_GPU=0`으로 기존 CPU 경로를 그대로 사용할 수 있다.

## Correctness

CPU와 GPU 경로는 다음 항목에서 동일했다.

| Check | Result |
|---|---:|
| Proxy topology fields (`child`, valid mask, parent, sibling) | byte-identical |
| Padded input/slot rows | identical |
| Depth-based RoPE positions | identical |
| FlashInfer little-endian packed mask | byte-identical |
| Prefix lengths 1, 7, 8, 31, 255, 1023, 2047 | all pass |
| P1 matched-width 70B TP2 output hashes | 7/7 match |
| P2 matched-width 70B TP2 output hashes | 7/7 match |
| AL / hit rate in A/B | identical |

CUDA regression suite: 110 tests passed, including P2 arena/proxy/tree contracts.

## Direct latency effect

Same tiny7, seed 42, output 256, matched target width, detailed profile. Values are
conditional means over tree-hit steps. CPU and GPU profiles are separate runs, so
only directly replaced preparation spans are used for this comparison.

| Tree-hit phase | CPU proxy topology | GPU proxy topology | Change | CPU verify setup | GPU verify setup | Change |
|---|---:|---:|---:|---:|---:|---:|
| P1, 8 nodes | 0.882 ms | 0.369 ms | **-0.512 ms** | 0.997 ms | 0.945 ms | **-0.052 ms** |
| P2, 4 nodes | 0.646 ms | 0.367 ms | **-0.279 ms** | 0.951 ms | 0.902 ms | **-0.049 ms** |

The fused verify kernel itself is slightly slower than CPU mask construction alone
(P1 0.495 vs 0.422 ms, P2 0.475 vs 0.408 ms), but it also removes the separate
input/RoPE/slot copy (P1 0.103 ms, P2 0.101 ms). Consequently the complete verify
setup improves by about 0.05 ms.

The total directly removed target preparation is therefore approximately 0.56 ms
per P1 tree hit and 0.33 ms per P2 tree hit.

## End-to-end interpretation

Profiler-off runs with the final fused kernel did not show a stable end-to-end
gain. A P1 pair improved, while the final order-reversed P2 pair regressed slightly:

| Final fused profiler-off pair | CPU | GPU | Change |
|---|---:|---:|---:|
| P1 decode TPS | 66.123 | 67.021 | +0.899 tok/s |
| P1 target step | 68.471 ms | 67.518 ms | -0.953 ms |
| P2 decode TPS | 55.656 | 55.389 | -0.267 tok/s |
| P2 target step | 67.326 ms | 67.869 ms | +0.543 ms |

Thus priority 1 removes the measured CPU topology bottleneck without changing the
output, but its end-to-end TPS effect on tiny7 is smaller than run-to-run variation.
Another P2 GPU run was nearly identical to an earlier CPU reference (67.869 vs
67.890 ms), while one P2 pair contained two transient long target-verify prompts and
measured +2.379 ms. None of these single-run values is used as a speedup claim.

The defensible result is the directly replaced-span reduction above plus exact output
parity. This optimization does not address the dominant cost: four additional target
verification nodes in the current P1/P2 tree configurations.

## Artifacts

- Runner: `../run_gpu_topology_ab.sh`
- A/B JSONL/log/profile files: this directory
- GPU parity tests: `ssd/tests/test_tree_topology_gpu.py`
- GPU implementation: `ssd/ssd/engine/helpers/tree_topology_gpu.py`
