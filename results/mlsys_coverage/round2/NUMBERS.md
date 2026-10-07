# Raw-result-derived round-2 numbers

All main rows: 480 first-turn inputs, B=8, T=0.7, max output=64. TPS mean ± sample SD; n=3 only after all three runs complete.

| Model | Variant | n | Decode TPS | AL | Hit |
|---|---|---:|---:|---:|---:|
| llama2 | base | 3 | 634.15 ± 2.45 | 1.9181 | 0.7929 |
| llama2 | packed | 3 | 651.70 ± 4.91 | 1.9197 | 0.7905 |
| llama2 | mixed | 3 | 651.04 ± 4.08 | 1.7959 | 0.8057 |
| llama2 | candidate | 3 | 652.45 ± 3.79 | 1.9238 | 0.8102 |
| llama2 | ssd | 3 | 649.08 ± 1.70 | 1.9161 | 0.3501 |
| llama3 | base | 3 | 646.76 ± 1.17 | 2.3733 | 0.7385 |
| llama3 | packed | 3 | 647.58 ± 2.82 | 2.3572 | 0.7353 |
| llama3 | mixed | 3 | 672.26 ± 4.02 | 2.1185 | 0.7570 |
| llama3 | candidate | 3 | 673.06 ± 5.54 | 2.1185 | 0.7570 |
| llama3 | ssd | 3 | 652.33 ± 4.16 | 2.3900 | 0.4537 |

## Greedy tree / chain (B=1, one full-corpus run per row)

| Run | Decode TPS | AL | Tree events |
|---|---:|---:|---:|
| llama2_tree_full/llama2_duet-chain_p10_t0_full.json | 123.57 | 2.2413 | 0 |
| llama2_tree_full/llama2_duet-tree_p11_t0_full.json | 108.22 | 2.3772 | 11083 |
| llama2_tree_host/llama2_duet-tree_p11_t0_full_host.json | 106.94 | 2.3772 | 11083 |
| llama3_tree_full/llama3_duet-chain_p10_t0_full.json | 128.96 | 2.7589 | 0 |
| llama3_tree_full/llama3_duet-tree_p11_t0_full.json | 104.18 | 2.8570 | 6319 |
| llama3_tree_host/llama3_duet-tree_p11_t0_full_host.json | 101.85 | 2.8570 | 6319 |
