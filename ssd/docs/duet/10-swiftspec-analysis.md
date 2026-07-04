# 10 — SwiftSpec (ByteDance, ASPLOS'26) analysis: what DUET can import

**Date**: 2026-07-04
**Sources**: arXiv 2506.11309 (paper), github.com/ByteDance-Seed/SwiftSpec
(code, cloned + full mechanism trace). SwiftSpec = disaggregated async
spec decoding, 348-369 tok/s Llama3-70B-int4 on 8×H800 NVLink.
Ablation: pipeline+tree-reuse 1.43×, latency kernels 1.16×, total 1.75×
over vLLM/SGLang/TRT-LLM.

## The "Evolving Tree Cache" mechanism (their name: tree-aware KV mgmt)

Draft keeps ONE persistent tree across rounds. KV layout: `[prefix cache
| tree cache]` — all layers in ONE monolithic tensor
(`dist_llama.py:2074`, `whole_cache`), so any reindex is a single
fancy-index copy across every layer at once (`update_kvcache`,
`dist_llama.py:2578`).

Per round (`tree_manager.py:167-255` + C++ `tree_manager.cc`):
1. `traverse(accepted_ids)` — walk the tree along the verified tokens.
2. **fall-in** (landing node exists — their "hit", counter tracked):
   `prune_and_remap_tree` + `update_kvcache(root, path+subtree_idxs)` —
   accepted path promoted into prefix AND the surviving subtree's KV
   compacted, **one gather, zero forwards**. The subtree (speculatively
   grown during the verify window) becomes the new tree.
3. **fall-out** (their "miss"): reuse the partial path's KV, forward
   ONLY the novel suffix (their only "glue"), rebuild.
4. Node state: `{token, cum-logprob, depth, ancestor-mask row}` + a
   global best-first frontier heap of `(score, token, parent)`; per-node
   "logits" = top-x child candidates in the heap. KV/hidden live
   positionally in the cache — that's why gather-reindex carries them.
5. While the target verifies, the draft keeps expanding `depth` more
   best-first levels past the snapshot it sent
   (`ea_model_pipe.py:362`); the snapshot is non-destructive
   (`get_target_candidates` pushes everything back).
6. Draft is a standalone small LM (token-ids only wire) — NOT EAGLE —
   which is what lets the pipeline decouple (target→draft payload =
   accepted ids + 1 bonus token).

## Direct answer: can DUET drop the glue decode? YES

Our glue does two jobs; both are replaceable on OUR data structures:

| glue job | replacement | status |
|---|---|---|
| draft KV for the response chain | chain = last step's cache row → its KV lives in scratch → **index_copy scratch→glue slots** (our KV is also one monolithic tensor `[2, L, blocks, …]` → single gather). Masks (`[persistent | glue | diag]`) don't care how the slots were filled → **zero CG changes** | design ready |
| fork logits at each accepted position | `tree_cache_logits` of the matched row — we ALREADY store full logits (richer than SwiftSpec's top-x) and ship them as logits_q | already stored |
| logits after the LAST chain token (full-accept fork, 29% of P1 hits) | never computed (the row stopped there). Fix: cover position K via P2's all-accept slot (Policy B already seeds it) or a 1-token mini-forward | one design wrinkle |
| miss steps | JIT already writes KV at the REAL prefix positions and produces out_logits → glue after JIT is pure duplication → skip | free |

Expected saving: glue fwd 1.78 + glue prep ~0.9 − gather ~0.05 ≈
**−2.5 ms draft busy on ~100% of steps** (hit via promotion, miss via
dedup). Caveat from the campaign frontier: at B=1 champion the draft has
slack, so this converts to TPS only through (a) the wide-early/overlap
bundle's "free draft budget", (b) draft-bound regimes (bigger draft,
B>1) where it is a direct period cut, (c) removing the glue CG family
(code simplification).

## Full import list, prioritized

1. **KV promotion glue-removal** (above) — cheapest, unconditional
   draft-time win, ~1-2 days. Do first.
2. **Fused GEMM+AllReduce kernels** (`GemvAR`/`GemmARLayerV2`,
   IPC-buffer AR fused into o_proj/down_proj epilogue, gated bsz==1
   seqlen≤16 — exactly our verify shapes; their 23-43% op cut, 1.16×
   e2e). **This is the only known lever that attacks our measured
   frontier (1.9 ms/verify-pos)** — cheaper verify positions would
   re-open deep-narrow depth (E10 failed at 1.9; at ~1.3 it wins), and
   only DUET wants depth → differentially helps DUET over SD even
   though the kernel is shared. Risk: their kernels target
   H800/NVLink IPC; PCIe 3090 port is nontrivial. Heavy but
   frontier-moving.
3. **Tree-attention kernel with in-kernel RoPE+tree-mask**
   (`DecodeAttnOp`): takes position_ids+mask directly — no FlashInfer
   plan() per position. Would kill our per-position prep
   (phase1_prep 3.1 + phase2_prep 1.6 ms/step) and the step-0 numpy
   mask build. Medium effort, draft-side (slack-bound) value.
4. **Best-first global frontier expansion** (SpecExec-style, C++
   TreeManager, O(k log s)): dynamic budget allocation vs our static
   fan_out_list. Deep-narrow is our static approximation; the dynamic
   version needs variable-depth rows → CG-shape family per level
   (they key 25 CUDA graphs by shape with warmup counters). Big work,
   marginal over a tuned static list at our scale.
5. **DUET 2.0 direction — proxy-boosted evolving tree**: unify P2 into
   a persistent best-first tree: proxy residual candidates enter the
   frontier heap with ĥ·r̂ scores (instead of a separate K2-deep phase),
   the tree evolves with subtree reuse, and accepted proxy-seeded
   branches KEEP DEEPENING across rounds instead of being rebuilt at
   fixed K2 — structurally lifts the K2-truncation on P2 rows (though
   per-token off-policy quality still needs draft adaptation). This
   absorbs items 1+4 and dissolves the P1/P2 phase split. Research-
   grade redesign; the natural "next paper" framing.
6. Minor adopts: fall_in/fall_out counters (≈ our hit/miss metrics ✓
   have), packed single-broadcast wire (✓ have), shape-keyed CG with
   warmup (✓ have), `-1` sentinel control channel (nice-to-have).

## Honest context

Their 348 tok/s is 8×H800 + NVLink + 3B draft on its own TP group —
different hardware class; the transferable knowledge is the mechanism
split above and the ablation shape (pipeline/tree-reuse >> kernels).
Their draft runs ~7 tree levels per verify window because H800 makes
draft forwards ~0.5 ms; on our 3090s the same window fits ~13 forwards
at 2.5 ms — the balance that drove our whole frontier analysis.
