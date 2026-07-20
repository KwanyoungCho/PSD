# 11 — KV promotion (glue removal), SwiftSpec-style: design [REMOVED]

**Status (2026-07-04): implemented, verified correct, measured as a
performance WASH at B=1, and REMOVED from the codebase** (user decision —
the 10-token glue forward is already batch-free on GPU, so replacing it
with a 1-token tip forward + KV gather nets zero; the code only added
complexity). Full implementation preserved in git history
(commits 80eb896 → 41bee95, removal commit follows 41bee95).
Measured results: experiments/proxy_async_overlap/kv_promo/RESULTS.md.
This document is kept as the design/correctness record.

**Original design below** — Gate: `SSD_DUET_KV_PROMO=1` (default OFF).
Motivation: docs/duet/10 §import-list item 1.

## What glue does today (hit steps, 82%)

Decodes the response chain `[rec, c0..c_{K-1}]` (K+1 wide, one forward
~1.78ms + prep ~0.9) at REAL positions `nt-1 .. nt-1+K` to (a) write
draft KV there, (b) produce K+1 fork-logit vectors.

## Why promotion is correct (verified against the code)

1. **KV exists**: the matched cache row's seed KV was written by tree
   forward 1 (scratch slot s=0), chain token c_i by forward i+2 (slot
   s=i+1) — EXCEPT c_{K-1} (the tip): the forward that would write its
   KV / produce its after-logits never ran. (Same physics as SwiftSpec
   landing on an unexpanded leaf; our chain rows ALWAYS land at a tip.)
2. **RoPE consistency**: a row matches iff its seed position k_idx ==
   chosen_pos == L (first-reject). Seed rope = nt_prev-1 + k_idx + 1
   = nt_prev + L = nt_new - 1 = glue position 0's rope. Chain token i's
   rope = seed+1+i = glue position i+1. Exact match, all rows.
3. **Attention-context exactness**: tree mask is tril over glue —
   a row seeded at j attends glue[0..j] only, i.e. exactly the tokens
   that are accepted prefix by the time it matches. No contamination
   from the rejected glue suffix.
4. **Fork logits already stored**: glue logits position i == the row's
   stored logits (`tree_cache_logits`, = out_logits at the hit path),
   shifted: glue_logits[:K] = out_logits[:K]; glue_logits[K] must come
   from the tip mini-forward.

## The minimal-form replacement (hit steps only)

1. **KV gather**: src slots = row scratch slots {seed s=0, c_i s=i+1,
   i≤K-2} from the PREVIOUS step's layout (base = nt_prev-1 + glue_w_prev
   ... + s*MQ_LEN + row_offset; physical via dbt — block tables only
   grow, old positions stable). dst = glue slots nt-1 .. nt-1+K-1.
   Src/dst ranges OVERLAP (dst tail ≥ nt_prev+K enters the old scratch
   region) → gather to a temp buffer, then scatter. KV tensor is
   monolithic `[2, L, blocks, bs, H, D]` → view `[2, L, blocks*bs, H, D]`
   → two fancy-index copies across all layers at once.
2. **Tip mini-forward**: 1-token standard decode (existing decode CG
   family) of c_{K-1} at position nt-1+K → writes tip KV + yields
   glue_logits[K]. (Latency-bound ~1.2-1.3ms — the irreducible part;
   folding it into tree forward 1 is a later optimization with mask
   surgery + a 1-shallower position-K sub-row.)
3. **Fork logits** = cat(out_logits[0,:K], mini_logits) — no glue CG.
4. **Miss steps unchanged** (JIT + legacy glue): JIT writes KV at real
   positions already; glue dedup there is a separate, smaller follow-up.

## Prerequisite: un-overlay Phase 2 scratch under the gate

Current lookahead `(K1+1) + max(K1*mq1, K2*mq2)` lets Phase 2 overwrite
Phase 1's first K2*mq2 scratch slots — fine when tree KV is disposable,
FATAL for promotion (P1 rows' depth 0..~2 KV destroyed). Under the gate:
lookahead = `(K1+1) + K1*mq1 + K2*mq2` and the Phase-2 region bases
after Phase 1's extent. Cost: +K2*mq2 (=40) reserved tokens per seq —
negligible at B=1 / max_model_len 2048.

## Expected value (honest)

Saves glue prep+width (~2.65ms) minus mini-forward (~1.3) minus gather
(~0.05) ≈ **+1.1-1.3ms draft busy on hit steps (82%)** ≈ −1ms/step
average. At the B=1 champion this is draft slack (frontier: no direct
TPS); it funds the wide-early/overlap bundle and is a direct period cut
in draft-bound regimes. Also deletes the glue CG family from hit steps.

## Step plan (one commit each)

1. docs (this file)
2. promo metadata: record per-step scratch bases + row-slot arithmetic
   helpers + non-overlay lookahead under gate
3. hit-path: gather + tip mini-forward + logits assembly, glue skip
4. CPU unit test of slot arithmetic; py_compile; TRACE parity mode
   (SSD_DUET_KV_PROMO_PARITY=1: run BOTH paths, compare glue_logits/KV)
5. GPU smoke (ns=4) + parity run + A/B vs champion
