# Exact baseline and benchmark adaptations

Engine files in `/home/chokwans99/PSD-paper-review` remain at `a82f7d2`.
`mlsys_paper_ab.py` prepends that checkout before any SSD import, including
spawned workers. Every result records module source SHA256, adapter SHA256,
parameters, prompts, GPU selection, seed and the clean paper-worktree status.

Common adapters, identical across each comparison:

- Unique shared-memory namespace and distributed port. Disable the old global
  `/dev/shm/sem.*` deletion at shutdown, which could corrupt unrelated jobs.
  These change process isolation/cleanup, not inference kernels.
- Preserve target execution dtype in the AMD draft config after
  `dataclasses.replace`. The literal original AMD configuration advertises
  float32 and fails FlashAttention initialization. Both failed initial smoke
  attempts are retained. FP16 compatibility is shared, not credited as a
  verification speedup.
- Disable automatic prefix lookup in verification/cold-path comparisons.
  The paper paged-prefix implementation has independently reproduced errors;
  wrong-output cached runs cannot be latency baselines. Decode still uses its
  ordinary KV cache, tree glue and accepted-path commits. Prefix reuse is a
  separate correctness/performance experiment, not disabled production policy.
- Submit fixed groups of B requests through the unmodified old scheduler.
  This bounds its resident population without importing the new scheduler.
  All prompts finish; partial-B tail and cap/clip batch-steps are excluded
  only from steady metrics. This is a controlled decode comparison, not a
  continuous-serving latency benchmark.
- Observe actual emitted lengths and old engine time counters externally.
  The old inaccurate raw token counter is not used for TPS. AL/hit/token/time
  all use the same eligible batch-steps. Generated outputs are retained.

Primary model trials use T=1, so the original proxy's fixed T=1 also equals
the actual temperature. Changing the root law while testing verification
would confound this comparison. This is not a claim that the temperature
fix is unnecessary; its already approved port remains separate.

Narrow experimental arms:

- `base`: original verification, forward, draft/tree construction and KV.
- `chain`: current fast verifier with CPU temperature hints; no packed query,
  batched proxy graph, mixed-miss policy or whole executor replacement.
- `tree`: original B1 tree executor with a captured conditional residual
  ladder. Preserve original parent-q references, sibling order, CPU coins
  on visited nodes and one terminal multinomial. No change to target forward,
  proxy, root ranking, tree generation or KV commit. V1 and V2 pack-transfer
  prototypes are separately recorded. This is an optional *new minimal port*
  motivated by Round6, not the same path as unified graph OFF→ON.
- `kv`: only last-query preservation, original request boundary, completed
  block hash, paged-prefix attention. No tree glue replacement, defragmentation,
  KV promotion, new scheduler or new batched tree executor.

The final `tree_chain_b1` candidates exercise physical patch01 by itself in
`PSD-paper-chain-review`. Their job treatment is `chain`, but raw `arm` is
`base`: `PAPER_ARM=base` intentionally installs only the common adapter and
does not replace the physically patched function again. Source hashes and
the dirty diff (01 only) identify the treatment. Both candidates share the
corresponding original-engine controls from the unified contrasts.

Output determinism is checked across arms; source/RNG/coupling distinctions
will be reported rather than presumed. Microbenchmark observations are not
whole verification or serving TPS claims. Dense7B/8B on4090 does not reproduce
the original70B/Blackwell TP2 environment.
