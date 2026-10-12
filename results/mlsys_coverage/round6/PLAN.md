# Round6: itemwise approval, verification and KV performance review

Date: 2026-10-12 KST. Base: feat/duet-mlsys-coverage@2fe4596.

User approved actual-temperature root scores, G>M proposal-law correction,
resident/prefill B limit, and int64 wire extension. Output-cap batch steps
must be excluded from performance counters, with both tokens and time
removed. User separately confirmed excluding steps with actual active batch
less than nominal B while completing all remaining requests. Keep raw totals
and exclusion counts. Verification and KV performance changes remain under
review; this campaign is authorized, adoption is not yet approved.

## Frozen procedure before measurements

1. Add explicit step batch metadata and additive full-batch/boundary-excluded
   performance metrics. Keep old/raw fields and historical results intact.
   Validate union of exclusions and consistent event/token/time populations.
2. Run existing correctness regressions on GPU0. Use GPU1 for isolated
   verification microbenchmarks, with no model workloads on the same GPU.
   Measure chain B1/2/4/8 and tree B1/2/4/8, 32k/128256 vocabulary, T=.7/0,
   multiple tree sizes and branching widths. Separate eager/graph acceptance
   from whole target verification and from end-to-end TPS. Use alternating
   A/B order and repeated timing; compare mathematical/output references.
3. Whole-model fixed-parameter comparisons on PIX GPU pairs2,3 and4,5:
   full dense Llama2-7B+AMD135m and Llama3-8B+Qwama0.5B. Keep K4/2,
   exit21 (L2)/16 (L3), P1/P2 nodes8/4, C3, G=M, residual/topm/mix0,
   actual T=.7, q-path policy and miss2 fixed. Disable optional stream and
   fusion changes to isolate acceptance graph/trim. Compare B1 legacy and
   unified, B2/4/8 unified as appropriate. Use stratified48 for diagnosis,
   full480 for final representative B1/B8 comparisons, same input caps,
   output128, independent process repeats, warmup then measured pass.
   Algorithm/RNG changes must not be described as identical-output A/B.
4. Distinguish (a) generic prefix-cache/re-prefill correctness, (b) old
   removed draft KV promotion replacing glue, (c) required target accepted
   path commit and current batched-draft stage/restore. Audit fragmented
   page tables. Measure stage, restore, glue forward and target commit
   separately. Add explicit profiling only, no unsafe switch that disables
   required KV operations. Benchmark gather/scatter versus recomputation
   under equal context/sequence semantics; do not revive removed promotion
   or approve a new serving algorithm based only on a microbenchmark.
5. Report actual correctness and latency evidence, negative/failed runs,
   comparable timing scopes, limitations and itemwise approval status in
   REPORT.md and append to ../MERGE_REVIEW.md. Do not claim a universal
   speedup or original70B/TP2 reproduction on this eight-4090 server.

Hardware at start: all eight RTX4090 idle;2,3 and4,5 and6,7 PIX;0,1 NODE.
Use pair6,7 for independent KV/model diagnosis if needed; preserve per-pair
comparisons and monitor external processes. No other user's processes are
terminated. GPU work resumed by the user's explicit instruction this turn.

## Diagnostic extensions after observed failures (not acceptance criteria changes)

- Qwama BF16 retained-tree vs shorter recomputation produced softmax TV above
  the initial 1% investigation trigger and a greedy argmax difference. Keep
  this failure. Additional diagnosis records all 48 cases without declaring
  numeric equivalence: same one-token tip shape, equal-width path recomputation,
  BF16 versus FP16. Equal-width comparison isolates shape-dependent arithmetic
  from incorrect logical/physical context. No production dtype/policy change.
- Add separate full target-model cached-prompt prefill measurements on both
  pairs, B1/B8, prompt lengths512/768/800, three alternating cold/cached repeats.
  Both arms use the corrected implementation. Clear only completed-request
  block hash lookup in the cold arm; do not benchmark knowingly invalid old
  paged attention as an acceptable baseline. Measure actual cached-token counts,
  prefill latency, greedy output equality and preemption count. This is separate
  from tree continuation KV promotion and is not a full-dataset TPS claim.
- All model/profile comparisons remain the frozen configuration above. Failed
  benchmark fixtures (parent qref, greedy wire, draft-attention dispatch,
  diagnostic canvas width) are retained; none required production logic changes.

- Fixed-input results exposed slower graph acceptance than legacy serial greedy
  walking at T0 B1/B2. Prepare a **benchmark-only** batched-argmax/CPU-walk
  alternative (32 shape cases, exact output equality against production
  batched greedy). Run when a campaign GPU is free. No production dispatch
  modification. This checks a concrete remaining optimization direction; its
  micro speed is not a full-serving TPS claim.

- To answer the user's **whole chain verification** question with current
  model paths as well as acceptance-only micro, add24 profiled48 controls:
  two models × B1/2/4/8 × general/fast/fast+packed. Reuse K4/2, exit21/16,
  T.7, output96, fixed seeds8300/8301, draft73; tree disabled, proxy graph
  enabled in all arms. Queue these after the B8 full runs on the same pairs
  so they do not delay the existing B1 full critical path. No runtime edits.

- The greedy-only host prototype was faster in all32 micro shapes, with
  identical path/recovery/terminal. Validate it on real models using an
  explicitly opt-in **benchmark adapter**, not production file/default edits.
  Two models × B1/B8 × host0/1 × profile48/full480 =16 model executions.
  K4/2, exit21/16, C3, actual T0, output64, target seeds8400/8401 (second
  measured), draft74. Original accept graph capture remains in both arms;
  only replay changes, using CPU metadata already read by the verifier.
  Candidate scoring/probability budgeting is unchanged. Queue after chain
  diagnostics so these use GPUs otherwise freed before long B1 repetitions.
