# Round7 — original paper-code adoption review

Base: `a82f7d24fb36827a9a81a3567f344dccb71f193e` in an isolated detached
worktree. Systems source: `df8ac6e`. Neither paper branch nor production
defaults will be changed during measurement.

Question: which verification/KV changes should be ported to the paper code?
Do not use unified-executor graph OFF as the paper baseline.

1. Run the original paper engine with a common benchmark-only adapter for
   process isolation, actual emitted-token/nominal-B/cap-excluded accounting,
   and checkpoint dtype compatibility. Record every adapter and source hash.
2. Compare original chain acceptance with the separately ported fast chain
   verifier. Compare original B1 tree walk with a minimal captured residual
   ladder preserving the paper tree executor, target forward, proxy scoring,
   draft construction, and KV commit. No transplant of the whole B>1 executor.
3. First verify fixed-input mathematical parity, proposal-law handling,
   sampled outputs where RNG order is preserved, and malformed/degenerate
   cases. Diagnose failures before scheduling expensive corpus runs.
4. Profile 48 saved inputs, then all 480 inputs with repeated independent
   processes and reversed A/B order for supported representative settings.
   Include B1 tree/chain and original-supported B>1 chain; B>1 dynamic tree
   was unsupported in the paper baseline and cannot be labelled a direct A/B.
5. Isolate common KV fixes (paged-prefix lookup, last query, original request
   boundary, completed-block hash) from draft KV promotion. Reproduce old
   correctness failures, measure cold-path overhead and correct prefix reuse;
   invalid outputs are not a performance baseline. Do not replace tree glue.
6. Produce individual portable patches, negative findings, latency scopes,
   raw results and a recommendation matrix in REPORT and MERGE_REVIEW.

Hardware: eight idle RTX4090 24GiB; PIX pairs2,3 /4,5 /6,7, NODE0,1.
Use identical GPU pair within each contrast; isolate process ports/shared
memory; retain telemetry. Dense7B/8B available; paper70B/Blackwell TP2
latency is not reproduced by a dense7B run or an AWQ70B checkpoint.

Performance inclusion: entire batch-step has actual B equal nominal B and
no cap/clip. Tokens, time, AL and hit use the same included steps. Preserve
raw totals and all generated requests. No result-dependent prompt removal.

## Diagnostic decisions before additional runs

- First48-input original-engine profiles showed chain gains around0.6–0.8ms,
  much smaller and workload-dependent gains for the minimal B1 tree ladder,
  and sign-changing cold-path differences for KV fixes. Run whole480 paired
  controls rather than treating micro or one profile as an adoption verdict.
- Add a direct original-engine versus **current unified engine** contrast on
  the same GPU pair and T1/fixed tree parameters. This is a bundle comparison,
  distinct from the minimal ladder port. Stochastic RNG consumption differs.
  Two full480 repeats in reversed order are the confirmation; omit redundant
  extra48 profiling and retain separate16-input detailed timing diagnosis.
- Verify portable physical patches in a second sparse checkout, plus full
  model prefix cold/cached output parity with the narrow KV port. These are
  integration/correctness checks, not extra policy tuning.
- Workspace disk space was low when a second full checkout copied historical
  artifacts. Both review checkouts use sparse `ssd/ssd` paths now. Inference
  source bytes did not change; no user data or original results were removed.
- Unified R0 uses PIX2,3; R1 uses NODE0,1 after physical-port checks release
  that pair. This replaces the proposed wait for PIX4,5 before any R1 run.
  Each original/candidate contrast stays on the same pair. R0 is A→B, R1 is
  B→A. This adds a topology control while using otherwise idle GPUs. Report
  the pair-specific contrasts; an average across these two is descriptive,
  not a repeated-trial estimate for one fixed topology.
- Final isolation: apply only01 chain-fast inside the unchanged paper tree
  workload, using the unified comparison's original-engine controls (same
  480 prompts, seeds, parameters and GPU pair). Explicit `control` fields
  reference those controls rather than counting reused baselines as new runs.
  Candidate R0 follows the PIX diagnostic queue, R1 follows the NODE full
  queue. This tests whether whole-executor replacement is necessary for the
  observed tree-workload gain. These are two topology-specific confirmations,
  not extra tuning or two repeats on one topology.
  Both final candidates use a third sparse checkout with **physical patch01
  only**, installed before either candidate starts. `PAPER_ARM=base` selects
  only the common benchmark adapters; it avoids double-installing the fast
  function. Job treatment is still `chain`. Raw source hashes identify this
  precisely. No tree/KV patch is present in this checkout.
