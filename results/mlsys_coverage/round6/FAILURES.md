# Round6 failed probes and interpretation

Every failed raw file/log remains available. No failed probe is counted as a
passing correctness/performance result.

| File / run | Failure | Follow-up |
|---|---|---|
| `tree_micro.json` | Fixture assigned each sibling its own parent-q reference; violates shared parent proposal law | Correct fixture uses first sibling index for the shared parent q |
| `tree_micro_retry1.json` | Greedy fixture omitted `u_valid`, required by production wire contract | Add both `valid` and `u_valid`; `tree_micro_retry2.json` completes64 cases |
| `llama3_profile/llama3_b1_legacy_profile` | Nonexistent local target path | Resolve actual `/data/chokwans99/models/layerskip-llama3-8B`; retry in a separate directory, preserve initial plan |
| `amd_kv.json`, `qwama_kv.json` | Standalone model loader selected ordinary AR attention; ancestor wrapper has incompatible query dispatch | Set local Attention modules to existing async-draft dispatch in the isolated probe, no production attention changes |
| `qwama_kv_retry1.json` | Investigation threshold exceeded: BF16 logit/softmax and argmax disagreement between retained tree and shorter recomputation | Keep failure. Diagnose shape-dependent arithmetic; do not declare bitwise equivalence or enable KV promotion |
| `qwama_kv_diagnostic_{bf16,fp16}.json` | Equal-width diagnostic requested width N+1 for a chain with N+2 tokens including tip | Correct diagnostic uses max(N+1, chain length); final diagnostic retry1 files complete48 cases each |

The 1% TV value was an investigation trigger in a synthetic numerical probe,
not an accepted losslessness tolerance. Diagnostic completion does not mean
that a 1% or larger deviation is acceptable. The equal-width control and FP16
comparison are used to explain the source of the deviation; adoption still
requires the intended serving semantics and separate review.

The KV probe excludes network, cache lookup, lifetime of all candidate arenas,
and copy costs outside the single retained tree. It cannot justify replacing
production glue merely because its measured gather+tip time is shorter.

Current production tree glue covers all served tree contexts with an ancestor
mask, while this micro selects one path from already-computed tree KV and
adds one tip. It is not an A/B replacement of all production tree glue or
its future-context logits. A branching-tree promotion design would also
need correct missing-leaf forwards and candidate KV lifetimes.

These Qwama numerical comparisons concern draft KV/logits. A changed draft
proposal distribution alone does not violate target losslessness: the actual
sampling law q must match the q used in verification, and target p must use
the correct target context. The probe does not establish a new promotion
service's complete q/lifetime contract and does not certify bitwise identity.
Do not equate draft argmax disagreements with target output-law violations.
