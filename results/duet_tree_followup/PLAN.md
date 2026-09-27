# Tree follow-up validation campaign — 2026-09-22

Status: CPU and one-GPU executor fixture validation complete. Real-model full
inference awaits five free GPUs. See REPORT.md and queue_status.json.

Objective: accepted descendants conditional on a cache hit (AL), with unchanged
root selection, target/draft, temperature, depth 4/2, node budgets 8/6, C=3 and
G=M. TPS is descriptive. Preserve all previously frozen results.

## Policies fixed before new online outcomes

1. `q_path`: existing q product, breadth allocation, depth == round frontier.
2. `reach`: original eight-prompt frozen acceptance/reach score only.
3. `q_gain`: q product plus calibrated C=3 gain allocation.
4. `reach_gain`: frozen reach plus gain allocation.
5. `reach_frontier`: frozen reach, breadth allocation, all unexpanded frontier.
6. `reach_gain_frontier`: both gain allocation and all unexpanded frontier.

All preserve the existing per-root future-round reserve, raw-q/root thresholds,
round widths, root quotas and physical forward schedule. Gain allocation solves
the finite per-root node allocation problem BEFORE sampling new child tokens.
This is a one-round objective under a multi-round budget, not a proof of globally
optimal expected AL. No child is deleted based on its realized score. Selecting
zero children is allowed, but that queried parent is marked expanded as before.

## Calibration and numerical validation

- Use only the original eight prompts / 77 calibration trees for gain curves.
- Compute gamma(1), gamma(2) exactly. Derive gamma(3) by conditioning on the first
  rejection; exact finite-vocabulary checks and controlled, seeded numerical
  integration on real distributions. Record numerical uncertainty separately
  from the much larger calibration/context uncertainty. No extrapolating C=2.
- Compare allocation with exhaustive enumeration, including nonconcave gains,
  zero/unequal remaining budgets, ties, invalid lanes and multiple roots.
- Verify finite-vocabulary output distribution for legal pre-sampling allocation,
  plus a deliberately illegal post-sampling selection negative control.
- Check CUDA graph/eager tensor parity and tree masks, q references, sibling
  prefixes and ancestry before full runs. GPU checks are required, not inferred
  from CPU success.

## Online comparison

All 480 Spec-Bench questions / 560 turns, seeds 1, 42, 123, six policies. Interleave
policy order by seed; no early efficacy stopping. Natural EOS, 128-token cap,
actual previous-turn generation, native 2048 context; same raw base-model format
as the previous campaign. This is not the paper's 1024-token setup.

Primary: equal-six-task mean of pooled nonfinal hit-conditional descendant AL.
Question-clustered, task-stratified paired bootstrap (5000 samples), all turns and
seeds together. Report all policy contrasts, P1/P2, task/seed variation, emitted
AL, all-event sensitivity, hit rate, node counts and coverage. Use simultaneous
intervals for the prespecified primary contrasts before claiming a winner.

## Causal diagnosis

Audit full eligible frontier/allocation decisions in a separate traced run;
tracing timings are not performance evidence. Same-prefix super-tree replay
requires new p/q for unexpanded branches. Old hit-only traces cannot substitute
for that observation. Keep unmeasured questions explicitly pending.

Follow-up feasibility check: reusing saved leaf target p needs only draft q, so
we tried reconstructing the prefix and recomputing TinyLlama q on one idle GPU.
Both HF and production-model/SDPA routes failed the prespecified known-parent
q agreement gate (median TV <.005, p95 <.02). Preserve the failed checks; do not
use them to fill missing oracle labels. Same-engine/KV multi-round causal replay
remains unimplemented and is NOT included in queue completion status.

## Resource constraint at start

All eight GPUs were occupied by other users. Do not terminate their jobs or load
the 70B trial onto occupied devices. Implement, calibrate and validate on CPU
while waiting for five available GPUs. Never label a queued or unrun trial as
completed, or a local expected gain as measured online AL.
