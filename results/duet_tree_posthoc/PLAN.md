# Tree score posthoc study — 2026-09-22

Aim: diagnose expected descendant AL conditional on a cache hit, with root policy fixed.
The original three-seed performance campaign stays frozen.

Before collecting new outcomes:

- Run all 480 Spec-Bench questions / 560 turns for both old q-path and frozen phase/sibling/q-bin policies, seed 1. Same 128-token cap, natural EOS, actual multi-turn history, 4/2 depth, 8/6 nodes, G=M, and models as `duet_tree_al_full/plan.json`.
- Observe **every served tree**, save exact full-distribution-derived alpha, reach, terminal probabilities, coin variance, draft-only features and target-only explanatory features. Save raw distributions for every 32nd tree for independent reproduction. Exclude warmup. Audit against verification events. Do not use observer timings as performance evidence.
- Primary diagnostic uses nonfinal verification events. Conditioning on an event being nonfinal depends on its outcome: exact expectations describe the tree before that event's coins, not a censoring-adjusted conditional estimator. Also report all-event prediction/noise diagnostics.
- Assess q-path and the frozen score on identical observed trees. Break errors down by phase, depth, sibling, q confidence, and target/draft overlap. Separate target-only oracle information from features available when the node is created.
- Examine exact conditional coin variance; lack of predictability of an individual random walk must not be called score error.
- Counterfactual prefix/ancestor-constrained subset DP at node budgets 2/4/6 diagnoses a **fixed sampled pool**, not safe post-sampling pruning or optimal generation. Separate score error from optimizer error. Do not claim alternative ungenerated branches are known.
- Test scalar lookup calibration extensions (depth, conditional WOR q, parent-q entropy) using five question-grouped folds. Both turns and both policies of one question stay in one fold. Fold assignment is fixed by each task's question index modulo 5; no neural model or extra target forward. Compare both within-policy and cross-policy generalization. These are exploratory cross-fitted diagnostics, not confirmation of a new online AL gain.
- Calibration-budget control (specified before full outcome analysis): also fit depth/rich tables using only the original eight calibration prompts / 77 trees. Compare them with the original frozen table on full diagnostic trees. A benefit from 384-question cross-fitting must not be described as a benefit from the original tiny calibration budget.
- Analyze actual continuation gain among observed expanded nodes. Do not substitute unavailable future q/p as a free online feature. State selection/positivity limits explicitly.
- Additional analysis fixed while full runs are collecting: integrate one future child over q, giving exact g1 = sum min(p,q). Compare known expanded nodes at equal depth; fit a scalar g table from earlier-node features in the same question folds. This does not observe unexpanded nodes, and is not an online gain trial. The g table is reach-weighted and trained only on observed expansions.
- Report confidence intervals from task-stratified question clustering, coverage, raw audit, mathematical identities, failure examples, and reproducible source hashes.

No efficacy stopping, model selection by a single task, forced positive conclusion, or change to production proposal/verification rules.

Additional mathematical audit during collection: test the breadth-first one-child allocation rule.
Residual/WOR can have g(2)-g(1)>g(1). Compute exact g(1)/g(2) on the deterministic raw 1/32
sample using an O(V log V) formula, independently checked against full finite-vocabulary enumeration.
Compare two known expanded parents at equal depth under a two-new-node budget. This is an
oracle allocation diagnostic on a sample of full-corpus traces, not a full online allocation trial.

Exploratory follow-up to the first-policy readout: for the same two-parent/two-new-node diagnostic,
derive phase-only g(1)/g(2) curves from the ORIGINAL eight calibration prompts, weight by exact
parent reach, and select among (1,1)/(2,0)/(0,2) using frozen estimated reach. This tests an
allocation decision with online-available scores. It is not a new inference rollout and no
full-corpus outcome is used to fit these two phase curves. Report all preexisting comparisons.
