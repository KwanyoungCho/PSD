# Direct residual versus proxy comparison

2026-09-12. Development: existing 4 datasets x 32 prompts x seeds 42/123,
layer 56, T=1, B=1, chain K=4, M=17 and global root budget 15. The previous
train/test split has been inspected repeatedly; treat ALL existing snapshots
as development for NEW policy selection.

Fresh confirmation: 32 additional distinct prompt texts per dataset, starting
at source row 33, seed 7, identical model/settings/collector, maximum output
256, stride 8. Exclude texts seen in any development dataset and duplicates
within confirmation. Freeze selected policies before evaluating confirmation
outcomes. Report every prespecified candidate family, including failures.

Main outcomes: prompt-balanced global top-15 coverage, paired prompt-cluster
bootstrap within dataset; equal-position top-3 correction coverage weighted
by true first-rejection probabilities for mechanism decomposition. Do not
equate these with live cache hit rate or TPS. Mirror-SD refers here only to
the proxy-score comparison inside DUET, not a whole-system implementation.

Analyses: direct TV to the same true residual; exact budget-aware support and
ranking regret decomposition; ideal-residual advantage and proxy-induced
regret; sufficient dominance/ranking conditions; model-free calibration
diagnostics; TV-optimal per-context uniform-mixture fit (oracle diagnostic);
oracle gradual repair of actual proxy error; dependence on actual error,
draft overlap, and observable proxy features.

Deployable-score families to investigate: subtraction shrinkage lambda,
support-preserving mixtures of normalized proxy/residual, smooth relative
draft suppression, temperature correction, and step-wise switching between
the two original policies using only e/q-derived features. A best single
threshold rule and a ridge-linear gain predictor are fitted on development
only. All diagnostics using p are non-deployable and labeled oracle.

No engine sampling changes. Existing lossless chain audit remains applicable;
new policy comparisons are offline on the exact same stored paths. Numerical
theorems are independently tested on synthetic probability simplices. Fresh
prompt provenance, code hashes, run completion, dtype/probability validity,
and existing-policy parity are checked before final reporting.
