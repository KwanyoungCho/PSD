# Round5: temperature-correct improved DUET versus independently tuned SSD

User request 2026-10-08. Base c4f05a6, feature branch feat/duet-mlsys-coverage.

1. Correctness first: use request target/draft temperatures for cache-root scores; retain soft ranking at greedy T=0 and unchanged exact verification. Integrate e(1-q), full allowed-vocabulary normalization, and 0.75 observed + 0.25 overlap terminal-position mass. For a branching tree, generalize the two terminal-mass recursions along ordered siblings; this is an estimator, not an exact conditional probability theorem. Verify chain reduction, ragged lengths, sibling topology and graph replay with changing temperatures.
2. Freeze a stratified tuning subset (8 per group = 48 questions), exclude it from the separately reported 432-question confirmation. Full480 tables are also required. All experiments use the same original480 first-turn corpus, T=.7, dense two model pairs, initial input512/output96 screen; final output128 and separate longer-output checks. Preserve returned-token and boundary-excluded metrics. Same output/RNG criteria apply only to identical algorithms.
3. Independently profile/tune SSD K={2,4,6,8}, F={1,3,5}, B={1,8}. Examine target verify, draft tree preparation/forwards, cache hit/miss and waits. Expand a boundary optimum once in the improving direction. SSD keeps the same shared correctness/backend/graph optimizations; it is not an untuned strawman.
4. DUET initial breakdown anchors: exits16/21/26/30 at K4/2; K2/1, K4/2, K6/3, K8/4 around useful exits; tree node/width, P1 root count and P2 root budget interventions follow the measured bottleneck. Hold miss length4 explicit while comparing K. Start reach_gain_frontier and existing fused/bulk options. Every new parameter family has a timing/coverage/AL reason. Evaluate neighboring settings until the selected finite neighborhood is closed or a memory/correctness constraint is recorded. Finite measured optimum, never a global perfection claim.
5. Keep a Pareto comparison of AL and time. Report an AL-priority setting and a throughput-efficient setting if they differ; do not quietly replace the user's AL objective with TPS. Selection uses tuning only, separately from full/held-out confirmation. Profiles are diagnostics and never reported as uninstrumented TPS.
6. Freeze chosen SSD/DUET B1 and B8 configurations before confirmation. Compare on the same GPU pair per model and batch, same input order/seed/output caps. Run at least two independent draft-process seeds, with a warm pass and a measured pass; report paired question bootstrap for AL and process-level throughput range (not token-level CI). Add B2/B4 with declared parameter transfer or tuning. Also compare DUET implementations with/without previous batched optimizations at fixed algorithms.
7. Analyze final breakdown before further optimization. Implement only evidenced overhead fixes, verify identical outputs for same algorithm, then rerun affected comparisons including SSD when shared code changes. Keep failures and negative results.
8. Deliver one Round5 REPORT and append the current conclusions, exact changes, artifacts, presets, merge conflicts and remaining scope to ../MERGE_REVIEW.md; commit/push after validation. No remaining planned job is called complete before it actually finishes.

Implementation follow-ups added after profiling: (a) eliminate unused tree ladder copies/final normalizations, (b) overlap batched-tree proxy scoring with target post layers. Both are opt-in/default0, preserve frozen algorithmic parameters, and have separate profile parity gates and full480 paired process repetitions. The stream arm is compared against trim-only; trim is compared against the original implementation. Do not conflate the two increments or kernel-only and end-to-end speedups.

Final parameter closure after implementation changes: on tuning48 only, profile K1−1, K2−1, exit−2 and exit+2 around each frozen fast point with trim/stream enabled. Warm-check the top two plus unchanged stream/trim controls. Switch only for a measured >=2% warm gain (an experimental noise guard, not a runtime parameter or significance test). Freeze this decision before a new full480 confirmation; retain existing paired full data when the selected parameters/implementation are unchanged. New parameters get both same-seed/same-GPU full repeats. This finite check addresses a changed overlap window without feeding full480 AL/TPS back into parameter selection.

Hardware: eight idle RTX4090, full models on one target + one draft GPU. Pair2,3 and4,5 are primary PIX pairs;0,1 NODE and6,7 PIX can screen/profile independently. Pin comparisons to the same pair, record topology and external processes; never pool pair-specific absolute TPS.

Temperature audit: the preexisting sampling/verifier paths already used request T; cache-root proxy scoring explicitly used T=1. The modification changes cache allocation, not the lossless target verification law.

## B8 stability amendment (05:49 KST, before new selection)

The last-pass timing audit found that rare long steps dominate some short
B8 tuning runs. Example: L2 B8 postopt trim control, three steps >5×median
contribute19.7% of measured time; ordinary median is28.24ms, maximum1068.78ms.
The original L3 B8 K3/1 finalists also have large tails, so restricting the
recheck to the newer K3/2 neighborhood would perpetuate that selection bias.
The original SSD and B1 warm finalists have no >5×median steps. The threshold
is a descriptive audit label, not a filter, runtime parameter or diagnosis
of the tail's cause. No final480 throughput sample is discarded.

Amended bounded protocol: B8 only, original trim/stream controls plus the
two postopt profiled finalists (four L2 settings); for L3 also K3/1 at exits
24/26 with trim and trim+stream (eight settings total). Each candidate uses
tuning48/output128, six passes in one process, target seeds7000..7005 and
startup draft seed29. First three passes warm; last three measured. Choose
the largest median whole-pass TPS*, switching only for>=2% over the original
trim control. This guard is not a significance claim. No per-step tail
trimming is permitted. Run the frozen SSD fast control on the same six-pass
protocol. Each DUET candidate gets a separate detailed profile on the other
GPU pair; timing results from those profiles do not select the winner.

Freeze the amended B8 choice before new full480 confirmation; reuse existing
full repeats only when parameters AND implementation switches match exactly.
Otherwise perform both original paired-GPU/seed full480 repeats. Keep all
old frozen decisions and full measurements as provisional/history. B1 keeps
its postopt result, and AL-priority points remain independently reported.
No full480 score is used to choose the amended parameters. This amendment
addresses a measured selection flaw, not a claim of global optimality.
