# Round 4: correctness, phase overlap, tree-policy transfer and miss fallback

- Working base: feat/duet-mlsys-coverage@696e92c; paper base a82f7d2.
- User requests: boundary exclusion, paper Figure 5 timing, apply previous tree improvements, shorter/branching miss drafts for B1/B>1, audit legacy B1 G>M.
- Priority: proposal-law correctness before performance; boundary accounting before comparisons; policy smoke before full dataset.
- Hardware initially idle: 8 RTX4090. Pair 2,3 / 4,5 / 6,7 use PIX links. GPU0 reserved for regressions. Same comparison stays on same pair. Capture occupancy and external processes.
- Model pairs: full dense LayerSkip-Llama2-7B / AMD-Llama-135m and LayerSkip-Llama3-8B / Qwama-0.5B-Instruct. This is not the paper's dense70B/Blackwell setup.
- Dataset: all 480 first-turn questions, input cap512, output cap64. Small subsets are smoke/profile only and explicitly labeled. T=.7 primary policy comparison; T=0 correctness checks.
- Primary metrics: returned-token TPS and AL including recovery. Retain full raw accounting. Auxiliary boundary-excluded AL excludes capped terminal events; throughput exclusions remove the entire affected batch-step numerator and duration. Request-level censoring is reported separately, never silently substituted for full-run TPS.
- Tree transfer: q_path, reach, q_gain, reach_gain, reach_frontier, reach_gain_frontier. Hold root policy, physical rounds/widths and G=M constant. Frozen historical calibration is copied with provenance; transfer results do not imply it was calibrated for the new dense models.
- B1 G>M: replace realized-score pruning by a generation-order prefix in both cached/precomputed and on-demand routes. Enumerated finite-vocabulary negative control plus serving tests and dense-model smoke.
- Miss: independently vary JIT chain length; compare shallow ordered-WOR branching if correctness checks pass. This is inspired by tree SD, not a full SpecInfer reproduction.
- Timeline: record P1 start/end, proxy ready, P2 start/end and target model/acceptance completion per matching step. Separate capture from steady execution. Never infer overlap by adding unrelated medians. Profile TPS is diagnostic only.
- Paper Fig5: K1/K2=8/4, exit56, dense70B/TinyLlama; hit normalized DS42%, proxy wait22%, PS27%, target verify61%+24%, sampling6%, sync9%. PS may finish after model forward yet before next verify-ready boundary; measure both deadlines.
- All failures, uncompetitive policies and remaining limitations stay in REPORT and the unified ../MERGE_REVIEW.md.


## Follow-up addendum (2026-10-08)

- Corrected the inferred miss default after checking actual source0 events: `duet_jit_short=True` means baseline chain2. Explicit chain2 is a reproducibility repeat. Added explicit chain4 for equal-node comparison with tree2x2; no completed arm discarded.
- Pair0,1 (NODE within NUMA0) became available for Llama2 B1; Llama3 B1 uses6,7 PIX. Absolute cross-model TPS is not topology controlled. Each within-model contrast stays on its pair.
- Added eight-question diagnostic dense calibration, frozen before evaluation. Report both all480 and held-out472 excluding those eight from both compared arms. Historical table transfer remains separately reported.
- Kernel trace justified exact mask/fanout fusion, metadata bulk export and existing parallel insertion kernel ablations. Full comparisons use a fresh baseline and one engine with two target seeds; show each pass separately. Never claim independent process repetitions or inference-wide optimum.
- B1 follow-up stage retains the directory name `*_b1_short` but now contains explicit chain4, chain1 and same-policy implementation optimization. The proposed redundant legacy-short arm was canceled before execution.
- Runtime source manifest now hashes tracked AND new Python source files. Earlier reports predate this field and retain their original source/history metadata; they are not retroactively relabeled as exact snapshots.

- Final combination check added after component results: historical reach+gain+frontier, explicit miss chain4, K4/2, fused+bulk, same480 questions and2target passes. Do not add individual AL/TPS percentages; measure the combination directly. This is an exploratory combination confirmation on the same corpus, not a newly untouched test set. The short-budget stage now contains two jobs per model.

- Spare pairs0,1 (Llama2 NODE) and6,7 (Llama3 PIX, after B1) are used for a bounded independent-process baseline/fused+bulk confirmation. Each arm is a fresh engine with draft seed1 and target3030/3031, all480 questions. These are secondary-pair checks; never pool absolute TPS with the primary pair. No further GPU arms are planned after these and the final combinations.

- Causal-audit amendment: K2/1 implicitly changes default miss2 to miss1 through `duet_jit_short=True`. Add exactly one matched-miss control per model (K2/1, explicit miss2, fused+bulk, full480/two target passes) on the primary pair. This resolves the discovered coupling; it is not another optimizer sweep. P1/P2 node budgets still change8/4 to4/2 with the phase configuration, which remains explicit.
