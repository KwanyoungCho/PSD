# DUET tree parameter postmortem and optimization

This study searches P1 and P2 independently where the implementation permits,
then validates only the winner on full Spec-Bench.  Diagnostic traces are never
used as throughput measurements.

## Fixed reference

- native context: 2,048; no draft RoPE extension
- target/draft: LayerSkip Llama-2-70B / TinyLlama-1.1B
- seed 42, temperature 0.7, top-p 1.0, raw prompts, batch 1
- K1/K2 = 8/4, exit layer 56
- P1: backbone, roots-per-position 3, N1/M1 = 14/12
- P2: W/R = 15/10, N2/M2 = 8/8
- shared branch width C = 2; P1 thresholds 0/0; P2 thresholds 0.01/0.01

## Data split and iteration

1. `forensic_subset.jsonl`: 10 questions per subtask (60 questions, 70
   turns), deliberately spanning long prompts, low/high AL, P1-heavy,
   P2-heavy, and miss-heavy outcomes in the existing seed-42 reference.  It
   is used to explain failure modes and is not a headline evaluation set.
2. `screening_subset.jsonl`: a fixed seed-20260814 sample of 12 different
   questions per subtask (72 questions, 84 turns).  It is used for paired
   candidate checks after the trace-based narrowing.
3. Full Spec-Bench: 480 questions / 560 turns, output cap 1,024.  Run only the
   unchanged reference and the single selected candidate under the same
   native-2,048 stopping contract.
4. After the first full comparison, use `postfull_forensic_subset.jsonl` (48
   questions) and `postfull_screening_subset.jsonl` (60 questions), both
   disjoint from the original 132 tuning questions, for a single residual-cap
   audit.  Any second full candidate must first pass the same overlap-tail
   gates; these post-full subsets are never reported as headline results.

## Post-hoc questions

- **Depth:** how often does the accepted path reach the last generated depth?
  A high censoring rate means another draft round has measurable opportunity.
- **Width:** how often is an accepted/expanding node ranked outside the current
  round width, and how often does an alternative sibling rescue acceptance?
- **Search versus verify budget:** replay closure-valid M caps on an N-node
  generated tree.  Then use live N>M arms before increasing target verify rows.
- **Candidate allocation:** inspect P1 hit context/root rank and P2 served-root
  rank.  Change P1 roots-per-position or P2 R only if useful roots concentrate
  at an allocation boundary.
- **Marginal value:** compare relative AL gain with relative target-step gain.
  Sending more nodes is useful only if AL grows faster than step latency.

## Overlap gates (distribution, not mean)

For every profiled arm, signed gaps are computed per aligned step.

- P1 gap = proxy arrival on draft - P1 ready.  Negative means P1 was late.
- P2 gap = next target request - P2 cache ready.  Negative means P2 exposed a
  target-side wait.

Report late frequency plus signed p01/p05 and overrun p90/p95/p99/max.  A new
arm must not increase either late frequency by more than 0.5 percentage point
or p99 overrun by more than 0.3 ms relative to the profiled reference.  This is
checked separately for P1 and P2.

## Candidate order

The search is sequential rather than a full grid.

1. Collect a reference topology/calibration trace for structural analysis.
   Its overlap/TPS values are discarded because trace D2H and file I/O alter
   scheduling.  Collect a second clean CUDA-event profile with all topology,
   calibration, and node-audit hooks off for overlap-tail analysis.
2. Required K2+1 balance check: K2=5 at exit 56, then move exit to 54 or 52
   only as far as needed to restore the P2 tail gate.  Earlier exit also
   shortens P1 overlap, so both phase gates must pass.
3. Hold the selected K2/exit pair fixed and separate P2 search from verify:
   N2/M2=10/8, then 10/10 only if the extra searched nodes look useful.
4. Independently test P1 only if the reference trace is depth/width limited:
   K1=9 or N1/M1=16/12, followed by M1=14 only when target-side marginal
   verification is justified.
5. Change P2 R in {8,12} (W remains 15) or P1 roots-per-position only when the
   root-rank audit shows budget pressure.  C is shared by the current engine,
   so a C change is not treated as phase-independent.
6. Validate at most two survivors on `screening_subset`; run reference and one
   winner on full Spec-Bench.

Final selection requires the same UIDs, seed, context contract, and profiler
off.  TPS and AL are question-level metrics; overlap gates come only from the
separate profiled run.

## Executed post-full residual-cap audit

The first full winner still saturated P2 search at 9.535 generated nodes per
10-node cap.  On disjoint data, N2/M2=12/10 improved AL without increasing
target verification rows, while 12/12 raised target latency for negligible
extra AL.  Clean timing for 12/10 measured P1 late 0.474% and P2 late 0.378%,
with zero p99 overrun in both phases.  Therefore only 12/10 proceeds to the
second full validation; P1 remains unchanged.

The second full validation rejected 12/10 (TPS -1.47%, AL -1.16%, target step
+0.35% versus 10/10), despite a slight majority of question-level wins.  Since
only 6/560 turn outputs remained identical under the same seed, a final medium
check repeats N2 in {10,11,12}, M2=10 on the balanced 60-question subset at
seeds 1, 42, and 123.  This is used to judge the stable direction and the N2=11
midpoint, not as a paper headline result.

That check rejects N2=11 consistently and does not override the larger full
rejection of N2=12.  The next bounded candidate keeps N2/M2=10/10 and tests P2
confidence floors 0.02 and 0.03 at three seeds.  The current trace shows that
these floors suppress 20.29%/23.02% of expansion candidates while risking
1.08%/1.85% of useful expansions.  Only a seed-stable A/B winner can proceed
to full validation.

The three-seed A/B selects 0.02: mean TPS +3.29% and AL +3.38% over 0.01,
with all six subtask means improving and target verify unchanged.  Confidence
0.03 is rejected because its seed-42 regression outweighs the higher P2 AL.
The final full run therefore changes only P2 confidence 0.01 -> 0.02 while
keeping K1/K2=8/5, exit 49, P1 N/M=14/12, and P2 N/M=10/10.

The final 480-question/560-turn validation rejects confidence 0.02.  Relative
to 0.01 it reduced TPS by 2.11% and AL by 1.89%, increased target-step latency
by 0.212 ms, and improved only MT-Bench while the other five subtasks declined.
The bounded search is complete.  The selected configuration is K1/K2=8/5,
exit 49, P1 N/M=14/12, P2 N/M=10/10, and P2 proxy/confidence thresholds
0.01/0.01.
