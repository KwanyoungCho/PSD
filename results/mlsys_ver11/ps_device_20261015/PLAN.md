# PS metadata readback removal

Baseline: immediately preceding optimized runtime, not the original paper runtime. See BASELINE.json.

Changes: reuse P1 host prefix length in P2; defer fan-out host lists until an eager/diagnostic consumer. No policy/RNG/graph math/guard changes.

Validation: final full regression; forced executor-decline fallback parity; paired AB/BA native event breakdown (24 SpecBench turns each, fixed settings); 48-turn profile-OFF check. Run GPU performance arms sequentially on 4/5 after regression, preserving raw outliers. No TPS-based adoption claim.

Counters under study: phase2_build, dispatch gap, root metadata preparation, executor buffer preparation, PS graph, total PS. Source-conditional breakdown and output equality also required.
