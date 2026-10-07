# What changes and what must remain exact

## Actual sampling temperature and root ranking

Let e=softmax(z_exit/T_target) and q be the actual draft sampling distribution,
including T_draft and sampler_x when configured. The old root code deliberately
used T=1 even for T=.7 generation. Verification already used the actual T.
This round corrects cache ranking in unified B1/B>1, legacy tree, and chain
graph/eager paths. It does not replace q by a root score in verification.
At greedy T=0 there is no softmax(z/0): token generation/verification is greedy;
budget ranking retains the untempered soft scores as previously requested.
Legacy raw/top-M wire gates remain T=1-only and reject incompatible requests;
the new formulas require full vocabulary and reject those gates at config.

For an internal context in the proxy phase, score s(v)=e(v)(1-q(v)); leaves use s(v)=e(v).
P1 starts before e exists and retains its draft-based root scores.
Exclude already proposed children, then normalize by the sum over ALL allowed
tokens, not just the stored top-M. Top-M truncation is a transport/computation
limit; it must not give diffuse contexts an artificial larger position weight.

## Position estimator on chains and trees

For a chain a_i=min(1,e_i(y_i)/q_i(y_i)), and abar_i=sum_v min(e_i(v),q_i(v)).
Define h(a) by the first-rejection hazard plus the all-accepted bonus.
The frozen prior improvement is htilde=.75 h(a)+.25 h(abar).

A branching tree requires the ordered-WOR sibling ladder. At context u and
sibling s, R_us and D_us are the current proxy residual and proposal law after
earlier sibling rejection/exclusion. Compute a_us=min(1,R_us(y)/D_us(y)) and
abar_us=sum min(R_us,D_us). Propagate two separate reach/terminal recursions:

rho(child_s)=rho(u) [product_{j<s}(1-a_uj)] a_us;
tau(u)=rho(u) product_j(1-a_uj).

Use either a or abar throughout each recursion, then mix the two resulting
terminal masses .75/.25. Do NOT mix per-edge alphas instead. Each terminal
mass sums to one on a valid tree; the single-child case reduces to the chain
formula. The sibling distributions depend on the realized earlier tokens;
the overlap recursion is a regularized estimator, not the exact conditional
terminal law of the observed whole tree. Token scoring uses original parent
q for complement; residual ablation retains the final sibling residual.

An independent miss chain may be deeper than K1/K2. The terminal recursion
must cover max(K1,K2,miss_depth), otherwise deep miss contexts lose reach.
Likewise ragged chains use e at their real bonus context and do not subtract
the padded q row or exclude the arbitrary padded token. These are cache-score
fixes, not changes to target acceptance/recovery.

## How timing constrains parameters

At one common step origin, C1=a+D1 and C2=max(C1,P)+J+D2, where J is the
measured gap from both prerequisites being ready to P2's recorded start.
The shorter C2=max(C1,P)+D2 notation absorbs J into effective phase cost;
it must not silently treat a raw replay/phase span as the whole cost.
P1 fits when C1<=P;
P2 fits strictly in target forward when C2<=F_model, and before the next
verification boundary when C2<=F_ready. Measure both windows per step.
For SSD measure glue/build/decode/cache export and its completion relative
to F_ready. Miss JIT and target wait remain separate from future-cache build.

The cache built during step t is consulted by the next request. A negative
P2-before-ready slack is a missed conservative boundary, not an identity
for exposed wait in the same step: host work or prefill before step t+1 can
hide a tail. Final diagnostics also compare C2(t) with the actual request
start of t+1, restricting this pairing to the same unchanged full batch
without boundary/capture steps. The wait belongs to t+1. These observations
remain diagnostics; reported TPS uses the benchmark's measured step times.

Cross-process endpoints use the existing per-process CPU/CUDA anchor and
CUDA event elapsed times. They are not an independent hardware-wide clock
calibration. Small slack very close to zero should not be interpreted as a
proven hard deadline margin; event placement/anchor offsets and profiling
overhead remain. The next-request endpoint is the recorded target request
span start, not a separate measurement of network/IPC arrival. Same-process
durations and the final uninstrumented TPS are retained alongside the
cross-process timeline rather than inferred from it.

Neither target nor draft latency is a constant across K, B, root budget and
tree shape. Changing K can change node capacity and query buckets. A depth
increase increases the upper bound AL<=depth+1, but the realized gain is
sum_d P(accepted_depth>=d). Broadening roots targets cache hit; broadening
continuation targets accepted depth. Parameter decisions must cite the
affected terms, not only an aggregate TPS improvement.

Some CLI parameters also have derived effects. With an unspecified P2
budget, W=proxy_fan_out*(max(K1,K2)+1); the continuation-forward width
depends on draft_fan_out*(K1+1), capped by available roots. Thus K1-down
neighbors can reduce W/width as well as serial rounds even when explicit
node caps are fixed. Report these as setting interventions, not isolated
per-forward causal effects. Saved args and presets preserve the exact
dependencies; final tables resolve default W numerically.

Exit values in the saved CLI/config are zero-based indexes: exit21 means
the first22 layers have run. All figures retain the literal CLI index.
The screen's linear exit suggestion was only an approximate neighbor seed;
its selected integer neighbors were actually profiled. It is not an exact
layer-cost model. Likewise P1-replay/K1 is an effective average under the
current forest width, not an isolated single-token draft-forward latency.

P1 need not finish before proxy arrival for the full draft work to remain
hidden: C1>P is permissible if max(C1,P)+D2 still finishes before F. Earlier
proxy can wait for an unfinished P1; later proxy gives less P2 runway. This
is why checking only P1 or only exit-layer cache quality is insufficient.

Under dynamic/eagle global tree selection, tree_beta is not an active
parameter: each positive-score root gets capacity NV, and the global
frontier/gain selector distributes expansions. The alternative beta-based
root allocator belongs to other policies. Its measured beta0/beta1 runs
serve as identical-algorithm timing repetitions, not beta optimization.

The existing reach/gain/frontier tree estimator remains frozen. Its tables
come from older experiments, not newly fitted test outcomes. G=M throughout
the primary tuning to avoid post-sampling value-based pruning. Optimize
execution only after recording the current breakdown; compare identical
algorithms/output streams when testing a pure implementation change.

## Interpreting the proxy side stream

CUDA overlap is not itself a speedup guarantee. Proxy head/scoring and
target post layers share one GPU and may each become slower when concurrent.
Measure the new proxy arrival P', P1 completion C1', P2 cost D2' and target
completion F' rather than subtracting the old proxy duration from latency.
The same relation C2'=max(C1',P')+J'+D2' applies. If P1 is still later than both
proxy arrivals, a modest proxy delay may be hidden by P1. If proxy arrival
already determines P2 start, contention can delay the future cache even
while final target logits finish earlier. The two completion boundaries
and measured next-step waits must therefore be examined together. Full
uninstrumented TPS, with identical outputs, decides whether to enable the
option; no overlap-only theoretical speedup is claimed.
