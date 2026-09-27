# Training-free candidate improvement campaign

2026-09-12. No new head, regression gate, or model training. Changes affect
prefetch candidate selection only, never target sampling or draft verification.
Baseline is the proxy-score ablation in DUET; this is not a full Mirror-SD
system benchmark.

Development includes ALL previously inspected prompts (source rows 1–64 per
dataset), all previous T=1 snapshots, and newly generated T=0.7/0.5 snapshots
on 16 of these prompts per dataset. New independent confirmation uses 24
previously unseen prompts per dataset (source rows 65–88 unless duplicates
require skipping), at actual generation T=1/0.7/0.5. Interleave datasets in one
input file to amortize model loading; preserve dataset/prompt provenance.
Output limit 256, B=1 chain K=4, exit=56, top-M=17, global root budget 15.
Store pT, pD, pE56 in float32 at per-prompt stride 8.

Before confirmation policy outcomes are inspected, freeze one winner per
temperature for each family and the best overall policy, chosen using
development prompt-balanced coverage. Include optimized proxy-only allocation
as an equal-treatment control. Report all tried policies; use paired,
dataset-stratified prompt bootstrap and mark multiple-comparison exploration.
Never select a new winner using confirmation data and call it confirmed.

Prespecified families: (1) separate source and position-weight changes;
position weight powers, uniform hedges, bonus boosts, expected-rejection
hazards, and hybrids; (2) full-vocabulary vs top-M normalization; (3) bounded
draft discounts, positive-score floors, rank/ratio-gated subtraction;
(4) conservative interval-certified exchanges protecting proxy-head tokens;
(5) local agreement-gated residuals; (6) correction/bonus separation and
residual-tail diversification. Scalars are experimental hyperparameters,
selected on development, not learned model parameters.

Development-only extension before policy freeze: add two-sided residual
scores, capped negative-part rescue, complement-of-draft weighting, and a
square-root uncertainty floor. Preserve initial replay/results as v1. These
test whether sign errors can be handled without simply choosing a global
subtraction coefficient. No confirmation policy outcomes were inspected.

Implementation finding: production chain candidate logits currently omit
sampling-temperature scaling. New replay uses the correct e_T/q_T and also
reports the reconstructed legacy-temperature candidate control. Generation
sampling and the recorded p/q/e use the requested T; policy changes do not
alter the committed target distribution. This distinction must remain visible.

Mechanism: exact candidate-exchange identity Delta=(Gamma-E)/Z, measuring
positive and negative signed errors on included/excluded candidates, rather
than inferring performance from scalar proxy TV or entropy alone. Distinguish
source ranking gains from reallocation and from oracle information.

Runtime additions: four frozen/control arms at T=0.7, 16 prompts, seeds
801/802, forward/reverse order. Add the canonical unscaled residual as a
secondary control; no new rule tuning. Because parts of this timing screen
overlapped GPU2 replay/check work, mark it exploratory. Repeat the main
matched-temperature proxy-source allocation contrast on seeds 903/904 with
GPU2 idle and no concurrent analysis. This is a limited timing replication,
not another unseen-prompt claim. Use emitted output tokens / generate wall
time as the primary timing metric (engine decode counts can include tokens
discarded at the output-length boundary).

Numerical robustness amendment: keep the original frozen topk(15) replay.
Evaluate the same frozen rules with the actual topk(17) wire prefix as well.
BF16 logit ties can change token identity without changing predicted score.
Do not retain a small source-score win unless it survives this check; do not
use target probabilities to resolve ties in a deployable policy.

Primary outcome: global 15-root expected correction/bonus coverage. Secondary:
fixed-position top-3 coverage, true-h control, local switch oracle, candidate
pool oracle, support/clipping and exchange-error diagnostics. Any deployment
claim requires runtime integration/parity and timing. New best policies must
be tested against both original proxy baseline and equally optimized proxy
allocation before attributing gains to residual information.
