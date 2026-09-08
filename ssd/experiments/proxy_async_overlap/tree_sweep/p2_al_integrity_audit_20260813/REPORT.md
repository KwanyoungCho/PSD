# P2 accepted-length integrity audit (2026-08-13)

## 결론

- 전체 Spec-Bench에서 `P2 AL < miss AL`로 보였던 주된 원인은 지표 정의 불일치다. 기존 `accept_len_on_miss`는 recovery token 1개를 포함하고, P1/P2 AL은 recovery를 제외한다.
- 같은 정의(accepted speculative tokens only)로 맞추면 세 seed 모두 P2가 miss보다 높다.
- summarization의 일부 긴 생성에서는 실제 P2 AL 붕괴가 존재한다. 그러나 같은 후반 구간에서 P1도 함께 붕괴하고 draft 확률도 급격히 퍼진다. P2 token/logit/cache wiring 오류는 관찰되지 않았다.
- 특히 `317_t0`의 후반에는 miss가 전혀 발생하지 않았다. 보고된 miss AL은 앞쪽 쉬운 context의 값이고, 낮은 P2 AL은 뒤쪽 어려운 context의 값이므로 직접 비교할 수 없다.

## 1. 지표 정의를 맞춘 전체 결과

아래 값은 prompt별 `n_verify_steps × phase_rate`로 phase step 수를 복원한 step-weighted 평균이다.

| Run | P1 AL | P2 AL | legacy miss AL (recovery 포함) | miss AL (spec-only) | P2 - miss |
|---|---:|---:|---:|---:|---:|
| tree seed 1 | 4.539 | 1.988 | 2.759 | 1.759 | +0.229 |
| tree seed 42 | 4.497 | 1.842 | 2.734 | 1.734 | +0.108 |
| tree seed 123 | 4.660 | 1.965 | 2.787 | 1.787 | +0.178 |
| matched chain seed 42 | 4.212 | 1.800 | 2.701 | 1.701 | +0.099 |

엔진은 miss에 `suffix_len`을 저장하지만 P1/P2에는 `suffix_len - 1`을 저장한다. 향후 runner는 기존 필드를 유지하면서 `accepted_spec_len_on_miss`와 recovery 포함 명시 필드를 함께 기록한다.

## 2. 왜 P2 hit와 P2 AL은 별개인가

P2 후보 점수는 대략 다음 사건을 근사한다.

`P(i,v) = Pr(proxy가 위치 i에서 reject를 예상) × residual_proxy_i(v)`

이 점수는 correction token `v`가 실제 recovery가 될 가능성을 위한 값이다. `v` 이후 target/draft 분포가 얼마나 잘 겹치는지는 포함하지 않는다. 따라서 P2 hit는 “recovery root를 맞혔다”는 뜻이고, P2 AL은 그 root **다음** draft token들이 수락된 길이다. correction root를 잘 맞히면서 그 직후 TinyLlama가 다시 틀리는 상태는 수학적으로 가능하다.

고정 horizon `K`에서 phase별 기대 AL은 다음과 같다.

`E[L | phase] = sum_{d=1..K} Pr(A_1 ∩ ... ∩ A_d | phase)`

P2 hit와 miss는 서로 다른 조건부 context 집합이다. 따라서 `P1 >= P2 >= miss` 같은 일반적인 순서는 없다. 또한 현재 P1 horizon은 8, P2/miss horizon은 4이므로 raw P1/P2 AL도 그대로 비교하면 안 된다.

## 3. 현재 코드 경로 검증

- target은 early-exit `p_E`, draft `p_D`, ratio acceptance 근사, residual `(p_E-p_D)+`, first-reject mass `h`, `P_iv=h×residual` 순으로 P2 후보를 만든다.
- draft selector는 P1 중복을 제거하고 score 순서를 유지한 채 정확한 budget만 선택한다.
- P2 continuation token/logit은 실제 TinyLlama forward로 생성된다.
- cache key/token/logit/valid-k는 동일 row namespace로 합쳐지고, P2 row는 K2 폭으로 표시된다.
- hit 시 matched cache row의 token/logit을 그대로 사용하며 phase를 P1/P2로 분류한다.
- tree verification은 각 node의 실제 parent q-logit을 `parent_q_ref`로 전달한다.
- miss는 actual recovery token을 받은 뒤 그 token에서 TinyLlama를 즉시 다시 실행하는 JIT-SD다. “P1/P2가 못 맞춘 문장을 그대로 쓰는 경로”가 아니다.

정확성 trace 결과:

| Check | Result |
|---|---:|
| P1 audited roots | 8,265 / 8,265 pass |
| P2 audited roots | 2,750 / 2,750 pass |
| internal node actually forwarded | all pass |
| node token has parent logit | all pass |
| target parent-q mapping exact | all pass |
| sampled token/q row exact | all pass |
| served topology == walked topology | 204 / 204 |
| selected P2 token ID OOB / zero | 0 / 710, 0 / 710 |
| special tokens among selected roots | 1 / 710 |
| selector budget mismatch | 0 / 71 |

GPU eager/captured replay, page boundary/page swap, sentinel KV write, sampler-q parity, GPU/CPU topology 및 verify mask 테스트 37개가 전부 통과했다. 관련 CPU 계약 테스트는 157개 통과, 8개 CUDA-only skip이다.

## 4. 긴 summarization의 실제 붕괴

Full seed 42에서 P2 hit가 20회 이상이고 P2 AL < 0.7인 prompt는 9개였으며 전부 summarization이었다. 모두 `prefill + completion > 2048`이었다.

동일 설정의 `317_t0` 진단에서는 prompt 1,734 token에서 시작해 총 2,315 token까지 생성됐다.

| Context 구간 | P1 steps / AL | P2 steps / AL | P1 attempted-q median | P2 attempted-q median |
|---|---:|---:|---:|---:|
| `< 2048` | 40 / 5.600 | 13 / 2.769 | 1.000 | 0.865 |
| `2048–2239` | 25 / 5.520 | 7 / 2.286 | 1.000 | 0.828 |
| `>= 2240` | 15 / 0.533 | 40 / 0.325 | 0.149 | 0.146 |

2240 이후:

- P2 hit 40회 중 34회가 `k_idx=0`, 즉 첫 draft token reject의 correction root였다.
- 40회 중 33회는 root 다음 token을 하나도 수락하지 못했다.
- P1도 15회 중 9회가 accepted length 0이었다.
- target recovery token을 decode하면 `murder trial begins`, `life in prison`, `illegal possession of firearms and ammunition`처럼 정상 문장 조각이었다.
- P2 node token도 정상 Llama subword였고, 550개 long-context P2 root의 node/q mapping 검사가 전부 통과했다.
- 이 요청의 cache miss 4회는 context 1,735–1,885에서만 발생했다. 2,048 이후 miss 표본은 0개다.

따라서 이 현상은 P2가 이상한 token을 생성한 것이 아니라, TinyLlama가 긴/퇴화된 continuation에서 target과 계속 불일치하고 proxy가 매번 그 correction token을 잘 맞혀 P2 hit로 분류되는 correction loop다. analytic RoPE cache 확장은 crash를 막지만 TinyLlama의 native 2,048-token 학습 분포 밖 품질을 보장하지 않는다.

## 5. 남은 알고리즘 이슈

- P2 후보의 `p_E/p_D` 계산은 현재 plain softmax이고 실제 verification은 temperature 0.7을 적용한다. 또한 `p_E`는 final target이 아니라 early-exit proxy다. 따라서 `P(i,v)`는 정확한 확률이 아니라 ranking score/근사치로 표현하는 것이 맞다.
- 현재 P2 score는 root hit probability를 목표로 하며 root 이후 target-draft overlap을 직접 최적화하지 않는다. P2 AL을 높이려면 future-overlap 또는 draft-confidence penalty를 추가하는 별도 알고리즘 실험이 필요하다.
- 공정한 후속 분석은 context-length bin별로 `P1@K2`, P2, miss survival curve를 기록해야 한다. 특히 같은 context 구간에 miss가 없으면 P2-vs-miss 결론을 내리지 않아야 한다.

## Artifacts

- 7-prompt current-code correctness trace: `../p2_al_integrity_audit_20260813/`
- two-long-prompt trace: `../p2_al_long_context_audit_20260813/`
- context-position trace with `num_tokens`: `../p2_al_boundary_pair_trace_20260813/`
- full three-seed inputs: `../p1_p2_tree_full_rerank_3seed_20260812/`
- matched chain input: `../p1_p2_tree_matched_chain_seed42_20260813/`
