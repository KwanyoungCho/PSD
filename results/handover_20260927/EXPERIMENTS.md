# 작업·실험 이력과 해석

인계 기준일: 2026-09-27. 날짜는 원래 실험 기록 기준이다. 이 문서 작성일에 모든
실험을 다시 수행했다는 뜻이 아니다. 원본 수치/설정은 각 디렉터리의 REPORT,
JSON/CSV, `runs/*/command.json`을 따른다. 아래 CI는 각 원본의 통계량에 대한
구간이므로 서로 다른 집계 방식의 수치를 직접 뺄셈하지 않는다.

## 공통 목표, 환경, 비교의 단위

- Root 후보 선정: 실제 correction token이 미리 만든 cache에 있는 확률을 높인다.
- Tree 구성: cache hit 후 이미 선택된 root를 제외한 accepted descendants의 AL을
  높인다. TPS는 별도로 기록한다. 많은 노드를 검증해 TPS가 낮아져도 AL 결과는
  그대로 보고하되 예산이 다른 비교를 같은 알고리즘 효과로 설명하지 않는다.
- 추가 neural head 학습은 하지 않는다. 보정표/스칼라를 calibration에서 추정한
  방법은 “추가 모델 학습 없음”이지 “데이터로 정한 값이 전혀 없음”은 아니다.
- 역사적 환경: eslab19, RTX 4090 24GB × 8, target AWQ 70B TP4 + draft
  TinyLlama-1.1B BF16 TP1. 주 tree 캠페인은 physical GPU 3–7을 사용했다.
  Python 3.12.7, torch 2.8.0, transformers 4.57.1, triton 3.4.0;
  개별 run의 환경 snapshot이 우선이다.
- `proxy-only`는 여기서 주로 후보 점수 `e`라는 의미다. 동일 DUET 안에서의
  점수 ablation이며, Mirror-SD 전체 구현과 동일하다는 뜻이 아니다.
- 과거에 이미 있던 P1/P2 tree executor, backbone, only-proxy 모드, latency
  profiler 등과 이번 브랜치에서 새로 만든 연구 도구를 구분한다. 기반 commit은
  `a82f7d2`; 이번 변경 목록은 [CODE_MAP](CODE_MAP.md)에 있다.

## 1. 초기 후보 정책과 early-exit 분포 측정 — 9/9–10

자료: [초기 보고서](../residial_dist/REPORT.md),
[초기 인계](../residial_dist/HANDOVER.md),
[entropy](../residial_dist/entropy/REPORT.md),
`ssd/experiments/proxy_source_ablation/`.

한 context의 draft `q`, target 최종 `p`, 중간 layer proxy `e`를 비교할 수 있도록
chain 후보 source `residual|proxy|draft`와 진단 probe를 추가했다. 전 layer의
hidden/residual을 graph capture 전에 확보하고 같은 최종 norm/head로 확률을
계산했다. Rank0만 실행하는 probe에서 TP collective head를 호출하면 멈추므로
복제된 full head를 사용하도록 했다. 과거 NCCL timeout도 기록에 남아 있다.

초기 coverage 실험 자체는 4 datasets × 3seed = 12 runs, **34,150 steps**였다.
Humaneval/alpaca/c4/gsm 각각32 prompts, cap256, T1, B1, only-proxy chain,
exit56/요청 K1,K2=8,4/실제 verify K4/roots15 조건이다. 참 `h·R`로 위치-token
쌍을 선택하면 coverage .8208로 같은 예산의 oracle 상한 .8210에 근접했고,
참 target만의 순위는 .7928이었다. Proxy를 넣은 실용 비교에서는 residual 쪽이
약 .038 낮아 방향이 반대였다. 이론적 최적 목적과 근사 입력의 오차를 구분한
첫 실험이다. 이후의 full-distribution/actual-wire 분석이 원인과 수치를 더 엄밀히
확인했으며, 이 첫 probe의 제한된 top64 정규화 차이도 기록돼 있다.

별도로 시행한 Entropy 실험은 12 runs, 128개 서로 다른 prompt × 3seed = 384 생성,
34,702 verify steps / 173,510 contexts를 사용했다. B=1, T=1, PS-only chain,
exit index 56, 요청 K1/K2=8/4, 실제 verify K=4, 최대 256 생성 token,
80개 layer를 측정했다. 다음은 layer56의 `H(e)-H(p)`다.

| 통계 | 결과 |
|---|---:|
| Context 평균 | +0.1589 nats |
| 차이 절댓값의 중앙값 | 0.3042 nats |
| Proxy entropy가 더 높은 비율 | 52.6% |
| 더 낮은 비율 | 47.4% |
| Prompt/dataset 균등 평균 | +0.1512, CI [+0.1236,+0.1783] |

따라서 “proxy는 항상 평평하다”는 설명은 틀리다. 평균 entropy 차이만으로 residual
후보 순위가 나빠지는 원인도 확정할 수 없다. 동일 head 통제에서도 비슷한 결론이었다.
초기 coverage probe의 T=1 관례와 이후 실제 request temperature를 사용하는
entropy/distribution probe를 구분해야 한다.

## 2. 공유 분석 검토, residual 오차와 정확성 역할 분리 — 9/10–11

자료: [REPORT](../residial_dist/shared_review/REPORT.md),
[FOLLOWUP](../residial_dist/shared_review/FOLLOWUP.md).

4 datasets × 2seed, 256 생성, 22,990 steps를 측정하고 그중 2,995 sampled
steps / 14,975 contexts의 full `p/q/e`를 저장했다. 단순 평탄화 가정,
temperature 보정, residual support 손실, candidate/cache/verification 구현을
점검했다. 이 split의 held-out coverage는 residual 0.6056, proxy 0.6426,
temperature τ=0.9 residual 0.6103이었다. KL을 최소화한 τ=1.055193은 KL
0.4637→0.4556을 개선했지만 coverage는 0.5965로 떨어졌다. Rejection 가중
residual support 누락은 22.04%였다. KL/entropy 개선과 cache 목적의 개선은 다르다.

중요한 역할 구분:

1. `e` 기반 분포는 correction 후보의 cache 계산 위치를 고르는 데 사용한다.
2. 실제 correction token은 target와 draft의 **참 residual**에서 뽑는다.
3. 그 token이 cache root와 일치하면 해당 root는 이미 선택된 token이다.
4. 그 뒤 draft가 생성한 descendants를 그들의 실제 draft proposal로 검증한다.

따라서 proxy score를 descendant acceptance 분모에 넣지 않는다. 이 설명은
지원되는 구현 경로와 올바른 proposal 보존에 대한 것이며, 아래에서 찾은 G>M
사후 pruning 편향까지 없다고 보장하지 않는다. CPU sampler 점검(기록별 600k,
300k draws)은 별도 검사이며 전체 분산 GPU engine의 보편적 증명은 아니다.

## 3. Residual 대 proxy 직접 수학 비교 — 9/12

자료: [THEORY](../residial_dist/direct_comparison/THEORY.md),
[REPORT](../residial_dist/direct_comparison/REPORT.md),
[TABLES](../residial_dist/direct_comparison/TABLES.md).

`a=[p-q]+`, `b=[e-q]+`, `Z=Σa`, `Ẑ=Σb`, `δ=TV(p,e)`로 놓고 Z,Ẑ>0일 때

\[
R=a/Z,\quad \widehat R=b/\widehat Z,\qquad
TV(R,\widehat R)\leq\min\left(1,\frac{\delta}{\max(Z,\widehat Z)}\right)
\]

를 유도했다. 양의 부분을 취하기 전후의 한 방향 오차를 비교하고 정규화의 영향을
분리한 bound다. Proxy 오차가 작고 residual mass가 크면 bound가 작지만,
이것만으로 `TV(R,Rhat)<TV(R,e)`나 top-k coverage 우위를 보장하지 않는다.
실제 token 순위의 margin, support 삭제, 위치별 거절 확률도 필요하다.

새 128 prompts, exit56/T1 직접 검증에서는 global coverage가 residual 61.297%,
proxy 65.618%, 차이 −4.321pp였다. 별도의 rejection-weighted top3 분해에서는
oracle headroom 13.016pp − 삭제 손실 10.665pp − 순위 손실 7.360pp = −5.009pp였다.
두 수치는 목적/집계가 다르므로 동일 delta로 읽지 않는다. 25개 점수/selector도
비교했다. 모든 early-exit, 모든 방법에서 proxy를 이길 수 없다는 정리는 아니다.

## 4. 추가 모델 학습 없이 후보를 개선 — 9/12

자료: [REPORT](../residial_dist/training_free/REPORT.md),
[THEORY](../residial_dist/training_free/THEORY.md),
[HANDOVER](../residial_dist/training_free/HANDOVER.md),
[LIVE_CLEAN](../residial_dist/training_free/LIVE_CLEAN.md).

55개 source/ranking × 2개 normalization × 24개 hazard 변형 = 2,640 조합을
development 512 생성/5,796 sampled steps에서 비교했다. 독립 96 prompts
(기존 파일의 rows65..88, hash로 중복 검사) × T={1,.7,.5} = 288 생성,
3,112 sampled steps에서 확인했다.

Token score는 `[e-λq]+`, `e(1-q)^β`, residual floor 등을 비교했다.
거절 위치 배분은 `αbar=Σmin(e,q)`에서 얻은 예상 hazard와 기존 hazard를
`h̃=(1-ω)hhat+ωhbar`로 섞었다. Top-M 안에서만 정규화하는 대신 허용된
full-vocab mass로 정규화하는 방법도 분리했다. 실제 wire의 top17→15 절차를
재현한 최신 표는 다음과 같다(ω=.25, full normalization).

| T | 기존 proxy | 배분/정규화 개선 proxy | 같은 배분의 수정 score | 수정 score의 추가 차이 |
|---|---:|---:|---:|---|
| 1 | .65126 | .66801 | .66811 | +.011pp, CI [−.217,+.162]pp |
| .7 | .82020 | .84363 | .84431 | +.068pp, CI [−.110,+.274]pp |
| .5 | .89005 | .91335 | .91321 | −.014pp, CI [−.112,+.068]pp |

각 온도의 수정 score는 각각 `e sqrt(1-q)`, `e(1-q)`,
`max([e-q]+,.75e)`다. 가장 일관된 개선은 **후보 예산을 위치에 배분하는 방식**이다.
토큰 점수만으로 동일 배분의 proxy보다 좋다는 결론은 확정하지 못했다.
초기의 T1 +.125pp는 실제 wire 재현 후 +.011pp로 정정됐다. BF16 동점 때문에
한 사례에서 coverage가 .982만큼 바뀐 예도 있어 CPU 순위만으로 결론내리면 안 된다.

`[e-q]+ ≤ e(1-q) ≤ e`와
`e(1-q)=[e-q]+ + min(e,q)-eq`는 정확한 관계다. 과도한 subtraction을 완화하는
근거는 있지만 독립한 두 draw의 불일치 해석은 SD rejection 분포와 다르다.
심지어 e=p에서도 일반적으로 true residual은 아니다. 최적 residual 추정식이라는
정리 대신 목적/한계를 밝힌 robustness ablation으로 제시해야 한다.

별도 16 prompts × 2seed 무계측 live 실험은 seed별 TPS +4.72% / −7.81%,
합산 −1.61%, hit +.55pp였다. 후보식 수정으로 속도가 좋아졌다고 주장할 근거는 없다.

## 5. DUET 파라미터 calibration 새 prototype — 9/13

자료: [REPORT](../duet_calibration/REPORT.md),
[THEORY](../duet_calibration/THEORY.md),
[CONFIRMATION](../duet_calibration/CONFIRMATION.md),
[HANDOVER](../duet_calibration/HANDOVER.md).

기존 저장소 추천을 그대로 사용하지 않고, exit 전후 사용 가능한 draft 시간과
verification 비용을 모델링하고 소수 anchor로 후보를 줄이는 prototype을 만들었다.
전체 54개 성공 inference, campaign 내 서로 다른 56 prompts를 사용했다.

- 5개 anchor로 freeze 후 19개 grid를 validation으로 쓴 첫 모델:
  cycle MAPE 1.58%, reward MAPE 17.90%. 첫 추천 exit40/K1,K2=4,2는 측정 best
  exit56 대비 regret 4.65%. 시간 예측이 좋아도 reward 예측이 충분하지 않았다.
- 추가 6점(exit40/56/72 × 4/2,4/1)으로 좁힌 뒤 exit56/4,2를 고정했다.
- 독립 16 prompts × 2seed × 256 tokens, profiler off:
  exit56/4,2/roots15는 TPS 92.154, 기존 exit56/8,4/roots15는 64.811,
  +42.19%, CI [+36.60,+47.52]%. exit40/4,2는 90.244,
  exit72/4,2는 88.226이었다.
- Tree 비용의 단순 보간은 MAPE 67.89%로 실패했다. Nv8의 .69/1.50초 outlier는
  반복 시 중앙값 약 28ms로 재현되지 않았다. 사후 median refit을 독립 검증된
  탐색 성공으로 계산하지 않는다.
- 후보 계수 보조 분석은 8 calibration prompts/47 snapshots, 16 validation/104,
  7 layers, 1,050 policies였다. exit56/T.7/roots15에서 원래 proxy84.13%,
  개선 proxy87.75%, e(1-q)88.57%; score 추가 +.828pp의 CI [−.221,+2.658]pp.
  P1 dedup 전 offline replay이고, 최종 TPS confirmation에 적용한 정책이 아니다.
  layer72에서는 선택한 정책이 −.964pp였다.

결론: 제한된 chain/hardware에서는 전수 sweep보다 적은 측정으로 좋은 설정을
좁힐 가능성이 있다. 새로운 tree/hardware/precision의 전역 최적 파라미터를
보장하지 않는다. 시간 모델 + 실제 shortlist 비교 + 독립 confirmation이 필요하다.

## 6. 논문과 tree 구현 감사 및 reach 점수 — 9/21

자료: [REPORT](../duet_tree_analysis/REPORT.md),
[THEORY](../duet_tree_analysis/THEORY.md),
[`TREE_IMPLEMENTATION`](../../ssd/docs/duet/TREE_IMPLEMENTATION.md).

논문의 root prior × q 경로 곱은 코드에 있으나 frontier, quota, threshold,
G→M rerank는 논문 식만으로 설명되지 않았다. 같은 부모의 WOR 형제 j는 현재
target residual Rj와 실제 조건부 proposal Dj를 사용해야 한다.

\[
\alpha_j=\min(1,R_j(t_j)/D_j(t_j)),\qquad
\rho_j=\rho_{parent}\prod_{k<j}(1-\alpha_k)\alpha_j,\qquad
\mathbb E[AL]=\sum_j\rho_j.
\]

여기서 tree root는 이미 cache hit로 선택됐으므로 descendant AL 합에서 제외한다.
q 경로 곱은 실제 acceptance/reach와 다르다. 이미 관측된 tree의 exact ladder,
ancestor/sibling-prefix subset DP, 실제 production 함수의 완전열거 검사를 만들었다.
Oracle subset DP는 사후 진단이며 그대로 sampling 정책에 넣으면 안 된다.

24 prompts의 238 full-p/q trees를 8 calibration(77 trees) /16 validation(161)로
나눴다. `phase_sibling_q_bin` 보정 reach의 MSE는 19.27% 감소했다. 새 neural
head가 아니라 phase/sibling/q-bin 스칼라 표다. 16 prompts × 2seed 무관측 생성에서
AL 1.905→2.092(+9.82%), CI [−.03824,+.40389]; TPS +1.91%, CI [−4.02,+8.01]%.
관측을 켠 별도 실행은 AL 2.247→2.023이었다. 관측 유무를 혼합하면 안 되며,
작은 live 결과는 아래 full corpus 결과가 우선한다.

실제 production rerank/verifier 함수의 18가지 작은 tree 완전열거에서 G>M
score-dependent pruning의 TV=.01875를 재현했다. Closure만으로는 proposal이
보존되지 않는다. 모든 신규 정책 비교는 **G=M**으로 고정했다.

## 7. 기존 두 점수 full Spec-Bench 생성 비교 — 9/21–22, 완료

자료: [REPORT](../duet_tree_al_full/REPORT.md),
[OBJECTIVE](../duet_tree_al_full/OBJECTIVE.md),
`analysis.json`, `audit.json`, `task_results.csv`, `runs/*/records.jsonl`.

480문항(6 tasks 각80), MT의 두 번째 turn을 포함한 560턴 전부,
`q_path`/`phase_sibling_q_bin` × seeds1,42,123 = **3,360턴**을 완료했다.
원래 base-model raw continuation, 실제 이전 turn 출력 사용, natural EOS,
cap128, draft native context2048; 입력 절단/누락 없음. 논문의 cap1024/chat 재현은 아니다.

고정 조건: exit index56, K1/K2=4/2, C=3, P1 G=M=8 / P2 G=M=6,
legacy residual root 정책, P2 roots15, P1 위치당 roots3. 실제 P1 root bucket은
9/21/27이고 width는 `(9,9,9,9)/(21,15,15,15)/(27,15,15,15)`였다.
P1 threshold 0/0, P2 root .01/raw-q .03. 총138,357 verify events,
113,783 raw cache hits를 기록했다.

| 결과 | q_path | 보정 reach | 해석 |
|---|---:|---:|---|
| 주지표: task 균등, nonfinal hit descendant AL | 2.0651327 | 2.1003084 | +1.703%; delta .0351757, CI [−.0024888,+.0720564] |
| Hit rate | 82.4107% | 81.7271% | −.6835pp; root 식 고정이어도 trajectory는 달라짐 |
| P1 AL delta | — | +.0464 | 이 secondary 구간은 양수 |
| P2 AL delta | — | +.01317 | CI에 0 포함 |

Seed별 delta는 +.0333,+.0685,+.0031. 주지표는 마지막 verify event의 EOS/cap
censoring을 제외하고 `accepted_spec_len`의 descendants를 센다. 실제 emitted AL,
all-event 지표는 별도다. 특정 phase의 결과만으로 전체 우위를 주장하지 않는다.
62개 파일의 frozen manifest를 보존한다.

## 8. Full tree 사후 진단 — 9/22, 완료

자료: [FINDINGS](../duet_tree_posthoc/FINDINGS.md),
[REPORT](../duet_tree_posthoc/REPORT.md), [THEORY](../duet_tree_posthoc/THEORY.md).

새로 두 정책 × full480/560 × seed1에 관측기를 켜서 37,869 hit trees,
262,145 nodes(36,896 nonfinal trees)를 수집했다. 매32개 수준의 full분포 저장으로
1,184 raw snapshots, hit가 있는 question 합집합476을 확보했다. 이 trace의 timing은
성능 비교에 사용하지 않는다. Frozen 결과 manifest는 1,273파일이다.

- q MSE .07967258 → frozen reach .06769848, −15.03%.
- 보정표의 첫 형제 α는 모든 bin에서 .5 초과(P1최소 .61738, P2 .50853).
  따라서 이 점수는 같은 부모에서 첫 형제를 항상 우선한다. 실제 oracle은 해당
  비교의 14.2249%에서 다른 순서를 선호했다.
- 풍부한 metadata의 OOF 모델은 MSE를 줄여도 고정 pool B=4의 AL 추가 효과는
  약 .00534(+.25%)였다. 기존8개 prompt만으로 feature를 늘린 점수는 개선되지 않았다.
- 관측 acceptance와 exact expectation 차이의 MSE 약 .336은 대부분 조건부
  sampling coin variance였다. 이미 확률이 주어진 뒤의 무작위 결과까지 예측해야
  한다고 해석하지 않는다.
- 새 자식을 뽑기 전의 `g1=Σmin(p,q)` 및 WOR 재귀의 `g2`를 구했다.
  두 번째 marginal이 더 큰 비오목 사례가 contexts6.13%(가중4.60%)였다.
- 원래8개/77개 calibration trees만으로 추정한 γ1/γ2는
  P1 .841473/.912016, P2 .704568/.782921.
  두 부모·새 노드2개, 1,282 pairs/369 questions에서 frozen reach×γ 배분은
  국소 expected gain .5148469→.5300555(+2.954%), delta CI [.00691877,.02394132].

이것은 가능한 다음 한 round 배분의 결과다. 실제 C=3/multi-round 정책이나
end-to-end AL +2.95%로 외삽하지 않는다. 관측하지 않은 branch에는 p/q가 없어
그 branch의 효과를 기존 hit trace만으로 복원할 수 없다.

## 9. C=3 배분·frontier 개선과 남은 검증 — 9/22–23

자료: [REPORT](../duet_tree_followup/REPORT.md),
[THEORY](../duet_tree_followup/THEORY.md),
[PLAN](../duet_tree_followup/PLAN.md),
[최신 실행 상태](../duet_tree_followup/STATUS_20260927.md).

| 정책 | Reach priority | Fanout | Frontier |
|---|---|---|---|
| q_path | q 경로 곱 | 기존 breadth | depth == round |
| reach | frozen 보정 | 기존 breadth | depth == round |
| q_gain | q 경로 곱 | 사전 gain 최적화 | depth == round |
| reach_gain | frozen 보정 | 사전 gain 최적화 | depth == round |
| reach_frontier | frozen 보정 | 기존 breadth | 미확장 depth <= round |
| reach_gain_frontier | frozen 보정 | 사전 gain 최적화 | 미확장 depth <= round |

Root 정책, threshold, per-root 미래 round reserve, round width, G=M 예산은
고정한다. 새 token을 보기 전에 `max Σparent ρhat_parent γ(c_parent)`를 푼다.
C=3/B≤8의 단조 공통 γ에서 정수 partition으로 해당 one-round 목적을 정확하게
최적화한다. Full-horizon/global AL optimality 정리는 아니다. c=0이어도 기존
selector처럼 해당 queried parent를 expanded로 표시하는 동작을 유지한다.
현재 P2는 두 round이며 첫 round에서 활성 root를 모두 평가하므로, frontier 변경
자체가 P2 내부에 재고할 이전-depth node를 추가하지 않는다. P2 관측 통계가
변한다면 앞선 trajectory/cache 변화와 구분해 해석한다.

γ3는 C=2를 외삽하지 않고 첫 거절에 조건부인 재귀로 구했다. 작은 vocab은 exact,
큰 vocab은 head32 exact/tail512 seeded Monte Carlo. 원래 calibration8/77 trees의
230 descendant contexts에서 P1 γ3=.93601196, P2=.83081316을 얻었다.
수치 SE가 작더라도 context/calibration 표본 불확실성은 남는다.

완료한 검사:

- CPU γ 완전열거 484 cases, 최대오차 1.31e−13; 배분 600 cases, 2.22e−16;
  두 token 유한분포 100 cases, 2.22e−16. 잘못된 사후선택 negative control TV=.01875.
- 실제 selector CPU 1,000 cases, mixed-depth mask 3 cases.
- GPU allocation CPU/eager/capture 200 cases.
- 작은 모델의 실제 executor/attention, 6정책 × 8조건, 72 node/q-reference audits.
  이를 full 70B 검증이라고 부르지 않는다.

기존 raw에서 expanded descendant3,438 contexts/460 questions를 확보했다.
602 observed round groups/369 questions, 실제 현재 B에서:

| 국소 expected gain | 값 | 비교 |
|---|---:|---|
| frozen breadth | .6851094 | 기준 |
| frozen gain | .6965150 | +1.66%, delta .0114056, CI [−.0020208,+.0245624] |
| q gain | .69823345 | 기준 대비 delta .013124, CI [.001269,.024967], exploratory |
| Oracle | .7458523 | 배분 진단용 상한 |
| True reach + calibrated gain | .7424098 | reach 오차와 gain 오차 분리 |

q gain − frozen gain은 +.001718, CI [−.007816,+.011075]로 승자 미확정이다.
옛 two-parent/B2 결과를 B3에 적용하면 delta .000842, CI에0을 포함한다.
현재 B 비교에서 배분 변경59.17%, 승50.08%, 패8.71%; 양의 기여+.029233과
음의 기여−.017827가 상쇄됐다. γ3 수치불확실성의 delta bounds는
[+.011276,+.011945], calibration prompt LOO의 delta 범위는 [+.009005,+.012281].
이것들이 question bootstrap의 CI를 대체하지 않는다.

Draft-only shadow를 24 trees/91 known parents에서 시도했다. HF q의 평균 TV
.008356/p95 .029608, production model+SDPA의 평균 .007805/p95 .030744로
사전 gate(median<.005, p95<.02)를 통과하지 못했다. 효과 근거에서 제외했다.
BF16 때문이라고 단정하지 말고 prefix, KV, mask, RoPE, backend를 진단해야 한다.
동일 engine/KV 다중 round causal replay는 아직 구현하지 않았고 queue에도 없다.

원래 계획은 6정책 × 3seed × 560턴 = **10,080턴** 및 별도 두 full trace였다.
9/23 첫 smoke가 모델 로딩 OOM으로 실패했으므로 online 결과는 없다.
9/27에는 이 상태를 확인하고 STOP 및 인계 문서를 추가했다. 실패의 자세한 로그는
[상태 문서](../duet_tree_followup/STATUS_20260927.md)에 연결돼 있다.

## 아직 끝나지 않은 결론과 사용자 설명

다음 항목은 논문 주장으로 확정하지 않는다: 수정 token score의 matched-proxy 우위,
Mirror-SD 전체 대비 우위, 새 tree gain/frontier의 online AL 우위, dense target
분포 보존/성능, tree 전역 최적 calibration, causal shadow의 재현 성공.

사용자와 다시 설명할 때는 (1) 참 residual과 근사 오차, (2) proxy-only와의 직접
목적 비교, (3) token score와 위치 배분을 분리한 결과, (4) cache root와 descendant
verification, (5) q-product와 reach, (6) WOR 형제/AL 합, (7) 사전 gain 배분,
(8) 통계적으로 확정된 것과 남은 실험 순서로 진행한다.
[`TODO.md`](../residial_dist/training_free/TODO.md)의 이해 확인은 그대로 미완료다.
