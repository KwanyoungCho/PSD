## 2026-09-27 서버 인계 — 현재 상태를 먼저 읽을 것

전체 브랜치의 작업/실험/미완료 검증과 full-model 재개 절차를
[루트 HANDOVER](../HANDOVER.md)에 정리했다.
[실험 이력](handover_20260927/EXPERIMENTS.md),
[코드 지도](handover_20260927/CODE_MAP.md),
[새 서버 검증 순서](handover_20260927/FULL_MODEL_VALIDATION.md),
[원시 데이터 이관](handover_20260927/ARTIFACTS.md)을 함께 본다.

아래 9/22의 “GPU 대기열 가동”은 과거 상태다. **9/23 첫 70B smoke가 AWQ 모델
로딩 중 다른 프로세스와 GPU 메모리를 공유한 상태에서 OOM으로 끝났고 queue는
stopped다. 새 6정책×3seed full480과 두 frontier trace는 미실행이다.**
[최신 상태/실패 원인](duet_tree_followup/STATUS_20260927.md)에 기록하고 STOP을
추가했다. 기존 두 정책의 3,360턴 full AL 평가와 사후 분석은 완료 상태로 보존했다.
인계 기록과 사용자의 수식 이해 확인은 별도이며 TODO는 여전히 미완료다.

## 2026-09-22 추가: C=3 fanout 검증 및 GPU 대기 캠페인

- [통합 보고서](duet_tree_followup/REPORT.md), [수식](duet_tree_followup/THEORY.md), [대기열 상태](duet_tree_followup/queue_status.json).
- 기존 raw에서 expanded descendant 3,438문맥/460질문을 검증. 배분 비교 602라운드/369질문에서 보정 reach+gain의 국소 기대 AL +1.66%, 차이 CI [-0.002021,+0.024562]; online 개선 확정 아님.
- C=3 완전열거/배분 최적성/유한분포 검증 및 6정책 GPU executor·q-reference 검사는 통과. 실제 70B full inference는 아직 미실행.
- GPU 5개가 비면 6정책×3seed×full480(10,080turn)과 별도 2정책 full frontier trace를 실행하는 대기열 가동(PID 3274225, `duet_tree_followup/queue_status.json`). 기존 root 정책·G=M·8/6 node·4/2 depth 유지.
- Draft만 재계산한 shadow는 known-parent q 재현 gate 실패: 효과 근거에서 제외. 동일 engine/KV 상태의 다중 라운드 causal replay는 별도 미해결 항목.
- 사용자에게 전체 수식을 설명하는 TODO는 이해 확인 전까지 계속 미완료.

# DUET 연구 진행 기록 및 재개 안내

**9/22 추가 — tree 사후 분석 완료:** [결론과 다음 개선 방향](duet_tree_posthoc/FINDINGS.md),
[전체 수치](duet_tree_posthoc/REPORT.md), [수식](duet_tree_posthoc/THEORY.md).
Full480문항/560턴×두 정책을 새로 실행해 37,869개 tree/262,145개 node를 분석했다.
Frozen reach는 q-path보다 MSE를 15.03% 줄였다. 그러나 첫 형제의 예측 alpha가
모든 bin에서 .5를 넘어 같은 부모의 첫 형제를 항상 우선한다는 한계를 증명했다.
관측된 해당 비교의 14.22%에서는 target을 아는 oracle이 뒤 형제를 선호했다.
더 큰 자료로 확률을 보정해도 위치 선택 이득은 작았고, 기존8문항에서 feature만
늘린 개선은 확인하지 못했다. 기존8문항의 gain curve×frozen reach로 fanout을 배분하면
두 부모·두 새 node의 국소 expected gain은 .51485→.53006(+2.95%), 차이 CI
[+.00692,+.02394]였다. **새 정책의 end-to-end AL +2.95%라는 뜻은 아니다.**
실제 C=3/다중 round 배분 구현과 online AL 검증은 다음 과제다. Root 정책은 유지했고,
새 neural head/production selector 교체는 하지 않았다. 기존 무관측3-seed 자료도 보존했다.

**최신 사용자 지시 — AL 우선, full dataset:** root 후보 선정은 cache hit,
tree draft 구성은 AL 향상을 목표로 분리한다.
[Full Spec-Bench AL 평가](duet_tree_al_full/REPORT.md)를 9/22 완료했다.
480문항/560턴 전부 × 두 점수 × 3 seeds = 3,360턴을 실행했고, 입력 절단·누락은 없다.
기존 root 정책과 예산 설정에서 AL은 **2.0651→2.1003(+1.70%)**지만,
차이의 95% CI **[−0.0025,+0.0721]**이 0을 포함해 전체 우위는 미확정이다.
P1의 깊이 3/4 수락 비율 증가가 관측되며 P2 개선은 불확실하다.
관측 hit rate는 82.4107%→81.7271%다. 동일 root 정책이어도 생성 경로·cache 상태가
달라질 수 있으므로 AL과 별도로 추적한다. 출력 상한은 128이며 논문의 1,024-token
조건을 재현한 것은 아니다. TPS는 보조 기록으로만 사용한다.
수식 목적은 [OBJECTIVE.md](duet_tree_al_full/OBJECTIVE.md), 실행 무결성 검사는
[audit.json](duet_tree_al_full/audit.json)에 있다. 설명 이해 확인 TODO는 계속 미완료다.

기록일: **2026-09-21**. 처음에는 기존 보고서·코드 상태와 대화에서 확인한 결론을
통합한 인계 기록으로 작성했으며, 이후 사용자 요청으로 tree 감사·새 실험을 추가했다.
사용자는 아직 모든 내용을 따라오지 못했으며, 다른 작업 후 함께 설명을 재개하거나
연구를 이어가려 한다. **작업 완료와 사용자의 이해 확인은 별도 상태로 관리한다.**

연구 목표는 MLSys 논문을 위한 DUET 개선이다. 실제 correction을 예측하는 후보 선정,
early-exit·P1/P2·tree 예산의 선택, 최종 처리율을 연결한다. 추가 모델/head 학습은
원하는 방향이 아니다. 기존 저장소의 calibration 추천을 그대로 채택하지 말라는
요청에 따라 새 연구용 prototype을 만들었다. Tree 선정 방식은 아래 신규 감사와
실험 단계까지 진행했으며 production 정책 교체는 아직 하지 않았다.

**9/21 추가 — tree 연구:** [새 보고서](duet_tree_analysis/REPORT.md),
[수식·정확성 분석](duet_tree_analysis/THEORY.md). 논문 식 (4)의 q 경로 곱은
코드에 있으나 frontier/cap/threshold/rerank 차이가 있다. 올바른 node reward는
root prior × parent 도달 × 앞선 형제 거절 × 현재 형제 수락이다. 24 prompts의
238개 full-p/q tree를 수집하고 8/16 split으로 scalar calibration을 검증했다.
Node reach MSE는 19.27% 감소했다. Full-p/q 계측 실행에서는 hit당 AL이
2.247→2.023으로 낮았으나, 분포 저장을 끈 16 prompts×2-seed 비교에서는
AL 1.905→2.092, prefill 포함 처리율 68.968→70.286 tokens/s(+1.91%)였다.
AL 차이의 95% CI는 [−0.03824,+0.40389]로 우위 미확정이다.
TPS 변화의 95% CI는 [−4.02%,+8.01%]다. 최신 지시에 따라 tree의 채택 판단은
AL을 우선하며 TPS 증가를 필수 조건으로 두지 않는다.
두 계측 조건을 혼합하지 않는다. 추가 head 학습은 하지 않았다.

**이전 lossless 설명의 범위 정정:** 현 G>M score-dependent tree rerank는
ancestor/sibling closure만으로 분포 보존을 보장하지 않는다. 실제 production
rerank와 tensor verifier의 완전 열거 반례에서 TV=0.01875를 재현했다. 과거
chain/G=M 결과까지 반박한 것은 아니며, 새 실험은 G=M으로 수행했다.
전 frontier·causal fanout·continuation gain·graph 비용을 결합하는 구현은 후속 과제로
남아 있다. 완료된 새 full-corpus 실행은 고정한 두 점수의 더 큰 문항/seed AL 비교다.
전체 frontier와 배분 알고리즘의 최적화를 완료한 것으로 읽지 않는다. 상세 raw 결과와
재현 명령은 새 보고서에 있다. **사용자의 수식 이해 확인 TODO는 계속 미완료다.**

**현재 확보한 결과는 후보 coverage 개선과 파라미터 조정에 따른 속도 개선이며,
두 성과를 같은 원인으로 설명하면 안 된다.**

| 질문 | 현재 판단 |
|---|---|
| Proxy는 항상 target보다 평평한가? | 아니다. Entropy 증가·감소가 모두 관찰된다. |
| 정확한 residual을 쓰는 목적은 타당한가? | 같은 위치·같은 비용·같은 후보 수에서 true residual top-k가 correction coverage에 최적이다. 전역 위치 배분에는 true h도 필요하다. |
| 현재 early-exit의 `[e-q]+`가 proxy-only보다 좋은가? | 주요 exit56 실험에서는 평균 열세다. 모든 layer/상황/방법의 불가능성을 증명한 것은 아니다. |
| 후보 선정은 개선했는가? | 기존 proxy 정책 대비 coverage 개선을 확인했다. 가장 일관된 기여는 거절 위치별 후보 배분이었다. |
| `e(1-q)` 자체가 같은 배분의 proxy보다 좋은가? | 평균 추가 이득이 있는 조건은 있으나 신뢰구간이 0을 포함하여 우위 미확정이다. |
| 후보 수식 변경으로 실제 TPS가 높아졌는가? | 확인하지 못했다. 별도 2-seed 실행의 합산 변화는 −1.61%였다. |
| Calibration으로 파라미터 탐색을 줄일 수 있는가? | 제한된 chain 조건에서 feasibility를 확인했다. 비용 모델 + 소수 실제 비교가 필요하다. |
| Calibration의 최종 설정은? | 측정 범위에서 exit56, K1/K2=4/2, P1 fanout3, P2 roots15. |
| 새 tree 점수의 AL 우위는? | Full Spec-Bench 3-seed 평균 +1.70%, 95% CI에 0 포함. 전체 우위 미확정이며 P1 깊은 수락 경로에서 개선 신호가 있다. |
| Mirror-SD 전체 시스템을 이겼는가? | 실제 Mirror-SD 전체 시스템과의 비교는 아직 하지 않았다. |

**지금까지 수행한 작업과 근거 문서는 다음과 같다.** 날짜는 해당 연구 기록의 날짜이며,
이번 기록일에 모두 다시 실행했다는 뜻은 아니다.

| 시기 | 작업 | 완료한 내용 | 주요 문서 |
|---|---|---|---|
| 9/9–10 | 기존 DUET 후보 정책·계측 확인 | residual/proxy/draft source 비교 스위치, CUDA Graph 경로 확인, 전 layer probe, 초기 coverage 분석 | [초기 보고서](residial_dist/REPORT.md), [환경·구현 인계](residial_dist/HANDOVER.md) |
| 9/10 | Proxy–target entropy 실측 | 동일 context의 layer별 entropy와 여러 가중 방식, 최종 head 통제 | [Entropy 보고서](residial_dist/entropy/REPORT.md) |
| 9/10–11 | 공유 분석 검토 및 lossless 경로 점검 | 전체 p/q/e 저장, temperature 보정·support/순위 진단, 실제 correction/cache/verification 역할 확인 | [공유 분석 검토](residial_dist/shared_review/REPORT.md), [후속 점검](residial_dist/shared_review/FOLLOWUP.md) |
| 9/12 | Residual 대 proxy 직접 비교 | 충분조건·반례·후보 삭제/순위 손실 분해, 새 128 prompts 독립 검증, 25개 점수 정책과 선택기 비교 | [직접 비교 결과](residial_dist/direct_comparison/REPORT.md), [증명](residial_dist/direct_comparison/THEORY.md) |
| 9/12 | 학습 없는 후보 개선 | 2,640개 조합 탐색, 새 96 prompts × 3 temperature 검증, 실제 wire/동점 민감도, 별도 실행 속도 측정 | [개선 실험 결과](residial_dist/training_free/REPORT.md), [수식](residial_dist/training_free/THEORY.md), [인계](residial_dist/training_free/HANDOVER.md) |
| 9/13 | 새 calibration 설계·실험 | 시간/보상 모델, anchor profiling, shortlist 선택, 독립 confirmation, root/tree/후보 계수 보조 실험 | [Calibration 결과](duet_calibration/REPORT.md), [수식](duet_calibration/THEORY.md), [인계](duet_calibration/HANDOVER.md) |
| 9/14까지의 후속 설명 | 수정 점수의 논문 근거 검토 | `e(1-q)`의 차감 완화·독립 불일치 해석·편향과 한계 설명 | 본 기록의 수식 논의 및 [설명 TODO](residial_dist/training_free/TODO.md) |
| 9/21 | 진행 기록 통합 | 완료/미확정/미착수 구분, 재개 순서와 자료 연결, 설명 TODO 보존 | 이 문서 |
| 9/21 | Tree 구성 감사·새 점수 실험 | 논문/구현 비교, 정확한 reach/AL·DP, rerank 편향 완전 열거, full-p/q holdout과 실제 점수 교체 | [Tree 보고서](duet_tree_analysis/REPORT.md), [Tree 수식](duet_tree_analysis/THEORY.md) |
| 9/21–22 | Full dataset의 AL 중심 검증 | 480문항/560턴×2정책×3seeds, 자연 EOS·실제 대화 이력, 138,357 verification events, paired CI·phase/tail·node 분석 | [Full AL 보고서](duet_tree_al_full/REPORT.md), [목적함수](duet_tree_al_full/OBJECTIVE.md) |

초기 보고서의 “Mirror-SD”는 동일 DUET 엔진에서 proxy 점수만 사용하는 ablation을
가리킨다. 초기의 “실행 불가 layer”, 단일 TV 임계값, 평탄화 중심 설명 등을 보편적
결론으로 인용하지 않는다. 후속 직접 비교와 training_free 보고서의 조건·한계가
현재 판단의 기준이다. 과거 인계의 “calibration 미착수”, “T=1만 측정” 상태도 이후
작업으로 갱신되었다.

**Entropy 측정에서는 평균 방향과 개별 context의 변화가 다르다는 점을 확인했다.**

12런, 128개 prompt ID × 3 seed = 384 generation, 34,702 verification step,
173,510 context를 측정했다. Exit56의 draft 위치 동일 가중 결과는 다음과 같다.

- `H(e)-H(p)` 평균 **+0.1589 nats**, 절대차 중앙값 **0.3042 nats**.
- 엄격한 부호 기준 증가 위치 **52.6%**, 감소 위치 **47.4%**.
- 따라서 평균적으로 entropy가 높다는 사실만으로 모든 proxy를 sharpen해야 한다거나,
  residual 실패가 평탄화 때문이라고 결론 내릴 수 없다.

**Residual 분석은 실제 correction을 정답으로 삼아 진행했다.**

같은 prefix에서 p=target, q=draft, e=proxy라 하면

\[
R_i(v)=\frac{[p_i(v)-q_i(v)]_+}{Z_i},\qquad
J^*_{i,v}=h_iR_i(v).
\]

고정된 draft 경로와 동일 root 비용에서 `J*`의 전역 top-B가 이론적 root coverage를
최대화한다. Bonus 위치는 residual 대신 target 분포를 사용한다. 이 목적은 아직
준비되지 않은 root나 재사용 suffix 길이·비용까지 포함한 TPS 최적화와 다르다.

수학적으로는 residual TV 상계, top-k 순위 보존의 충분조건, 후보 교환 이익과 오차의
정확한 분해를 정리했다. `e,q`만 같아도 가능한 target에 따라 승자가 달라지므로,
proxy 오차 조건 없이 항상 proxy-only를 이기는 규칙은 일반적으로 보장할 수 없다.

새 128 prompts의 exit56, T=1 직접 비교에서는 global root coverage가 residual
**61.297%**, proxy **65.618%**, 차이 **−4.321%p**였다. 별도의 위치별 top-3,
reject-event 가중 분석은 다음처럼 분해됐다. 서로 다른 지표이므로 앞 숫자와 합치지 않는다.

\[
\text{residual−proxy}
=\underbrace{13.016}_{\text{oracle 개선 여지}}
-\underbrace{10.665}_{\text{후보 삭제 손실}}
-\underbrace{7.360}_{\text{순위 손실}}
\simeq-5.009\text{%p}.
\]

Draft≈target인 작은 residual 사례는 존재하지만, 그 사례만으로 전체 실패를 설명할
수는 없었다. Target–proxy KL을 줄인 temperature 보정도 correction coverage를
반드시 높이지 않았다. 초기의 간단한 임계값/Ridge 선택기는 진단용으로 비교했으며
추가 이득을 얻지 못했다. 이를 새 모델/head 학습을 채택한 결과로 해석하지 않는다.

**후보 선정 개선은 토큰 점수와 위치 배분을 나누어 시험했다.**

1. 토큰 점수: `[e-q]+`, `e`, `[e-λq]+`, 혼합, `e(1-q)^β`,
   `max([e-q]+,ρe)` 등.
2. 위치 배분: 관측 draft token으로 만든 `h_hat`에 전체 분포 overlap으로 만든
   `h_bar`를 혼합한다. `alpha_bar=sum_v min(e_v,q_v)=1-TV(e,q)`이며,
   `h_tilde=(1-ω)h_hat+ω h_bar`다. 한 prefix에서의 기대 수락률 항등식은 정확하지만
   `h_bar`가 관측 경로의 정확한 거절 분포라는 보장은 없다.
3. 정규화: 위치별 top-M 점수 합 대신, sampled token 제외 후 전체 허용 vocabulary
   점수 합으로 나눠 위치 간 비교를 수행한다. 정확한 입력의 결합확률 의미를 보존하지만,
   근사 입력에서 모든 context에 이긴다는 정리는 아니다.

아래는 `training_free`의 독립 96 prompts, 세 temperature에서 실제 wire 방식
(topk17 후 앞15)까지 반영한 replay다. P1을 끈 PS-only chain, exit56, roots15 조건이다.

| T | 기존 proxy | 같은 배분·정규화의 개선 proxy | 수정 토큰 점수 포함 조합 | 토큰 점수 추가 효과, 95% CI |
|---|---:|---:|---:|---:|
| 1.0 | 65.126% | 66.801% | 66.811% | +0.011 [−0.217, +0.162]%p |
| 0.7 | 82.020% | 84.363% | 84.431% | +0.068 [−0.110, +0.274]%p |
| 0.5 | 89.005% | 91.335% | 91.321% | −0.014 [−0.112, +0.068]%p |

각 temperature에서 선택한 점수는 순서대로 `e sqrt(1-q)`, `e(1-q)`,
`max([e-q]+,0.75e)`다. 세 조건 모두 full 정규화와 ω=.25를 사용했다.

전체 조합은 기존 proxy 대비 +1.685/+2.411/+2.315%p였으나, 같은 배분의 proxy 대비
토큰 점수의 추가 우위는 미확정이다. 초기 T=1의 +0.125%p는 wire/동점 점검 후
+0.011%p로 줄었으므로 초기 숫자를 최종 증거로 사용하지 않는다. 별도 단독 실행은
seed별 TPS +4.72%/−7.81%, 합산 **−1.61%**로 속도 개선을 확인하지 못했다.

최근 full-DUET calibration의 후보 replay는 별도의 작은 표본이다. 8개 calibration
prompts/47 snapshots와 16개 validation prompts/104 snapshots에서 7 layers를
저장하고, 총 1,050개 정책 조합을 비교했다. Exit56, T=.7, roots15의 선택 정책은
`s=e(1-q), full normalization, ω=.75`였다.

| 방식 | Coverage |
|---|---:|
| 기존 proxy | 84.13% |
| 같은 배분·정규화의 개선 proxy | 87.75% |
| 같은 배분·정규화의 `e(1-q)` | 88.57% |

기존 대비 +4.444%p, 같은 배분 proxy 대비 **+0.828%p, CI [−0.221,+2.658]%p**다.
이는 **P1 중복 제거 전 저장 경로의 이론적 coverage**이며 wire/dedup 후 실제 cache
hit나 TPS가 아니다. Layer72에서는 선택 정책이 기존 proxy보다 −0.964%p였다.
Layer79의 residual 추가 이득은 마지막 layer 조건이므로 실용적인 early-exit 속도
개선의 증거로 사용하지 않는다. 정책을 최종 TPS 추천 설정에 자동 적용하지 않았다.
원본: [candidate_summary.json](duet_calibration/candidate_summary.json).

**최근 대화의 미해결 질문은 “`e(1-q)`가 논문 기여로 충분히 정당화되는가?”이다.**

이 질문에 대해서는 다음 구분까지 설명했다. 사용자가 모두 이해·동의했다고 처리하지 않는다.

\[
[e-q]_+\le e(1-q)\le e,\qquad
e(1-q)=[e-q]_++\min(e,q)-eq.
\]

따라서 차감을 완화한다는 해석과, 고정 q에서 상대 민감도 1이라는 성질은 있다.
그러나 실제 residual을 더 잘 근사한다는 정리나 최적 목적함수에서의 유도는 아니다.
독립적으로 V~e, Y~q를 뽑으면 `P(V=v,Y≠v)=e(v)(1-q(v))`지만, 독립 불일치는
SD의 거절 사건이 아니다. Y=y를 이미 관측한 독립 모형에서는 이 질량이
`e(v) 1{v≠y}`이므로, 현재 후보 정책의 근거로 그대로 대입할 수 없다.

특히 e=p여도 `p(1-q)`는 일반적으로 true residual과 다르다. 설명용 반례는
`p=e=(.6,.3,.1), q=(.5,.1,.4)`다. Draft가 C를 제안했고 root 하나를 고르면,
true residual 점수 `(.1,.2,0)`는 B, 수정 점수 `(.3,.27,.06)`는 A를 고른다.
C를 제외해도 이 차이가 남고, 실제 correction 확률은 A=1/3, B=2/3이다.
이는 실측 사례가 아니라 정확한 proxy에서도 수정식의 편향이 남는 수학적 예다.

현재 논문에서의 위치는 **민감도 완화를 위해 평가한 후보 점수/ablation**이다.
“이론적으로 더 정확한 residual”, “동일 배분 proxy를 확실히 능가하는 핵심 수식”이라는
주장은 아직 뒷받침되지 않는다. 실제 proxy 오차와 correction coverage를 연결해
차감 강도를 정하고, 정확한 proxy 극한에서 residual로 돌아가도록 하는 설계는
후속 연구 방향이지 완료한 방법이 아니다.

**Lossless와 관련해서는 후보 준비와 실제 sampling의 역할을 확인했다.**

실제 correction은 최종 p와 실제 proposal q의 residual에서 뽑는다. 그 토큰으로
cache를 조회하며, hit root는 이미 확정된 토큰이다. 그 뒤 실제 q에서 생성한
continuation만 p/q로 검증한다. 후보 점수 s를 verification 분모로 사용하지 않는다.

Production CPU sampler/verify, cache lookup, 후보 hook과 CUDA Graph 경로,
분포 tap 및 hash를 점검했다. 9/11 CPU proposal 60만 회와 후속 세 temperature
합계 30만 회 검사는 각각의 캠페인 범위다. 모든 분산 GPU/KV/race/옵션의 end-to-end
정확성을 완전히 증명한 것은 아니다. 기준 target도 양자화 전 원본이 아닌 실행한
AWQ target이다. 상세 경로와 한계는 [FOLLOWUP](residial_dist/shared_review/FOLLOWUP.md)에 있다.

**Calibration은 기존 추천 로직과 별도로 새로 설계하고 실제 실행으로 확인했다.**

모델쌍은 LayerSkip Llama2-70B AWQ TP4 + TinyLlama-1.1B BF16 TP1,
RTX4090 다섯 장, B1, T=.7, P1+P2 chain이었다. 입력 16–768 tokens,
max_model_len2048 범위다. Layer 값은 엔진의 0-based 설정이다.

시간·보상 모델의 핵심은 다음과 같다. 기호의 상세 정의는 [THEORY](duet_calibration/THEORY.md)에 있다.

\[
C=\max(x+D_1,P)+D_2,\qquad T=\max(F,C)+S,
\]
\[
U=1+\pi_0A_0+\pi_1A_1+\pi_2A_2,\qquad G=\frac{\mathbb E[U]}{\mathbb E[T]}.
\]

P1 완료와 proxy 도착 중 늦은 시점 뒤에 P2가 진행되며, miss에서도 JIT-short로
토큰을 생성한다. Hit×AL만으로 보상을 계산하거나 forward 시간을 단순 합산하지 않는다.

- 5 anchors → 모델 동결 → 19개 설정 검증: cycle time MAPE **1.58%**,
  tokens/step MAPE **17.90%**, output TPS MAPE **15.98%**.
- 최초 추천 exit40·4/2는 해당 격자 최고 exit56·4/2보다 **4.65%** 낮았다.
  최초 추천의 실패를 최종 보완 결과로 지우지 않는다.
- 이후 별도 선택 prompts에서 exit40/56/72 × K1/K2=4/2 또는 4/1의 6개를 실행했다.
  이 절차는 최초 validation을 본 뒤 추가한 secondary 분석이며 다시 동결했다.
- 최종 별도 16 prompts × 2 seeds, 256 output tokens, profiler OFF 확인:

| 설정 | Output TPS | 비교 기준 대비 |
|---|---:|---:|
| exit56, K1/K2=4/2 — 선택 | **92.154** | **+42.19%**, CI [+36.60,+47.52]% |
| exit40, K1/K2=4/2 | 90.244 | +39.24% |
| exit72, K1/K2=4/2 | 88.226 | +36.13% |
| exit56, K1/K2=8/4 — 비교 기준 | 64.811 | 기준 |

**이 +42.19%는 후보 수식 변경이나 Mirror-SD 대비 속도 개선이 아니다.** 기존 residual
후보 구현을 유지하며 phase forward 예산을 조정한 결과다. 주 실험의 후보 softmax는
legacy T=1이고 실제 생성/검증은 T=.7이었다. 별도 후보 비교는 후보 T도 .7로 맞췄다.
최종 추천의 candidate는 여전히 `legacy`다.

Exit56이 모든 seed에서 유일한 최고는 아니었고, 전역 최적 설정을 증명하지 않았다.
조건 밖의 모델/GPU/batch/context/tree에 이 추천을 그대로 적용하지 않는다.
[추천 JSON](duet_calibration/recommended.json), [CLI 인자](duet_calibration/recommended_args.txt).

Calibration의 제한과 보조 결과도 남긴다.

- Low-K=2/2 구간의 시간 오차는 **8.10–13.56%**로 커졌다. Draft-wait와
  proxy-wait 양쪽 실행 구간을 profiling해야 한다.
- Root 수 8/24, fanout1/2, P2 tree 및 양 phase tree를 시험했다. 공동 최적값은 미확정이다.
  Tree verify 평균 시간 보간은 **67.89% 오차**로 실패했다. 큰 지연의 반복 여부와
  중앙값 진단을 독립 예측 성공으로 대체하지 않는다.
- 조건부 기대 AL `1+sum_j product_{i≤j} alpha_i`로 accept 난수의 조건부 분산을
  제거할 수 있음을 확인했다. 이 값으로 현재 추천을 다시 만들거나 17.90%의
  품질 예측 오차를 해결한 것은 아니다.
- 5개 profiling + 6개 shortlist 선택 비용 **632.5초(10.54분)**,
  19개 검증 비용 **1312.4초(21.87분)**. 실행당 prompts와 탐색 영역이 달라
  일반적인 동등 예산 우위로 해석하지 않는다. 관측 TPS가 지속될 때 비용 회수에
  약 **138,162 output tokens**가 필요했다.
- 유효 inference 실행 **54개**, 캠페인 내 중복 없는 **56 prompts**.
  성공한 GPU 실행 wall 합은 약 **59.98분**. 과거 연구에서 사용한 prompt가
  포함되므로 전체 연구에 걸친 완전한 신규 데이터라는 주장은 하지 않는다.
- 요청마다 verifier/collector가 재생성되어 초기 분포 probe의 이전 요청이 유실된
  문제를 찾아 요청별 저장·명시적 flush 후 재수집했다. 제외한 시도는
  `duet_calibration/failed_attempts/`에 보존했다. 주요 TPS의 prompt metric과는 별개다.

**사용자와 함께 재개할 설명은 모두 미완료로 유지한다.** 세부 체크리스트의 정본은
[training_free/TODO.md](residial_dist/training_free/TODO.md)다. 설명을 제공했다는 이유만으로
이해 확인을 완료 처리하지 않는다. 재개 순서는 다음을 권장한다.

1. 목표·용어: p/q/e, 실제 R, 후보 점수 s, `(위치,토큰)` root, coverage/hit/AL/TPS.
2. 정확한 residual의 최적성, proxy 대입으로 생기는 삭제·순위·정규화 오차.
3. 거절 위치 h와 h_hat/h_bar 혼합, 전체 정규화, 토큰 점수 변경을 분리해 이해.
4. 공정한 세 대조군과 결과: 기존 proxy / 같은 배분의 proxy / 수정 점수.
5. `e(1-q)`의 설계 근거·편향·논문 주장 범위 및 실제 sampling과의 분리.
6. Calibration의 시간·보상 모델, 최초 실패와 보완, +42.19%의 정확한 비교 대상.
7. 사용자가 바꿀 tree 선정 방식과 다음 연구의 우선순위를 함께 결정.

**추가 연구는 아래 상태에서 멈춰 있다. 이번 기록 작업에서 자동으로 실행하지 않았다.**

- [ ] 현재 후보 수식의 설명을 사용자와 마치고, 핵심 방법/ablation의 위치를 결정한다.
- [ ] Proxy 오차를 반영한 residual 추정 또는 후보 순위 규칙을 실제 coverage 목적에서
  도출하고, 기존의 단순 혼합·할인보다 나은지 별도 개발/검증 자료로 확인한다.
- [ ] Full-DUET에서 wire/tie/P1 dedup과 root 준비 시각·재사용 suffix 길이를 포함해
  위치 배분 개선의 실제 이득을 측정한다. Counterfactual coverage를 TPS로 대체하지 않는다.
- [ ] 새 tree 선정기의 `shape → 비용`, `정책 → coverage/AL` 인터페이스를 정한다.
  기존 chain 시간 모델을 새 tree에 그대로 외삽하지 않는다.
- [ ] Calibration에 양쪽 대기 구간과 조건부 기대 AL을 반영하고 동일 예산의
  대조군 대비 추천 regret/비용을 독립 평가한다.
- [ ] 모델쌍·하드웨어·문맥 bucket을 확대한다. B>1 및 다른 모델 지원도 본 캠페인에서
  완료한 것으로 취급하지 않는다.
- [ ] 최종 정책의 실제 TPS 개선과 Mirror-SD 전체 시스템 비교를 수행한다.

**재개 시 사용할 작업 공간과 실행 근거도 보존한다.**

- 2026-09-21 확인 branch: `feat/duet-proxy-source-ablation`,
  HEAD `a82f7d24fb36827a9a81a3567f344dccb71f193e`.
  기존 tracked 파일 9개에 수정이 있고 결과 디렉터리·probe 등이 untracked다.
  연구 산출물이 커밋된 상태로 가정하지 않는다. 이번 기록 작업은 엔진을 수정하지 않는다.
- Python: `ssd/.venv/bin/python`.
- Target: `/home/chokwans99/awq_calibrated/layerskip_llama2_70b`.
  AWQ artifact: `/home/chokwans99/awq_artifacts/layerskip70b_awq_tp4`.
  Draft: `/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0`.
  이전 AWQ norm 보정 이슈 때문에 원본 target 경로로 조용히 바꾸지 않는다.
- 과거 실행은 주로 GPU3–7을 사용했다. 현재 점유 상태는 이번에 재측정하지 않았으므로
  재개 시 확인한다. 다른 사용자의 프로세스를 종료하지 않는다.
- `--duet_only_proxy`는 **P1을 끄는 실행 모드**다. 토큰 점수 source=e는
  `--duet_proxy_source proxy`다. 두 의미를 혼용하지 않는다.
- 후보 변경은 실제 CUDA Graph 경로에 적용되어야 한다. Eager 코드만 바꾼 비교를
  유효한 ablation으로 간주하지 않는다. `training_free/runtime_policy.py`는 실험 hook이다.
- Raw 데이터는 `ssd/experiments/proxy_source_ablation/` 및
  `results/duet_calibration/runs/`, 동결 정책·hash·통계는 각 결과 디렉터리에 있다.
  이전 확인 데이터를 다시 탐색에 쓰면 이후에는 독립 검증 자료로 부르지 않는다.
- 완료한 `frozen*.json`/원시 결과를 덮어쓰지 않고 새 실험은 별도 디렉터리에 기록한다.
  단순 follow-up 설명을 위해 GPU 실험이나 전체 집계를 다시 돌릴 필요는 없다.
- [Calibration 재현 절차](duet_calibration/REPORT.md),
  [후보 실험 인계·재현](residial_dist/training_free/HANDOVER.md),
  [Calibration checks](duet_calibration/checks.json),
  [후보 실험 audit](residial_dist/training_free/audit.json)를 기준으로 이어간다.
