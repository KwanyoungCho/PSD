# 사후 분석의 결론과 다음 개선 방향

2026-09-22. **Reach를 AL 점수로 쓰는 근거는 타당하다. 그러나 현재 보정표는
첫 형제가 틀릴 문맥을 구별하지 못하고, 현재 fanout 규칙은 node별 추가 AL을
비교하지 않는다. 이번 결과에서 다음 구현의 우선순위는 더 복잡한 확률 보정보다
추가 AL에 따른 fanout 배분이다.**

이는 새 배분 정책의 end-to-end AL 우위를 확정했다는 뜻은 아니다.
다음 결론은 기존 정책 두 개로 수집한 전체 corpus의 정확한 사후 분석과,
online에서 사용할 수 있는 정보로 수행한 국소 배분 실험에 근거한다.

## 1. 무엇을 실제로 측정했는가

- Spec-Bench 480문항/560턴 전체 × q-path/frozen-reach 두 정책, seed1: **1,120턴**.
- **37,869개 hit tree, 262,145개 node**. 마지막 verification event 제외 주 진단은
  36,896개 tree다. 두 정책 중 적어도 한쪽에 hit가 있는 476문항에서 conditional
  지표가 정의됐다. 나머지 4문항도 실행했고 undefined 값을 0으로 채우지 않았다.
- 모든 tree의 p/q로 정확한 수락 확률·reach·기대 AL·coin 분산을 계산했다.
  원본 전체 분포 1,184개를 다시 계산한 차이는 0이었다. 실제 float32 업데이트와
  double 정규화 참조의 reach 차이는 최대 1.52e-6이었다.
- 기존8문항 보정과, 평가 문항 전체를 제외한 5-fold scalar 보정을 비교했다.
  후자는 fold당 최대384문항으로 보정하므로 원래8문항보다 calibration 비용이 크다.
- Target LayerSkip70B AWQ / TinyLlama1.1B, T=.7, exit56, depth4/2, node8/6, G=M.
  128-token cap, natural EOS, 실제 multi-turn history. Raw base-model continuation이다.
- Observer가 비동기 스케줄에 영향을 주므로 기존 무관측 성능 결과와 합치지 않는다.
  모든 score 비교는 동일한 관측 tree에서 수행했다. TPS는 채택 지표가 아니다.

## 2. 현재 reach 수식은 의미가 있다

정확한 수식은

\[
\rho(v_j)=\rho(u)\prod_{k<j}(1-\alpha_k)\alpha_j,
\qquad E[AL\mid T,p,q]=\sum_v\rho(v).
\]

Node reach MSE는 q-path **0.079673 → frozen 0.067698**, **15.03% 감소**했다.
차이 CI는 [-0.013297,-0.010684]다. 고정 sampled-pool의 예산4 DP에서도
q-path 점수의 **2.08020 → frozen 2.12461**, 차이 +0.04441, CI [+.03882,+.04990]였다.

따라서 수락 법칙에 맞춘 보정이 전혀 의미 없었다는 결론은 맞지 않는다.
단, 이 subset 지표는 실제 생성 tree를 새로 실행한 AL이 아니고,
사후 token-dependent pruning을 배포해도 된다는 뜻도 아니다.

## 3. 무엇을 못 맞추는지 구체적으로 확인됐다

**현재 frozen table은 같은 부모의 첫 형제를 항상 우선한다.**
첫 형제의 예측 alpha 최솟값이 P1=.617383, P2=.508532로 모두 .5를 넘는다.
뒤 형제의 reach 비중을 전부 합쳐도 1−alpha0를 넘지 않으므로 첫 형제보다 작다.
이것은 관측 경향에 그치지 않고 table에 대한 수학적 결과다.
343개 q-bin 조합×2 phases의 완전 열거도 통과했다.

첫 형제가 실제로 대안 집합에 포함된 depth1의 **18,019개 비교**에서,
frozen은 첫 형제를 **100%** 골랐지만, target까지 아는 one-child-gain oracle은
**14.22%**에서 뒤 형제가 더 좋았다. P1 13.18%, P2 20.95%였다.
비율은 task→question→관측 비교의 균등 평균이다.

이 oracle은 현재 이후 분포까지 아는 **관측된 expanded-node subset**의 oracle이다.
전체 미확장 frontier에서 14.22%의 손실을 다 회복할 수 있다는 뜻은 아니다.

오류는 단순히 깊이가 커서 확률 곱이 작아지는 것만의 문제가 아니다.
q≥.9인 첫 형제 중 실제 alpha<.1인 attempt mass 비율이 P1 **1.04%**, P2 **2.67%**였다.
첫 형제·attempt≥.3으로 제한해도 이런 강한 오판 node가 **602개** 있었다.
반대로 q<.1인데 alpha>.95인 node는 **2,929개**였다.
확신이 높은 draft와 target이 다른 토큰을 지지하는 문맥을 현재 feature로 구분하지 못한다.

최대 오차 사례는 q=.999997인 `|`에 target 확률이 약2.29e-9였는데,
frozen alpha는 .861578이었다. 이 극단 사례 자체를 전형적인 빈도라고 주장하지 않는다.
빈도 판단에는 위 전체 집계와 [실제 사례 파일](examples.json)을 사용한다.

## 4. Feature를 더 넣으면 해결되는가

| 보정 | Reach MSE | 예산4 subset의 기대 AL |
|---|---:|---:|
| 기존 frozen, 8문항 | .067698 | 2.12461 |
| 동일8문항 + depth | .067890 | 2.12404 |
| 동일8문항 + depth/entropy | .069266 | 2.12152 |
| Cross-fit, 같은 phase/sibling/q feature | .062085 | 2.12974 |
| Cross-fit + depth/entropy | .061475 | 2.12995 |
| Cross-fit, direct local transition | .061143 | 2.13061 |

**8문항이라는 같은 보정 비용에서 feature를 늘린 개선은 확인하지 못했다.**
Depth/entropy까지 넣으면 오히려 subset AL이 -0.00309 낮았다(CI [-.00490,-.00134]).

보정 자료를 늘리면 magnitude/bias는 더 잘 맞춘다. Rich cross-fit은 frozen보다
MSE를 9.19% 더 줄였지만 subset AL 증가는 **+0.00534(+0.25%)**였다.
통계적으로 양의 차이가 있어도 효과 크기는 작다. MSE가 가장 좋은 direct-beta가
subset AL도 항상 가장 좋은 것은 아니다. 예산4에서는 conditional-q/depth의
2.13110이 점 추정상 가장 높았지만, 이것으로 정책 전체 우위를 선언하지 않는다.

동일-depth 위치 선택은 rich 보정을 해도 **0.44%**만 바뀌었고,
그 one-child-gain 변화 +0.000266의 CI는 [-.000102,+.000696]이었다.
현재 `depth==round` 제약 아래 같은 깊이 점수의 공통 배율을 고쳐도 순위는 바뀌지 않는다.
노드별 확률 예측과 실제 의사결정을 별도로 평가해야 하는 이유다.

## 5. 다음 위치의 g를 모르는 것이 주 병목인가

한 후보의 정확한 확장 가치는 rho(u) g1(u)이며
g1(u)=Σmin(p_u,q_u)다. 따라서 g가 수식에 필요하다는 주장은 맞다.

하지만 **이번 관측 범위에서 g를 더 정교하게 예측하는 것이 가장 큰 병목이라는
근거는 약하다.** Exact rho만 사용한 선택과 rho×exact g의 선택이 다른 비율은
1.79%, 평균 gain 차이는 .00173이었다. Frozen rho에 예측 g를 곱했을 때도
gain 변화는 +.000066, CI [-.000197,+.000290]이었다.

반면 frozen rho를 진짜 rho로 교체한 위치 선택의 차이는 +.06504였다.
현재 feature가 첫 후보의 강한 거절을 구분하지 못하는 문제가 훨씬 크게 나타났다.
이는 관측된 expanded-node 비교에 관한 결과이며 미확장 leaf 전체의 결론은 아니다.

## 6. 더 유망했던 것은 fanout 배분이다

**“어느 부모를 확장할까”와 “그 부모에게 몇 후보를 줄까”는 다른 문제다.**
현재 구현은 선택된 부모마다 자식 하나씩을 먼저 주고 남은 예산으로 둘째/셋째를 준다.
이 규칙은 첫째/둘째 후보의 **추가 기대 AL**을 서로 비교하지 않는다.

Residual/WOR에서는 둘째 후보의 추가 이득이 첫째보다 클 수도 있다.
p=(0,1), q=(.9,.1)이면 g(1)=.1, g(2)=1이다. 실제 원본 snapshot의
**4,595개 expanded context / 470문항**에서 g1/g2를 정확히 적분했을 때,
둘째 추가 이득이 첫째보다 큰 비율은 context 기준 **6.13%**, pooled reach 가중 **4.60%**였다.
이 정밀 검사는 원본을 저장한 1/32 systematic 표본이다.

추가 이득이 감소하는 경우에도, 높은 reach 부모의 둘째 후보가 낮은 reach 부모의
첫째 후보보다 유리할 수 있다. 따라서 위 비오목성이 모든 개선의 원인이라는 뜻은 아니다.

시험한 작은 개선식은 다음과 같다.

\[
\max_{\{c_u\}}\sum_u\widehat\rho(u)\gamma_\phi(c_u),
\qquad\sum_uc_u\le B,\quad\gamma_\phi(0)=0.
\]

Gamma는 **기존8개 calibration 문항만** 사용한 도달 가중 평균 gain이다.
P1의 gamma1/gamma2=.84147/.91202, P2=.70457/.78292다. 새 neural head는 없고,
online 의사결정에 미래 target p를 넣지 않는다.

두 부모·새 node 예산2에서는 (1,1), (2,0), (0,2)의 점수를 비교한다.
A의 둘째 후보와 B의 첫째 후보를 비교하는 식은

\[
\widehat\rho(A)[\gamma(2)-\gamma(1)]
\mathop{\gtrless}\widehat\rho(B)\gamma(1).
\]

**369문항의 1,282개 관측 대안 쌍**에서 실제 p/q로 그 결정을 평가한 결과:

| 결정 규칙 | Exact expected added AL |
|---|---:|
| 각 부모에 하나씩 | .51485 |
| q-path × calibration gain curve | .52716 |
| Frozen reach × calibration gain curve | **.53006** |
| Exact reach × calibration curve | .56376 |
| Exact reach × exact gain의 oracle | .56568 |

Frozen reach×curve의 차이는 **+.01521, 국소 gain +2.95%**, CI **[+.00692,+.02394]**다.
P1은 +.01632 [+.00680,+.02616], P2는 +.00982 [-.00465,+.02327]였다.
**P2만의 우위는 확정하지 못했다.**

이것은 미래 target을 알아야만 가능한 oracle 개선과 다르게, 이미 있는 reach와
작은 calibration 통계로 내릴 수 있는 결정에서 관측한 개선이다.
다만 첫 정책의 오차 분석 후 추가한 탐색적 실험이며, 실제 C=3/다중 round/전체 frontier를
다시 생성한 결과가 아니다. **전체 inference AL이 2.95% 증가했다는 뜻이 아니다.**

## 7. 원래 못 맞추는 부분과 아직 측정하지 못한 부분

| 구분 | 이번에 확인한 것 | 대응 |
|---|---|---|
| 수락 coin의 운 | Exact conditional variance .33602 token², 실제 oracle 잔차 MSE .33680 | 개별 AL 한 번 대신 기대 AL로 점수 평가 |
| 현재 입력 정보의 부족 | 같은 q/phase/bin으로 다른 target 수락 결과가 가능; frozen은 첫 형제를 역전시킬 수 없음 | 더 나은 사전 신호/보정 여부 검증. 유한 실험으로 모든 개선 불가능을 선언하지 않음 |
| 배분 알고리즘 | Reach만 정렬한 breadth-first fanout은 추가 이득 비교가 아님 | 후보 수별 gain을 비교하는 사전 배분 |
| 기록의 부족 | 미확장 leaf의 다음 q, 미적중 root의 전체 frontier/eligibility 경쟁은 없음 | Shadow 확장 또는 causal replay, 전체 frontier 선택 사유 기록 |

완전한 oracle과 현재 정책의 차이는 `정보 부족으로 남는 차이 + 현재 정보로도
개선할 수 있는 추정/선택 차이`로 나뉜다. 이번에 그 둘을 전부 수치로 분리한 것은 아니다.
Oracle gap 전체를 향후 달성 가능한 개선량이라고 말하지 않는다.

## 8. 무엇을 다음에 구현하고 검증할 것인가

1. **현재 frozen reach를 기준으로 유지하면서 fanout 배분만 먼저 바꾼다.**
   같은8문항에서 더 복잡한 depth/entropy table을 채택할 근거는 약했다.
2. 실제 C=3에 맞는 gamma(1..3), node budget, 선택된 부모 수, future-round reserve를
   포함해 배분한다. 두 후보 실험의 결과를 셋째 후보에 임의 외삽하지 않는다.
   결정은 새 child token을 샘플하기 전에 내리고 G=M을 유지한다.
3. 가능하면 고정 prefix의 shadow/super-tree에서 causal replay로 비교한다.
   정책은 그 시점에 드러난 feature만 보고, 아직 생성하지 않은 child/target 정보를 보지 않는다.
   그래야 실제 놓친 branch와 observed pool의 한계를 구별할 수 있다.
4. 그다음 고정 root 정책/같은 예산의 full480문항×여러 seed online AL을 비교한다.
   P1/P2와 hit rate를 분리해 추적한다. Root→cache hit, tree→AL 목표는 유지한다.

P1은 terminal mass의 25.93%가 cap 전 leaf에서, 36.52%가 depth4 cap에서 끝났다.
P2는 각각15.64%/55.85%였다. Cap 확대/전 frontier 허용에도 여지가 있지만,
이를 현재 예산에서 무료로 얻는 개선량으로 보지 않는다.

기존 무관측 3-seed 결과 **AL +1.70%, CI에 0 포함**은 변하지 않았다.
그 실험의 62개 파일 checksum도 재확인했다. 이번에 production selector는 바꾸지 않았다.
사용자와의 전체 수식 설명 TODO 역시 이해 확인 전까지 미완료다.

전체 표·실행 검사: [REPORT.md](REPORT.md), 수식 유도: [THEORY.md](THEORY.md),
그림: [diagnostics.pdf](diagnostics.pdf), [allocation.pdf](allocation.pdf).
