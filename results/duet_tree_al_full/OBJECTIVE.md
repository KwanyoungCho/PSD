# Root는 hit, tree는 AL: 이번 평가의 목적함수

사용자 지시(2026-09-21)에 따라 TPS가 아니라 AL을 tree 정책의 주 평가 기준으로
삼는다. 이 문서는 결과를 보기 전에 고정한 [PLAN.md](PLAN.md)를 풀어 설명한다.
확률 및 tree 검증식의 전체 유도는 [이전 THEORY.md](../duet_tree_analysis/THEORY.md)에 있다.

## 1. 두 문제를 분리한다

현재 target 검증이 끝나고 이어갈 context를 root 사건 r이라고 하자. Cache lookup
우선순위를 적용한 후 서로 배타적인 사건으로 정의하고 그 확률을 pi_r로 쓴다.
선택한 cache root 집합이 C이면

\[
H(C)=P(\text{cache hit})=\sum_{r\in C}\pi_r.
\]

Root 선정은 이 hit 확률을 높이는 문제다. 지난 residual/proxy 후보 선정 연구가
여기에 해당한다. 이번 비교에서는 root source와 후보 예산을 바꾸지 않는다.

Root r 아래 만든 tree를 T_r, 실제 수락된 descendant 수를 A(T_r)라고 하면

\[
L_r=E[A(T_r)\mid r\text{ hit}],\qquad
L_{\mathrm{hit}}=\frac{\sum_{r\in C}\pi_r L_r}{H(C)}.
\]

**이번 tree 구성의 목적은 고정된 root 정책과 예산 아래 L_hit를 높이는 것이다.**
이상적인 한 context에서 C와 pi가 고정되어 있으면 분모는 상수이므로
\(\sum_r\pi_rL_r\)를 최대화하면 된다. Root prior를 tree 배분에 계속 쓰는 것은
적중 가능성이 높은 root 아래 더 긴 수락 경로를 준비하기 위해서다.

독립적인 전체 생성 실행에서는 tree 변경에 따라 생성 경로, 이후 context,
cache lookup 및 P1/P2 hit 구성도 달라질 수 있다. 따라서 동일한 root **정책**은
모든 실행에서 동일한 root **표본 및 hit 확률**을 보장하지 않는다. 이번 측정은
전체 생성 정책의 비교이며, 모든 context를 고정한 개별 tree의 인과 실험은 아니다.

## 2. AL은 node 수가 아니라 수락되는 경로의 길이다

실현된 tree와 각 context의 target/draft 분포를 고정한다. Node v가 검증에서
도달되어 수락될 확률을 rho(v)라 하면

\[
E[A(T)\mid T,p,q]=\sum_{v\in T}\rho(v).
\]

예를 들어 parent u 아래 첫 자식의 조건부 수락률이 0.7이고, 그 자식이 거절된
후 둘째 자식의 조건부 수락률이 0.5라면 자식들의 도달·수락 확률은
rho(u)×0.7과 rho(u)×0.3×0.5다. 같은 parent의 두 자식을 동시에 수락할 수는
없다. 따라서 형제의 draft 확률 q를 그냥 더하거나 곱한 값과 AL은 다르다.

깊이가 K인 tree에서는 또 다음 항등식이 성립한다.

\[
E[A]=\sum_{d=1}^{K}P(A\ge d).
\]

이에 따라 결과에는 평균뿐 아니라 수락 길이 분포와 tail 확률을 함께 저장한다.
깊이 1에서의 실패를 줄였는지, 깊이 2 이상을 더 자주 이어갔는지 구분할 수 있다.
P1은 K=4, P2는 K=2이므로 phase별로 해석해야 한다.

## 3. 올바른 확장 우선순위는 추가 AL의 기대값이다

Leaf u를 한 번 forward해서 c개 자식을 새로 만들 때, 실현된 자식의 조건부
수락률이 alpha_1,...,alpha_c라면 추가 기대 AL은

\[
\Delta L(u,c)=\rho(u)\left(1-\prod_{j=1}^{c}(1-\alpha_j)\right).
\]

아직 자식을 뽑기 전에는 괄호 안의 값을 future proposal에 대해 평균한
g(u,c)를 예측해야 한다. 따라서 forest 내 추가 AL은

\[
\boxed{\Delta L_{\mathrm{forest}}(u,c)
\ \propto\ \widehat\pi_r\widehat\rho(u)\widehat g(u,c).}
\]

이번에 검증하는 기존 prototype은 phase/sibling/q-bin별 scalar calibration으로
rho를 추정하고, 같은 확장 크기에서 g가 비슷하다고 단순화한 점수다. 별도의
neural head를 학습하지 않는다. **g를 추정하는 완성된 AL 최적 알고리즘을 이미
구현했다는 뜻은 아니다.** 이번 full test에서 실패하더라도 AL 목적함수가
틀렸다는 결론이나 q-path가 전역 최적이라는 결론은 나오지 않는다.

## 4. TPS를 목표에서 빼도 자원 제약은 필요하다

제약 없이 depth나 node 수를 늘리면 AL이 늘 수 있다. 점수 또는 구조 선택의
개선인지 확인하려면 비교할 수 있는 예산을 고정해야 한다. 이번에는 exit56,
forward depth K1/K2=4/2, C=3, root 수와 P1/P2 node cap=8/6을 고정했다.

목적함수를 AL/time으로 바꾸지 않는다. 다음 단계의 shape 탐색 역시 정해진
forward·node·round 제약 아래 AL을 최대화하는 문제로 설계하면 된다. 향후
예산 자체를 바꾸는 실험은 각 예산에서 달성한 AL 곡선으로 분리해 보고한다.

정확히는 P2 root 상한이 15이고 P1은 context마다 root 3개를 만든다. P1의
context bucket에 따라 physical forward 폭도 달라진다. 따라서 서로 다른
생성 경로에서 총 root/forward cell 수가 같다는 주장은 하지 않는다.

Nominal cap이 같아도 실제 이용한 node 수는 달라질 수 있다. 이번 기본 event
로그의 valid_k는 hit tree의 검증 node 수를 담는다. 따라서 그 분포를 보조로
집계할 수 있다. 반면 모든 미적중 root의 생성 node 수와 draft forward 수는
현재 무관측 실행 로그에 없으며, 같다고 추정해서 수치로 채우지 않는다.

## 5. 판정 기준과 해석 범위

- 주 지표: 각 task의 hit당 descendant AL을 구한 뒤 6 tasks를 균등 평균한다.
- P1/P2 조건부 AL과 50:50 phase 평균을 함께 본다. Phase 구성 효과를
  점검하는 진단이며, 같은 context의 counterfactual 평가를 대신하지는 않는다.
- 종료 시점 이후까지 계산된 token이 AL에 섞이지 않도록 각 요청의 마지막
  verification event를 주 집계에서 제외하고, 모든 event 집계도 함께 낸다.
  이 제외 규칙 자체도 완벽한 unbiased stopping-time 보정은 아니다.
- 480 original questions를 task 안에서 paired bootstrap한다. 두 turn과 세
  seed는 같은 question cluster에 보존한다. 3,360턴을 모두 독립 표본으로
  취급하지 않는다. 신뢰구간은 지정한 세 seed 결과에 조건부인 문항 불확실성이다.
- 두 번째 turn은 실제 첫 응답에 의존하므로 양 정책에서 입력 context가
  달라질 수 있다. 질문 ID에 대한 pairing과 동일 입력 pairing은 구별한다.
- AL 차이와 신뢰구간으로 개선 근거를 판단한다. TPS 증가를 요구하지 않는다.
- Full은 480문항/560턴 전체를 뜻한다. 출력 상한 128, natural EOS, raw base
  model continuation을 사용하며 공식 1,024-token/chat-template 결과와 동일시하지 않는다.

결과가 좋으면 이 설정에서 rho 기반 확장 순위의 효과가 확인되는 것이다.
차이가 불확실하거나 나쁘면 먼저 phase·깊이별 실패 지점을 확인하고,
g 예측 또는 사전 shape 배분을 독립 calibration/validation에서 개발해야 한다.
이번 full corpus에 맞춰 다시 튜닝한 뒤 동일 데이터의 개선을 최종 검증으로
제시하지 않는다.
