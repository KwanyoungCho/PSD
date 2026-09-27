# Tree 점수의 의미, 실패 원인, 개선 가능성을 구별하는 수식

범위: root가 cache hit한 뒤 검증하는 tree의 **accepted descendants**를 AL로 센다.
Correction root는 제외한다. Root 선택의 목적은 별도로 cache hit다.
아래 `정확`은 실제 구현의 ordered without-replacement residual ladder와 주어진
target/draft 분포에 대한 정확성이다. 그 분포를 사전에 알고 있다는 뜻은 아니다.

## 1. 후보의 draft 확률과 수락 확률은 다르다

부모 문맥 u의 target/draft 분포를 p_u, q_u라 하고, 형제를 v_1,…,v_c 순으로 뽑는다.

\[
D_1=q_u,\quad R_1=p_u,\qquad
\alpha_j=\min\{1,R_j(v_j)/D_j(v_j)\}.
\]

앞선 형제가 거절되면

\[
R_{j+1}=\frac{[R_j-D_j]_+}{\sum_v[R_j(v)-D_j(v)]_+},\qquad
D_{j+1}(v)=\frac{D_j(v)\mathbf1\{v\ne v_j\}}{1-D_j(v_j)}.
\]

확률 0인 거절 사건 뒤의 residual 정의는 기대값에 영향을 주지 않는다.
q(v_j)를 그대로 모든 형제의 proposal 확률로 쓰는 것도 잘못이다. 실제 j번째 proposal은
앞선 토큰을 제거한 D_j다. 실제 verifier는 이 ladder를 계산하고 있으며, 여기서는
그 수락 법칙에 맞는 **tree 생성 우선순위**가 무엇인지 묻는다.

고정된 실현 tree에서 부모의 도달 확률을 rho(u)라 하면

\[
\beta_j=\Bigl[\prod_{k<j}(1-\alpha_k)\Bigr]\alpha_j,
\qquad \rho(v_j)=\rho(u)\beta_j,\quad \rho(root)=1.
\]

논문의 q-path=product q는 토큰이 draft에서 얼마나 그럴듯한지를 반영하지만,
target의 수락/거절과 형제 순서까지 포함한 rho와 일반적으로 다르다.
예컨대 p=q이면 첫 후보는 q(v)가 작더라도 항상 수락되고, 뒤의 형제에 도달할 확률은 0이다.
q-path는 이 상황에서도 첫 후보에 q(v)<1, 뒤 형제에 양의 값을 준다.

## 2. 왜 rho가 AL 목적함수에 맞는가

I_v를 검증 경로가 v를 수락했는지의 indicator라 하면 실제 AL은 L=sum_v I_v다.
선형성으로

\[
\mu(T,p,q)=E[L\mid T,p,q]=\sum_{v\in T}\rho(v).
\]

각 노드의 독립성은 필요 없다. 같은 경로의 노드는 강하게 의존해도 성립한다.
따라서 q-path보다 rho를 잘 추정하는 것은 **올바른 방향**이다. 다만 이것은
전체 tree의 평가식이지, rho만 크게 보아 모든 예산 배분을 최적화할 수 있다는 정리는 아니다.

예측 MSE 감소와 선택 개선도 다르다. 현재 selector는 depth==round인 후보끼리 비교한다.
같은 phase/depth의 모든 priority를 같은 양수 c_d로 곱해도 그 round의 순위는 바뀌지 않는다.
그런 scale 오차를 고치면 MSE는 줄어들지만 생성 tree는 그대로일 수 있다.
이 주장은 공통 배율 보정에 대한 정확한 불변성이고, 이번 alpha table 변경 전체가
공통 배율이었다고 가정하는 것은 아니다. 실제 변경률은 별도로 기록한다.

현재 scalar 보정은 phase, sibling order, raw-q bin으로 alpha를 추정하고,
형제 거절과 경로 도달을 곱한다. 수락 법칙과 연결되어 있지만 alpha 자체의 정확성을
보장하지 않는다. 평균 alpha를 곱한 값은 일반적으로 실제 확률들의 곱의 조건부 평균과 다르다.

이번 frozen table에는 더 구체적인 한계가 있다. 첫 형제의 alpha_hat 최솟값이
P1에서는 .617383, P2에서는 .508532로 **모든 q bin에서 .5보다 크다**.

\[
\widehat\beta_0=\widehat\alpha_0>1/2,\qquad
\sum_{j\ge1}\widehat\beta_j\le1-\widehat\alpha_0<1/2.
\]

따라서 같은 부모 아래에서 첫 형제가 후보에 있으면 뒤 형제를 우선하는 점수를
현재 frozen table로는 만들 수 없다. 모든 q-bin 조합 343개×2 phase도 완전 열거했다.
이는 전체 root 간 우선순위가 고정이라는 뜻은 아니다. 동일 부모의 형제 순위에 대한 결과다.
또한 true alpha가 항상 .5 이상이라는 뜻도 아니다. Target이 첫 후보를 강하게 거절하는
문맥을 현재 table이 개별적으로 식별하지 못하는 표현력/입력 정보의 한계다.

## 3. 확률적 수락과 점수 오류를 분리한다

u에서 끝날 확률은 h(u)=rho(u) product_children(1-alpha)다. leaf에서는 h(u)=rho(u).
root의 depth는 0이다. 이 terminal mass는 합이 1이며

\[
E[L]=\sum_u h(u)d(u),\qquad
\operatorname{Var}(L\mid T,p,q)=\sum_u h(u)d(u)^2-\mu^2.
\]

이 분산은 p,q,tree를 모두 정확히 알아도 다음 coin을 보지 않고 실제 L 한 번을
완벽하게 맞출 수 없는 부분이다. 기대값을 잘 맞추는 점수와 개별 실행 길이를 맞추는
점수를 혼동하면 안 된다.

X를 online 점수가 사용하는 정보, Z=(T,p,q)를 완전한 진단 정보라 하자. X가 Z에
포함되고 새 검증 coin을 보지 않는다면

\[
E[(s(X)-L)^2]
=E[\operatorname{Var}(L\mid Z)]+E[(s(X)-\mu(Z))^2].
\]

추가로 두 번째 항은

\[
E[\operatorname{Var}(\mu(Z)\mid X)]
+E[(s(X)-E[\mu(Z)\mid X])^2]
\]

로 분해된다. 앞 항은 **선택한 정보 X만으로는** 구분되지 않는 차이, 뒤 항은 그 정보로도
보정할 수 있는 추정 오류다. 유한한 데이터에서 feature bin 안의 분산을 보았다는
이유만으로 첫 항을 정확히 측정했다거나 더 풍부한 feature도 소용없다고 주장하지 않는다.

종료 event를 제외하는 조건은 현재 수락 결과와 연관될 수 있다. 따라서 위 분산/평균
검사는 all-event 결과도 함께 보고, nonfinal 진단만으로 coin의 편향을 주장하지 않는다.

## 4. 도달 확률만 맞춰도 확장을 잘못 고를 수 있다

현재 leaf u에 c개의 자식을 추가한다고 하자. 자식 토큰이 이미 정해졌을 때 추가 기대 AL은

\[
\Delta\mu(u,c)=\rho(u)\left[1-\prod_{j=1}^{c}(1-\alpha_{u,j})\right].
\]

괄호는 도달한 뒤 적어도 한 자식이 수락될 확률 g(u,c)다. 앞으로 손자도 더 만들면
그 후속 길이의 가치가 추가된다. 따라서 rho가 높아도 다음 문맥의 p/q가 서로
어긋나 있으면 확장 이득이 작다.

한 후보를 **뽑기 전**에는 다음 항등식이 정확하다.

\[
g_1(u)=E_{v\sim q_u}[\min(1,p_u(v)/q_u(v))]
=\sum_v\min(p_u(v),q_u(v))=1-\operatorname{TV}(p_u,q_u).
\]

따라서 p/q를 아는 one-child oracle의 위치 선택 점수는 rho(u)g_1(u)다.
예를 들어 rho(A)=0.8, g(A)=0.1, rho(B)=0.4, g(B)=0.8이면
A/B의 이득은 0.08/0.32라서 B를 골라야 한다. **진짜 rho만 알아도 A를 고르면 틀린다.**

실제 online에서는 아직 확장하지 않은 u의 다음 draft forward와 target 분포가 없으므로
g_1(u)를 무료로 계산할 수 없다. 부모에서 u를 만들 때 이미 얻은 q(u), 부모 q의
entropy, depth, phase, 형제 순서 등으로 g를 예측할 수 있는지는 별도 실험 문제다.
이번 cross-fit g table은 그러한 이전 문맥의 feature만 입력으로 쓴다.

더 엄밀히 online 정보 F에서 필요한 것은

\[
S(u,c)=E[\rho(u)g(u,c)\mid F]
=P(\text{reach }u\mid F)E[g(u,c)\mid F,\text{reach }u].
\]

일반적으로 E[rho|F]E[g|F]와 같지 않다. 다음 단계에서는 rho·g의 조건부 의존성을
다루거나, **추가 AL 자체를 직접 보정**하는 방법도 후보가 된다. Forest 간 배분에서는
해당 root의 실제 사용 확률까지 함께 고려한다. 이번 hit-only 기록으로 미적중 root의
예산 배분을 평가한 것처럼 해석하지 않는다.

## 5. 형제 수와 깊이 배분도 별도 의사결정이다

이미 정해진 형제들에서 j번째 형제가 더하는 값은
rho(u) product_(k<j)(1-alpha_k) alpha_j다. 모든 부모에게 한 명씩 먼저 주고,
그다음 같은 순위로 둘째/셋째를 주는 현재 규칙이 이 값의 최적 배분이라는 보장은 없다.

단일 단계에서 forward/node 비용이 같으면 가장 큰 **추가 기대 AL**을 고른다.
여러 단계와 서로 다른 비용에서는 단순 gain/cost 순위도 일반적으로 정수 예산 최적해가
아니다. 부모 확장 여부의 비용, 자식 수의 비용, 잔여 깊이, round 폭 및 deadline을
제약으로 둔 DP/beam 또는 예산 배분 알고리즘을 평가해야 한다.

더 구체적으로, residual/WOR에서는 g(c)가 항상 오목하다는 보장도 없다.
p=(0,1), q=(.9,.1)이면 g(1)=.1, g(2)=1이다. 둘째 후보의 추가 이득 .9가 첫째 .1보다 크다.
부모 A/B의 rho=.8/.2이고 A의 g=(.1,1), B의 g=(.9,1)이면 새 node 예산2에서
하나씩 주는 이득은 .8*.1+.2*.9=.26, A에 둘 다 주면 .8이다.
단순한 첫 후보 이득 순위의 greedy도 B를 먼저 골라 .26에 머무른다.
이는 synthetic 반례이며 실제 빈도/효과는 저장 분포로 따로 검사한다.

두 후보의 기대값은 vocab² 완전 열거 없이 정확히 계산할 수 있다. A=Σmin(p,q),
R=[p-q]+/(1-A), z_t=[q(t)-p(t)]+라 하면

\[
g(2)=A+\sum_t z_t\sum_v\min\left(R(v),\frac{q(v)\mathbf1\{v\ne t\}}{1-q(t)}\right).
\]

z_t>0인 t에서는 R(t)=0이므로 안쪽 합은 f(1/(1-q(t)))다.
f(c)=Σmin(R(v),c q(v))는 R(v)/q(v)를 정렬한 prefix-R/suffix-q 합으로 계산한다.
따라서 O(V log V)에 g(2)를 얻는다. 단일-support/퇴화 경우는 별도 처리한다.
`fanout.py`는 작은 vocab의 모든 두-token 순열과 대조하고 실제 원본 snapshot에 적용한다.
이 계산에 필요한 미래 p/q는 oracle 진단 정보이므로 online에 그대로 넣는 무료 계산이 아니다.

실제로 사용 가능한 작은 근사도 시험한다. 기존 8문항의 calibration에서 phase별
gamma_phi(1), gamma_phi(2)를 reach 가중 평균으로 구하고, online에는 frozen rho_hat만 사용한다.

\[
\gamma_\phi(c)=
\frac{\sum_{u\in\mathrm{calibration},\,\phi(u)=\phi}\rho(u)g_u(c)}
{\sum_{u\in\mathrm{calibration},\,\phi(u)=\phi}\rho(u)}.
\]

여기서는 이미 생성된 descendant 중 실제로 확장되어 다음 p/q를 아는 위치만 쓴다.
즉 calibration의 관측 expanded 문맥에서 **도달했을 때의** 평균 gain을 근사한다.
모든 미관측 frontier의 모집단 평균을 정확히 알았다는 뜻은 아니다.

\[
\max_{c_u\in\{0,1,2\},\;\sum_uc_u\le B}
\sum_u\widehat\rho(u)\,\gamma_{\phi}(c_u),\qquad\gamma_\phi(0)=0.
\]

두 부모 A,B, 예산2에서 1+1과 2+0을 비교하는 조건은

\[
\widehat\rho(A)[\gamma(2)-\gamma(1)]
>\widehat\rho(B)\gamma(1).
\]

즉 A의 둘째 형제에 쓰는 node와 B의 첫째 형제에 쓰는 node의 **추가 기대 AL**을 비교한다.
기존 단순 breadth 우선 규칙과 다르게, node마다 무엇을 포기하고 무엇을 얻는지 계산한다.
Calibration에서 target을 쓰지만 online 점수에 미래 target 분포를 넣지 않는다.
다만 phase 평균 g와 rho를 분리하는 근사이며, 관측된 expanded-node 분포에서 추정한 값이다.
실제 C=3/다중 round 전체 알고리즘으로 확대하려면 gamma(3), 잔여 깊이의 가치,
전체 frontier·root quota를 함께 다뤄야 한다. 여기서는 두-node 국소 결정을 검증한다.

현재 구현에는 depth==round, root별 quota, future-round node reserve, q threshold도
있다. 점수만 바꾸면 이 제약들로 제외된 후보를 되살릴 수 없다. 이번 비교는 이
제약을 바꾸지 않았으므로 완전한 tree 최적화 결과가 아니다.

## 6. 이번 oracle 분석이 말해 주는 것과 말해 주지 않는 것

고정된 샘플 pool에서 ancestor와 ordered-sibling prefix를 지킨 subset A는
기존 노드의 rho를 유지하므로 그 tree의 조건부 기대 AL은 sum_(v in A)rho(v)다.
이를 DP로 최대화하면 그 pool 안에서의 oracle subset을 얻는다.

- 같은 점수의 greedy와 DP 차이: 그 pool에서 알고리즘 선택으로 남은 차이.
- 같은 DP의 추정 점수와 oracle rho 차이: 그 pool에서 순위 추정으로 남은 차이.
- 보정 점수 MSE가 감소해도 subset AL이 증가하지 않을 수 있다. 결정 경계에서의
  순서가 중요하고, 이미 모두 보존하는 노드의 확률을 잘 맞춰도 선택은 바뀌지 않는다.

이는 원래 sampling 과정에서 token 값에 따라 사후 삭제해도 lossless라는 뜻이 아니다.
기존 분석은 G>M rerank가 실제 proposal 조건부분포를 바꾸는 반례를 확인했다.
따라서 여기의 DP는 **진단용**이고 실제 개선은 proposal draw 전의 예산/확장 결정에
적용해야 한다. 이번 수집은 G=M이다.

또한 관측하지 않은 대안 가지의 p/q, 미적중 root의 상태, 다른 스케줄에서 달라질
candidate pool은 이 기록에 없다. 고정-pool oracle과 관측된 expanded-node oracle은
전역 최적 tree나 실제 달성 가능한 AL 증가량이 아니다.

## 7. 원래 못 맞추는가

q=(0.9,0.1), 첫 후보 v=A를 고정해도 p=q이면 alpha=1이고 p=(0,1)이면 alpha=0이다.
q, q entropy, sibling, depth가 같아도 정답이 달라질 수 있다. 따라서 **이 feature들만으로
모든 target/context에서 정확한 점수를 보장하는 수식은 없다.**

그러나 이는 더 좋은 근사가 불가능하다는 정리가 아니다. 실제 workload에서 보정이
가능한 규칙성이 있는지 cross-fitting으로 검사하고, 이어 독립적인 online 생성에서
AL 향상을 검증해야 한다. 원문 context/hidden state/추가 target 정보를 사용한다면
정보 집합 자체가 달라진다. 그 비용과 사용자의 새 모델 학습을 원하지 않는 방향을
함께 고려해야 한다.

선택 문제 자체에서도 oracle gap을 둘로 나눌 수 있다. F를 결정 전에 아는 정보,
G_a를 합법적인 예산 배분 a의 실제 조건부 기대 gain이라고 하면

\[
V_{oracle}=E[\max_aG_a],\qquad
V_F^*=E[\max_aE[G_a\mid F]].
\]

현재 정책 a_hat(F)에 대해

\[
V_{oracle}-E[G_{\hat a(F)}]
=\underbrace{V_{oracle}-V_F^*}_{\text{정보 부족에 따른 차이}}
+\underbrace{V_F^*-E[G_{\hat a(F)}]}_{\text{추정·선택 알고리즘의 차이}}.
\]

두 항 모두 음수가 아니다. Oracle 차이가 있다고 해서 그것을 전부 현재 입력 F만으로
회복할 수 있다는 뜻이 아니다. 이번 유한한 보정 실험으로 V_F^*를 정확히 알아낸 것도 아니다.

예를 들어 같은 q=(.5,.5), 같은 후보 순서 A,B에서 target이 (1,0)/(0,1)로 반반이고
이를 구분하는 정보가 F에 없다면, 어느 root-child를 확장해야 하는지도 반반이다.
확장 후 이득이 1로 같고 한 곳만 확장할 수 있으면 F만 쓰는 최선의 평균 gain은 .5,
target을 아는 oracle은 1이다. 이 차이는 scalar 보정량을 늘리는 것만으로 사라지지 않는다.

## 8. 관측되지 않은 가지를 다음에 어떻게 검사할 것인가

현재 hit-only trace에서 확장되지 않은 leaf의 다음 q는 없다. 이 부분은 물리적으로
예측 불가능하다고 결론내릴 문제가 아니라, **이번 기록으로는 반사실을 식별할 수 없는 문제**다.

후속 진단은 고정 prefix에서 추가 가지를 shadow 생성해 target으로 평가하거나,
작은 super-tree를 만들고 후보 정책을 causal replay하는 방식이 가능하다. 정책은 실제
online 시점에 알 수 있는 feature만 보고 다음 확장/예산을 결정하고, 아직 드러나지 않은
자식 token이나 oracle p를 보지 않아야 한다. 그 뒤 exact AL로 정책을 평가한다.

같은 prefix/candidate random stream을 사용하는 이런 비교는 서로 다른 생성 경로의
평균 AL만 보는 것보다 무엇 때문에 선택이 달라졌는지 명확히 해 준다. 그래도 원래
시스템의 전체 round 폭, root quota, phase 도착 시점은 별도 반영해야 한다.
이번 보고서가 그러한 전 frontier의 counterfactual 평가까지 끝냈다고 주장하지 않는다.
