# DUET tree의 목적함수와 안전한 구성 규칙

2026-09-21. 수식은 현재의 ordered without-replacement(WOR) draft sampling과
residual-ladder verification을 기준으로 한다. 구현·실험은 [REPORT.md](REPORT.md).

## 1. 무엇을 최적화하는가

Root 후보를 고르는 일과 그 root 아래 tree를 만드는 일은 다르다.

- Root (r=(i,v)): 진행 중인 target 검증이 실제로 이 correction/context로 끝날 사건.
- \(\pi_r\): 그 사건의 확률. 기존 proxy 후보 점수는 이 확률의 추정치이다.
- Root 아래 노드: root가 실제로 적중했을 때, 다음 target 검증에서 수락될 draft token.
- \(A(T_r)\): root 자체와 마지막 recovery/bonus를 제외한 **수락 descendant 수**.

같은 시간·같은 root 집합이라면 유용한 목적함수는

\[
\sum_r\pi_r\mathbb E[A(T_r)\mid r\text{ hit}].
\]

Root hit가 확인된 뒤에는 해당 tree에 대해 \(\pi_r=1\)이다. 이번 full-distribution
probe는 실제 적중한 tree만 관측하므로, hit 이전 전체 forest의 \(\pi_r\) 품질이나
한 번도 적중하지 않은 root에 낭비한 연산량까지 추정하는 자료는 아니다.

실제 시스템에서는 miss 시 JIT draft도 토큰을 낸다. 따라서 cache의 순이득은
\(\pi_r\{E[A(T_r)]-E[A_{\mathrm{miss},r}]\}\)와 시간 차이로 평가해야 한다.
P1/P2가 같은 cache key를 저장할 수 있다면 사건을 중복 계산하지 않는다.

## 2. 논문의 점수는 무엇인가

사용자 논문 4절 식 (4):

\[
s_{\mathrm{old}}(v_d)
=\widehat\pi_r\prod_{\ell=1}^{d}q(v_\ell\mid r,v_{<\ell}).
\]

이것은 root 점수에 **draft가 그 경로를 생성할 확률**을 곱한 값이다.
단일 경로를 q에서 순차적으로 한 번씩 뽑는 경우에는 경로 생성 확률이라는 해석이
맞다. 여러 형제를 WOR로 뽑는 현재 tree에서 두 번째 이후 형제의 proposal은
제외·재정규화된 분포이므로 원래 q의 곱이 그 표본 사건의 확률과도 같지는 않다.

더 중요한 차이는 **생성 확률과 검증 수락 확률이 다르다**는 것이다.
q를 WOR proposal로 바꿔 곱하는 수정만으로는 아래 수락 목적함수가 되지 않는다.

## 3. 현재 검증을 그대로 수식화한다

하나의 parent context u에 자식 \(v_1,\ldots,v_m\)이 표본 순서대로 있다.
\(t_j\)는 자식의 token이다. 처음에는

\[
R_{u,1}=p_u,\qquad D_{u,1}=q_u.
\]

j번째 자식을 **실제로 시도했을 때**의 조건부 수락 확률은

\[
\alpha_{u,j}=\min\left(1,\frac{R_{u,j}(t_j)}{D_{u,j}(t_j)}\right).
\]

그 자식이 거절되면 target residual과 다음 draft proposal을 각각 갱신한다.

\[
R_{u,j+1}(x)=\frac{[R_{u,j}(x)-D_{u,j}(x)]_+}
 {\sum_z[R_{u,j}(z)-D_{u,j}(z)]_+},
\]

\[
D_{u,j+1}(x)=\frac{D_{u,j}(x)\mathbf1\{x\ne t_j\}}
 {1-D_{u,j}(t_j)}.
\]

분모가 0인 residual 갱신은 정상 확률 과정에서는 도달 질량이 0인 거절 분기다.
코드의 수치 fallback과 구분해서 이해해야 한다. 양의 질량으로 시도하는 token의
proposal 확률은 양수여야 한다.

첫 형제에서는 p/q이지만, 이후 형제에서는 **갱신된 R/D**다. 모든 형제에서
원래 p/q를 반복해서 사용하는 모델은 현재 verifier와 다르다.

## 4. 노드의 정확한 가치는 ‘도달해서 수락되는 확률’이다

실현된 tree의 token, parent, sibling order, p, q가 고정되어 있다고 하자.
\(\rho(v)\)를 남아 있는 verification coin의 무작위성에 대한 노드 수락 확률로
정의한다. Root context에는 \(\rho(r)=1\)을 둔다.

\[
\boxed{\rho(v_j)=\rho(u)
 \underbrace{\prod_{k<j}(1-\alpha_{u,k})}_{\text{앞선 형제들이 모두 거절}}
 \underbrace{\alpha_{u,j}}_{\text{현재 형제 수락}}.}
\]

이 식은 독립성을 임의로 가정해서 marginal 확률을 곱한 것이 아니다.
각 \(\alpha_{u,j}\)는 앞선 거절에 따라 residual을 갱신한 **조건부** 확률이고,
조건부 확률의 연쇄 법칙을 적용한 것이다. 모델 p/q와 실현 tree를 고정한 상태다.

노드 v의 수락 indicator를 \(I_v\)라 하면

\[
A(T)=\sum_{v\in T}I_v,\qquad
\boxed{\mathbb E[A(T)\mid T,p,q]=\sum_{v\in T}\rho(v).}
\]

이는 기대값의 선형성으로 정확하다. 노드들의 독립성은 필요 없다.
따라서 cache hit 이전 node reward는 \(\pi_r\rho(v)\)다.

모든 자식이 거절되어 context u에서 끝나는 확률은
\(h(u)=\rho(u)\prod_j(1-\alpha_{u,j})\)이다. 자식이 없는 leaf에서는
\(h(u)=\rho(u)\). 전체 h의 합은 1이다. 이 관계와
\(\sum_v\rho(v)=\sum_u h(u)\operatorname{depth}(u)\)도 구현 검사에 사용했다.

## 5. 작은 예: 왜 q 경로 곱과 다를까

모든 context에서 p=q=(0.4,0.35,0.25)라고 하자. Root 아래 첫 두 표본이
token0, token1이고, 첫 자식 아래 두 표본이 token1, token0인 tree를 생각한다.

| 노드 | 기존 q 경로 점수 | 정확한 도달·수락 확률 |
|---|---:|---:|
| root의 첫 자식 | 0.40 | 1 |
| root의 둘째 자식 | 0.35 | 0 |
| 첫 자식 아래 첫 자식 | 0.14 | 1 |
| 첫 자식 아래 둘째 자식 | 0.16 | 0 |

p=q이면 첫 proposal은 어떤 token이든 수락된다. 따라서 둘째 형제는 시도되지
않는다. 그런데 기존 점수는 마지막 행을 그 위 행보다 높게 평가한다.
이 반례는 q 점수가 항상 나쁘다는 뜻이 아니라, 수락 길이와 동일한 목적함수라는
보편적 정당화가 성립하지 않는다는 뜻이다.

## 6. 실제 inference에서 쓸 수 있는 개선 점수

미래의 모든 draft context에 대한 target p는 온라인에서 무료로 주어지지 않는다.
현재 검증 중인 context의 early-exit e를 아직 target이 계산하지 않은 모든 미래
분기의 p 대용으로 그대로 사용할 수도 없다. 따라서 4절은 정확한 사후 평가와
oracle용이고, 실제 구성에는 \(\widehat\alpha\)가 필요하다.

이번 prototype은 새 neural model/head 없이, calibration의 full p/q에서 계산한
정확한 \(\alpha\)를 phase·형제 순서·q 구간별로 모았다. 구간 b의 추정은

\[
\widehat\alpha_b=
 \frac{\sum_{n\in b}w_n\alpha_n+\lambda\mu_{\mathrm{parent}}}
 {\sum_{n\in b}w_n+\lambda},\qquad
w_n=P(\text{node n is attempted}),\quad\lambda=5.
\]

시도조차 하지 않은 형제를 거절 표본 0으로 잘못 넣지 않는다. \(\lambda\)는
관측 개수가 아니라 **기대 시도 질량** 5에 해당하는 shrinkage다.
global → phase → phase/sibling → phase/sibling/q-bin으로 backoff한다.
이는 학습된 추가 언어모델은 없지만, 데이터에 맞춘 scalar calibration이라는
사실은 명시해야 한다. ‘아무 fitting도 없는 식’이라고 부를 수는 없다.

실제 교체한 점수는

\[
\boxed{s_{\mathrm{new}}(v_j)=s_{\mathrm{new}}(u)
 \left[\prod_{k<j}(1-\widehat\alpha_{u,k})\right]
 \widehat\alpha_{u,j},\quad s_{\mathrm{new}}(r)=\widehat\pi_r.}
\]

CUDA graph 내부에서는 log로 계산한다.

\[
\log s(v_j)=\log s(u)+\log\widehat\alpha_{u,j}
 +\sum_{k<j}\log(1-\widehat\alpha_{u,k}).
\]

이 점수는 **앞으로 어느 node 아래를 더 생성할지**에만 사용한다. 이미 뽑은
token의 raw q, proposal sampler, verifier의 R/D를 이 점수로 바꾸지 않는다.
점수 추정이 틀리면 성능이 나빠질 수 있지만, 그것만으로 acceptance 확률을
틀리게 만드는 방식은 아니다. 다만 기존 엔진 전체의 losslessness를 새로
인증한 결과라는 뜻은 아니다.

Calibration 평균을 곱하면 정확한 기대 도달 확률이 된다는 보장도 없다.
Context별 수락률의 상관, sparse bin, 정책 변경에 따른 context 이동 때문에
\(E[\prod\alpha]\ne\prod E[\alpha]\)일 수 있다. 실제 검증이 필요한 이유다.

## 7. 노드 점수만 높이면 최적 tree가 되는가

아니다. 노드 자체의 수락과 **그 노드를 확장해서 추가로 얻는 수락 길이**는
다르다. 현재 leaf u에 새로운 자식 c개를 한 단계 추가한 실현 결과의 이득은

\[
\Delta A(u,c)=\rho(u)\left\{1-\prod_{j=1}^c(1-\alpha_{u,j})\right\}.
\]

아직 자식을 뽑기 전이라면 위 중괄호에 대해 future proposal을 평균 낸 값
\(g(u,c)\)가 필요하다. 따라서 더 완전한 확장 점수는

\[
\boxed{\widehat\pi_r\widehat\rho(u)\widehat g(u,c)}
\]

이며 비용이 다르면 \(\widehat\pi_r\widehat\rho(u)\widehat g/\Delta C\)를
휴리스틱 우선순위로 쓸 수 있다. 같은 c·같은 continuation 품질·같은 비용이면
\(\widehat\pi_r\widehat\rho(u)\) 순위로 단순화된다. 이번 실행 prototype은
이 단순화만 시험했으며 g나 비용 모델까지 최적화했다고 주장하지 않는다.

예를 들어 같은 root 아래 두 leaf의 reach가 0.6과 0.3이고, 한 자식을 만들 때
그 자식의 조건부 수락 가능성이 각각 0.1과 0.8이라고 하자. Reach만 보면 첫
leaf가 좋지만 한 forward의 기대 이득은 0.06 대 0.24다. 두 번째 leaf를
확장해야 한다. **정확한 node reach를 알아도 확장 순위는 틀릴 수 있다.**

형제를 추가할 때의 한계 이득은
\(\rho(u)\prod_{k<j}(1-\alpha_{u,k})\alpha_{u,j}\)다. 따라서 모든 parent에
무조건 같은 C=3을 주기보다, 깊이를 늘릴 가치와 두 번째·세 번째 형제를 늘릴
가치를 비교해야 한다. 단 이 비교에서 **새로 뽑을 형제의 token을 미리 보고
포함 여부를 정하면 안 된다**. 사전 예측/예산으로 결정한다.

CUDA graph의 fixed width 때문에 논리 node를 조금 줄였다고 latency가 비례해서
줄지는 않는다. 이 경우 한 graph slot의 기회비용을 비교하거나 width/verify
bucket을 넘는 변화의 비용을 실측해야 한다. 단순한 value/cost greedy가 모든
선행·예산·deadline 제약의 전역 최적해라는 보장은 없다.

### Sampling 전에 shape를 정하는 DP도 가능하다

사후 token 선택을 피하는 다른 설계는 **shape를 먼저 결정**하는 것이다.
어떤 phase/depth에서의 조건부 형제 수락률을 \(a_1,a_2,\ldots\)로 예측하고
\(b_j=a_j\prod_{k<j}(1-a_k)\)라고 하자. 남은 depth h, 저장할 child-node
예산 B, draft parent-forward 예산 Q에 대해 다음 대리 모델을 쓸 수 있다.

\[
V(h,B,Q)=\max_{m,B_j,Q_j}\sum_{j=1}^{m}
 b_j\{1+V(h-1,B_j,Q_j)\},
\]

\[
m+\sum_j B_j\le B,\quad 1+\sum_jQ_j\le Q,\quad m\le C.
\]

h=0 또는 B=0 또는 Q=0이면 V=0이다. Parent forward 한 번으로 m개 child
proposal을 만들고, 그 아래 예산을 나누는 재귀다. 노드를 생성할 때 원래 q에서
정해진 m개를 WOR로 뽑고 모두 보존한다. Root 간에는 \(\widehat\pi_r V_r\)를
보상으로 예산을 나눌 수 있다. 이는 Sequoia 계열의 사전 shape 최적화와 연결된다.

단 위 식의 ‘최적’은 **예측 b와 총 forward 예산이라는 대리 모델 안에서**의
최적이다. 실제 DUET는 매 round 폭 W와 phase deadline이 있으므로 총 Q 제약만
으로 실행 가능성을 보장할 수 없다. 각 depth/round의 query 수 벡터까지 상태로
넣거나, 작은 후보 shape들의 실제 schedule을 검사해야 한다. Context별 b가
다르다는 문제도 남는다. 이번에 실행한 prototype은 이 shape DP가 아니다.

## 8. 실현된 tree 안에서의 최적 사후 선택

관측된 후보 pool의 품질을 비교하려고 exact DP를 구현했다. 포함하는 node의
모든 조상과 각 단계의 앞선 형제를 함께 포함하는 subset만 허용한다.
\(F_u(B)\)를 u 아래 최대 B개 node의 최대 reward 합이라고 하면

\[
F_u(B)=\max_{k,b_1,\ldots,b_k}
 \sum_{j=1}^k\{w(v_j)+F_{v_j}(b_j)\},\quad
 \sum_{j=1}^k(1+b_j)\le B.
\]

자식은 순서상 prefix \(v_1,\ldots,v_k\)만 선택한다. w=q-path이면 기존
대리 목적함수의 정확한 해, w=추정 reach이면 개선 점수의 해,
w=정확한 \(\rho\)이면 **현재 후보 pool에서** 가능한 최적 사후 수락 길이다.
선택한 node 이전의 조상·형제가 유지되므로 해당 node의 \(\rho\)는 원래
tree와 같다. 이것이 reward를 그대로 더할 수 있는 이유다.

이 oracle은 만들지 않은 다른 token/분기에 대한 전역 최적 tree가 아니다.
또한 **이 사후 선택을 현 residual verifier에 그대로 배포해서는 안 된다.**
아래 분포 보존 문제가 별개로 존재한다. DP는 진단용이다.

### 예측 MSE 감소가 tree 개선을 보장하지 않는 수학적 이유

같은 고정 pool과 같은 feasible subset 집합에서, 모든 node reward의 추정
오차가 \(|w_v-\widehat w_v|\le\epsilon\)이고 최대 B개를 선택한다고 하자.
정확한 reward의 최적 subset을 S*, 추정 reward의 최적 subset을 S_hat이라 하면

\[
W(S^*)-W(\widehat S)\le 2B\epsilon.
\]

증명은 \(\widehat W(S^*)\le\widehat W(\widehat S)\)를 중간에 넣고, 두 subset의
reward 오차를 각각 B epsilon으로 묶으면 된다. Closure 제약이 있어도 두
subset의 feasible 집합이 같으면 성립한다.

이번 MSE 감소는 이런 **모든 node의 오차 상한**을 보장하지 않는다. 게다가
온라인 생성 정책이 바뀌면 pool과 방문 context도 바뀌고, greedy 확장은 추정
reward의 전역 최적 DP도 아니다. 따라서 낮아진 MSE에서 AL 개선 정리를 바로
얻을 수 없다. 다음 calibration 목표는 전체 node MSE뿐 아니라 실제로 경쟁하는
parent들의 **확장 이득 순위와 선택 regret**여야 한다.

## 9. Closure만으로는 사후 pruning의 losslessness가 보장되지 않는다

한 proposal X~q를 받아들일 때 p/q를 쓰는 증명은, 시도 시점의 X의 조건부
분포가 실제 q라는 사실을 사용한다. X를 본 후 score에 따라 시도할지 버릴지
결정하면 \(P(X=x\mid\text{retained})\)는 일반적으로 q(x)가 아니다.
조상·앞선 형제를 보존하는 것은 tree walk의 구조적 필요조건이지만 이 조건부
분포까지 보존하는 충분조건은 아니다.

현재 production `rerank_tree_indices`와 `tree_verify_walk_tensor`를 그대로
로드해 다음 작은 상태 공간을 전부 계산했다.

- p=(0.4,0.3,0.3), q=(0.6,0.25,0.15).
- Root에서 두 형제를 WOR로 뽑고 높은 q의 형제 아래 한 자식을 추가한다.
- 그 context의 q'=(0.3,0.3,0.4). 총 생성 G=3, 최종 검증 cap M=2.
- 18개 tree 표본과 모든 verification coin 분기를 열거한다.

| 선택 규칙 | 첫 출력의 주변분포 |
|---|---|
| 3개 모두 유지 | (0.4, 0.3, 0.3) |
| 생성 순서상 첫 2개를 내용과 무관하게 유지 | (0.4, 0.3, 0.3) |
| 현 production q-score rerank | **(0.4, 0.31875, 0.28125)** |

마지막 행의 TV는 **0.01875**이다. Monte Carlo 오차가 아니라 유리수 정확
열거 결과이며, float tensor verifier를 직접 분기 열거한 결과도 일치한다.
이는 실제 LLM에서 편향 크기를 측정한 실험은 아니고, 현재 알고리즘의 보편적
lossless 주장에 대한 구현 수준의 반례다. G=M 경로나 기존 chain 결과까지
동일하게 잘못되었다는 뜻이 아니다.

안전한 구성의 충분조건은 각 context에서 ‘다음 proposal을 시도할지/몇 개를
예산으로 둘지’를 **그 proposal을 보기 전에** 결정하고, 시도하는 proposal이
그때의 올바른 q 또는 WOR D를 따르게 하는 것이다. 이전에 관측한 부모 token과
그 수락 가치로 미래 자식 생성을 정하는 것은 허용된다. 실현된 자기 token을
보고 자기 자신을 포함할지 정하는 것은 다른 문제다. 서로 다른 분기의 미리
계산된 token도 선택 후 조건부 proposal을 바꿀 수 있으므로 일반적인 arbitrary
lookahead pruning을 안전하다고 확장 해석하지 않는다.

이번 prototype은 기존의 sampling **이전** fanout 결정 순서를 유지하고 G=M을
강제한다. 변경점은 이미 보존된 parent 중 어디에 다음 연산을 쓸지뿐이다.

## 10. 자유로운 사후 pruning이 꼭 필요하다면

Target-only matching은 별도 선택지다. 각 도달 context에서 새로 X~p를 뽑고,
그 token이 tree 자식에 있으면 계속하고 없으면 출력 후 끝낸다. 새 target
표본과 tree의 독립성을 유지하면 arbitrary tree에도 분포가 보존된다.
이 경우 고정 tree의 도달 확률은 target 경로 확률 곱이며, q≈p라면 q-path는
자연스러운 근사다. 하지만 현재 residual coupling과는 다른 verifier다.

예를 들어 자식 1개, p=q이고 독립 target-matching이면 평균 match 확률은
\(\sum_x p(x)^2\), residual 검증이면 수락률은 1이다. 따라서 verifier를
바꾸면 기존 점수·AL·성능 비교도 다시 해야 한다.

## 11. Calibration 및 논문 주장과의 연결

시간당 진전은 대략 \(J=E[U]/E[T]\)로 평가한다. 추가 tree 연산의 보상·시간
증가가 \(\Delta U,\Delta T>0\)라면 J가 증가할 조건은
\(\Delta U/\Delta T>J\)이다. 실제 overlap은 max 형태의 timing과 CUDA graph
bucket을 포함하므로, 독립 node latency를 단순 합산하지 않는다.

따라서 전체 설계의 순서는 root 사건 확률 → 조건부 tree 수락 가치 → causal
확장/fanout → deadline과 검증 비용 → 실측 TPS다. 이번 연구가 수식적으로
확정한 것은 목적함수와 편향 반례이며, scalar estimator의 일반적 우위나
Mirror-SD 전체 시스템 대비 성능 우위는 아니다.

관련 1차 문헌:

- [EAGLE-2](https://arxiv.org/html/2406.16858v2): draft confidence와 acceptance의
  경험적 관계에 기반한 동적 tree와 경로 점수. 그 모델에서의 관측 관계를 다른
  draft와 verifier의 항등식으로 가져오면 안 된다.
- [Sequoia](https://arxiv.org/html/2402.12374v3): 수락 토큰 기대값과 tree 최적화,
  residual sampling과 target matching의 구별. 이번 식은 현재 DUET의 실현
  token별 residual ladder에 맞춘 것으로, 일반적인 기대 수락 목적함수 자체를
  새로운 발견이라고 주장하지 않는다.
- [D-cut, Appendix B](https://arxiv.org/html/2607.14647v1): 현재 token confidence를
  보고 유지 여부를 정하는 것의 proposal 편향과 causal stopping의 조건을
  다룬다. 위 반례는 이 저장소의 함수를 대상으로 별도로 재현한 결과다.
