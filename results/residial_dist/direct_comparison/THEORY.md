# Residual과 only-proxy를 직접 비교하는 수학

추가 개선 분석은 [training_free/THEORY.md](../training_free/THEORY.md),
실제 검증 결과는 [training_free/REPORT.md](../training_free/REPORT.md)에 있다.
기존 직접 비교 정리에 위치 배분, 상대 민감도 제한, 실제 wire 동점 검증을 덧붙였다.

2026-09-12. 아래 정리들은 별도 표기한 경우를 제외하면 **한 위치, 같은 후보 비용,
같은 후보 수 k, 전체 vocabulary**에 대한 것이다. 실제 엔진의 sampled-token 제외,
위치별 top-M 재정규화, 전역 예산 배분은 별도로 평가한다.

**1. 비교 대상과 목적함수**

Target, draft, early-exit proxy를 각각 p, q, e라 두자.

\[
a_v=[p_v-q_v]_+,\quad b_v=[e_v-q_v]_+,
\quad Z=\sum_v a_v,\quad\widehat Z=\sum_v b_v.
\]

Z와 Zhat이 양수인 경우 실제 correction 분포와 추정 분포는

\[
R_v=a_v/Z,\qquad \widehat R_v=b_v/\widehat Z.
\]

Z=0이면 이상적인 SD에서 그 위치의 unconditional rejection 확률이 0이다.
Zhat=0이면 추정 residual 정규화가 정의되지 않으므로 fallback 규칙이 필요하다.
이 경우에 양의 정규화 상수를 가정한 정리를 적용하면 안 된다.

후보 집합 S가 실제 correction token을 포함할 확률은

\[
C(S)=\Pr(c\in S\mid\text{rejection})=\sum_{v\in S}R_v.
\]

따라서 직접 비교는 S_r=Top-k(b), S_e=Top-k(e)에 대해

\[
\boxed{\Delta=C(S_r)-C(S_e)
=R(S_r\setminus S_e)-R(S_e\setminus S_r).}
\]

공통 후보는 차이에서 사라진다. residual이 새로 넣은 후보의 실제 correction 질량이
뺀 후보의 질량보다 커야 이긴다. TV(p,e)만 비교하거나 entropy 차이만 비교해서는
이 부호를 알 수 없다. **우리의 평가 대상은 target 자체의 다음 토큰이 아니라,
현재 SD에서 rejection 이후 나오는 correction token**이라는 점이 출발점이다.

**2. Proxy가 정확할 때의 우위와 그 한계**

S*=Top-k(R)는 k개 후보 중 C를 최대화한다. e=p이면 S_r=S*이므로

\[
G_k=R(S^*)-R(\mathrm{Top}_k(p))\geq0.
\]

즉 정확한 proxy를 얻으면 residual 선택은 이 목적함수에서 최적이다. 엄격한 우위는
선택 집합의 차이가 실제 R 질량 차이를 만들 때 생긴다. 동률이면 같은 성능이다.

예를 들어 p_w>p_v라서 target-only는 w를 더 좋아하더라도, 두 residual이 양수이고

\[
q_w-q_v>p_w-p_v
\]

이면 p_v-q_v>p_w-q_w여서 correction에는 v가 더 중요하다. **draft가 이미 충분히
제안하는 토큰을 뒤로 보내고, draft가 부족하게 제안하는 토큰을 앞으로 보내는 것**이
residual의 이익이다.

Z가 크다는 것 자체는 이 이익을 보장하지 않는다. p=(0.6,0.4,0), q=(0,0,1)이면
Z=1이지만 R=p다. 이 경우 residual과 target-only의 순위는 같다. 따라서
“draft와 target 차이를 무조건 키우자”는 결론도 나오지 않는다. draft를 나쁘게 만들면
acceptance와 전체 속도가 떨어질 수 있다.

**3. 기존 TV 상한이 말하는 것과 직접 비교에 필요한 추가 항**

\[
\delta=\operatorname{TV}(p,e),\qquad
B=\min\left(1,\frac{\delta}{\max(Z,\widehat Z)}\right).
\]

앞에서 증명한 결과는 TV(R,Rhat)≤B다. 요점은 clipping이 정규화 전의 한쪽 오차를
키우지 않지만, 작은 질량으로 나누면 정규화 후 오차가 커질 수 있다는 것이다.
이것만으로 TV(R,Rhat)와 TV(R,e)의 대소관계가 결정되지는 않는다.

삼각부등식을 함께 쓰면

\[
\operatorname{TV}(R,e)\geq\operatorname{TV}(R,p)-\delta.
\]

따라서

\[
\boxed{B<\operatorname{TV}(R,p)-\delta}
\]

이면 residual이 only-proxy보다 **분포 TV에서** 가깝다는 충분조건을 얻는다.
오른쪽의 TV(R,p)는 target 자체를 correction 분포 대신 사용해서 생기는 차이다.
이 조건이 실패했다고 only-proxy가 이긴다는 뜻은 아니다. 충분조건은 보수적이다.

후보 목적함수로 옮기면, S_r가 Rhat의 top-k라는 사실 때문에

\[
\begin{aligned}
R(S^*)-R(S_r)
&=[R(S^*)-\widehat R(S^*)]
 +[\widehat R(S^*)-\widehat R(S_r)]
 +[\widehat R(S_r)-R(S_r)]\\
&\leq 2\operatorname{TV}(R,\widehat R)\leq2B.
\end{aligned}
\]

중간 항은 Rhat의 top-k 최적성 때문에 0 이하이고, 나머지 두 집합의 확률 오차는
각각 TV 이하이다. 그러므로

\[
\Delta\geq\underbrace{R(S^*)-R(S_e)}_{G_e}-2B.
\]

**G_e>2B이면 후보 coverage에서도 residual이 이긴다.** 여기서 G_e는 *현재 실제
proxy 후보*를 기준으로 한 개선 여지다. 2절의 *정확한 target 후보*를 기준으로 한
G_k와 다른 값이다. 두 값을 혼용하면 안 된다.

실제 구현처럼 y를 먼저 제외했다면 Rhat 대신 제외 후 정규화한 후보 분포를 사용하여
regret≤2TV를 적용할 수 있다. 그러나 제외 전 분포에 대해 구한 B를 그대로 이 TV의
상한으로 쓰면 안 된다. 전체 vocabulary 상한과 구현상 제외 후 TV를 분리해 측정했다.

**4. 정규화 오차보다 직접적인 순위 조건**

ε_v=e_v-p_v, η=max_v|ε_v|라고 하자. clipping의 비팽창성으로
|b_v-a_v|≤|ε_v|≤η이다. 실제 residual 점수의 k번째와 k+1번째 차이가

\[
\boxed{a_{(k)}-a_{(k+1)}>2\eta}
\]

이면 top-k 집합은 보존된다. 안쪽 토큰은 최대 η 내려가고 바깥 토큰은 최대 η 올라가도
경계가 뒤집히지 않기 때문이다. 이때 residual은 oracle 후보 집합을 그대로 선택한다.
단, η가 작다는 것만으로 충분하지 않다. 후보 경계의 실제 점수 차이에 비해 작아야 한다.

또한 현재 두 후보 집합의 추정 residual 점수 차이를

\[
\Gamma=b(S_r)-b(S_e)\geq0
\]

라고 하면, 두 집합의 대칭차에서만 오차가 작용하므로

\[
\left|[a(S_r)-a(S_e)]-\Gamma\right|
\leq\min(2\delta,2k\eta).
\]

첫 번째 상한은 전체 |a-b|의 합이 |p-e|의 합=2δ 이하이기 때문이고,
두 번째 상한은 최대 2k개 토큰에 각각 η만큼 오차가 있기 때문이다. 따라서

\[
\boxed{\Gamma>\min(2\delta,2k\eta)\ \Longrightarrow\ \Delta>0.}
\]

이 조건은 같은 후보 집합이 허용되는 한 y 제외 후에도 쓸 수 있다. 하지만 δ,η는
full target을 알아야 측정할 수 있다. 따라서 이번 분석에서는 진단용이다. 실행 중
사용하려면 별도로 검증한 오차 상계나 예측기가 필요하다. 평균 δ를 모든 context의
상계로 대입해서 “보장”이라고 부를 수는 없다.

**5. 언제 0으로 지우거나 순서를 틀리는가**

실제로 필요한 토큰 a_v>0을 b_v=0으로 지우는 조건은

\[
\boxed{\epsilon_v\leq-(p_v-q_v).}
\]

즉 target 확률을 과소평가한 정도가 그 토큰의 실제 residual 점수보다 크면 지워진다.
전체 Z가 충분히 커도 어떤 중요한 토큰의 p_v-q_v는 작을 수 있다.

두 토큰이 clipping 이후에도 살아 있다면

\[
b_v-b_w=(a_v-a_w)+(\epsilon_v-\epsilon_w).
\]

따라서 순위 오류는 entropy가 아니라 **해당 토큰 쌍의 오차 차이**와 **실제 residual
순위 간격**에 의해 정해진다. 모든 토큰에 대한 평균 KL/TV를 줄이는 것이 이러한
중요 후보들의 순위를 개선한다는 보장도 없다.

이를 정확히 계측하기 위해, b>0인 토큰 안에서 true R로 가장 잘 고른 k개 후보의
coverage를 C_support라 두자. 구현이 0점 토큰을 채우면 실제 선택된 0점 토큰도
허용 집합에 포함한다. 그러면 항상

\[
\begin{aligned}
L_{\rm support}&=C(S^*)-C_{\rm support}\geq0,\\
L_{\rm rank}&=C_{\rm support}-C(S_r)\geq0,\\
\boxed{\Delta&=G_e-L_{\rm support}-L_{\rm rank}.}
\end{aligned}
\]

첫 손실은 후보를 지워서 생긴, 현재 예산 k에 관한 손실이다. 두 번째 손실은 살아 있는
후보 안에서도 순위를 틀려서 생긴 손실이다. “지운 토큰 전체의 R 질량”과 첫 손실은
다르다. 전체 질량에는 원래 k개 예산으로 고르지 못할 꼬리 토큰도 포함되기 때문이다.

**6. e,q만 알면 항상 이기는 규칙을 만들 수 있는가**

일반적으로 불가능하다. e=(0.40,0.35,0.25), q=(0.39,0.10,0.51)을 고정하자.
only-proxy는 A, residual은 B를 top-1으로 고른다.

| 가능한 target | 실제 residual R | 승자 |
|---|---|---|
| p₁=(0.49,0.10,0.41) | (1,0,0) | only-proxy |
| p₂=(0.39,0.20,0.41) | (0,1,0) | residual |

두 상황에서 관측한 e,q는 완전히 같다. draft가 C를 제안한 경우도 두 target 모두에서
실제로 reject될 수 있다. 그러므로 e,q에만 근거한 동일한 선택으로 양쪽에서 모두
이길 수는 없다. 실질적인 우위 주장은 proxy 오차에 대한 조건 또는 새로운 데이터의
평균 성능으로 뒷받침해야 한다.

**7. 개선식의 의미**

뺄셈을 약하게 한 sλ=[e−λq]+는 λ=0에서 only-proxy, λ=1에서 residual이다.
clipping 전 정확한 residual 점수와의 차이는

\[
e-\lambda q-(p-q)=\epsilon+(1-\lambda)q.
\]

λ를 줄이면 음수 clipping을 완화할 수 있지만, q가 큰 토큰을 다시 밀어 올리는 bias도
추가된다. 과소평가 오류를 보완하는 정도와 원래 residual 이익을 희석하는 정도 사이의
절충이다. 따라서 작은 λ가 반드시 이기는 것도, λ를 튜닝하면 반드시 개선되는 것도 아니다.

또 다른 방법은 정규화된 두 후보 분포를

\[
s_w=(1-w)\,e_{\rm eligible}+w\,\widehat R_{\rm eligible}
\]

로 섞는 것이다. 0<w<1이면 e가 양수인 토큰의 support를 유지하면서 residual 정보를
반영할 수 있다. 그러나 support 보존만으로 순위 정확성이나 제한된 예산에서의 성능은
보장되지 않는다. 이번 실험에서 두 계열을 모두 검증했다.

**8. 여러 위치와 실제 시스템으로 확장할 때**

위치 i가 최초 rejection 위치일 확률을 h_i라 하면, 전체 후보 목록 S의 coverage는

\[
C_{\rm global}(S)=\sum_{(i,v)\in S}h_iR_i(v).
\]

마지막 bonus 위치는 rejection residual 대신 target p를 사용한다. oracle는 같은
비용의 (위치,토큰) 15쌍 중 h_iR_i(v)가 큰 것을 고른다. 실제 구현은 hhat 및 위치별
top-17 점수의 재정규화를 사용한다. 따라서 local top-k 정리가 그 global 구현의
정확한 최적성을 자동으로 증명해 주지는 않는다.

이미 DS가 준비한 집합 D가 있으면 새로운 후보의 추가 coverage는

\[
\sum_{(i,v)\in S\setminus D}h_iR_i(v).
\]

실제 지연시간은 여기서 한 단계 더 나아가 후보의 완료 시점, cache 유효성, 절약되는
draft 시간, 분기 비용을 반영해야 한다. 이번 coverage 분석은 분포 기반 선택의 성능을
검증한 것이며, 전체 Mirror-SD 대비 TPS 우위의 증명이 아니다.

표준 stochastic SD의 acceptance/recovery 근거는
[Leviathan et al., 2023](https://proceedings.mlr.press/v202/leviathan23a.html)이다.
[Mirror-SD §3.1](https://arxiv.org/html/2510.13161v2#S3.SS1)은 early-exit top-κ를 후보로
전달한다. 본문에서 말하는 only-proxy 비교는 이 후보 점수 아이디어를 현재 DUET의
같은 stochastic SD 조건에 놓은 ablation이다. Mirror-SD 전체 시스템을 그대로 실행한
결과로 해석하지 않는다.

위 부등식, 충분조건, 정확한 손실 분해 및 반례는
[theory_checks.py](theory_checks.py)에서 60,000개 float64 합성 분포로 추가 확인했다.
증명은 위 대수식이며, 합성 검사는 구현상 실수를 점검하는 보조 수단이다.
