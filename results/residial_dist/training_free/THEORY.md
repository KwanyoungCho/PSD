# 학습 없이 후보 선정을 개선하는 수학

이 문서는 기존 residual TV 증명을 반복하기보다, **어떤 계산을 바꾸면 후보 선정이
좋아질 수 있는지**를 설명한다. 증명되는 성질과 실험으로 판단할 사항을 구별한다.
실측 결과는 [REPORT.md](REPORT.md), 실험 선택 규칙은 [PROTOCOL.md](PROTOCOL.md) 참조.

## 1. 최적화해야 할 것은 다음 검증의 재사용 확률이다

같은 prefix의 target, draft, early-exit 분포를 각각 $p_i,q_i,e_i$라 하자.
모두 실제 sampling temperature를 적용한 확률이다. 검증할 draft token은 $y_i$다.
표준 확률적 SD의 수락률과 첫 거절 위치 확률은

\[
\alpha_i=\min(1,p_i(y_i)/q_i(y_i)),\qquad
h_i=\left(\prod_{j<i}\alpha_j\right)(1-\alpha_i).
\]

모두 수락할 확률은 $h_K=\prod_{j<K}\alpha_j$이다. $Z_i>0$인 거절 위치에서

\[
a_{iv}=[p_i(v)-q_i(v)]_+,\quad Z_i=\sum_v a_{iv},\quad R_i(v)=a_{iv}/Z_i.
\]

Bonus 위치의 분포는 $R_K=p_K$로 표기한다. 따라서 동일 비용의 캐시 root 집합 $S$의
이론적 coverage는

\[
C(S)=\sum_{(i,v)\in S}h_iR_i(v).
\]

**정확한 $h_iR_i(v)$의 전역 top-B가 최적**이다. 고정 위치에서는 $R_i$의 top-k가
최적이지만, 여러 위치의 예산을 함께 고르면 $h_i$도 필요하다. $Z_i=0$인 거절
위치는 실제 거절 확률이 0이므로 기여를 0으로 둔다. 실제 구현의 $y_i$ 제외는
후보 점수에 적용하고, 평가에는 원래의 $R_i$를 사용한다. 정확한 산술에서
$h_i>0$이면 $p_i(y_i)<q_i(y_i)$이므로 $R_i(y_i)=0$이다. 반대로
$R_i(y_i)>0$이면 그 토큰의 수락률은 1이어서 $h_i=0$이다. 따라서 이 제외는
정확한 결합질량 $h_iR_i$를 잃게 하지 않는다. 근사 후보 분포의 정규화와 순위에는
영향을 줄 수 있으므로 양쪽 후보 정책에 같은 제외를 적용한다.

참고: [Leviathan et al., 2023](https://proceedings.mlr.press/v202/leviathan23a.html).
이 문서의 목적함수는 root 재사용 가능성이다. root 준비 완료 시점, 재사용할 suffix
길이, GPU 실행 시간까지 포함한 TPS 목적함수와는 다르다.

## 2. Residual이 지는 정확한 이유를 후보 교환에서 측정한다

한 위치에서 $b_v=[e_v-q_v]_+$, $S_r=\operatorname{Top}_k(b)$,
$S_e=\operatorname{Top}_k(e)$라 하자. 두 정책에 같은 $y$ 제외를 적용한다.
$I=S_r\setminus S_e$, $O=S_e\setminus S_r$, $\eta=b-a$라 두면

\[
\Gamma=b(I)-b(O)\ge0,\qquad E=\eta(I)-\eta(O),
\]

\[
\boxed{C(S_r)-C(S_e)=\frac{\Gamma-E}{Z}.}
\]

이는 상한이 아니라 정확한 항등식이다. $\Gamma$는 proxy residual 점수가 주장하는
교환 이득이며, $E$는 그 주장에 포함된 실제 점수 오차다. 들어온 후보를 과대평가하거나
빠진 후보를 과소평가하면 $E$가 커진다. **$\Gamma>E$여야 이긴다.**

실제로 필요한 토큰을 0으로 지우는 조건은
$e_v-p_v\le -(p_v-q_v)$이다. 확률이 낮은 토큰만 사라지는 것이 아니다.
$p_v$와 $q_v$가 둘 다 큰 토큰도 그 차이가 작으면 지워진다.

이번 실험은 top-3 교환을 직접 계산하고, 모든 기록에서 위 등식을 검사한다.
Entropy나 전체 TV의 평균을 오차의 방향으로 대신 사용하지 않는다.

## 3. 첫 번째 개선: top-M 안에서의 재정규화를 제거한다

후보 점수를 $s_{iv}\ge0$라 하고, $y_i$를 제외한 전체 점수 합을 $A_i$,
그중 상위 M개 점수 합을 $m_i$라 하자. $\rho_i=m_i/A_i$이다.

기존 전역 점수와 전체 질량으로 정규화한 점수는 각각

\[
J^{\rm old}_{iv}=\widehat h_i\frac{s_{iv}}{m_i},\qquad
J^{\rm full}_{iv}=\widehat h_i\frac{s_{iv}}{A_i}.
\]

그러므로

\[
\boxed{J^{\rm old}_{iv}=J^{\rm full}_{iv}/\rho_i.}
\]

상위 M개 밖에 점수 질량이 많이 남는 위치일수록 $\rho_i$가 작다. 기존 방법은
그 위치의 상위 후보 점수를 더 크게 부풀린다. 같은 위치 안의 순위는 그대로지만,
**다른 위치와 비교할 때 예산 배분이 달라진다.**

간단한 예: 두 위치의 확률이 $(0.6,0.4)$, 각 위치 최고 후보의 실제 residual
확률이 $(0.4,0.9)$이고 M=B=1이라고 하자. 재정규화하면 점수 $(0.6,0.4)$로
첫 위치를 고른다. 실제 coverage는 $0.6\times0.4=0.24$다.
전체 확률 점수 $(0.24,0.36)$로 고르면 두 번째 위치를 골라 0.36을 얻는다.
이 예는 정확한 입력을 주어도 top-M 재정규화가 전역 최적 선택을 깨뜨릴 수 있음을 보인다.

다만 실제 입력 $e,\widehat h$는 근사치다. 기존 편향이 우연히 오차를 보상하는
상황도 가능하므로 **full 정규화가 모든 실제 context에서 이긴다는 정리는 아니다.**
전체 vocabulary를 이미 계산하는 현재 chain에서는 추가 모델 forward가 필요 없다.
또한 $y$ 제외 뒤 전체 합으로 나누는 근사 점수가 곧 실제 $R$이라는 뜻은 아니다.
반면 정확한 $a,h$를 쓰면 $h_i>0$인 위치의 제외 질량은 원래 0이므로, full 점수는
정확한 결합확률과 일치한다.

## 4. 두 번째 개선: 한 토큰의 비율로 만든 위치 추정을 완만하게 보정한다

현재 $\widehat\alpha_i=\min(1,e_i(y_i)/q_i(y_i))$로 $\widehat h$를 만든다.
만일 proxy가 $y_i$의 확률을 조금 높게 추정해 $e_i(y_i)\ge q_i(y_i)$가 되면,
추정 거절 확률이 정확히 0이 된다. 실제로는 거절 가능한 위치인데 후보 예산을
거의 받지 못할 수 있다. 앞 위치의 작은 수락률 오류는 뒤 위치의 누적곱에도 영향을 준다.

고정 prefix에서 $Y\sim q_i$라면 다음은 정확하다.

\[
\begin{aligned}
\mathbb E_Y\!\left[\min(1,e_i(Y)/q_i(Y))\right]
&=\sum_vq_i(v)\min(1,e_i(v)/q_i(v))\\
&=\sum_v\min(e_i(v),q_i(v))\\
&=1-\operatorname{TV}(e_i,q_i)=:\bar\alpha_i.
\end{aligned}
\]

$q_i(v)=0$인 좌표는 기대값에 기여하지 않으며 마지막 합으로 정의하면 된다.
$\bar\alpha_i$로도 같은 누적곱 형태의 $\bar h$를 만들고

\[
\boxed{\widetilde h=(1-\omega)\widehat h+\omega\bar h}
\]

를 사용한다. 실험에서 탐색한 $\omega$는 0.25, 0.5, 0.75이다.
이는 관측된 $y_i$ 정보와 전체 proxy–draft overlap 정보를 섞는 방법이다.

**주의할 정확한 경계:** $\bar\alpha_i$의 한 prefix 기대값 등식은 정리지만,
$\bar h$가 관측된 draft 경로의 정확한 첫 거절 분포라는 주장은 하지 않는다.
앞서 생성한 토큰에 따라 뒤 prefix가 바뀌므로, 평균들의 곱을 전체 경로의 정확한
기대값으로 해석해서도 안 된다. $\widetilde h$는 실험으로 검증하는 추정 규칙이다.
Target 샘플러의 실제 수락률은 $\alpha_i$ 그대로다.

## 5. 세 번째 개선: draft를 빼면서 중요한 후보를 삭제하는 민감도를 제한한다

단순 subtraction의 문제를 더 직접적으로 보려면 확률의 **상대 변화**를 보자.
$e_v>q_v$에서

\[
\frac{\partial\log(e_v-q_v)}{\partial\log e_v}
=\frac{e_v}{e_v-q_v}.
\]

$e_v\approx q_v$이면 이 값이 커진다. $e_v$가 조금만 변해도 residual의 상대
크기는 크게 변하고, 경계를 넘으면 0이 된다. 이는 뺄셈이 **절대 오차를 키운다**는
주장이 아니다. 절대 변화는 clipping 전에 그대로이고 clipping은 비팽창적이다.
작은 residual을 기준으로 한 상대 민감도의 문제다.

### 5.1 부드러운 draft 할인

\[
\boxed{s_v=e_v(1-q_v)^\beta,\qquad\beta\ge0.}
\]

$\beta=0$이면 proxy-only다. $\beta>0$이면 draft 확률이 높은 토큰을 낮춘다.
$e_v>0,q_v<1$인 토큰을 0으로 잘라내지 않으며,

\[
\frac{\partial\log s_v}{\partial\log e_v}=1.
\]

즉 고정 q에서 proxy 확률의 상대 오차가 score에 그대로 전달된다. 차의 상대 오차로
바뀌면서 커지는 현상을 피한다. 다만 이는 $s$가 $R$에 더 가깝다는 보장이 아니다.
Residual의 강한 draft 차감을 덜 적용하는 대신 대체 목적의 편향이 생긴다.

$q_v$가 작을 때 전개하면

\[
s_v=e_v-\beta e_vq_v+O(e_vq_v^2).
\]

차감량이 $q_v$ 자체가 아니라 대략 $\beta e_vq_v$이다.
예를 들어 $e_v=0.1,q_v=0.12$이면 원래 residual은 0이고,
$\beta=1$ 점수는 $0.1\times0.88=0.088$이다. 다른 토큰보다 얼마나 낮출지에
draft 정보를 쓰면서 proxy가 중요하다고 본 후보를 즉시 없애지 않는다.

$\beta=1$에는 $V\sim e,Y\sim q$를 **독립적으로** 뽑을 때
$\Pr(V=v,Y\ne v)=e_v(1-q_v)$라는 해석도 있다. 그러나 SD의 실제 correction은
이 독립 불일치 분포가 아니라 $[p-q]_+/Z$다. 이 식을 새 lossless correction
샘플러라고 부르면 틀리며, 캐시 후보 순위를 위한 heuristic으로만 사용한다.

### 5.2 Residual에 proxy 비례 하한 부여

\[
\boxed{s_v=\max\{[e_v-q_v]_+,\;\rho e_v\},\qquad0<\rho\le1.}
\]

점수는 $\rho e_v\le s_v\le e_v$이므로 최대 할인율을 $1-\rho$로 제한한다.
예를 들어 $\rho=0.75$면 proxy 점수의 25% 이상을 깎지 않는다.
0점 삭제를 막으면서 residual이 충분히 큰 후보에는 residual을 그대로 쓴다.

미분 가능한 지점의 상대 민감도는 1 이상 $1/\rho$ 이하이다. 하한이 선택되면 1이고,
residual이 선택되면 $e_v-q_v\ge\rho e_v$이므로
$e_v/(e_v-q_v)\le1/\rho$이다. 경계에서는 미분이 없지만 함수는 연속이다.
이 역시 순위 정확도 보장은 아니며, proxy 오차에 대한 **민감도 제한**이다.

## 6. 조건부로 안전한 순위 교환도 가능하지만, 오차 상계가 필요하다

어떤 후보에 대해 $(1-\kappa)e_v\le p_v\le(1+\kappa)e_v$가 참이면

\[
L_v=[(1-\kappa)e_v-q_v]_+\le a_v\le
U_v=[(1+\kappa)e_v-q_v]_+.
\]

현재 후보 w 대신 v를 넣을 때 $L_v>U_w$라면 $a_v>a_w$가 보장된다.
뒤 순위 v와 앞 순위 w를 **서로 교환**하면 그 사이의 모든 prefix 후보 집합의
실제 질량이 증가하고, 두 토큰을 모두 포함하는 prefix 질량은 같다.

실험 구현은 proxy 순위의 점수 목록을 유지하고 토큰 ID만 교환하여 source 순위와
위치 배분을 분리한다. 그러나 실제 proxy에 모든 토큰의 상대 오차 상계를 확인한 것은
아니다. 따라서 이 조건부 정리를 무조건 안전한 실전 교환 규칙으로 주장하지 않는다.
전역 동점 처리로 prefix가 아닌 슬롯을 고르면 prefix 정리가 직접 적용되지 않는 점도 있다.
$\kappa=1$이면 $L_v=0$이어서 교환이 일어나지 않는다. 이 설정은 baseline과 완전히
동일한 no-op이며, 동점 순서 변화가 개선으로 집계되지 않도록 검사했다.

## 7. Temperature와 lossless 조건

분포는 $p_T=\operatorname{softmax}(z_p/T)$,
$q_T=\operatorname{softmax}(z_q/T)$,$e_T=\operatorname{softmax}(z_e/T)$로 일치시켜 비교한다.
T를 낮추면 proxy와 target의 argmax가 같은 위치에서는 집중도가 높아질 수 있지만,
argmax가 틀린 위치에서는 잘못된 후보가 더 확신 있게 선택될 수 있다.
따라서 T 감소가 residual의 우위를 보장하지 않는다. 실제 생성 T를 바꾼 실험이 필요하다.

특히 **현재 후보 정책은 이미 제안된 draft token $y$를 제외한다**. Draft가 거의
결정적이 되어 $q\to\delta_y$이면 허용 후보 $v\ne y$에 대해 $q_v\to0$이므로

\[
[e_v-q_v]_+\to e_v,\qquad e_v(1-q_v)^\beta\to e_v.
\]

정확히 $q=\delta_y$라면 제외 후 residual, proxy-only, 부드러운 할인 점수는 동일하다.
따라서 낮은 T에서 **강한 draft 차감의 이점이 반드시 커질 것**이라는 기대도 성립하지
않는다. 이미 제외한 y가 draft 질량 대부분을 차지하면, 남은 후보 사이에서 q로
차등을 줄 여지가 줄어든다. 이는 T=0의 별도 greedy 구현을 이번에 실험했다는 뜻은 아니다.

현재 chain 후보 계산의 raw-logit softmax는 T를 생략한다. 실제 sampler/verification은
T를 적용하므로, 이 상태는 출력 분포 변경보다는 후보 추정 불일치 문제다.
실험 hook은 후보 계산만 동일 T로 맞춘다. 이는 반드시 바꿔야 하는 lossless 정확성
버그라는 뜻은 아니다. 기존 코드의 tree 경로 주석에는 temperature 반영을 과거 속도
실험 후 원복한 이력도 있다. T=1 후보 계산이 경험적인 운영점일 가능성까지 포함해
실제 TPS로 비교해야 한다. 이번 실험은 tree에 대한 결과가 아니다.

변경한 $s,\widetilde h$는 **어느 root 아래의 draft suffix를 미리 만들지**에만 사용한다.
실제 correction은 여전히 target의 $R_T$에서 나온다. 그 토큰이 캐시에 있으면 해당
prefix의 suffix를 재사용하며, suffix 토큰은 실제 제안에 쓰인 $q_T$로 검증한다.
후보 점수 $s$를 correction 샘플러나 $p/q$ 검증의 분모에 넣지 않는다.

이 분리는 후보 정책 변경이 sampling rule을 바꾸지 않는 이유다. 저장 분포 replay는
출력을 만들지 않으며, 실제 실행 검사는 기존 verifier/sampler 파일을 수정하지 않은
실험 hook으로 수행한다. 이는 GPU/TP/KV 전체 구현의 완전한 정확성 증명을 대신하지 않는다.

## 8. 이번 분석이 증명하지 않는 것

- Only-proxy가 모든 분포 또는 실제 데이터 분포에서 최적이라는 정리.
- 부드러운 할인이나 하한 규칙이 모든 context에서 residual/proxy-only보다 좋다는 정리.
- 낮은 entropy, 낮은 T 또는 큰 TV(p,q)만으로 후보 우위를 결정할 수 있다는 주장.
- 후보 coverage 이득만으로 Mirror-SD 전체 구현의 TPS를 이겼다는 주장.

[Mirror-SD 원문](https://arxiv.org/html/2510.13161v2#S3.SS1)은 early-exit 후보 외에도
분기 확장, speculative streaming, 실행 배치를 포함한다. 이번 직접 비교 대상은
동일 DUET 엔진 안의 proxy-score 후보 기준선이다. 완전한 시스템 비교는 별도다.

## 9. 수학적으로 같은 top-k라도 동점 처리가 결론을 바꿀 수 있다

동일한 추정 점수를 받은 두 토큰 v,w를 경계에서 고를 때, 점수 목적함수에서는
둘 중 어느 것을 골라도 최적이다. 그러나 실제 $hR$ 질량은 다를 수 있다.
따라서 proxy 점수 최적성만으로 **동점 선택을 포함한 실제 coverage의 동일성**까지
보장할 수 없다. BF16 logits의 동일한 값은 float32 softmax 후에도 동일 확률을 만든다.

현재 wire는 중복 제거 여유분을 포함해 17개를 전송하고, 이 PS-only 설정에서는
앞의 15개가 root 예산에 들어간다. GPU topk(17)의 앞 15개와 topk(15)의 선택이
경계 동점에서 달라질 수 있다. 서로의 점수 합은 같아도 실제 coverage가 달라진다.

실측 예([numerical_case.json](numerical_case.json)): T=1, step 331의 마지막 후보는
token 285와 437이 같은 추정 결합점수 약 0.00555783으로 동률이었다. 전자의 선택과
후자의 선택 간 실제 coverage 차이는 약 0.98193이었다. 이 값은 한 step의 차이이며
전체 평균 차이는 훨씬 작다. 그렇더라도 0.1%p 정도의 추가 이득을 주장할 때는 무시할 수 없다.

따라서 동결한 replay 결과를 보존하고, 같은 동결 정책을 실제 wire 방식으로 전부
다시 평가했다. 후보 점수의 최적성 검사는 통과했지만, 작은 draft 할인 이득의 통계적
우위는 유지되지 않았다. 이 검사를 단순한 실행 오류로 간주해 좋은 쪽만 남기지 않는다.
재현 가능한 결정적 동점 규칙은 향후 엔진 통합 시 고정해야 할 명세이며, target을 보고
유리한 동점 토큰을 고르면 oracle 정보 누출이 된다.

## 10. Coverage 다음에는 실제 절약 시간을 목적함수에 넣어야 한다

이번 실제 실행에서 coverage 개선이 속도 개선으로 재현되지는 않았다.
Coverage 목적함수 $\sum h_iR_i(v)$는 모든 root hit의 가치를 1로 간주한다.
그러나 아직 준비되지 않은 root, 짧은 suffix만 준비된 root, 원래 miss 비용이 다른
연산에 가려지는 root는 실제 절약 시간이 다를 수 있다.

후보 간 실행 간섭을 무시하고 각 root의 조건부 평균 절약 시간 $g_{iv}$를 고정한
단순 모형에서는, 기대 절약 시간이

\[
\boxed{U(S)=\sum_{(i,v)\in S}h_iR_i(v)g_{iv}}
\]

가 된다. 준비가 끝나지 않아 재사용하지 못하면 그 사건의 절약 시간은 0으로 포함한다.
같은 후보 비용에서 이 목적함수의 최적 선택은 $h_iR_i(v)g_{iv}$의 top-B다.
$g$가 일정할 때만 원래 coverage 최적화와 일치한다. 후보별 비용이 다르거나 같은
batch 안의 후보끼리 간섭하면 단순 top-B 정리는 적용되지 않는다.

실행에서 측정한 decode 처리량에는 다음 항등식도 성립한다.

\[
\mathrm{TPS}_{\rm decode}
=\frac{\sum_t L_t}{\sum_t\tau_t}
=\frac{\operatorname{mean}(L_t)}{\operatorname{mean}(\tau_t)}.
\]

단독 실행에서 step 시간은 약35ms로 비슷했지만 수락 토큰 수의 표본 평균이 달라져
속도 결과가 반대로 나왔다. 이 작은 실험만으로 그 길이 차이의 인과 원인을 확정할 수
없다. 특히 같은 seed가 후보 정책 사이에서 같은 출력 경로를 보장하지 않는다.

$g_{iv}$를 추정한 시간 가중 정책은 **후속 가설**이다. 이번에 학습하거나 실제 우위를
검증한 새 정책으로 부르지 않는다. 이번 결과가 제공하는 것은 q 기반 위치 추정의
coverage 이득과, 이를 TPS 목표로 연결할 때 추가로 필요한 측정 항목이다.
