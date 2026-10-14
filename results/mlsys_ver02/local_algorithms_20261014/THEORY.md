# 이번 반영의 수학적 의미와 한계

기호: target 분포 p, early-exit proxy e, 실제 draft proposal q. 이 문서의 점수는 **미래 cache root와 draft tree를 선택**한다. 최종 토큰의 target 수락·correction 분포를 이 점수로 바꾸지 않는다.

## 1. Root 토큰: residual에서 complement로

기존 점수는 `[e(v)-q(v)]+`다. 실제 correction residual `[p-q]+`를 직접 근사하는 자연스러운 식이지만, p와 q가 가까워 residual의 전체 질량이 작으면 정규화가 e의 작은 오차를 증폭한다.

`Z=TV(p,q)`, `Zhat=TV(e,q)`, `delta=TV(p,e)`이고 두 Z가 양수이면

`TV(normalize([p-q]+), normalize([e-q]+)) <= min(1, delta/max(Z,Zhat))`.

이는 상한이다. 상한이 커졌다고 실제 오차가 반드시 크다는 뜻도, proxy-only보다 반드시 나쁘다는 뜻도 아니다. 반대로 이 상한만으로 특정 후보식의 우위를 증명할 수도 없다. Z=0이면 거절 사건 자체가 없으므로 그 조건부 residual을 정의할 필요가 없다.

새 점수는 `s(v)=e(v)(1-q(v))`다. 독립인 두 표본 Y~e, X~q에 대해 `Pr(Y=v, X!=v)=e(v)(1-q(v))`라는 해석을 갖는다. Draft가 이미 강하게 지지하는 토큰의 점수를 부드럽게 낮추면서, 작은 e-q의 부호 오류로 후보를 완전히 지우는 현상을 줄인다.

그러나 실제 SD 거절은 두 독립 표본의 불일치 사건이 아니다. 따라서 이 해석은 **설계 동기**이며, 새 식이 실제 residual의 정확한 근사 또는 최적 근사라는 증명이 아니다. 고정 위치식에서 proxy-only와 비교하는 실험이 별도로 필요하다.

이미 제안한 자식은 cache root에서 제외하고 남은 vocabulary 전체 합으로 정규화한다. 기존 top-M 합 정규화는 확률이 넓게 퍼진 문맥에서도 선택된 M개의 합을 1로 만들어 해당 문맥의 후보 가치를 부풀릴 수 있다. 예컨대 두 위치의 질량이 .6/.4이고 top1 확률이 .4/.9이면 top1 재정규화는 .6/.4로 첫 위치를 고르지만 full 정규화는 .24/.36으로 두 번째 위치를 고른다.

## 2. 위치: 관측 경로와 평균 overlap의 혼합

Chain에서 관측 token y_i에 대한 proxy 수락률은 `a_i=min(1,e_i(y_i)/q_i(y_i))`다. 이에 따른 종료 위치 질량은 `hhat_i=prod_{j<i}(a_j)*(1-a_i)`이며, 끝까지 수락하는 bonus 위치는 `prod_j a_j`다.

다른 추정은 `abar_i=sum_v min(e_i(v),q_i(v))`를 사용해 같은 종료 위치 재귀를 계산한 hbar다. 관측된 한 token의 비율에만 의존하는 것을 완화하기 위해 `h=.75*hhat+.25*hbar`를 사용한다.

Tree에서는 형제 순서별 실제 WOR 조건부 proposal과 residual 갱신을 반영한다. hhat과 hbar의 **재귀를 각각 계산한 후** 혼합한다. alpha부터 섞어 곱하면 교차항이 추가되어 다른 식이다.

두 질량이 각각 합 1이면 혼합도 합 1이다. 하지만 .25가 이론적으로 최적이라는 보장은 없다. 이전 로컬 탐색에서 동결한 스칼라이며 이번 전체 평가에서 다시 고르지 않는다. 이번 default는 단순화를 위해 T>0에서 공통 e(1-q)를 쓰며, 과거 온도별 우승 수식 세 가지를 그대로 재현하는 정책은 아니다.

## 3. Tree: 경로 확률에서 기대 수락 길이의 근사로

기존 q-path는 경로에서 draft 확률을 곱한다. 이는 draft가 그 경로를 제안할 점수이며, target이 형제들을 순서대로 거절·수락하여 그 경로에 도달할 확률과 다르다.

수락률 추정 alpha를 사용하면 j번째 자식의 도달 확률은

`rho(child_j)=rho(parent)*prod_{k<j}(1-alpha_k)*alpha_j`.

이번 alpha는 phase·형제 순번·q 구간별 과거 scalar 보정표다. 실제 forward의 q를 이 값으로 바꾸지 않는다. 예측 오차가 있으므로 실제 AL을 완벽히 예측하는 oracle이 아니다.

부모 u에 c개 자식을 주었을 때 하나 이상 수락될 확률을 과거 자료로 추정한 공통 곡선을 gamma_phase(c)라 두면, 현재 round의 배분 목적은 `max sum_u rho(u)*gamma_phase(c_u)`, `sum_u c_u<=budget`다. 같은 phase의 공통 단조 gamma 곡선에서는 높은 rho에 큰 c를 배정하는 것이 유리하다. 두 부모 A/B와 a>=b에 대해 교환 이득은 `(rho_A-rho_B)*(gamma(a)-gamma(b))>=0`이기 때문이다.

따라서 작은 예산의 정수 partition을 열거하면 비오목 gamma도 그 round에서 정확히 최적화할 수 있다. 미래 logits와 실제 target 수락률을 모르므로 **전체 tree의 실제 AL 전역 최적화**는 아니다.

이전 깊이의 미확장 node도 경쟁시킨다. 이미 forward한 node는 제외하고, 그 forward에서 생성한 미확장 자식만 남는다. 이전에 한 forward를 취소·반복하지 않지만, 만들어 둔 모든 후보를 결국 확장한다는 보장도 없다. 기존 물리 입력 폭·request별 예산·남은 round 예약을 유지한다.

## 4. 정확성의 경계

검증에서 현재 목표 분포를 r, 실제 proposal을 d라 쓰면 수락 확률은 `a(v)=min(1,r(v)/d(v))`다. 전체 수락 질량 `A=sum_v min(r(v),d(v))`에 대해

`d(v)*a(v) + (1-A)*normalize([r-d]+)(v) = min(r(v),d(v)) + [r(v)-d(v)]+ = r(v)`.

Tree의 다음 형제를 시도할 때는 해당 단계의 조건부 d와 갱신된 r로 같은 항등식을 적용한다. 새 점수는 이 d나 r에 들어가지 않는다. Parent/fanout을 자식 샘플 전에 결정하고 ordered-WOR prefix를 보존하는 이유가 여기에 있다. 이 항등식은 정확한 확률 연산에서의 정당화이며 floating-point 구현의 검증을 대체하지 않는다.

Cache root도 proxy 점수에서 뽑았다는 이유로 그 점수를 p/q 분모에 넣지 않는다. 실제 target 검증이 correction/bonus를 먼저 결정한 후, 그 token과 일치하는 cache를 조회한다. 그 뒤에 이어지는 draft 자식들만 실제 proposal 법칙으로 검증한다.

- Cache root 점수는 cache hit를, tree 구조는 AL을 목표로 한다. 두 효과와 비용을 2x2 대조로 분리한다.
- 새 tree의 parent 선택/fanout은 새 자식을 샘플링하기 전에 결정한다. 실제 ordered-WOR 순서와 raw q를 보존한다.
- G>M이면 생성 순서 prefix만 검증한다. 샘플을 사후 점수로 재정렬하지 않는다.
- T=0은 target argmax 경로로 검증한다. Soft 확률은 예산 점수용이다.
- 알고리즘이 달라지면 RNG 소비와 방문 문맥이 달라질 수 있다. 같은 seed의 출력 일치율은 기록하되 losslessness의 필요조건으로 삼지 않는다.
- 이번 두 소형 dense 모델·B8·고정 예산 결과를 70B의 모든 설정에 일반화하지 않는다. 보정표와 평가 prompt의 완전 분리 여부도 확인 전에는 held-out 성능이라고 부르지 않는다.
