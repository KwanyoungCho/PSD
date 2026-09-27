# C=3 사전 fanout 배분과 검증 범위

## 1. 무엇을 최적화하는가

root의 cache hit 확률은 기존 root 선정 정책의 목적이다. 여기서는 주어진 root를
검증할 때 수락되는 descendant 수 AL을 다룬다. 노드 u에 도달할 확률을 rho_u라 하고,
그 문맥에서 draft q로 WOR 후보 c개를 새로 생성했을 때 하나라도 수락될 확률을
g_u(c)라 하면, **이번 확장으로 추가되는** 기대 AL은 rho_u g_u(c)다.
여러 부모에 대해 합하면 sum_u rho_u g_u(c_u)다. 후속 라운드의 가치까지 포함한
전체 최적 제어 문제와는 다르다.

실제 배분에는 target p를 볼 수 없으므로 기존 frozen reach 추정값과, 기존 8개
calibration 문항에서 계산한 phase별 gamma(c)를 사용한다.

    maximize  sum_u rho_hat_u gamma_phase(c_u)
    subject to  0 <= c_u <= 3,
                sum_{u in root r} c_u <= now_r,
                now_r = max(1, remaining_r - future_rounds) if remaining_r > 0 else 0.

부모는 기존 forward 폭/선택 quota 안에서 선택된다. 상위 parent 선택을 포함한
forest 전체의 전역 최적해라는 주장은 하지 않는다. Root prior는 그대로 유지하며,
저장하는 priority에는 reach만 누적한다. gamma를 매 세대 reach에 곱해 재누적하지 않는다.

## 2. 세 번째 후보의 이득을 계산하는 이유와 방법

g(1)=sum_v min(p_v,q_v). 첫 후보 t가 뽑혔지만 거절되는 결합확률은
z_t=[q_t-p_t]+다. 거절 이후 target residual R=normalize([p-q]+)는 t에 독립이고,
다음 draft 분포 D^t는 q에서 t를 제외하고 정규화한 것이다. 따라서

    g(2) = g(1) + sum_t z_t g_1(R,D^t)
    g(3) = g(1) + sum_t z_t g_2(R,D^t)
         = g(2) + sum_t z_t [g_2(R,D^t)-g_1(R,D^t)].

마지막 식은 C=2 값을 임의로 외삽하지 않고, 실제 residual/WOR verifier를 적분한다.
조건부 g2는 이전 연구의 정확한 식으로 계산한다. 계산을 가속하기 위해 R_v/q_v를
한 번 정렬한다. 첫 t가 거절되면 c=1/(1-q_t)이고, 둘째 거절 후 residual의 분자는
[R-cq]+이므로 동일한 정렬 순서를 재사용할 수 있다. 둘째 거절 토큰 u에서도 residual은
0이고, 세 번째 draft의 scale은 1/(1-q_t-q_u)다. 누적합으로 overlap을 계산한다.

작은 vocabulary에서는 모든 t,u 및 verifier coin 경우를 완전 열거해 대조한다.
실제 32k vocabulary에서는 z가 큰 32개 t를 정확히 더하고 나머지를 고정 seed의
IID importance sampling으로 적분한다. Calibration은 context당 512개, 진단은 256개다.
추정 표준오차와 별개로, 적분하지 않은 tail의 전체 z 질량에 따른 **결정론적 구간**도
저장한다. Gamma(1/2)는 정확, gamma(3)는 수치적분 결과라고 구분한다.

Gamma는 관측된 expanded descendant 문맥에서 true rho로 가중한 평균이다.
이는 해당 calibration 문맥 분포에서 기대 추가 AL에 맞춘 scalar다. 어떤 문맥에서도
같은 gamma가 참이라는 가정은 아니며, 신규 frontier로의 분포 이동도 자동 보장하지 않는다.

## 3. 한 라운드의 배분을 정확히 풀 수 있는 이유

공통 gamma는 c에 대해 단조 증가하지만, 그 증가량은 감소할 필요가 없다. 따라서
첫 후보를 모두에게 하나씩 주거나, 단순 marginal greedy를 쓰는 것에는 일반적
최적성 보장이 없다. 가중치 a>=b, 후보 수 c>=d에 대해

    [a gamma(c)+b gamma(d)] - [a gamma(d)+b gamma(c)]
      = (a-b)[gamma(c)-gamma(d)] >= 0.

그러므로 rho_hat 내림차순의 부모에게 c도 내림차순으로 배정하는 최적해가 존재한다.
현재 B<=8, C=3에서는 정수 partition을 모두 만들어 비교하면 충분하다. 이는
gain의 concavity를 가정하지 않는다. CUDA 구현은 고정 크기 tensor 연산만 사용하고,
같은 입력에 대해 CPU 완전탐색 및 GPU eager/captured replay와 비교한다.

**부모별 gamma가 다르면 위 정렬 논리는 일반적으로 성립하지 않는다.** 현 구현은
phase별 공통 곡선에 한정한다. 부모별 true g를 쓰는 사후 oracle에는 별도의 전수열거를 쓴다.

## 4. Lossless 조건

c_u는 그 부모의 새 child sample을 보기 **전에** 결정한다. 부모의 token, 기존
sibling 정보, draft-only feature는 사용 가능하지만 앞으로 뽑힐 child나 verification
coin을 보고 c_u를 줄이거나 유리한 child만 남기지 않는다. G=M, sibling prefix를 유지한다.
그러면 해당 문맥의 WOR proposal와 residual verifier 규칙은 그대로다. 기본 한 토큰
정확성을 문맥별로 적용하면 생성 토큰의 결합 분포도 target의 자기회귀 분포가 된다.

이 설명은 수학적 알고리즘의 조건이다. 구현의 q-reference/attention mask/graph 동작은
별도 검사한다. CPU 유한분포 검사가 실제 70B 엔진 전체의 정확성을 대신하지 않는다.
기존 post-sampling pruning의 TV=0.01875 반례는 negative control로 유지한다.

## 5. Frontier와 사후 비교의 한계

기존: depth == round인 unexpanded 노드만 확장. 후보 변경: depth <= round인
모든 unexpanded 노드를 고려. Depth cap, root/raw-q gate, root quota, future reserve,
물리적 forward 폭은 유지한다. Position은 실제 depth, attention은 실제 ancestry
cell로 계산하므로 mixed depth에서도 올바른 prefix를 봐야 한다. 이를 실제 mask 함수와
GPU 실행기로 검사한다.

기존 raw 파일은 관측된 tree의 **모든 노드에서 target p**를 가지고 있지만 draft q는
그중 실제 확장한 부모에서만 가진다. 따라서 기존 파일만으로 계산한 C=3 비교는
관측된 부모 pool 안의 국소 비교다. 전체 frontier 또는 새 tree 전체의 causal replay가 아니다.

한 GPU로 draft를 다시 실행하면 관측된 leaf의 q를 보완할 수는 있다. 다만 새 forward의
q가 저장된 production q를 수치적으로 재현하는지 먼저 확인해야 한다. BF16 및 실행경로가
달라 생기는 차이를 무시한 채 개선 효과로 읽지 않는다. 이 검증에 실패하면 shadow
결과를 정책 우월성 근거에서 제외하고, 동일 engine/KV 상태를 이용한 별도 추적이 필요하다.
