# Round4 분석 기준

## 1. 논문 그림과 현재 시간 제약

한 step의 동일한 시간 원점에서 다음 시각을 측정한다.

- `a`: draft context 정렬(glue) 종료
- `P`: proxy NCCL 수신 완료. P1 후 wait를 시작한 시각과 구분한다.
- `D1`: P1 준비·실행에 걸린 시간
- `D2`: P2 준비·실행·cache metadata 반영 시간
- `F_model`: target 최종 logits 완료
- `F_ready`: target accept/reject와 postprocess 완료

\[
C_1=a+D_1,\qquad C_2=\max(C_1,P)+D_2.
\]

P1이 proxy 이전에 끝나는 조건은 `C1<=P`, P2가 다음 step 준비 전 끝나는 조건은 `C2<=F_ready`다. 엄격히 model forward 안에 숨기는 조건은 `C2<=F_model`이다. 후자는 전자보다 강하다. 논문 Fig5에서도 sampling 구간이 따로 있으므로 둘을 섞지 않는다. `D1~K1*t_d`, `D2~K2*t_d`는 1차 근사일 뿐, tree query 폭·vocabulary·selection/mask kernel·전송·준비 비용을 따로 측정해야 한다.

정상상태에서의 노출 시간 근사는 `[C2-F_ready]_+`다. 다음 요청이 miss이면 JIT와 통신도 더해진다. 실제 target wait와 대조한다. Batch all-hit는 JIT가 사라질 조건이지, 이전 cache 생성 완료까지 보장하는 충분조건은 아니다. 요청 hit를 독립이라고 가정할 때에만 all-hit 확률이 `h^B`이며, 실제 측정치와 구분해야 한다.

Profile은 graph capture가 없는 step, 실제 활성 요청 수가 설정 B인 step을 따로 선택한다. 작은 corpus 끝부분의 B=1/2 drain을 B=8 성능으로 보고하지 않는다. CUDA/host anchor에는 작은 시계 정렬 오차가 있으므로 마감 차이 0 부근은 해석에 주의한다. Profile을 켠 TPS를 성능 결론에 쓰지 않는다.

## 2. 이전 tree 개선의 적용 범위

Root 후보 수식은 고정한다. 기존 q-path continuation score는 부모 경로의 draft 확률 곱이다. 이전 reach 정책은 ordered-WOR sibling 검증에서 실제 방문 확률을 근사한다.

\[
\widehat\rho_{v_j}=\widehat\rho_u
\left(\prod_{i<j}(1-\widehat\alpha_{v_i})\right)\widehat\alpha_{v_j}.
\]

`alpha_hat`는 과거 phase/sibling/raw-q-bin별 조건부 수락률 보정값이다. 새로운 모델에서의 참 수락 확률이라는 보장은 없다. 1차 비교는 과거 보정값을 변경 없이 이식한다. 별도 후속 비교는 각 dense 모델의 사전 지정 8개 질문으로 보정하고, 이 질문을 제외한 472개 결과도 보고한다. 신경망 학습은 없으나 보정 자체는 empirical estimation이며, 학습하지 않은 순수 정리라고 쓰면 안 된다.

Gain allocation은 선택된 부모들의 독립 root별 예산 안에서

\[
\max_{c_i\in\{0,\ldots,C\}}\sum_i\widehat\rho_i\gamma(c_i),
\quad\sum_{i\in root(r)}c_i\le b_r
\]

를 푼다. `gamma(c)`는 그 부모에서 c개의 ordered-WOR 자식 중 하나를 수락할 기대 확률이다. 같은 nondecreasing curve를 적용할 때 큰 fanout을 큰 reach에 배정하는 재배열 성질로 작은 integer partition을 열거할 수 있다. 이는 **현재 round의 추정 이득** 최적화다. 남은 모든 round와 실제 target 분포에 대한 전역 AL 최적성 증명은 아니다. 이후 round를 위한 기존 reserve를 유지한다.

Frontier 확장은 `depth==round`를 `depth<=round`로 바꾸어 이전 round에서 확장되지 않은 얕은 후보도 선택할 수 있게 한다. q_path/reach, gain on/off, frontier on/off를 구분해 어떤 변경이 효과를 냈는지 본다. Cache hit나 context 구성은 tree가 달라지면서 간접적으로 바뀔 수 있으므로, root 수식이 같다고 hit 결과까지 같다고 가정하지 않는다.

## 3. G>M에서 보존해야 하는 proposal 법칙

Ordered-WOR의 j번째 형제 proposal은 이미 뽑은 형제를 제외한 실제 조건부 q다. 샘플을 본 뒤 자신/후속 자식의 점수로 생존 여부를 정하면, target에 보이는 token의 법칙이 원래 q와 달라질 수 있다. 원래 parent q를 그대로 검증에 사용하려면 그 선택 편향이 없어야 한다.

이번 수정은 고정된 생성 순서의 앞 M개를 전송한다. 생성기는 부모 먼저, 형제 순서대로 기록한다. 따라서 ancestry/sibling prefix가 유지되고, 해당 자식의 실현된 값에 따라 그 자식을 버리지 않는다. 생성한 뒤 점수 좋은 subtree를 고르는 것은 제거했다. 미래 확장 위치/자식 수를 **그 자식의 샘플링 전에** 정하는 reach/gain 정책과는 구분한다.

18개 유한 상태와 모든 수락/거절 분기를 전수 계산한 반례는 기존 pruning의 편향 존재를 보인다. LLM에서 얼마나 자주/얼마나 크게 나타나는지는 이 반례만으로 알 수 없다. 실제 serving helper의 두 phase, stale precompute cache, qref remapping을 별도 검사하고 dense B1 실행도 확인한다.

## 4. Miss의 chain 길이와 얕은 tree

현재 baseline은 `duet_jit_short=True`에 의해 chain2다. 명시적인 chain2 arm은 같은 조건의 반복이며 새로운 개선으로 계산하지 않는다.

- chain4: draft forward4, node4, 최대 깊이4.
- chain2: forward2, node2, 최대 깊이2.
- chain1: forward1, node1, 최대 깊이1.
- star3: forward1, node3, 최대 깊이1.
- tree2x2: forward2, node4, 최대 깊이2; 첫 형제만 다음 round에서 확장.

`tree2x2 vs chain4`는 node 수를 맞춘 비교다. 더 적은 직렬 draft forward와 첫 두 위치의 넓은 coverage를 얻지만 긴 수락 경로를 포기한다. `star3 vs chain1`은 query/node 수도 달라지는 tradeoff 실험이다. 작은 forward 수가 무조건 높은 AL이나 TPS를 의미하지 않는다.

현재 target query bucket은 K1/K2/최대 node 폭을 기준으로 하므로 chain1과 chain2가 같은 physical verify bucket을 쓸 수 있다. Miss policy가 다음 P1의 context/root 수와 cache 내용에도 영향을 준다. 총 AL, miss 조건부 AL, hit 비율, physical node 폭, TPS를 함께 봐야 한다.

SpecInfer의 token-tree/parallel verification 발상을 참고하지만, 이번 single-draft ordered-WOR shallow tree는 [SpecInfer](https://arxiv.org/abs/2305.09781)의 전체 시스템 재현이 아니다.

## 5. 경계 제외 집계

전체 TPS는 반환한 토큰 수를 전체 decode 시간으로 나눈다. 검증했으나 EOS/출력 상한에서 버린 토큰을 분자에 넣지 않는다. 보조 AL은 상한 도달 또는 suffix clipping이 발생한 마지막 sequence event를 제외한다. 보조 TPS는 하나라도 경계 event가 있는 **배치 step 전체**의 토큰과 시간을 함께 제외한다.

상한에 도달한 요청을 통째로 제거하는 지표는 다른 추정 대상이고, 출력 결과에 따른 선택 편향이 있다. 긴 입력을 실험 전에 고정 기준으로 제외한 것과 동일하지 않다. 전체 480개 결과와 cap 비율을 항상 같이 남긴다.

## 6. Kernel fusion에서 보존하는 fanout 수식

한 root의 이번 예산을 A, 선택된 유효 부모 수를 L이라고 하자. 부모는 기존과 같은 float32 priority 내림차순, 동점이면 lane 순으로 정렬한다. j번째 부모의 fanout은

\[
c_j=\min\left(C,\lfloor A/L\rfloor+\mathbf 1\{j<A\bmod L\}\right).
\]

이는 기존 C번의 round-robin 배분과 같다. 먼저 모든 부모에게 하나씩, 남으면 같은 순서로 둘째·셋째를 준다는 연산을 몫/나머지로 표현한 것이다. 실제 kernel은 root별 rank/count와 기존 future-round reserve를 유지한다. 예산/priority를 바꾸는 새 정책이 아니다. Gain 정책은 별도의 목적함수를 쓰므로 이 round-robin fusion으로 대체하지 않는다.

Mask fusion도 기존 prefix, glue visibility, 선택된 부모의 ancestor bits, 현재 self 위치를 그대로 little-endian packed byte로 작성한다. 매 replay마다 변경된 prefix·glue 폭·valid 값을 읽어야 하며 capture 당시 값으로 고정해서는 안 된다. 여러 ancestor word를 넘는 가변 round 폭, invalid lane, CUDA graph replay에서 byte 단위 동등성을 검사한다.

## 7. Miss tree가 짧다고 AL이 커지지 않는 이유

한 verification에서 수락한 draft token 수를 L이라 하면 recovery를 포함한 AL의 기대값은 tail-sum identity로

\[
\mathbb E[AL]=1+\mathbb E[L]
=1+\sum_{d=1}^{D}\Pr(L\ge d).
\]

이다. 이 식은 위치별 수락 사건의 독립성을 가정하지 않는다. 폭을 늘리면 주로 얕은 위치의 도달 확률을 높이지만, 최대 깊이 D를 줄이면 뒤쪽 항을 제거한다.

- star3는 D=1이므로 AL<=2. 첫 위치 coverage가 완벽해도 이 한계를 넘지 못한다.
- tree2x2는 D=2이므로 AL<=3. 구현상 첫 sibling만 깊이2 자식이 있고 둘째 sibling을 수락한 경로는 깊이1에서 끝난다.
- chain4는 D=4이므로 AL<=5. 반대로 모든 후보가 한 경로에 있어 첫 위치 rejection에 취약하다.

동일 시작 prefix s에서 tree2x2의 정확한 기대 수락 수는 `gamma2(s) + E[1{첫 sibling 수락} * gamma2(s+첫 sibling)]`이다. 두 번째 항은 첫 sibling 수락과 그 이후 context를 함께 평균하며, 서로 독립이라고 두지 않는다. 이 실험은 실제 first-child backbone만 확장하므로 두 sibling 모두 다음 깊이로 확장하는 완전2x2 tree와 다르다.

관측된 Llama3 B8 chain4의 miss 조건부 AL은 약2.31이다. 따라서 깊이1 star가 동일한 miss AL 수준에 도달하는 것은 그 구조의 상한2 때문에 불가능하다. 전체 AL은 별도로 P1/P2 hit mix까지 영향을 받는다. Depth2 tree는 상한3이므로 구조만으로 패배가 정해지는 것은 아니지만, 실측에서는 depth4 chain의 이득을 회복하지 못했다.

조건부 비교도 최종 prefix 분포가 서로 달라지므로 정밀한 same-prefix causal effect는 아니다. 이번에는 명시적 depth/width 개입, 같은 node 수 대조, 실제 miss/P1/P2 길이와 timing을 합쳐 해석한다. 추가 판단을 위해 산술 분해 `RESULTS.json:al_decomposition`과 task group별 `CATEGORY_RESULTS`를 제공한다.


T>0에서 tree 정책이나 depth가 바뀌면 RNG 소비 순서도 달라지므로 동일 seed의 개별 문장이 같을 필요는 없다. `OPTIMIZATION_OUTPUT_PARITY.json`의 bitwise 검사는 알고리즘이 같은18개 cell(8,640개 question 출력 쌍)에만 적용하며 모두 일치했다. 정책이 다른 short/combined arm의 낮은 동일출력 비율을 target 분포 보존 실패로 해석하지 않는다. 분포 보존은 실제 q와 비선견적 미래 확장/ordered-WOR 검증 계약으로 따로 판단한다.
