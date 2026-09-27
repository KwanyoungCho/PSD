# DUET tree 사후 분석 — 전체 데이터 진단

2026-09-22. Root는 cache hit, tree는 hit 이후 descendant AL을 목표로 구분한다.

**이 보고서의 MSE·고정-pool 선택 개선은 새 online tree 정책의 AL 개선율이 아니다.** 기존 관측 없는 3-seed 성능 실험은 AL 2.0651→2.1003 (+1.70%), 차이 CI [-0.0025,+0.0721]로 그대로 보존했다. 이번 진단은 왜 차이가 작고 무엇을 더 개선할 수 있는지 조사한다.

## 1. 실행 범위와 신뢰성

- 전체 Spec-Bench 480문항/560턴 × 두 생성 정책 × seed 1 = **1,120턴**, **37,869 served trees / 262,145 nodes**를 기록했다. Warmup 제외, 모든 hit를 한 번씩 수집했다.
- 같은 모델, exit56, T=.7, depth4/2, node8/6, G=M, 기존 root policy를 사용했다. 출력 상한128, natural EOS, 실제 이전 응답을 포함한 두 번째 turn이다. 공식 chat-template/1024 cap 재현은 아니다.
- Target/draft 전체 분포로 alpha·reach·terminal 확률과 coin 분산을 계산했다. 매32번째 tree의 원본 분포 **1,184개**를 재계산해 검증했다.
- 진단 observer의 CPU/GPU 동기화는 비동기 스케줄에 영향을 준다. 기존 무관측 성능 결과와 절대 AL/TPS를 합치지 않는다. 새 점수들은 **같은 tree**에서 비교했다.
- 기본 진단 가중치는 task 균등 → original question 균등 → 해당 question의 tree 평균이다. 두 생성 정책을 pooled하며 policy별 민감도도 제공한다. CI는 task 내 question cluster 2,000회 bootstrap; 양 turn과 두 정책을 같은 cluster로 유지한다.
- Cross-fit CI는 산출된 보정 table에 조건부인 탐색적 문항 불확실성이다. 각 bootstrap에서 보정 table을 다시 학습하지 않으며 calibration 표본/다른 seed의 불확실성까지 포괄하지 않는다.
- 표의 기본값은 종료 event 제외다. 종료 여부는 수락 결과에 의존할 수 있어, coin 변동/예측 분해는 전체 event도 함께 확인한다. 많은 보조 비교의 CI에는 다중 비교 보정이 없다.

| 수집 정책 | 문항/턴 | 모든 tree | 비종료 tree | node |
|---|---:|---:|---:|---:|
| q_path | 480/560 | 18,991 | 18,499 | 131,417 |
| phase_sibling_q_bin | 480/560 | 18,878 | 18,397 | 130,728 |

## 2. 현재 수식에 의미가 있는가

정확한 목적함수는 E[AL|tree,p,q]=Σ rho(v)다. rho(child)=rho(parent)×앞선 형제 전부 거절 확률×현재 형제 수락 확률이므로 기존 q-path보다 수락 법칙과 직접 연결된다. 그러나 보정한 alpha와 그 곱은 여전히 추정값이다.

**현재 frozen table의 형제 순위 한계는 수식으로 확인됐다.** 첫 형제의 예측 alpha는 P1에서 최소 0.617383, P2에서 최소 0.508532로 모두 .5보다 크다. 뒤 형제의 도달 질량을 전부 합쳐도 1−alpha0보다 클 수 없으므로, 같은 부모의 첫 형제가 대안에 있으면 frozen 점수는 항상 그 형제를 우선한다. 343개 q-bin 조합×2 phase를 모두 검사했다. 다른 root 간 순위까지 고정이라는 뜻은 아니다.

| 점수 | Node reach MSE ↓ | 기대 AL 예측 bias | 예산4 subset의 실제 기대 AL ↑ |
|---|---:|---:|---:|
| 기존 q-path | 0.079673 | -0.0760 | 2.0802 |
| 기존 frozen reach (8문항) | 0.067698 | +0.1500 | 2.1246 |
| 동일 8문항 + depth | 0.067890 | +0.1270 | 2.1240 |
| 동일 8문항 + depth/entropy | 0.069266 | +0.1222 | 2.1215 |
| Cross-fit phase/sibling/q | 0.062085 | -0.0061 | 2.1297 |
| Cross-fit + depth | 0.062103 | -0.0100 | 2.1302 |
| Cross-fit conditional q + depth | 0.061816 | +0.0021 | 2.1311 |
| Cross-fit + depth/entropy | 0.061475 | -0.0124 | 2.1300 |
| Cross-fit 직접 local transition | 0.061143 | -0.0195 | 2.1306 |

Frozen−q-path MSE 차이: **-0.01197 [-0.01330, -0.01068]**.
Cross-fit rich−frozen MSE 차이: **-0.00622 [-0.00669, -0.00575]**.
Cross-fit rich−같은 자료로 refit한 q table MSE 차이: **-0.00061 [-0.00072, -0.00050]**.

동일8문항 보정은 기존 77 calibration trees만 쓴다. Cross-fit은 각 task의 문항을 5개 fold로 나눠 최대 384문항의 관측 tree로 보정하고 나머지 96문항에 평가한다. 평가 문항의 양 turn·양 생성 정책을 보정에서 모두 제외했다. Hit가 없는 문항은 전체 실행 coverage에는 포함하고 undefined tree 지표를 0으로 대체하지 않는다. Cross-fit 개선을 원래 8문항의 저비용 calibration만으로 달성한 결과처럼 해석하지 않는다. 새 neural head나 추가 inference target forward는 없고 scalar lookup 통계 보정이다.

예산4 subset 값은 원래 node 수가 4보다 큰 tree에서만 계산한다. MSE 표본·다른 예산·기존 무관측 성능 실행과 절대 AL을 직접 비교하지 않고 같은 pool/예산 안의 정책 차이를 비교한다.

## 3. Phase·깊이에서 무엇을 못 맞추는가

| Phase | q-path MSE | Frozen MSE | Rich cross-fit MSE | Frozen AL bias | Rich AL bias |
|---|---:|---:|---:|---:|---:|
| P1 | 0.078105 | 0.067012 | 0.060237 | +0.2149 | -0.0062 |
| P2 | 0.086682 | 0.072356 | 0.067259 | -0.0561 | -0.0269 |

| 평가 tree 생성 정책 | q-path MSE | Frozen MSE | Pooled rich MSE | q-path 자료만 보정한 rich | Frozen-policy 자료만 보정한 rich |
|---|---:|---:|---:|---:|---:|
| q_path | 0.079392 | 0.066425 | 0.061076 | 0.061194 | 0.061212 |
| phase_sibling_q_bin | 0.079414 | 0.069043 | 0.062001 | 0.062076 | 0.062176 |

Policy별 보정에서도 평가 question은 다른 정책의 기록까지 보정에서 제외했다. 이는 관측된 두 정책 사이의 transfer 검사이며 새 알고리즘의 모든 frontier에 대한 off-policy 보장은 아니다.

아래 depth 표는 node를 합산한 보조 진단이다. 위 question-balanced 주 표와 가중치가 다르다.

| Phase/depth | Nodes | 정확 alpha | Frozen alpha | q-path MSE | Frozen MSE | Rich MSE |
|---|---:|---:|---:|---:|---:|---:|
| P1/d1 | 83,697 | 0.6400 | 0.7129 | 0.07077 | 0.05646 | 0.05194 |
| P1/d2 | 72,120 | 0.7307 | 0.7854 | 0.06591 | 0.05721 | 0.05159 |
| P1/d3 | 24,040 | 0.8304 | 0.8624 | 0.11281 | 0.10434 | 0.09168 |
| P1/d4 | 24,040 | 0.8413 | 0.8682 | 0.10077 | 0.09726 | 0.08499 |
| P2/d1 | 26,991 | 0.5380 | 0.5437 | 0.09374 | 0.07468 | 0.06972 |
| P2/d2 | 24,099 | 0.5937 | 0.5969 | 0.08454 | 0.06951 | 0.06573 |

| Phase / 형제 순서(0부터) | Nodes | 정확 alpha | Frozen alpha | q-path MSE | Frozen MSE |
|---|---:|---:|---:|---:|---:|
| P1/s0 | 124,151 | 0.7982 | 0.8457 | 0.09849 | 0.08874 |
| P1/s1 | 42,455 | 0.3660 | 0.4541 | 0.06010 | 0.04425 |
| P1/s2 | 37,291 | 0.2416 | 0.3515 | 0.02766 | 0.02150 |
| P2/s0 | 22,292 | 0.6857 | 0.7134 | 0.13411 | 0.11208 |
| P2/s1 | 15,678 | 0.3204 | 0.2490 | 0.07121 | 0.05294 |
| P2/s2 | 13,120 | 0.2118 | 0.2081 | 0.03518 | 0.02760 |

| Phase / target-draft overlap 구간 | Nodes | 정확 alpha | Frozen alpha | Frozen reach MSE |
|---|---:|---:|---:|---:|
| P1 / ≤.25 | 15,811 | 0.1582 | 0.5733 | 0.16686 |
| P1 / (.25,.5] | 25,673 | 0.3531 | 0.5890 | 0.11343 |
| P1 / (.5,.75] | 36,932 | 0.5725 | 0.6333 | 0.07954 |
| P1 / (.75,.9] | 27,811 | 0.7864 | 0.7134 | 0.06128 |
| P1 / >.9 | 97,670 | 0.9811 | 0.9439 | 0.03589 |
| P2 / ≤.25 | 5,889 | 0.1385 | 0.4035 | 0.14097 |
| P2 / (.25,.5] | 9,005 | 0.3348 | 0.4390 | 0.10572 |
| P2 / (.5,.75] | 11,601 | 0.5490 | 0.5072 | 0.07253 |
| P2 / (.75,.9] | 7,486 | 0.7707 | 0.6047 | 0.05905 |
| P2 / >.9 | 17,109 | 0.9640 | 0.8207 | 0.03653 |

Overlap=Σmin(p,q)는 이 분석에서만 사용하는 target 기반 설명 변수다. Online scalar alpha table의 입력에는 넣지 않았다. 전체 문맥의 overlap과 이미 샘플된 특정 토큰의 alpha도 같은 양은 아니다.

Alpha 평균은 exact attempt mass로 가중했다. 뒤 형제가 실제로 시도되지 않았다고 거절 label 0을 주지 않았다. 같은 phase/q라도 depth나 문맥의 p/q 불일치가 달라질 수 있다.

| Oracle 치환 진단 | Reach MSE |
|---|---:|
| Frozen 전체 | 0.067698 |
| 진짜 parent reach × frozen local transition | 0.046811 |
| Frozen parent reach × 진짜 local transition | 0.021815 |

치환 결과는 오류 요인을 알아보는 보조 실험이다. 두 감소량을 더해 전체 오차의 인과적 기여율이라고 부르지 않는다. 곱의 상호작용이 있다.

## 4. 실제 실패 사례와 정보 부족

**Draft가 확신했지만 target은 수락하지 않는 경우**: 첫 형제, attempt≥.3 조건에서 602 node를 찾았다. 이 조건에서 q-path 오차가 가장 큰 사례의 token `'|'` (문항 111_t1, P2, depth1): q=0.999997, p=0.000000, alpha=0.000000, 진짜 reach=0.000000, q-path=0.999997, frozen reach=0.861578.

**Draft 확률은 낮지만 target은 수락하는 경우**: 첫 형제, attempt≥.3 조건에서 2,929 node를 찾았다. 이 조건에서 q-path 오차가 가장 큰 사례의 token `' \\'` (문항 464_t0, P1, depth1): q=0.000000, p=0.000043, alpha=1.000000, 진짜 reach=1.000000, q-path=0.000000, frozen reach=0.652234.

q≥.9인 첫 형제만 제한했을 때의 오류도 확인했다. 아래 비율은 exact attempt mass 가중이다.

| Phase | Nodes | alpha<.5 비율 | alpha<.1 비율 |
|---|---:|---:|---:|
| P1 | 62,508 | 2.48% | 1.04% |
| P2 | 7,731 | 5.34% | 2.67% |
Phase/depth/q/entropy/qmax의 좁은 bin 안에서도 alpha<.05와 alpha>.95 사례가 함께 존재하는 bin을 1,414개 찾았다. 이는 현재 feature만으로 개별 정답을 구별하기 어렵다는 실제 사례다. 더 풍부한 context나 token identity를 써도 불가능하다는 증명은 아니다.

q=(.9,.1), 후보 A가 같아도 p=q면 alpha=1, p=(0,1)이면 alpha=0이다. 따라서 q와 그 entropy만으로 모든 문맥의 정확한 수락을 보장할 수 없다. 이것과 실제 workload에서 평균 예측을 개선할 수 있는지는 별개의 문제다.

## 5. 원래 맞출 수 없는 coin 변동

전체 event에서 exact conditional coin variance 평균은 **0.3360 token²**, 실제 AL−exact expected AL의 제곱오차는 **0.3368**다. 두 평균의 차이가 아닌 signed AL 잔차는 **-0.00206 [-0.00878, +0.00406]**다.

p/q를 완전히 알아도 새 수락 coin의 결과까지 맞힐 수는 없다. 이번 점수 평가는 실제 AL 한 번 대신 exact expected reach/AL을 정답으로 삼아 그 변동을 분리했다. 기대 AL의 점수 오류는 이 coin 변동으로 설명하거나 면책할 수 없다.

## 6. 점수와 선택 알고리즘 중 어느 쪽에 여지가 있는가

같은 실현 pool에서 ancestor+sibling-prefix 제약을 지키며 예산4 subset을 선택했다. 대안 노드를 새로 생성하지 않은 **진단용** 비교다.

| 방법 | 선택된 subset의 exact expected AL |
|---|---:|
| q-path greedy | 2.07823 |
| q-path DP | 2.08020 |
| Frozen reach DP | 2.12461 |
| Rich cross-fit DP | 2.12995 |
| Exact reach oracle DP | 2.24400 |

Rich−frozen: **+0.00534 [+0.00332, +0.00731]**. Oracle−frozen: **+0.11938 [+0.11470, +0.12385]**.

Rich 보정으로 subset 자체가 바뀐 비율은 15.06%, 기대 AL이 늘어난 비율은 3.62%, 줄어든 비율은 6.67%다.

현재 selector는 같은 depth==round 안에서 비교한다. 공통 scale 보정은 MSE를 줄여도 그 round의 순위를 바꾸지 않는다. 보정 table이 모두 공통 배율이라는 뜻은 아니지만, 확률값을 더 잘 맞추는 것과 tree를 바꾸는 것이 다른 이유다.

예산2/6 및 phase별 결과는 analysis.json에 함께 있다. 생성 예산 G=M인 실제 tree에서 모든 node를 유지하면 이 subset 최적화로 추가 AL을 얻지 않는다. 남은 차이를 실제 이득으로 바꾸려면 **샘플을 뽑기 전** 확장·fanout 배분을 개선해야 한다. Token 값을 본 뒤 사후 pruning하는 정책은 별도 분포 보존 문제가 있어 이번에 적용하지 않았다.

## 7. Reach만으로 충분한가: 다음 한 후보의 기대 이득

한 토큰을 추가하기 전의 정확한 이득은 rho(u)×g1(u), g1(u)=Σmin(p_u,q_u)=1−TV(p_u,q_u)다. 같은 depth에서 이미 확장되어 다음 p/q를 아는 위치들끼리 비교했다. 실현된 자식 token의 운을 적분한 one-child 진단이다.

| 선택 우선순위 | 선택 위치의 exact one-child gain |
|---|---:|
| q-path | 0.51329 |
| Frozen reach | 0.55580 |
| Rich reach | 0.55607 |
| Frozen reach × cross-fit g | 0.55587 |
| Rich reach × cross-fit g | 0.55604 |
| Exact rho만 사용 | 0.62085 |
| Exact rho×exact g | 0.62257 |

비교 집합 수 18,558. 진짜 rho만으로도 선택이 달라지는 비율 1.79%, 그 평균 gain 차이는 0.00173다.
Frozen×예측 g − frozen: **+0.00007 [-0.00020, +0.00029]**.
Rich×예측 g − frozen: **+0.00024 [-0.00034, +0.00085]**.

Rich 보정에 따른 위치 선택 변경률 0.44%, 예측 g를 추가했을 때 변경률 0.22%다. Depth1의 대안 집합에 첫 형제가 실제 포함된 18,019개 비교에서 frozen이 첫 형제를 고르는 비율은 100.00%, oracle의 최선이 뒤 형제인 비율은 14.22%다.

G 보정은 node 생성 때 이미 아는 phase/sibling/q/depth/부모 q entropy만 입력으로 썼다. 아직 forward하지 않은 node의 다음 q entropy를 입력으로 쓰지 않았다. 단, training label은 실제 확장된 node에만 있어서 selection bias/coverage 한계가 있다. 확장하지 않은 모든 leaf나 미적중 root에 이 결과를 일반화할 수 없다. 여기의 대안 집합도 실제 scheduler의 전체 frontier를 재현한 것이 아니다.

## 8. 형제 수 배분: 첫 후보부터 하나씩 주면 최적인가

그렇지 않다. p=(0,1), q=(.9,.1)이면 첫 후보까지의 기대 수락 g(1)=.1, 두 후보까지는 g(2)=1이다. 둘째 후보의 추가 이득 .9가 첫째 .1보다 크다. Residual/WOR에서는 추가 후보의 이득이 항상 감소하지 않는다.

부모 A/B의 reach=.8/.2, g_A=(.1,1), g_B=(.9,1)이면 새 node 예산2를 하나씩 주면 .26, A에 둘 다 주면 .8이다. 이는 정확한 반례이며 아래 표는 실제 분포에서 추가 검증한 결과다.

매32번째 원본 snapshot의 **4,595 expanded contexts / 470문항**에서 g(1), g(2)를 vocabulary 전체에 대해 정확히 적분했다. Monte Carlo가 아니다. O(V log V) 계산을 작은 vocab의 350개 완전 열거 사례와 대조했다. 이 정밀 fanout 진단의 표본 범위는 전체 tree를 쓴 앞 절의 범위와 다르다.

| Phase | Contexts | 평균 g(1) | 평균 g(2) | 둘째 추가 이득>첫째인 비율 | Reach 가중 비율 |
|---|---:|---:|---:|---:|---:|
| P1 | 3,813 | 0.7549 | 0.8366 | 5.45% | 4.03% |
| P2 | 782 | 0.6804 | 0.7730 | 7.52% | 7.12% |

이미 확장된 동일-depth 부모 두 개에 새 node 예산2를 줄 때, 1+1과 2+0/0+2를 비교했다. 모두 정확한 rho/p/q를 아는 oracle 진단이다. 새 자식 token을 뽑기 전의 기대값을 비교하며, 한 단계 뒤 손자의 추가 가치는 포함하지 않는다.

| Phase | 쌍 비교 수 | 하나씩 배분 | 최선 배분 | 차이 | 배분이 바뀌는 비율 |
|---|---:|---:|---:|---:|---:|
| P1 | 1,044 | 0.5360 | 0.5863 | 0.0503 | 76.85% |
| P2 | 238 | 0.4056 | 0.4637 | 0.0581 | 77.54% |

전체 차이 CI: **+0.05084 [+0.04309, +0.05972]**. 이 값은 해당 두 후보의 한 번 배분에서 얻는 oracle gain이며, 전체 검증 AL에 그대로 더할 수 없다. q/p가 알려진 expanded subset 및 systematic raw sample에 조건부이고, 실제 모든 frontier의 대안·forward/deadline 경쟁을 포함하지 않는다.

**미래 target 정보를 쓰지 않는 배분 점수도 시험했다.** 기존8문항만으로 phase별 gamma(1)/gamma(2)를 보정하고, frozen reach×gamma(c)의 합이 가장 큰 (1,1)/(2,0)/(0,2)를 고른다. 추가 neural head나 새 full-corpus fitting은 없다.

| Phase | Calibration gamma(1) | gamma(2) |
|---|---:|---:|
| P1 | 0.84147 | 0.91202 |
| P2 | 0.70457 | 0.78292 |

| 같은 두-node 배분 문제의 선택 기준 | 실제 exact expected gain | 기존 하나씩 배분 대비 차이 [95% CI] |
|---|---:|---|
| q-path × calibration curve | 0.52716 | +0.01231 [+0.00325, +0.02131] |
| Frozen reach × calibration curve | 0.53006 | +0.01521 [+0.00692, +0.02394] |
| Exact reach × calibration curve | 0.56376 | +0.04891 [+0.04100, +0.05798] |
| Frozen reach × exact g | 0.54339 | +0.02854 [+0.01921, +0.03810] |

이 작은 배분 검사는 첫 번째 정책의 점수 오차 분석 뒤 추가한 탐색적 follow-up이다. Curve를 맞추는 자료는 기존8문항뿐이고 새 corpus outcome으로 계수를 조정하지 않았다. 실제 생성/검증은 수행하지 않은 국소적인 한 단계 의사결정 검사다. C=3, 여러 round, root 간 배분까지 포함한 end-to-end AL 개선으로 주장하지 않는다.

## 9. 깊이·예산 때문에 멈춘 부분

| Phase | Depth cap에서 끝나는 질량 | Cap 전에 leaf에서 끝나는 질량 | 내부에서 모든 형제 거절 | Early-leaf 추가 길이 상한 |
|---|---:|---:|---:|---:|
| P1 | 0.3652 | 0.2593 | 0.3755 | 0.6696 |
| P2 | 0.5585 | 0.1564 | 0.2851 | 0.1564 |

| Phase | Node cap 미달 tree 비율 | 평균 미사용 node 예산 | Early leaf이면서 raw-q threshold 아래인 terminal mass |
|---|---:|---:|---:|
| P1 | 0.1336 | 0.6682 | 0.0000 |
| P2 | 0.1081 | 0.3244 | 0.0643 |

P2의 raw-q threshold=.03 아래 node는 이후 확장의 eligibility에서 제외된다. 위 질량은 그 조건을 충족하는 early leaf의 질량이다. Threshold를 없애면 모두 확장할 수 있다는 뜻은 아니며 global width/root quota/root-prior threshold도 작용한다. P1 raw-q threshold는 0이다.

첫 번째 종료 구조 표의 마지막 열은 early leaf를 모두 남은 depth까지 공짜로 확장하고 전부 수락시킨다는 낙관적 상한이다. 현재 node/forward 예산에서 얻을 수 있는 개선량이 아니다. 진짜 reach가 0인 node 비율은 45.48%이고, .01 미만은 54.44%다. 그렇다고 이 node들을 사전에 알 수 있거나 안전하게 사후 삭제할 수 있는 것은 아니다.

## 10. 결론의 범위와 재개 지점

- Reach의 합이라는 목적함수는 정확하다. alpha를 추정하는 입력/보정, 그 곱의 의존성, 확장 gain, 예산 알고리즘은 각각 추가 검증 대상이다.
- MSE 개선, 고정-pool 선택 개선, 실제 tree 생성 AL 개선을 서로 구별한다. 새 보정식의 online AL 개선은 이번 사후 분석만으로 확정하지 않는다.
- 다음 구현의 우선순위는 기존 frozen reach와 후보 수별 gain curve를 사용하는 fanout 배분이다. 동일8문항에서 depth/entropy feature만 늘린 개선은 확인하지 못했다. C=3/여러 round/전체 frontier로 확장한 뒤 실제 AL을 검증해야 한다.
- 전체 frontier·미적중 root·확장되지 않은 node에 대한 탐색/감사 기록이 있어야 점수 때문에 놓친 후보와 eligibility/quota 때문에 제외된 후보를 완전히 구별할 수 있다. 현재 저장 자료만으로 그 부분의 전역 최적성을 선언하지 않는다.
- 사용자와의 수식 설명 TODO는 여전히 미완료로 남긴다.

수식의 전체 유도: [THEORY.md](THEORY.md). 구체적 해석과 개발 우선순위: [FINDINGS.md](FINDINGS.md). 원본 실패 사례: [examples.json](examples.json), subset 사례: [failure_examples.json](failure_examples.json). 전체 통계: [analysis.json](analysis.json), 검사: [audit.json](audit.json), [math_audit.json](math_audit.json), 설계: [PLAN.md](PLAN.md).

```bash
ssd/.venv/bin/python results/duet_tree_posthoc/validate.py
ssd/.venv/bin/python results/duet_tree_posthoc/analyze.py
ssd/.venv/bin/python results/duet_tree_posthoc/fanout.py
ssd/.venv/bin/python results/duet_tree_posthoc/examples.py
ssd/.venv/bin/python results/duet_tree_posthoc/sibling_bound.py
ssd/.venv/bin/python results/duet_tree_posthoc/make_report.py
```
