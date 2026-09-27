# DUET Calibrator: 시간 제약과 생성 토큰 보상을 분리한 사전 설정

이 문서는 새 설계다. 기존 `calibrate_k_balance.sh`의 signed-gap 최소화 추천을
채택하지 않는다. 기존 코드는 측정 label/실행 계약을 확인하는 자료로만 사용한다.
실측 결과가 나오기 전부터 calibration의 성공을 가정하지 않는다.

## 1. 목적함수

한 verification step의 결과를 c∈{0,1,2}로 나눈다: cache miss, P1 hit, P2 hit.
π_c는 **서로 배타적인 실제 phase 비율**, A_c는 해당 경우 수락한 speculative
token 수의 조건부 평균(항상 붙는 recovery/bonus 한 개 제외)이다.

U = E[emitted tokens per step] = 1 + π_0 A_0 + π_1 A_1 + π_2 A_2.

정상 상태의 decode 처리율 목적함수는 G=E[U_s]/E[T_s]이다. 상태별 U/T를 평균하지
않는다. 유한 요청의 실제 지표는 잘린 마지막 step, prefill, 요청 경계까지 포함한
`sum(actual output tokens)/sum(generate wall seconds)`로 별도로 검증한다.

사용자 제안 `hit × AL_hit / timestep`은 유용한 출발점이지만, DUET은 miss에도
JIT draft를 생성하고 target verification을 수행한다. 따라서 miss 보상, P1/P2
길이 차이와 recovery 한 개를 빠뜨리면 후보 설정의 순위가 달라질 수 있다.
P2 hit는 P1 root와의 dedup/우선순위 이후 결과이므로 두 hit를 독립 확률로 더하지 않는다.

## 2. 실행 DAG와 시간 제약

현재 step의 target verification 시작을 시간 원점으로 놓는다.

- x: draft가 glue를 마치고 P1 계산을 시작하는 상대 시각.
- D1: P1 root 생성과 forward들의 전체 소요 시간.
- P: proxy 후보가 draft에 도착하는 상대 시각.
- D2: P2 선정과 forward들의 전체 소요 시간.
- M: cache merge/정리 시간.
- F: target이 verification과 sampling을 마치고 다음 request를 보낼 수 있는 시각.
- S: 두 파이프라인이 준비된 후 cache lookup/JIT/응답/다음 verify 준비 시간.

Draft의 cache 준비 시각은 C=max(x+D1,P)+D2+M이다.
개념적 cycle model은 T=max(F,C)+S이다. S의 상태 의존성과 이전 step에서 넘어오는
대기는 실제 trace로 분리해야 한다. 장치 간 CUDA-event anchor 오차도 기록한다.
Target compute와 draft compute를 모두 더하는 직렬 모델은 이 overlap을 놓친다.

이 max에는 중요한 평탄 구간이 있다. 다른 성분을 고정했을 때
`P(ell) <= x+D1`이면 `C=x+D1+D2+M`이므로 proxy가 조금 더 늦게 와도
P2 시작 시각은 변하지 않는다. 이 구간에서는 exit를 늦췄다는 이유만으로 P2의
forward 예산을 줄일 필요가 없다. 반대로 `P(ell)>x+D1` 구간에서는 도착 지연이
cache 준비 지연으로 그대로 이어진다. 다시 `C<=F`이면 그 차이도 target 실행 뒤에
숨는다. 따라서 exit 증가에 따른 시간 비용은 하나의 직선이 아니라 병목이 바뀌는
구간별 함수다. 실제 F/D2도 형상 및 proxy 부하로 변하므로 그 성분은 별도 측정한다.

P는 단순 `exit_layer/num_layers × target_full_forward`가 아니다.
Target prefix 실행 + norm/head + 후보 선정 + 통신/stream 의존성을 포함한다.
F 역시 검증 node 수, attention 형상, 실제 prefix 길이와 target 후처리에 의존한다.

모든 draft 일을 target 시간 안에 숨기는 조건은

    max(x+D1,P)+D2+M <= F.

한 forward 시간이 국소적으로 d1,d2이고 나머지 비용이 o1,o2라면

    K1 <= floor((P-x-o1)/d1)
    K2 <= floor((F-max(x+o1+K1*d1,P)-o2-M)/d2)

는 각각 proxy 도착 전 P1 완료와 cache 준비 완료의 근사 경계다. 음수 예산은
가능한 forward 0개라는 진단이며, 엔진의 K>=1 제약과 충돌하면 다른 exit/shape를
골라야 한다. 실제 d1,d2가 round/context/폭에 따라 변하면 누적 실측 비용을 쓴다.

이 경계를 전역 최적의 필요조건이라고 해석하면 안 된다. 시간 밖으로 한 round를
더 실행해도 보상이 충분히 늘면 TPS가 높아질 수 있다. ΔT>0일 때 정확한 판정은

    (U+ΔU)/(T+ΔT) > U/T  <=>  ΔU/ΔT > U/T.

따라서 gap이 가장 작은 설정과 TPS가 가장 높은 설정은 구별해서 평가한다.

## 3. forward latency는 무엇에 대해 고정적인가

같은 모델/quantization/TP, batch, query node 수, KV length bucket, kernel 및 graph
형상 안에서는 반복 시간을 측정해 대표값과 p10/p90을 얻을 수 있다. 형상이 바뀐
설정까지 한 상수로 놓는 것은 검증 대상이다.

현재 chain의 P1 forward 폭은 `3*(valid_k+1)`처럼 현재 요청의 long/short 상태와
fanout에 따라 변한다. K1 변경은 forward 횟수뿐 아니라 long 경로의 폭, target의
verification 길이, 다음 상태 확률도 바꾼다. K2 변경도 P2/miss verification 길이를
바꾼다. 단일 baseline의 `P1 전체 시간/K1`만으로 다른 설정을 정확히 예측할 수 없다.

Tree는 generation node 수 N_gen과 실제 verification node 수 N_verify가 분리되어
있다. N_gen 증가가 N_verify 증가를 반드시 의미하지 않는다. 반대로 threshold가
확장을 막아도 실행 graph 폭이 고정이면 GPU 시간이 줄지 않을 수 있다.

### 이번 하드웨어에서 얻은 예시 계수

5개 anchor의 trace에서 보정한 값은 다음과 같다. 단위는 ms, c=context_length/512,
n은 현재 target에서 검증하는 chain의 speculative 길이(K1 또는 K2)다.

    F  ≈ 20.238 + 0.697 n + 0.237 c
    D1 ≈ 0.071 + K1 (3.580 + 0.034 n + 0.248 c)
    D2 ≈ 0.849 + K2 (3.620 + 0.377 c)
    P  ≈ 2.642 + (ell/80) (15.160 + 0.708 n + 0.166 c)

이 계수는 보편적인 모델 상수가 아니라 현재 모델/하드웨어/형상의 calibration
결과다. D1의 K1*n 항은 횟수와 현재 경로의 폭을 함께 반영한다. D2는 root budget=15가
고정된 fit이므로 budget=24의 비용을 그대로 계산하는 데 쓰지 않는다. x/S와 계수 원본은
[frozen.json](frozen.json)에 있다. Transformer나 proxy head를 학습한 것은 아니다.

첫 19개 검증에서는 cycle MAPE=1.58%였지만, 별도의 K1=K2=2 경계 실험에서는
8.10–13.56%였다. 특히 F와 S를 작게 예측했다. 그러므로 측정 범위를 벗어나거나
어느 파이프라인이 마지막으로 준비되는지가 달라지면 새 anchor로 보정해야 한다.

## 4. 품질 관측과 식별 가능한 범위

π_c와 A_c는 시간 측정만으로 알 수 없다. 같은 latency의 두 proxy가 서로 다른
top-token 순위를 가질 수 있으므로, 시간-only calibration으로 최적 exit를 보장할
수 없다는 것은 식별 가능성의 문제다.

동일 prefix에서 여러 exit의 e_l과 p,q를 저장하면, 후보 점수/혼합 계수/root 예산에
따른 coverage를 모델을 다시 실행하지 않고 평가할 수 있다. 그러나 이 결과는
저장된 경로에서의 counterfactual root coverage이며 새 정책의 장기 π_c·A_c·TPS를
정확히 주지 않는다. P1 dedup, 선택한 root 이후의 continuation 품질, 시간 내 준비
완료 여부를 포함한 소수의 실제 실행으로 마지막 검증을 해야 한다.

깊은 exit의 품질 단조 증가도 가정하지 않는다. 실제 layer별 proxy가 바뀌면서
coverage가 감소하는 위치/평균이 있을 수 있다. 단조 회귀로 그 결과를 숨기지 않는다.

## 5. 새 추천 절차와 평가 계약

1. 하드웨어/모델/temperature/context 범위를 fingerprint한다.
2. 독립 calibration 프롬프트에서 소수 anchor 설정의 시간 성분과 실제 phase 보상을 측정한다.
3. Max 연산을 포함한 overlap 모델과 phase별 reward 모델로 미측정 조합을 점수화한다.
4. 단일 추천과 top-3 shortlist, 측정 범위/외삽 여부를 저장하고 동결한다.
5. 별도 validation 프롬프트에서 작은 격자 전체를 실행해 prediction error, shortlist
   recall, measured-oracle 대비 regret을 계산한다. 이 전체 격자는 **방법 검증용 비용**이며
   실제 calibration 비용에서 숨기지 않고 별도 보고한다.
6. 추천과 실제 격자 최고 설정 등을 또 다른 프롬프트/seed에서 재검증한다. 검증 격자에서
   고른 최고 설정은 validation oracle이지 미측정 조합까지의 전역 최적이 아니다.

비교군: 기존 기본값, 시간 균형만 보는 추천, calibration에서 직접 측정한 최고 anchor,
새 모델의 top-1/top-3. 성공/실패는 실제 regret과 실행 비용으로 판단한다.

## 6. 파라미터 분류와 tree 교체 경계

| 종류 | 처리 |
|---|---|
| exit layer, K1, K2 | 시간과 품질을 함께 비교하는 주요 탐색 축 |
| P1 fanout, P2 root/forward 폭 | graph 폭에 따른 비용 표와 품질 관측 필요 |
| N_gen, N_verify, C_tensor | tree adapter가 유효성/shape/보상 정보를 제공 |
| tree threshold | 같은 폭 안에서 선택 품질을 바꾸는 별도 선택기 설정 |
| h 혼합 계수, source 할인/floor | 저장 p/q/e replay로 저비용 후보 축소 후 실제 확인 |
| spec_k | 현재 split 구현에서 K1+K2로 유도; 독립 sweep하지 않음 |
| top-M, wire 길이, padding/layout | 후보/중복제거/예산 계약을 만족하도록 유도; 임의 축소 금지 |

Tree 선정기를 바꿀 때 calibration의 목적함수와 탐색기를 다시 작성하지 않도록,
adapter는 `validate(config)`, `shape(config,state)`, `cost(profile,config,state)`,
`reward(calibration,config,state)`를 제공한다. Cost-only adapter로는 품질 최적을
보장할 수 없으며, 미측정 shape/정책은 '추가 calibration 필요'로 표기한다.

## 7. 연구 의미와 선행 연구 범위

사전 hardware profiling과 acceptance 기반 예산 선택 자체는 새로운 개념이라고
주장하지 않는다. Sequoia는 hardware-aware tree size/depth optimization,
TurboSpec은 execution profiling과 feedback을 이용한 goodput 기반 speculation 양 선택을 다룬다.

- https://arxiv.org/abs/2402.12374
- https://arxiv.org/abs/2406.14066

DUET에서 별도로 검증할 부분은 early-exit 도착 시각과 두 cache phase의 overlap,
배타적 P1/P2/miss 보상, tree generation/verification 분리, 후보 점수와 실행 시간의
결합이다. 의미 있는 결과는 'calibration이 가능하다'는 주장보다 적은 anchor로
얼마나 낮은 regret을 달성하는지와 재calibration 비용이다.
