# Round5 — 실제 온도·개선 root/tree를 적용한 DUET와 독립 튜닝 SSD 비교

통합 인수인계: [MERGE_REVIEW.md](../MERGE_REVIEW.md) 20절. 원시 결과, 실패 기록, 실행 계획과 재현 script를 함께 보존한다.

**2026-10-09 표시 자료 추가:** [논문 형식의 Batch·AL·cache hit·TPS 표](paper_view/REPORT.md), [사용 파라미터](paper_view/PARAMETERS.md), [전체 실험 그림 index](paper_view/index.html), [최종 설정 breakdown PDF](paper_view/FINAL_BREAKDOWNS.pdf). 기존 완료487실험/691pass를 CPU에서 재분석했다. GPU 실험을 추가 실행하지 않았으며, 세부 trace가 있는337개와 wall/source 분해만 가능한150개를 구분했다.

## 최종 판단에 먼저 볼 결과

**Llama2 B1에서는 DUET의 AL과 처리량 개선을 확인했다. Llama3 B1은 AL 차이가 통계적으로 뚜렷하지 않았고 SSD가 더 빨랐다. B8은 두 모델 모두 최적화 후에도 SSD가 우세했다.** 이번 선택에서 제외한432개에서도 이 판단이 유지된다. 이는 측정한 workload/장비/설정의 결과이며 B8의 모든 가능한 DUET 변형이 열세라는 증명은 아니다.

아래는 추가 최적화와 마지막 시간 제약 재확인까지 반영한 **처리량 우선 설정**이다. AL 우선점은4·5절에 별도로 남겼다. 값이1보다 작으면 같은GPU에서 튜닝한 SSD가 더 빠르다는 뜻이다.

| Model | B | 최종 K1/K2, exit | DUET/SSD AL* | TPS* DUET/SSD 두 반복 |
|---|---:|---|---|---|
| llama2 | 1 | 11/2, 21 | 2.346 / 2.143 | 1.088x / 1.133x |
| llama2 | 8 | 3/2, 21 | 2.107 / 2.151 | 0.885x / 0.876x |
| llama3 | 1 | 4/2, 16 | 2.690 / 2.661 | 0.939x / 0.966x |
| llama3 | 8 | 3/1, 26 | 2.284 / 2.687 | 0.786x / 0.781x |

질문별AL 구간·432개 확인·실제 반환 TPS를 함께 확인해야 한다. Cache hit만 상승했다고 전체AL/처리량 우위로 결론내리지 않는다.

| Model | B | 반복/GPU | TPS* DUET/SSD | 실제 반환 TPS DUET/SSD |
|---|---:|---|---|---|
| llama2 | 1 | 0 / 2,3 | 127.03/116.73 | 127.88/116.73 |
| llama2 | 1 | 1 / 0,1 | 126.84/111.91 | 127.24/112.05 |
| llama2 | 8 | 0 / 0,1 | 604.86/683.09 | 606.62/687.46 |
| llama2 | 8 | 1 / 2,3 | 597.28/681.79 | 599.95/686.37 |
| llama3 | 1 | 0 / 4,5 | 115.88/123.40 | 115.30/122.71 |
| llama3 | 1 | 1 / 6,7 | 117.21/121.30 | 116.48/120.78 |
| llama3 | 8 | 0 / 6,7 | 537.68/684.36 | 539.27/682.66 |
| llama3 | 8 | 1 / 4,5 | 542.47/694.42 | 541.64/692.40 |

아래는 해당 최종 처리량 설정의 tree 예산이다. Exit는0부터 시작하는 CLI index이며 exit21은22개 layer 실행 후를 뜻한다.

| Model | B | C | P1/P2 nodes | P1 roots | draft fanout | 실제 P2 W | trim/stream |
|---|---:|---:|---|---:|---:|---:|---|
| llama2 | 1 | 3 | 16/4 | 2 | 2 | 12 | 1/1 |
| llama2 | 8 | 3 | 6/4 | 1 | 2 | 4 | 1/0 |
| llama3 | 1 | 3 | 8/4 | 2 | 2 | 5 | 1/1 |
| llama3 | 8 | 1 | 3/1 | 2 | 2 | 4 | 1/0 |

공통: miss chain4, reach_gain_frontier, e(1−q)/full 정규화/overlap mix .25, fused+bulk ON. P2 tree floor는 proxy .01/confidence .03, P1 start/confidence floor는0/0이다. Beta .5는 기록값이지만 이 expansion 정책에서는 비활성이다. **C=1은 각 cache 후보의 continuation에 sibling 분기가 없는 chain 형태**다. 여러 cache root를 유지하는 것과 내부 tree 분기를 늘리는 것을 구분한다.

전체AL*은 경계 event를 제외한 **source 비중×source별AL***의 합으로 분해된다. 아래 조건부AL 차이는 도달 문맥 자체도 다르므로 P1/P2만의 독립적인 인과효과로 해석하지 않는다.

| Model | B | DUET miss/P1/P2 비중 | DUET miss/P1/P2 AL* | SSD miss/hit 비중 | SSD miss/hit AL* |
|---|---:|---|---|---|---|
| llama2 | 1 | 15.3% / 30.9% / 53.8% | 1.937 / 3.541 / 1.776 | 45.4% / 54.6% | 1.777 / 2.447 |
| llama2 | 8 | 23.1% / 27.1% / 49.8% | 1.986 / 2.735 / 1.822 | 45.9% / 54.1% | 1.785 / 2.461 |
| llama3 | 1 | 32.1% / 45.5% / 22.5% | 2.413 / 3.173 / 2.108 | 34.5% / 65.5% | 2.346 / 2.827 |
| llama3 | 8 | 17.2% / 52.1% / 30.7% | 2.365 / 2.654 / 1.611 | 64.6% / 35.4% | 2.450 / 3.120 |

이번 비교는 개선 수식과 tree/구현을 **함께 적용한 시스템 비교**다. e(1−q) 또는 위치 혼합 한 항만의 기여를 SSD 대비 이득 전체로 주장할 수 없다. Root 수식의 독립 기여는 이전 root ablation과 구분한다.

같은 timing 포함 step 집합에서는 **TPS 비율=(반환token/step 비율)/(step 시간 비율)**로 정확히 분해된다. 이는 AL*의 event 제외 집합과 섞어 계산한 근사가 아니다. 아래 두 반복은 각각 같은GPU쌍이다.

| Model | B | 반환token/step DUET/SSD | step 시간 DUET/SSD |
|---|---:|---|---|
| llama2 | 1 | 1.073 / 1.117 | 0.986 / 0.985 |
| llama2 | 8 | 0.984 / 0.972 | 1.111 / 1.110 |
| llama3 | 1 | 1.000 / 1.022 | 1.065 / 1.058 |
| llama3 | 8 | 0.858 / 0.848 | 1.092 / 1.086 |

L2 B8은 timing 포함 step당 반환량이 SSD보다약1.6–2.8% 적고 step 시간은약11% 길었다. L3 B8은 반환량이약14–15% 적은 동시에 step 시간이약9% 길었다. 따라서 B8 열세를 proxy overhead 하나로만 설명하거나 cache hit 개선만으로 해결됐다고 볼 수 없다. 특히 L3 B8은 DUET cache hit82.7% 대 SSD35.5%인데도 전체AL은 낮다. DUET P2의 clean-event 비중30.7%, 해당AL1.611인 점을 함께 봐야 한다.


최종 선택된 **같은 파라미터의 튜닝48 진단**에서 측정한 시간은 아래와 같다. 모든 값은 경계 batch/capture/초기20step/미완성 batch를 제외한 median ms이며 비계측 TPS와 구분한다. 각 열의 median을 합쳐 하나의 실측 timeline으로 해석하지 않는다.

| Model | B | target pre/post | proxy head·score | P1 전체/replay | P2 전체/replay | target wait |
|---|---:|---|---:|---|---|---:|
| llama2 | 1 | 10.38/5.06 | 1.68 | 12.65/12.06 | 2.48/1.64 | 1.08 |
| llama2 | 8 | 13.14/5.99 | 0.81 | 10.82/9.01 | 7.42/5.52 | 5.47 |
| llama3 | 1 | 8.58/8.73 | 2.27 | 11.19/10.76 | 5.32/4.56 | 1.11 |
| llama3 | 8 | 15.32/2.86 | 1.81 | 15.20/13.54 | 5.93/4.29 | 10.42 |

Replay에는 phase graph 내 선택·sampling도 포함하므로 단일-token draft forward latency로 부르지 않는다. Bulk export를 켠 현재 구현에서는 P1/P2의 최종 export를 P2 전체 span에 함께 포함한다.

| Model | B | P1-before-proxy slack | P2-before-final slack | P2-before-ready/next-request slack | 전환 gap J | 실제/물리 query 비율 두 반복 | 전체 batch hit 비율 두 반복 |
|---|---:|---:|---:|---:|---:|---|---|
| llama2 | 1 | -0.51 | 1.15 | 1.71/1.90 | 0.004 | 64.1% / 64.0% | 85.0% / 84.5% |
| llama2 | 8 | 3.96 | -1.29 | -0.42/0.42 | 0.003 | 79.0% / 78.8% | 14.3% / 15.7% |
| llama3 | 1 | -1.58 | 0.82 | 1.49/1.68 | 0.005 | 99.2% / 99.2% | 68.1% / 67.7% |
| llama3 | 8 | 0.68 | -2.00 | -1.25/-0.50 | 0.004 | 74.3% / 74.2% | 21.2% / 22.5% |

음수 slack은 해당 경계를 넘었다는 뜻이다. Next-request는 동일 요청들로 이루어진 다음 batch까지 경계/capture가 없는 경우만 대응시킨다. 현재 step에서 만든 cache의 지연은 다음 요청의 대기에 영향을 준다. Query와 all-hit 비율은 전체 decode 기록의 진단 통계다. 서로 다른 process의 시간 축은 기존 CPU/CUDA anchor에 정렬한 event 시각이다. 0에 아주 가까운 slack을 엄밀한 deadline 여유로 단정하지 않으며 다음 요청 시점도 기록된 target request span의 시작이다. Query 비율을 곧바로 같은 비율의 TPS 손실로 해석하지 않는다. Shape별 횟수는 POSTOPT_RESULTS.json의 query_shape_counts에 있다.

## 1. 실험 범위와 판단 기준

- Full32layer LayerSkip-Llama2-7B + AMD-Llama-135m(FP16), LayerSkip-Llama3-8B + Qwama-0.5B(BF16). 양자화·축소 target을 사용하지 않았다.
- RTX4090 8개를 4쌍으로 나누어 사용했다. Target/draft 각각1GPU. GPU2,3·4,5·6,7은PIX,0,1은NODE. 모든 SSD/DUET 비교는 같은GPU쌍 안에서 대응하며 서로 다른 topology의 절대TPS를 무조건 평균하지 않는다.
- T=.7. 기존 보유480개 first-turn 입력(6분류×80개) 중48개로 파라미터를 선택했다. 최종 결과는 전체480개와 이번 선택에서 제외한432개를 구분한다. 432개는 과거 연구에서도 미관측인 새 데이터라는 뜻이 아니다.
- 기본 확인은 입력cap512/출력cap128, 추가 길이 확인은 입력1024/출력256. 원 Spec-Bench 전체·560turn·무제한context·논문의 출력1024 실험을 완료했다는 뜻이 아니다. 각 raw report의 truncated_prompts와 prompt_ids에 실제 입력을 기록했다.
- 각 최종 설정은 새 draft process 두 번, 각각 전체 입력 warm pass 후 measured pass를 수행했다. Target seed6100/6101 또는6200/6201, draft seed41/42. 두 반복은 GPU쌍을 바꾸되 각 DUET/SSD 비교쌍의 GPU는 동일하다.
- AL은 수락한 draft 토큰과 마지막 recovery/bonus 토큰을 포함한 검증 길이다. AL*은 출력cap 도달·clipping이 발생한 terminal event를 제외한다. TPS*은 그런 요청이 포함된 batch step 전체의 출력과 시간을 함께 제외한다. 실제 반환 토큰 기준 raw TPS도 모두 보존한다. 분자만 제외하거나 수락했지만 반환하지 않은 토큰을 더하지 않았다.
- AL의95% 구간은 질문 단위 paired bootstrap4000회다. 같은 질문의 process 반복을 함께 묶는다. TPS 반복 범위는 신뢰구간이 아니다. AL 우선점과 처리량 우선점을 별도로 비교한다.
- 최종 manifest 집계: 완료487 jobs, 실패4 jobs. 실패한 시도도 삭제하지 않았다. 상세 목록은 [INVENTORY.json](INVENTORY.json).

## 2. 실제 적용한 수식과 정확성 계약

- Proxy e=softmax(z_exit/T_target), draft q=실제 T_draft와 sampler 법칙으로 계산한다. Proxy phase의 내부 context token score는 e(v)(1−q(v)), leaf는 e(v). 이미 제안한 sibling을 제외하고 **전체 허용 vocabulary**로 정규화한다. e가 아직 없는 P1은 draft 기반 root 점수를 유지한다.
- 위치 점수는 .75×observed terminal mass + .25×overlap terminal mass. 여기서 observed는 이미 뽑힌 proposal token에 조건화했다는 뜻이며 실제 target 수락률을 미리 관측했다는 뜻이 아니다. Tree에서는 ordered-WOR sibling의 수락·기각 ladder를 따라 각각 별도로 reach를 계산한 후 terminal mass를 혼합한다. 두 alpha를 먼저 혼합하는 방식과 다르다.
- 위 점수는 cache 후보/예산 추정치다. Target verification의 proposal 확률을 이 점수로 대체하지 않는다. 검증은 실제 draft q와 sibling 조건부 법칙을 유지한다. T=0 생성/검증은 greedy이고 예산 순위에는 soft score를 사용한다.
- Tree는 이전 reach_gain_frontier를 사용한다. Historical8prompt reach/gain table을 고정하고 이번 평가로 재학습하지 않았다. C1/2는 기존C3 gain curve의 prefix를 사용한다. Primary에서는 G=M, NV≤16 제약을 유지한다.
- K2>K1은 independent capacities를 가진 unified tree 경로에서 허용했다. Legacy split-chain/B1-tree guard는 유지한다. 각 모델/B의 T0/.7 full-model smoke 이후 phase-transfer 탐색을 진행했다.
- Chain 환원, ragged bonus, sibling probability mass, live graph temperature 변경, FP16/BF16 모델 기본 dtype하의 float32 온도 buffer, exact correction 보존을 검사했다. 수식 상세는 [THEORY.md](THEORY.md).

## 3. Breakdown을 이용한 파라미터 탐색

- SSD: K={2,4,6,8}, fanout={1,3,5}부터 시작해 경계의 개선 방향을 추가 측정했다. DUET용 tree 설정을 SSD에 강제하지 않았다. 공통 correctness, fast verifier, CUDA graph 최적화는 양쪽에 적용한다. 선택 근거와 cache-build slack은 [SSD_BREAKDOWN.md](SSD_BREAKDOWN.md)에 있다.
- DUET: exit16/21/26/30 → K1/K2 → 초기 root 수와 후속 forward 폭 → C와 노드 예산 → P2 예산 → 결합/인접 설정 → K1을 줄이고 K2를 늘리는 phase 재배분 순서로 확인했다. 마지막48개 warm 비계측 실험으로 선택하고 설정을 동결했다.
- 시간 제약은 C1=a+D1, C2=max(C1,P)+J+D2로 해석한다. P는 proxy 도착, D1/D2는 측정한 phase span, J는 준비가 끝난 뒤 P2 시작까지의 전환·dispatch gap이다. 간단한 식에서 J를 생략할 때는 이를 D2의 유효 비용에 포함해야 한다. P1이 proxy보다 늦더라도 C2가 target 완료 전에 들어오면 전체 cache build는 숨겨질 수 있다.
- K/B/root 폭/tree shape가 바뀌면 target과 draft 비용도 바뀐다. D1/K1을 모델 고유의 단일-token forward latency로 간주하지 않았다. P1 slack, P2-final-logits slack, P2-ready slack을 각각 측정했다. K1 감소는 기본 W=proxy_fan_out×(max(K1,K2)+1), 후속 forward 폭에도 영향을 줄 수 있어 모든 변화가 직렬 forward 감소 하나의 효과라고 해석하지 않는다.
- CLI exit는 **0-based**다. 예를 들어 exit21은22개 layer 실행 후의 출구다. 표는 CLI 값을 그대로 사용한다.
- dynamic/eagle 경로에서는 **tree_beta가 사용되지 않는다**. beta0/1 실험은 네 모델/B 조합 모두 출력·수락 기록이 동일했다. 이 차원을 최적화했다고 주장하지 않으며 timing 차이는 반복 변동이다. [PARAMETER_EFFECTS.json](PARAMETER_EFFECTS.json).
- 유한한 탐색 영역에서 확인한 선택이다. 모든 정수/실수 조합에 대한 전역 최적해 또는 다른 GPU·길이·온도에서의 최적값을 보장하지 않는다. 실행한 모든 조건은 각 *_plan.json과 campaign.json에 있다.

새 모델의 target/draft 비용 비율은 같지 않았다. 동일 K4/2 anchor의 B1 P1 구간은 L2약4.1ms, L3약11.2ms였다. B8의 proxy head·score 경로도 L2약1.0ms, L3약5.2ms로 달랐다. 모델·vocabulary·dtype·kernel 조건이 함께 바뀐 비교이므로 차이를 parameter 수나 vocabulary 크기 하나의 순수한 효과라고 단정하지 않는다. 실제 측정한 두 phase의 비용을 기준으로 예산을 다시 정했다.

| 조건 | 처음 발견한 시간 제약 | 확인한 변경 방향과 의미 |
|---|---|---|
| L2 B1 | K4/2·exit21의 P1-before-proxy 여유 약6.8ms | 작은 AMD draft의 남는 P1 창을 쓰도록 K1을 늘렸다. K12 부근에서 P1 지연이 생겨 구현 변경 뒤 K11까지 재확인했다. K2만 늘리는 것과 분리해 AL/시간을 비교했다. |
| L2 B8 | 같은 K4/2·exit21에서 P2-before-ready 약−1.7ms | K1·P1 root 수·continuation 예산을 줄이고 node/width를 함께 확인했다. 마지막 반복 검증은 trim만 켠 K3/2·exit21을 유지했다. Stream 겹침은 P2 도착을 늦춰 기본으로 켜지 않았다. |
| L3 B1 | K4/2·exit16에서 P1은 proxy보다 약2.3ms 늦지만 P2-before-ready는 약+1.9ms | P1만 보고 실패라고 판단하면 안 된다. 전체 두 phase는 들어갈 수 있다. K6/4는 AL 우선점으로 보존하되 처리량 우선점과 구분했다. |
| L3 B8 | K4/2 anchor에서 P1 약25ms·P2 약12ms; exit16–30 모두 P2가 ready를 넘김 | Exit만 옮기는 것으로 해결되지 않아 K1, root/forward 폭, C, 노드 수를 줄였다. 이후 K2를2→1로 줄이는 후보까지 반복 확인했다. 깊이를 늘리는 AL 우선점은 비용도 함께 보고했다. |

위 숫자는 초기 anchor의 계측 median이다. 최종 선택 설정의 breakdown은 앞 표를 사용한다. 여러 예산을 바꾼 결과를 특정 파라미터 하나의 독립적인 효과라고 주장하지 않는다.

## 4. 추가 구현 최적화 전 최초 동결 설정

| Model | B | 목적/방법 | K1/K2 또는 SSD K | exit | draft fanout | C | P1/P2 nodes | P1 roots | P2 budget |
|---|---:|---|---|---:|---:|---:|---|---:|---|
| llama2 | 1 | ssd_fast | 4 | - | 7 | - | - | - | - |
| llama2 | 1 | ssd_al | 6 | - | 5 | - | - | - | - |
| llama2 | 1 | duet_fast | 12/2 | 21 | 2 | 3 | 16/4 | 2 | 13 |
| llama2 | 1 | duet_al | 12/2 | 21 | 2 | 3 | 16/4 | 2 | 13 |
| llama2 | 8 | ssd_fast | 4 | - | 7 | - | - | - | - |
| llama2 | 8 | ssd_al | 8 | - | 5 | - | - | - | - |
| llama2 | 8 | duet_fast | 3/2 | 21 | 2 | 3 | 6/4 | 1 | 4 |
| llama2 | 8 | duet_al | 6/2 | 21 | 3 | 3 | 12/4 | 2 | 7 |
| llama3 | 1 | ssd_fast | 4 | - | 7 | - | - | - | - |
| llama3 | 1 | ssd_al | 8 | - | 1 | - | - | - | - |
| llama3 | 1 | duet_fast | 4/2 | 16 | 2 | 3 | 8/4 | 2 | 5 |
| llama3 | 1 | duet_al | 6/4 | 16 | 2 | 3 | 12/8 | 2 | 7 |
| llama3 | 8 | ssd_fast | 4 | - | 1 | - | - | - | - |
| llama3 | 8 | ssd_al | 8 | - | 3 | - | - | - | - |
| llama3 | 8 | duet_fast | 3/2 | 26 | 2 | 1 | 3/2 | 2 | 4 |
| llama3 | 8 | duet_al | 6/3 | 30 | 3 | 3 | 12/4 | 2 | 7 |

P2 budget은 default도 실제 W로 풀어 적었다. K1/K2는 직렬 forward 횟수이고 nodes는 root마다 보존하는 continuation 노드 상한이다. P1 초기 roots와 후속 draft fanout은 별개다. Literal env·세부 floor·miss depth·동일 설정 alias는 *_FROZEN.json에 있다. AL 우선점이 더 느린 경우도 숨기지 않는다.

## 5. 전체480개 결과 — 추가 구현 최적화 전

| Model | B | 비교 목적 | DUET AL* | SSD AL* | ΔAL*95% 구간 | TPS* 비율 두 반복 |
|---|---:|---|---:|---:|---|---|
| llama2 | 1 | fast | 2.365 | 2.143 | [+0.175, +0.268] | 1.078x / 1.122x |
| llama2 | 1 | al | 2.365 | 2.259 | [+0.058, +0.152] | 1.088x / 1.088x |
| llama2 | 8 | fast | 2.107 | 2.151 | [-0.079, -0.009] | 0.882x / 0.872x |
| llama2 | 8 | al | 2.319 | 2.310 | [-0.036, +0.052] | 0.758x / 0.769x |
| llama3 | 1 | fast | 2.690 | 2.661 | [-0.012, +0.071] | 0.923x / 0.946x |
| llama3 | 1 | al | 2.977 | 3.011 | [-0.101, +0.032] | 0.985x / 0.963x |
| llama3 | 8 | fast | 2.407 | 2.687 | [-0.316, -0.243] | 0.732x / 0.716x |
| llama3 | 8 | al | 2.900 | 3.064 | [-0.228, -0.099] | 0.832x / 0.852x |

432개 별도 확인·6개 분류별 결과·source별 hit/AL·raw TPS는 [FINAL_RESULTS.json](FINAL_RESULTS.json), 표는 [FINAL_TABLES.md](FINAL_TABLES.md). 새 root 수식·tree·파라미터가 결합된 시스템 비교이므로 이 차이를 특정 수식 하나의 인과효과로 해석하지 않는다.

## 6. 추가 구현 최적화와 SSD 대비 최종 비교

- Ladder trim: 중복 clone, 마지막 sibling 이후 쓰지 않는 draft 정규화, proxy/complement에서 버리는 최종 residual 계산을 제거했다. Exact verifier와 residual 후보 방식의 최종 correction 분포는 유지한다. 기본off, SSD_TREE_LADDER_TRIM=1로 활성화한다.
- Proxy stream: unified batched tree의 TP1 proxy LM head·후보 점수 계산과 target의 후속 layer를 겹친다. Residual 원본을 후속 layer가 바꿀 수 있으므로 default stream에서 독립된 normalized hidden을 먼저 만든다. 최종 logits 뒤 acceptance/재사용 전에 완료 event를 기다린다. TP>1의 collective head는 원래 stream에 유지하며 이번 full 실험은TP1이다. 기본off, SSD_BATCH_TREE_PROXY_STREAM=1.
- 두 변경은 각각 프로파일 대조와 전체480개 동일 파라미터·시드·GPU 반복으로 확인했다. Token 출력과 step별 source·proposal node 수·수락/반환 길이 기록의 동일성을 두 pass 모두 검사한다. 모든 후보의 전체 확률 tensor를 full run마다 저장했다는 뜻은 아니다. Standalone kernel 이득을 전체 생성 이득으로 보고하지 않는다.
- Fused math/bulk export 기존 최적화도 B8에서 OFF 대조와 비교했다. 구체적인 속도 및 동일성 기록은 [FOLLOWUP_RESULTS.json](FOLLOWUP_RESULTS.json).

| Model | B | 기본 DUET/SSD TPS* | trim DUET/SSD | trim+stream DUET/SSD |
|---|---:|---|---|---|
| llama2 | 1 | 1.078x / 1.122x | 1.082x / 1.126x | 1.085x / 1.138x |
| llama2 | 8 | 0.882x / 0.872x | 0.885x / 0.876x | 0.831x / 0.812x |
| llama3 | 1 | 0.923x / 0.946x | 0.926x / 0.952x | 0.939x / 0.966x |
| llama3 | 8 | 0.732x / 0.716x | 0.742x / 0.720x | 0.705x / 0.701x |

표의 두 값은 각각 동일 GPU쌍의 대응 반복이다. 최적화 전후 AL은 출력 동일성 검사로 동일하며, 추가 기능이 항상 빨라진다는 가정을 두지 않는다.

관측된 fixed-setting 효과: ladder trim은0.3–1.3% 정도의 작은 TPS 증가였다. Proxy stream은 B1에서0.3–1.5% 증가했지만 B8에서는2.6–7.3% 감소했다. 이는 전체480 두 반복의 관측 범위이지 latency 유의성 구간이 아니다. B8의 stream 진단에서는 proxy 도착과 P2 완료가 늦어지는 현상이 동반된다. 동일 GPU 자원 경쟁과 일치하는 결과지만 hardware counter로 원인을 분리한 것은 아니다. 따라서 overlap이 생겼다는 사실만으로 stream을 기본 추천하지 않는다. Fused+bulk의 이번 B8 고정 설정 이득은 L2약1.4%, L3약3.8%이며 이전 Round4의 다른 설정 수치와 섞지 않는다.

겹침 구현 변경 후에는 K1−1/K2−1/exit±2를 **48개 튜닝 입력에서만** 재확인했다. 네 인접 설정의 breakdown을 확인하고 상위 두 개와 stream/trim 기본 설정을 warm 비교했다. 2% 이상 개선할 때만 바꾸는 일회성 선택 규칙을 사용했다. 이는 유의성 검정이나 serving threshold가 아니다. 바뀐 설정은 전체480개 두 반복을 추가하고, 그대로인 설정은 이미 완료한 대응 결과를 재사용했다. 다만 B8에서는 짧은 튜닝 pass의 드문 장시간 step이 순위를 왜곡할 수 있어 이 선택을 잠정 기록으로 보존하고 추가 재검증했다. 같은 process에서3회 예열+3회 측정하고, 개별 slow step을 제거하지 않은 pass별 TPS*의 중앙값으로 최종 선택했다. 기존 trim 기준보다2% 이상 높을 때 변경했다. Llama3의 기존 K3/1·exit24/26 후보도 포함했고 SSD의 선택 설정 역시 같은 반복 protocol로 재확인했다. [TUNING_TAIL_AUDIT.md](TUNING_TAIL_AUDIT.md)에 전후 tail을 기록했다. 최종 추천 값과 비교는 [POSTOPT_TABLES.md](POSTOPT_TABLES.md), 전체 기록은 [POSTOPT_RESULTS.json](POSTOPT_RESULTS.json)에 있다. [FINAL_PRESETS.json](FINAL_PRESETS.json)에는 실제 실행한 전체 CLI/env/engine 설정을 저장했으며, 각 *_RECOMMENDED_PLAN.json은 최종DUET와 SSD 처리량 설정을 바로 재실행하는 계획이다. AL 우선점은 최초 동결 지점을 그대로 별도 보존한다.

### B8 반복 측정으로 확인한 마지막 선택

아래는 **튜닝48** 결과이며 최종480 성능과 구분한다. 각 후보는 동일 process에서3회 예열 후3회 측정했다. TPS*는 slow step을 포함한다. 선택은 세 pass TPS*의 중앙값으로 하되 원래 trim 기준보다2% 이상 개선할 때 변경했다. SSD 행은 이미 독립 선택한 설정의 반복 대조이며 DUET 후보 선택의 기준값은 아니다.

| Model | 방법/선택 | K1/K2 또는 K | exit | nodes | stream | 측정 TPS* 3회 | 중앙값 AL* |
|---|---|---|---:|---|---:|---|---:|
| llama2 | DUET | 3/2 | 19 | 6/4 | 1 | 556.9 / 570.8 / 552.9 | 2.130 |
| llama2 | DUET | 3/1 | 21 | 6/4 | 1 | 562.4 / 500.1 / 546.6 | 2.042 |
| llama2 | DUET | 3/2 | 21 | 6/4 | 1 | 530.1 / 549.0 / 545.6 | 2.110 |
| llama2 | DUET **선택** (기준) | 3/2 | 21 | 6/4 | 0 | 565.9 / 575.2 / 576.6 | 2.110 |
| llama2 | SSD | 4 | - | - | 0 | 649.7 / 640.3 / 668.4 | 2.130 |
| llama3 | DUET | 3/1 | 26 | 3/2 | 1 | 436.8 / 557.8 / 529.4 | 2.436 |
| llama3 | DUET | 3/2 | 28 | 3/2 | 1 | 379.7 / 449.2 / 467.6 | 2.435 |
| llama3 | DUET | 3/2 | 26 | 3/2 | 1 | 474.4 / 474.8 / 477.0 | 2.438 |
| llama3 | DUET (기준) | 3/2 | 26 | 3/2 | 0 | 496.2 / 495.8 / 500.4 | 2.438 |
| llama3 | DUET **선택** | 3/1 | 26 | 3/1 | 0 | 470.6 / 591.6 / 556.4 | 2.436 |
| llama3 | DUET | 3/1 | 26 | 3/1 | 1 | 448.8 / 561.1 / 532.0 | 2.436 |
| llama3 | DUET | 3/1 | 24 | 3/1 | 0 | 550.3 / 564.7 / 547.8 | 2.352 |
| llama3 | DUET | 3/1 | 24 | 3/1 | 1 | 535.1 / 549.2 / 532.1 | 2.352 |
| llama3 | SSD | 4 | - | - | 0 | 700.0 / 753.6 / 639.4 | 2.801 |

세 번의 반복은 전체 모집단에 대한 정확한 latency 신뢰구간이나 전역 최적성을 보장하지 않는다. 전체 pass별 median/max step·tail 횟수는 TUNING_TAIL_AUDIT.json에 보존했다. L3 B8의 exit26/24 trim 후보 중앙값 차이는약1.1%에 불과하므로 exit26이 유일하게 최적이라고 주장하지 않는다.

**최종 확인에서 재튜닝 효과가 어디까지 재현됐는가:** L2 B1의 K12→11은 이미 trim+stream을 켠 K12 대비TPS가 +0.30%/−0.43%로, 추가 이득이 확실하게 재현되지 않았다. 원래 K12의AL2.365보다 K11의AL2.346이 낮아 AL 우선점은K12를 유지한다. 따라서 K11은 튜닝에서 선택된 처리량 후보이지 검증된 유일 최적점이 아니다. L3 B8의 K3/1·nodes3/1 최종 설정은 최초K3/2·nodes3/2 대비TPS가7.4–9.1% 증가했지만AL은5.1% 감소했다. 이 증가는 코드 옵션만의 효과가 아니라 phase/node 예산 변경까지 포함하며 SSD 대비TPS 우위로 이어지지는 않았다.

별도 계측 실행에서는 clean-step 중앙값의5배를 넘는 step 682개 중 675개가 같은 step 또는 직전 step의 capture event와 겹쳤다. [CAPTURE_TAIL_AUDIT.json](CAPTURE_TAIL_AUDIT.json). 이는 lazy capture가 큰 지연을 만들 수 있다는 계측 근거다. 다른 seed로 실행한 비계측 튜닝의 개별 tail 원인까지 모두 확정한 것은 아니다. Tree 구조마다 graph를 매번 새로 만드는 것이 아니라 batch/query/page bucket 등의 새 shape에서 capture하고 재사용한다.

## 7. B2/B4·긴 입력/출력·원인 분석 자료

- B2/B4는 **최초 동결한** B8 처리량 우선 파라미터를 그대로 이전한 full480 실험이다. B2/B4 각각 독립 최적화하거나 마지막 B8 재선택을 다시 이전했다는 의미는 아니다.
- 입력1024/출력256도 최초 B8 설정을 이전했다. 길이 효과와 설정 재튜닝 효과를 혼합하지 않는다.
- [FOLLOWUP_TABLES.md](FOLLOWUP_TABLES.md): 추가 batch/길이와 각 구현 대조.
- [BREAKDOWN_TABLES.md](BREAKDOWN_TABLES.md), [BREAKDOWN.json](BREAKDOWN.json): phase별 median, slack, all-hit/some-miss 분리, 실제 대표step timeline. 별개 구간의 median을 이어붙여 하나의 실측step이라고 표현하지 않는다. 초기 screening에는 나중에 추가한 target/glue capture label이 없어 일부 lazy capture가 남을 수 있다. 최종 선택과 속도 비교는 warm 비계측 결과를 사용한다.
- [figs/01_tuning_pareto.png](figs/01_tuning_pareto.png): 계측된 튜닝 후보의 AL/step-time 관계. 최종TPS 그림이 아니다.
- [figs/02_phase_deadlines.png](figs/02_phase_deadlines.png): exit 변경 시 두 phase의 시간 제약.
- [figs/03_frozen_full_comparison.png](figs/03_frozen_full_comparison.png), [figs/04_category_al.png](figs/04_category_al.png): 비계측 최종 비교 및 분류별AL.
- [figs/05_implementation_vs_ssd.png](figs/05_implementation_vs_ssd.png): 추가 최적화별 SSD 대비 처리량. [figs/06_proxy_overlap_timeline.png](figs/06_proxy_overlap_timeline.png): 동일 step의 실제 stream 전후 timeline.
- [figs/07_final_selected_comparison.png](figs/07_final_selected_comparison.png): 최종 추천 처리량 설정의 전체480/선택 제외432 AL 구간과 두 번의 GPU 대응 TPS 비교.
- 기존 논문은70B/TinyLlama, Blackwell targetTP2, B1, 출력1024이며 exit56/K8+4를 사용했다. 그 그림의 proxy1.7%·miss stall13%를 이번7B/8B4090 측정치로 그대로 대체할 수 없다. 이번에는 각 새 모델의 실측 window로 설정했다.

## 8. 검증·실패 기록·재현

- 최종 GPU 회귀는 trim0/stream0, trim1/stream0, trim1/stream1 세 구성에서 각각 실행했다. root/phase smoke와 full-model 비교는 별도다. [FINAL_REGRESSIONS.json](FINAL_REGRESSIONS.json), [FINAL_STREAM_REGRESSIONS.json](FINAL_STREAM_REGRESSIONS.json)과 각 log에 실제 테스트 수가 있다.
- B8 안정성 재검증의 동일 파라미터 stream ON/OFF 네 쌍도 각각6개 pass의 token/source/수락 기록이 동일한지 검사했다. 새 C1·작은 node shape도 포함한다. [STABILITY_PARITY.json](STABILITY_PARITY.json).
- 초기 GPU 회귀가 SSD 한 튜닝 조건과 겹친 기록은 timing에서 제외하고 독립 반복했다. Dtype 편집 중 import 실패, prototype CUDA capture의 scalar-temperature 실패도 보존한다. [EXCLUSIONS.json](EXCLUSIONS.json), [NOTES.md](NOTES.md).
- [TELEMETRY.json](TELEMETRY.json)에 GPU 점유·온도·clock·외부PID를 기록한다. 서로 다른 GPU쌍의 두 반복 범위를 latency CI라고 부르지 않는다.
- [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json), 각 raw report의 runtime_source_sha256, 모델 manifest, calibration SHA를 사용한다. 환경에 맞춘 model/GPU 경로 변경에는 relocate_plan.py를 사용한다. 기존 plan을 직접 덮어쓰지 않는다.

- 원본1169개를 빈 임시 checkout에 실제 복원해 SHA256·크기·mtime을 검사했고, 다른 경로에서 retry profile과 최종 선택 profile의 분석값도 일치했다. [ARCHIVE_RESTORE_AUDIT.json](ARCHIVE_RESTORE_AUDIT.json).

```bash
python results/mlsys_coverage/round5/archive_results.py --restore
python results/mlsys_coverage/round5/analyze_screen.py
python results/mlsys_coverage/round5/analyze_final.py
python results/mlsys_coverage/round5/analyze_followups.py
python results/mlsys_coverage/round5/analyze_postopt.py
python results/mlsys_coverage/round5/analyze_breakdown.py
python results/mlsys_coverage/round5/analyze_ssd.py
python results/mlsys_coverage/round5/audit_tuning_tails.py
python results/mlsys_coverage/round5/make_figs.py
```

다른 서버의 tree/root 변경과 merge할 때 duet_proxy_source↔duet_root_source 명칭, actual-temperature buffer, ordered-WOR q, miss depth와 recursion depth, G=M, request별 예산, side-stream 완료 event를 함께 검토해야 한다. 논문 환경70B/Blackwell 재현, 출력1024, 모델별 새 calibration, 더 넓은 데이터/온도/하드웨어와 전역 파라미터 최적성은 이 실험의 검증 범위 밖이다.
