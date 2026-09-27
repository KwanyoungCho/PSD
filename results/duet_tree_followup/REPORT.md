# DUET tree 후속 검증 — 통합 비교 보고서

**상태: 기존 데이터 기반 C=3 검증과 실행기 검사는 완료. 실제 70B 전체 생성 비교는 GPU 대기 중 또는 실행 중; 아직 결과 없음.**

이 보고서는 기존 점수, 새로운 fanout 배분, 전체 frontier 허용을 구분한다. root 후보 선정은 그대로 두고 descendant AL을 목적함수로 둔다. TPS로 승자를 고르지 않는다.

## 1. 현재 판단

- 기존 full480×3seed에서 q product→보정 reach의 AL은 **2.065133→2.100308 (+1.70%)**, 95% CI [-0.002489, +0.072056]였다. 전체 평균 우월성은 아직 확정되지 않았다.
- 실제 C=3 및 관측된 라운드 예산으로 확장한 국소 비교는 **0.685109→0.696515**, +1.66%, 차이 CI [-0.002021, +0.024562]다. 이 역시 확정적 개선이라고 말할 수 없다.
- 따라서 이전 두-node 국소 실험의 +2.95%를 C=3 또는 실제 전체 AL의 개선으로 확대 해석하면 안 된다.
- q 점수+gain과 보정 reach+gain을 모두 실제 비교해야 한다. Reach MSE가 더 작은 점수가 배분 후 AL까지 반드시 더 높이지는 않는다.
- 새 정책은 실험 전용 process hook으로 구현했다. Production 기본 정책과 기존 frozen 결과는 바꾸지 않았다.

## 2. 실제로 바꾼 결정

기존은 선택된 부모들에게 첫 자식을 하나씩 준 다음 둘째·셋째 자식을 순서대로 배정했다. 새 정책은 새 child token을 뽑기 전에, 같은 root의 남은 예산 안에서 `sum rho_hat(u) * gamma_phase(c_u)`를 최대화한다. C=3, B≤8의 정수 배분을 정확히 비교한다.

| 정책 | 확장 점수 | 자식 수 배분 | frontier |
|---|---|---|---|
| q_path | 기존 q product | 기존 breadth | depth=round |
| reach | 기존 8문항 보정 reach | 기존 breadth | depth=round |
| q_gain | q product | gain 최대화 | depth=round |
| reach_gain | 보정 reach | gain 최대화 | depth=round |
| reach_frontier | 보정 reach | 기존 breadth | 남아 있는 모든 unexpanded 노드 |
| reach_gain_frontier | 보정 reach | gain 최대화 | 남아 있는 모든 unexpanded 노드 |

모든 정책은 root 분포, threshold, depth 4/2, node 8/6, G=M, 물리적인 round별 forward 폭과 future-round reserve를 유지한다. 한 라운드의 예측 목적을 최적화하며, 미래 미관측 분기까지 포함한 전체 tree의 전역 최적성은 주장하지 않는다.
현재 P2는 2라운드이고 첫 라운드에서 모든 활성 root를 평가하므로, frontier 변경 자체는 P2 tree 내부에서 추가로 재고할 이전-depth 노드를 만들지 않는다. 실제 P2 통계는 앞선 생성 경로/캐시 구성 변화로 달라질 수 있다.

## 3. C=3 calibration

기존 **8문항, 77개 tree, expanded descendant 문맥 230개**만 사용했다. 추가 neural 학습이나 full benchmark를 사용한 fitting은 없다.

| phase | gamma(1) / gamma(2) / gamma(3) | gamma(3) 수치적분 SE |
|---|---|---|
| P1 | 0.841473 / 0.912016 / 0.936012 | 6.85e-10 |
| P2 | 0.704568 / 0.782921 / 0.830813 | 3.18e-09 |

gamma(1/2)는 정확하게 계산했다. gamma(3)는 첫 거절 토큰을 조건으로 한 정확한 재귀식을 수치적분했다. 실제 vocabulary에서의 결과를 모두 정확한 전수열거라고 부르지 않는다. SE 외에 미적분 tail 전체 질량에 의한 결정론적 구간도 저장했다. 수치적분 오차보다 8문항 calibration의 문맥 대표성이 더 큰 불확실성이다. [수식과 증명](THEORY.md), [calibration 원자료](gain_calibration.json).

## 4. 저장된 전체 데이터에서 C=3 비교

기존 full480 corpus의 두 정책 기록에서 추출한 deterministic 1/32 raw sample을 사용했다. **3,438개 관측된 확장 문맥, 460개 질문**에 C=3 기대 이득을 계산했다. 아래 배분 비교가 가능한 것은 **602개 라운드, 369개 질문**이다. 나머지 질문을 0점으로 채우지 않았다.

마지막 verification event는 제외했다. Nonfinal 여부와 cache-hit pool 자체가 생성 결과에 의존하므로, 이 조건부 표본의 prospective 값을 전체 policy intervention의 불편 추정치라고 주장하지 않는다.

같은 observed parent pool, 같은 해당 라운드의 node 수에서, 아직 뽑지 않은 새 자식의 기대 기여를 비교한다. 기존 sampled child의 realized AL을 재사용하지 않는다. **새 tree 전체를 생성한 online AL이 아니다.**

| 방법 | 이번 확장으로 얻는 기대 AL | 보정 reach+breadth 대비 차이, 95% CI |
|---|---:|---|
| q+breadth | 0.681903 | -0.003207, [-0.006402, -0.000305] |
| 보정 reach+breadth (기준) | 0.685109 | — |
| q+gain | 0.698233 | +0.013124, [+0.001269, +0.024967] |
| 보정 reach+gain | 0.696515 | +0.011406, [-0.002021, +0.024562] |
| true reach+calibration gain (oracle 입력) | 0.742410 | +0.057300, [+0.047048, +0.067990] |
| 보정 reach+true gain (oracle 입력) | 0.714697 | +0.029587, [+0.017492, +0.041735] |
| true reach+true gain (oracle) | 0.745852 | +0.060743, [+0.050246, +0.071577] |

두 부모/총3개 node로만 제한하면 보정 reach+gain의 변화는 **+0.000842**, CI [-0.002495, +0.003940]였다. 이전 두-node 문제보다 여지가 훨씬 작았다. 서로 다른 budget/cohort의 절대 AL을 직접 비교하지 않는다.

| phase | breadth | gain | 차이, 95% CI |
|---|---:|---:|---|
| P1 | 0.724120 | 0.736652 | +0.012532, [-0.003398, +0.028204] |
| P2 | 0.510590 | 0.521286 | +0.010696, [-0.000388, +0.022857] |

q+gain과 보정 reach+gain의 직접 차이는 **+0.001718**, CI [-0.007816, +0.011075]다. 점추정 순서만으로 두 정책의 우열을 확정하지 않는다.
결정을 바꾼 비율은 59.17%, 이득 사례 50.08%, 손해 사례 8.71%였다. 이득 기여 평균 +0.029233와 손해 기여 평균 -0.017827가 상쇄된다. 빈도가 낮아도 큰 손해 사례가 있어 승률만으로 정책을 선택하면 안 된다.
수치적분의 결정론적 구간 전체를 적용한 평균 차이 범위는 [+0.011276, +0.011945]다. Calibration 문항 하나씩을 제외했을 때 평균 차이는 [+0.009005, +0.012281]였다. 이는 test 일반화 CI가 아니라 calibration 민감도다.
통계는 task→original question을 같은 비중으로 두고, 질문 단위 bootstrap 2,000회다. 이 국소 진단은 탐색적이며 여러 비교에 대한 보정 CI는 아니다. [전체 수치](c3_analysis.json), [손해 사례/민감도](robustness.json).

## 5. 완료한 구현 검사

| 검사 | 결과 |
|---|---|
| C=3 완전열거 대조 484개 | 최대 오차 1.31e-13 |
| 배분 완전탐색 대조 600개 | 최대 오차 2.22e-16 |
| 유한 vocabulary의 2-token 결합 분포 100개 | 최대 오차 2.22e-16 |
| 잘못된 post-sampling pruning negative control | TV=0.01875를 검출 |
| 실제 selector CPU 1000개 + mixed-depth mask | 통과 |
| 배분 CPU/GPU/eager/CUDA graph 200개 | 통과 |
| 실제 tree executor+attention, 6정책×8 shape/page 조건 | 통과; node/q-reference audit 72개 |
| 실제 70B+TinyLlama 엔진 smoke 및 full online audit | 대기 |

GPU 실행기 검사는 저장소의 작은 모델 fixture를 사용했지만 실제 tree·attention kernel, variable round width, page bucket 전환, parent-q 연결을 실행했다. 이것을 실제 모델의 full inference 검증 완료로 바꾸어 표현하지 않는다.

## 6. 미확장 branch를 확인하려던 shadow 검증

기존 target p를 재사용하고 TinyLlama만 다시 실행해 관측된 leaf의 q를 보완하는 실험을 했다. 먼저 저장된 known-parent q를 재현하는지 24개 tree/91개 문맥에서 검사했다. HF 경로는 평균 TV 0.00836, p95 0.02961; production 모델 연산+SDPA 경로도 평균 TV 0.00781, p95 0.03074였다. 사전 기준(median<0.005, p95<0.02)을 통과하지 못했다.
**이 shadow 결과는 정책 개선 수치에서 제외했다.** 원인을 특정한 버그 또는 BF16만의 탓으로 확정하지 않는다. 동일 engine/KV 상태의 shadow 관측 또는 이에 준하는 q 재현 검증이 더 필요하다. 전체 frontier 추적은 어떤 후보가 탈락했는지는 보여주지만, 그 후보의 새 target/draft 분포까지 알려주지는 않는다.

## 7. GPU가 확보되면 자동 실행할 비교

- 여섯 정책, seed 1/42/123, **full480 질문·560turn×18 run = 10,080turn**. 새 캠페인 안에서 baseline도 다시 실행한다.
- 먼저 모든 정책의 실제 모델 smoke에서 graph/eager와 q-reference 검사를 통과해야 한다.
- 본 실험은 observer를 끄고, 별도로 baseline reach와 reach_gain_frontier의 full480 frontier trace를 각각 수집한다.
- 원래 질문 단위로 두 turn과 모든 seed를 묶어 paired bootstrap 5,000회. 미리 정한 6개 AL 비교에는 simultaneous CI를 사용한다.
- 자연 EOS, cap128, native2048, raw base-model continuation이다. 논문 cap1024/chat 설정 전체로 일반화하지 않는다.
- 다른 사용자의 작업을 종료하지 않는다. 5개 GPU가 각각 1GB 미만으로 비면 실행하고, 실행 중 오류/변경된 source를 발견하면 기록을 보존하고 중단한다.

현재 실행 상태: [queue_status.json](queue_status.json). 실행 조건: [PLAN.md](PLAN.md), [plan.json](plan.json). `STOP` 파일을 이 디렉터리에 만들면 대기 중 또는 다음 job 시작 전에 중단한다. 이미 실행 중인 job을 강제 종료하지 않는다.

