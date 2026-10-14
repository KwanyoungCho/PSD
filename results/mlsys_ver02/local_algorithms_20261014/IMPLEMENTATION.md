# 로컬 연구 알고리즘 반영 명세

기준: `MLSys-ver0.2@18e912f`. 다른 서버의 미병합 연구 코드는 사용하지 않는다.
이 변경은 실행 최적화만이 아니라 **후보·위치·미래 tree 확장 정책 변경**이다.
기능 테스트와 AL/cache hit 개선의 경험적 근거는 따로 판단한다.

## Root 후보와 위치

- 이전 후보: `[e-q]+`, 위치마다 top-M만 남긴 뒤 그 M개의 합으로 재정규화.
- 새 후보: 내부 문맥 `e(v)(1-q(v))`, leaf/실제 chain bonus는 `e(v)`.
- 이미 제안한 자식 토큰은 제외하고, 남은 **전체 vocabulary 점수 합**으로 정규화.
- 위치: `0.75 h_hat + 0.25 h_bar`. h_hat은 관측 proposal token의 proxy 수락률, h_bar는 `sum min(e,q)` overlap 추정에서 계산한다.
- Tree는 ordered-WOR 형제 사다리를 따라 residual과 조건부 q를 갱신하고, 두 terminal-mass 재귀를 각각 수행한 뒤 혼합한다. 두 alpha를 먼저 섞지 않는다.
- `e`에는 target 온도, `q`에는 draft 온도 및 설정된 sampler_x를 적용한다. T=0은 budget score만 T=1 soft score를 유지한다.
- 관측 토큰 조건의 정확한 target 거절 법칙이 아니라 **미래 캐시 root를 고르는 추정식**이다. 실제 수락과 correction sampling은 기존 target/proposal 법칙 그대로다.

B1 eager/graph, B>1 dense/flat, chain cache miss까지 같은 helper 계약을 사용한다.
부분 chain의 bonus는 batch 최대 폭이 아니라 실제 proposal 끝이다. Proxy의 tree reach 재귀는 실제 검증 폭을 포함하도록 구성한다.
GPU 고정 graph의 난수 상태는 기존 warmup 계약대로 보존한다.

이 root 점수 변경은 proxy를 받은 뒤 준비하는 P2 cache root에 해당한다. P1이 draft 확률로 root를 고르는 방식 자체는 바꾸지 않았다. Tree 확장 개선은 P1과 P2 양쪽에 적용한다. 따라서 총 hit뿐 아니라 P1/P2 source 비율과 조건부 AL도 함께 보고한다.

## Tree 확장

기존 q-path 점수 대신 다음을 사용한다.

`rho(child_j) = rho(parent) * product_{k<j}(1-alpha_hat_k) * alpha_hat_j`

alpha_hat은 기존 로컬 calibration의 phase/sibling/q-bin table이다. 새 신경망을 학습하지 않지만, **경험적 calibration을 사용하므로 순수한 무보정 이론식으로 표현하면 안 된다.**

선택된 부모들에게 같은 수의 첫 자식을 우선 주는 대신, root별 현재 예산에서 `sum rho(parent)*gamma_phase(c_parent)`를 최대화하는 정수 배분을 수행한다. Gamma가 비오목해도 작은 예산의 partition을 열거해 해당 round의 목적함수를 정확히 푼다. 전체 tree의 전역 AL 최적성은 보장하지 않는다.

현재 깊이뿐 아니라 이전 깊이의 아직 확장하지 않은 node도 경쟁한다. 이미 forward를 수행한 node의 `state`는 제외하므로 같은 node를 다시 forward하지 않는다. 같은 물리 forward 폭과 future-round reserve를 유지한다.

주의: 고정 폭 graph이므로 모든 입력 슬롯이 유용한 자식을 만드는 것은 아니다. 선택된 부모가 gain 배분에서 fanout 0을 받더라도 해당 슬롯의 forward는 수행하고, 원래 single-shot 계약대로 처리 완료 상태로 둔다. 따라서 이번 frontier 이식은 중복 forward를 늘리지 않지만 **낭비 슬롯을 모두 없앤 최적화도 아니다.** Fanout 0의 로그잇 재사용/재경쟁이나 물리 폭 압축은 다른 정책·실행 변경이며 이번 채택 효과에 포함하지 않는다.

선택과 fanout은 **새 자식 토큰을 뽑기 전** 결정한다. 저장 raw_q와 parent logits는 원래 proposal 법칙이며 보정 alpha나 root 점수로 대체하지 않는다. G>M은 생성 순서 prefix를 유지한다.

## 고정 calibration 출처와 지원 범위

`feat/duet-mlsys-coverage@1d9ace6`의 `results/mlsys_coverage/round4/calibration/{reach,gain}.json`을 byte-for-byte 이식했다. 기존 70B/TinyLlama 연구의 8문항 calibration이며, 이번 full 평가 결과로 다시 fitting하지 않는다. 이전 dense모델 새 calibration이 이 historical table을 확실히 이겼다는 근거가 없어서 기존 표를 유지한다.

기본은 하나: complement/full/.25와 reach_gain_frontier. 표는 설치 package에 포함하며 시작 로그에 SHA256을 출력한다. 기존 실험용 환경변수는 재현/ablation에만 사용한다.

- 연구 tree 정책: dynamic/eagle, C<=3, NV<=32. 기존 C<=3 curve의 prefix를 C1/C2에 사용한다. NV32는 calibration 범위가 아니라 allocation 계산량 상한이다.
- C>3/NV>32: q_path로 돌아가며 이유를 로그에 출력한다. 없는 calibration 값을 임의로 만들어 쓰지 않는다.
- 기존 backbone/hybrid 등의 재현 selector는 q_path 유지. P1/P2 tree가 꺼진 phase도 기존 방식 유지.
- Graph 미사용의 GPU arena도 같은 정책을 사용한다. 옛 CPU reference rollout은 q_path 전용이며 연구 정책으로 잘못 실행되지 않도록 명시 오류를 낸다.
- Full vocabulary가 없는 raw/top-M proxy 전송은 새 full normalization/overlap을 계산할 수 없으므로 명시적으로 거부한다.
- 옛 E0 root trace는 이전 점수 schema이므로 새 정책에서 명시 오류. 이번 비교 harness가 새 정책의 실측 근거다.

## 검증·비교를 읽는 법

1. 새 독립 검사는 scalar FP64 root/terminal 수식, chain-tree-flat 동일성, sampler_x/실제 온도, invalid/no-JIT 경계, 난수·입력 불변, gain 완전탐색, request/root 독립성, P1/P2 eager/graph와 raw q reference를 확인한다.
2. 과거 테스트 한 건은 옛 residual/top-M 수식을 명시한 reference 검사로 유지했다. 새 default는 별도 독립 scalar 검사로 검증한다. 과거 bench flat adapter 비교도 옛 정책을 명시한다.
3. 운영 선택 옵션을 늘리는 대신 `mlsys_local_algorithms.py`는 **benchmark process에서만** 정책을 교체한다. 검증·sampling·KV를 바꾸지 않는다.
4. 주 비교는 기존 ver0.2 / root만 / tree만 / 둘 다의 2x2다. 추가로 full 정규화만, 위치 혼합까지, 같은 위치식의 proxy-only를 비교한다.
5. 두 모델에서 B8/T.7/K4,2/exit21/N8,4/G=M/C3를 고정한다. P1은 문맥당 root 2개, P2 root budget은 5개다. CLI `--roots 2`를 전체 root 예산으로 오해하지 않는다. Miss 길이·threshold·모델·예산을 바꾸어 생긴 효과를 점수식의 효과로 섞지 않는다.
6. Root의 목적은 cache hit, tree의 목적은 AL이다. TPS와 proxy/P1/P2/verification 비용도 기록하지만 알고리즘 변화 후 출력/shape가 달라지므로 전체 latency 차이를 kernel 최적화의 인과 효과로 주장하지 않는다.
7. Full480은 첫 턴 전체 목록, input cap512/output cap64다. 560턴 평가나 70B 논문 재현이 아니다. 두 pass는 동일 target seed를 재설정하고 draft RNG는 이어지므로 독립 process 반복이 아니다. Bootstrap은 두 pass를 같은 원문 batch group으로 묶는다.
8. 미래 다른 서버와 합칠 때 후보식은 root_policy와 flat_tree_sampling을 함께 검토한다. Target verifier를 score식으로 바꾸거나 old rerank로 되돌리지 않는다.
