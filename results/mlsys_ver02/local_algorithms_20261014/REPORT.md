# 로컬 연구 알고리즘의 ver0.2 반영과 비교

상태: **코드 반영·최종 회귀·14개 전체 비교·4개 profile·최종 greedy smoke 3건 완료.** 구현의 완료와 모든 알고리즘의 우위 입증은 다르다. 아래 판정에 이득·불확실성·손실을 함께 적었다. 다른 서버의 진행 중인 연구 코드는 이번 변경에 합치지 않았다.

## 1. 무엇을 반영했나

| 구분 | 이전 ver0.2 (`18e912f`) | 이번 로컬 연구 정책 |
|---|---|---|
| P2 root 토큰 점수 | `[e-q]+` | `e(1-q)`; 이미 제안한 자식 제외 |
| 위치 내 정규화 | 위치별 top-M 합으로 정규화 | 남은 전체 vocabulary 합으로 정규화 |
| 위치 점수 | 관측 proposal의 hhat | `.75 hhat + .25 hbar` |
| Tree node 점수 | draft q의 경로 곱 | 형제 거절·수락을 반영한 calibrated reach |
| Tree fanout | 부모마다 먼저 자식 하나씩 배분 | 현재 round의 `sum reach*gamma(children)` 최대화 |
| Tree 경쟁 범위 | 현재 round 깊이 | 이전 깊이의 미확장 frontier 포함 |

운영 연구 preset은 위 조합 하나다. B1 eager/graph, B>1 dense/flat, 실제 온도·sampler_x·부분 proposal 길이를 같은 계약으로 연결했다. P1 root를 고르는 수식 자체는 유지하고, tree 확장 개선은 P1/P2 모두에 적용한다.

Root 점수는 cache hit를, tree는 AL을 목표로 한다. Root 개선이 hit를 높여도 짧은 P2 tree의 사용 비중이 커지면 전체 AL이 낮아질 수 있다. 따라서 총 hit·AL 외에 P1/P2 비중과 조건부 AL을 함께 비교한다.

**이론적 근거가 있는 추정식이지만, 실제 residual의 정확한 대체식이나 항상 우수한 알고리즘이라는 증명은 아니다.** `.25`와 보정표는 이전 로컬 실험에서 정한 값이며 이번 전체 결과로 다시 fitting하지 않았다. 자세한 수식·정확성 논리는 [THEORY.md](THEORY.md), 구현 계약은 [IMPLEMENTATION.md](IMPLEMENTATION.md)에 있다.

## 2. 무엇과 비교하나

기준은 같은 시스템 최적화가 들어 있는 `MLSys-ver0.2@18e912f`다. 새 알고리즘 외에 miss 길이, root 예산, tree node 수, exit layer, 모델, 온도를 함께 바꾸지 않았다.

1. 기존 ver0.2.
2. Root 개선만.
3. Tree 개선만.
4. Root + tree.
5. Full 정규화만.
6. Full 정규화 + 위치 혼합, 토큰식은 residual 유지.
7. 같은 개선 위치·정규화에서 proxy-only.

마지막 세 비교로 정규화/위치/토큰식의 효과를 분리한다. 여기서 proxy-only는 DUET의 후보 점수 ablation이다. **Mirror-SD 전체 시스템과의 직접 성능 비교가 아니다.**

두 full BF16 모델 쌍에서 전체 Spec-Bench 480문항 첫 턴을 각각 두 pass 평가한다. B8/T.7/K1=4/K2=2/exit21, P1 G=M=8, P2 G=M=4, C3다. P1은 문맥당 root 2개, P2 총 budget은 5개다. 입력 cap512/출력 cap64와 prefix OFF를 적용했다. [실제 파라미터](PARAMETERS.md)와 plan JSON에서 모든 설정을 확인할 수 있다. 모델별 최적 파라미터 탐색 결과는 아니다.

Batch가 실제 8이고 cap/clipping/graph 준비가 없는 step만 주 지표의 분자·분모에 함께 포함한다. 두 pass를 독립 문항으로 세지 않고 같은 원래 batch group 60개를 묶어 paired bootstrap한다. 여러 process/seed를 반복한 실험의 신뢰구간은 아니다. 알고리즘에 따라 EOS·길이와 방문 문맥이 달라져 포함 step 수도 달라진다. 동일한 문맥에서 root만 바꾼 offline 평가와 구분하며, 전체 문항의 별도 요청 평균도 함께 보존한다. 480개 문항을 평가했지만 560턴 전체 평가나 무제한 길이 serving은 아니다.

## 3. 결과

14개 primary arm이 모두 정상 종료했고 각 arm은 480문항 × 2 pass, 960개 출력을 기록했다. 외부 GPU process가 관측된 원래 Llama3 두 실행은 제외하고 정상 재실행을 선택했다. 전체 표·신뢰구간은 [TABLES.md](TABLES.md), 숫자는 [RESULTS.json](RESULTS.json)에 있다.

| 모델 | 변경 | Batch | AL | Cache hit (%) | TPS (보조) |
|---|---|---:|---:|---:|---:|
| Llama2-7B / AMD-135M | 기존 ver0.2 | 8 | 2.0263 | 74.507 | 552.19 |
| Llama2-7B / AMD-135M | Root만 | 8 | 1.9938 | 75.341 | 590.45 |
| Llama2-7B / AMD-135M | Tree만 | 8 | 2.0843 | 74.840 | 602.94 |
| Llama2-7B / AMD-135M | Root+tree | 8 | 2.1250 | 73.855 | 628.15 |
| Llama3-8B / Qwama-0.5B | 기존 ver0.2 | 8 | 2.3934 | 71.744 | 436.34 |
| Llama3-8B / Qwama-0.5B | Root만 | 8 | 2.3765 | 72.016 | 369.72 |
| Llama3-8B / Qwama-0.5B | Tree만 | 8 | 2.4412 | 70.783 | 437.71 |
| Llama3-8B / Qwama-0.5B | Root+tree | 8 | 2.4126 | 69.809 | 428.79 |

### 판정

1. **Tree의 P1 AL 개선 근거는 있다.** 기존 root를 유지하고 tree만 바꾸면 P1-hit AL은 Llama2 2.641→2.775, Llama3 2.837→2.999다. 차이의 문항 bootstrap 95% CI는 각각 `[+0.052,+0.218]`, `[+0.090,+0.235]`다. 총 AL은 각각 +0.058, +0.048이며 Llama2의 총 AL CI는 0을 포함한다. Llama3의 총 AL CI 하한은 +0.0003으로 작으므로 강한 보편적 우위라고 과장하지 않는다.
2. **Tree는 같은 cap을 더 활용했다.** P1 hit tree의 실제 node 평균은 Llama2 7.272→7.748, Llama3 7.175→7.680, 최대 깊이는 3.563→3.744 / 3.505→3.689다. 따라서 이번 실험은 “동일 실제 node 수에서 점수만 더 정확해졌다”는 증명이 아니다. Reach/gain/frontier 묶음이 남은 예산을 더 쓰며 구조를 바꾼 효과이고, 각 tree 구성 요소의 기여도는 별도 ablation이 필요하다.
3. **Root 변경의 cache hit 우위는 확인되지 않았다.** 기존 대비 root-only hit 차이는 Llama2 +0.834%p, Llama3 +0.272%p이며 두 CI 모두 0을 포함한다. Full 정규화만 추가하거나 `.25 hbar`를 추가한 단계에서도 보편적 개선은 확인되지 않았다. 특히 위치 혼합의 점추정은 두 모델에서 hit·AL 모두 낮아졌다. `.25`를 범용 최적값으로 주장하지 않는다.
4. **e(1-q)가 proxy-only를 이겼다고 쓸 수 없다.** 위치식·full 정규화를 맞춘 비교에서 hit 차이는 Llama2 −0.006%p `[−1.582,+1.516]`, Llama3 +0.775%p `[−0.695,+2.373]`다. 두 모델 모두 확정적인 우위가 없고 AL 점추정은 낮았다. 이것은 Mirror-SD 전체 시스템 비교가 아니라 DUET root 토큰식의 비교다.
5. **Root+tree 전체 조합은 모델 의존적이다.** 기존 대비 Llama2 AL은 +0.0987 `[+0.0295,+0.1695]`로 증가했다. Llama3 AL 차이 +0.0192는 CI가 0을 포함하고 hit는 −1.936%p `[−3.876,−0.161]`로 낮아졌다. 따라서 이 조합을 모든 모델에서 검증된 개선 기본값이라고 제시하지 않는다.
6. **이번 branch에는 요청한 연구 조합을 실제로 연결했다.** 구현 완료와 최종 논문 정책 채택은 구분한다. 코드의 운영 연구 preset은 complement/full/.25/reach-gain-frontier 한 조합이다. 최종 논문 채택의 근거는 현재 tree의 AL 방향이 더 강하고, root·위치 조합은 추가 검증이 필요하다. 다른 서버 결과를 합칠 때 이전 정책과 이 조합을 재비교할 수 있도록 2×2/분리 비교와 원시 자료를 함께 남긴다.

### 그림과 해석

- [2×2 비교 PNG](comparison.png) / [PDF](comparison.pdf): AL·hit·TPS.
- [분리 비교 CI PNG](ablations.png) / [PDF](ablations.pdf): 정규화, 위치, 토큰식, tree 추가의 변화.
- P1/P2 사용 비중과 조건부 AL의 대칭 분해는 RESULTS의 `al_accounting`에 있다. 이는 총 AL 차이를 정확히 나누는 집계 항등식이며, 같은 문맥에서 원인을 개별 조작한 인과 추정은 아니다.
- 실제 요청의 전체 문항 평균도 TABLES에 있다. 주 표는 B8/cap/clipping 제외 step 평균이고, 보조 표는 각 문항의 tail까지 포함하므로 가중치와 값이 다를 수 있다.


**TPS 해석 주의:** Llama3 root-only의 원래 제외 실행과 정상 재실행은 소스·960/960 출력·AL·hit가 같지만 TPS는 420.57→369.72로 달랐다. 원래 실행은 초기화 중 외부 process가 관측되어 주 비교에 재편입하지 않는다. 이 진단은 문항 bootstrap이 process 사이의 시간 변동을 설명하지 못한다는 근거다. 이번 표의 TPS를 새 수식의 확정된 속도 향상으로 인용하지 않는다. [RECHECK_STABILITY.json](RECHECK_STABILITY.json)에 보존했다.

새 root 계산은 공짜가 아니다. 최종 코드의 동일한 dense 점수 실행기에 기존/새 수식을 각각 넣은 고정 입력 CUDA graph 비교(N=8)에서 median 비용은 다음과 같다. Online flat 경로의 실제 shape를 그대로 replay한 측정은 아니다. 전송·Transformer forward는 포함하지 않은 **점수 계산 전체**의 차이이며 hbar 하나의 비용으로 해석하지 않는다.

| Batch | Vocabulary | 이전 점수 (ms) | 새 점수 (ms) | 증가 (ms) |
|---|---:|---:|---:|---:|
| 1 | 32,000 | 0.262 | 0.317 | 0.055 |
| 8 | 32,000 | 0.495 | 0.577 | 0.082 |
| 1 | 128,256 | 0.433 | 0.505 | 0.072 |
| 8 | 128,256 | 2.699 | 3.020 | 0.321 |

같은 입력/구조에서 순서를 교대해 반복했고 GPU 4 외부 process 관측 0이었다. 표의 logit은 합성 입력이므로 실제 online shape의 전체 latency 개선 근거로 사용하지 않는다. 원시 반복값·source hash·telemetry는 ROOT_COMPONENT.json/ROOT_COMPONENT_TELEMETRY.json에 있다. 품질이 바뀌면 방문 tree·source 구성·종료 시점도 달라진다. 따라서 online TPS/verification 평균 차이를 이 수식의 직접적인 kernel 가속으로 해석하지 않는다.

## 3-1. Phase 비용과 시간 예산

48문항(각 Spec-Bench group에서 8개)을 따로 측정했다. Profile 모드는 독립 capture audit를 비활성화하므로 무계측 primary와 같은 preparation-free 인증으로 취급하지 않는다. 모든 arm이 8개 generate epoch(2 warmup + 6 측정)를 기록했고 각 유효 step의 target/draft 구간이 빠짐없이 대응했다. 이벤트 상한에 의한 누락은 없었다. [구간 그림 PNG](phases.png) / [PDF](phases.pdf), [원시 집계](PROFILE_RESULTS.json).

| 변경 | 유효 step | Target pre | Proxy | Target post | Accept | P1 replay / total | P2 replay / total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기존 | 78 | 13.083 | 2.970 | 5.949 | 2.623 | 20.208 / 23.340 | 8.500 / 14.029 |
| Root만 | 48 | 13.069 | 3.124 | 5.949 | 2.840 | 20.287 / 23.902 | 8.638 / 13.783 |
| Tree만 | 58 | 13.142 | 2.967 | 5.977 | 3.089 | 22.796 / 25.822 | 9.561 / 15.131 |
| Root+tree | 47 | 13.136 | 3.025 | 5.972 | 4.295 | 24.017 / 32.257 | 10.659 / 23.593 |

단위 ms/호출. 서로 겹치거나 포함되는 구간이므로 모두 더해 한 step의 직렬 latency로 사용하지 않는다. Total에는 CPU launch gap·준비가 포함되고, P2 total에는 P1/P2의 공통 bulk cache export도 포함된다.

**이 측정은 tree 점수가 공짜인 최적화라고 보여주지 않는다.** Tree-only의 replay는 P1 20.208→22.796ms, P2 8.500→9.561ms로 늘었다. AL 향상과 비용 사이 tradeoff가 있다. 방문 문맥·shape가 다른 online profile이므로 이 전체 차이를 순수 점수 kernel 비용이라고 단정하지 않는다. Combined에서 늘어난 host gap/대기와 계측 영향도 겹쳐, 단일 원인으로 확정하지 않았다.

| 변경 | P1 완료→proxy 도착 여유 중앙값 (ms) | P2 완료→target 후처리 완료 여유 중앙값 (ms) |
|---|---:|---:|
| 기존 | -6.883 | -8.325 |
| Root만 | -7.898 | -9.369 |
| Tree만 | -9.727 | -12.017 |
| Root+tree | -13.484 | -23.467 |

음수는 draft 구간이 해당 target 기준시점보다 늦게 끝났다는 뜻이다. Proxy 도착은 P1 뒤의 blocking wait가 아니라 별도 관측 stream의 수신 완료 event로 잡았다. 따라서 P1 지연을 실제로 구분할 수 있다. GPU별 CUDA event를 host 기준시각에 맞춘 근사이므로 sub-ms 차이는 과해석하지 않는다.

**현재 K1=4/K2=2를 완전히 숨겨지는 최적 설정이라고 볼 근거는 없다.** 적어도 이 profile에서는 P1/P2가 늦게 끝나는 구간이 많다. 단, 계측이 추가된 별도 표본이므로 실제 무계측 deadline 비율이나 최적 K를 이 숫자만으로 확정하지 않는다. 정확한 시간 기반 재튜닝은 선택할 최종 root/tree 정책을 고정한 후 필요하다. 이번에는 품질을 구분하려고 같은 K와 예산을 유지했다.


## 4. 기능·수식 검증

- 최종 고정 소스의 **26개 suite, 278개 검사 통과**, failure/error/skip 모두 0. 검사 전후 195개 runtime/bench/test Python 파일 hash가 동일하다.
- 새 독립 검사: FP64 scalar root/terminal 수식, tree 형제 재귀, full 정규화, ragged/empty/no-JIT, 실제 온도/sampler_x, 난수·입력 불변, 정수 배분 완전탐색, request/root 독립성, P1/P2 eager/graph, 실제 q 보존.
- 기존 residual reference 검사는 기존 정책을 명시해 유지했다. 새 수식을 맞추려고 이전 reference의 기대값을 새 구현 결과로 바꾼 검사가 아니다.
- 실제 Llama2 B1/B8/T0/T.7, graph 미사용, G>M의 초기 smoke 6건 정상 종료. 그중 eager/G>M은 최종 소스이고 앞선 4건은 소스가 동결되기 전 실행이다. 차이 파일은 VALIDATION.json에 공개한다. 최종 소스로 Llama2 B1/T0, Llama2 B8/T0, Llama3 B8/T0의 16문항 smoke 3건을 다시 정상 완료했다. 이는 실행 검사이며 전체 T0 품질 평가나 AR bitwise 동일성 검사라는 뜻은 아니다. Full T.7 비교 14건도 최종 소스다.
- 보정표 두 JSON을 wheel에 포함하고 원본과 hash가 같은지 확인했다.
- FINAL_AUDIT: 14 primary + 4 profile + 3 final greedy, **21개 실행의 commit/source/adapter/출력 수/종료 상태 검증 통과**. Primary 14개 모두 decode 중 graph 준비가 겹친 step 0개다.
- Llama3 B8/T0 smoke는 16개 출력을 완료했지만 cap/clipping/부분 batch 제외 후 성능 유효 step은 0개였다. 따라서 이 smoke로 B8/T0 속도나 AL 우위를 주장하지 않는다.

최종 278개 검사 요약은 [CHECKS.json](CHECKS.json), 고정 소스 이력은 [CHECKS_PROVENANCE.json](CHECKS_PROVENANCE.json), 실행별 대조는 [FINAL_AUDIT.json](FINAL_AUDIT.json)에서 확인한다.

Source와 결과 출처는 [VALIDATION.json](VALIDATION.json), [SOURCE_PORT.json](SOURCE_PORT.json), [FROZEN_SOURCE.json](FROZEN_SOURCE.json)에 있다. 개발 중의 초기 실패 로그나 소스가 바뀌던 중의 검사 결과를 최종 278개 통과 근거로 사용하지 않았다.

## 5. 실험 중단과 제외

GPU 5 외부 process 진입으로 Llama2 초기화가 두 번 OOM 종료했다. 세 번째 초기화는 GPU 6·7만 사용하라는 사용자 지시에 따라 우리 process group만 종료했다. 이후 4·5 사용을 다시 허용받아 기존 간섭 없는 baseline은 byte-for-byte 보존하고 나머지를 재개했다. 완료 비교군 다음에 GPU가 점유된 경우도 자동으로 다음 실행을 시작하지 않고 기다렸다.

Llama3 root-only의 첫 실행은 생성은 완료했지만 초기화 중 외부 process가 관측되어 정식 비교에서 제외했다. 이후 proxy-only 초기화와 root-only 재시도에서도 외부 process가 진입해 OOM이 발생했고, 원시 기록을 보존한 새 폴더에서 재측정을 완료했다. GPU 6·7을 확보한 뒤 정상 완료한 두 arm만 주 분석에 포함했다. 실패·간섭·사용자 중단 기록은 보존한다. 이를 알고리즘 성능 회귀 또는 GPU 고장으로 분류하지 않는다. 다른 사용자의 process는 종료하지 않았다.

## 6. 적용 범위와 남는 한계

- Tree 보정표는 과거 70B/TinyLlama의 8문항 자료를 그대로 옮긴 scalar calibration이다. 새 neural head 학습은 없지만 경험적 보정이 전혀 없는 수식이라고 하면 안 된다.
- 지원되는 연구 tree 범위는 dynamic/eagle, C<=3, NV<=32. C/NV가 범위를 넘으면 기존 q-path로 돌아가며 로그로 알린다. NV32는 계산량 상한이고, 보정이 검증된 node 예산 범위라는 뜻은 아니다.
- 고정 폭 forward의 빈 슬롯을 모두 제거한 것은 아니다. Gain 배분에서 fanout 0인 부모 슬롯도 원래 single-shot 계약대로 forward된다. 이번 변경은 AL을 위한 정책 이식이다.
- 실제 target 검증·correction sampling은 기존 법칙을 유지한다. 새 score를 검증의 q로 쓰지 않는다. G>M은 생성 순서 prefix를 유지한다.
- Raw/top-M 전송과 옛 E0 trace는 새 full-vocabulary 점수와 맞지 않아 명시적으로 거부한다. 잘못된 이전 식으로 조용히 fallback하지 않는다.
- T=0 생성의 정확성과 T=0에서 이번 점수의 AL 우위는 다른 주장이다. 후자는 이번 T.7 비교로 확정하지 않는다.
- Miss DS/P1 tree, mixed-miss AR 채택, target 전체 packed, 다른 서버의 후속 알고리즘은 이번 작업과 별개다. 70B 논문 조건에서의 우위를 이 결과만으로 주장하지 않는다.

## 7. 코드와 인수인계

- Runtime commit: `cd2506c`.
- 독립 검사·ablation commit: `992b8b8`.
- 기준 commit: `18e912f`.
- 브랜치: `MLSys-ver0.2`.

[REVIEW_MAP.md](REVIEW_MAP.md)에 파일별 변경, 재현 명령, 다른 서버에서의 검토·병합 순서를 모았다. 이후 다른 서버 알고리즘을 합칠 때도 B1/B>1 dense/flat의 수식 일치, 실제 q 보존, G>M prefix를 함께 확인해야 한다.

## 8. 이번 완료와 이후 판단의 경계

이번 로컬 알고리즘 이식·기능 검사·계획한 비교는 완료했다. 실행기 구현 오류 때문에 남겨 둔 미완료 작업은 없다. 그러나 다음은 **입증하지 않은 연구 범위**이며 “모든 설정에서 최적화 완료”라는 뜻은 아니다.

- Root/위치 수식의 범용 우위와 .25의 최적성. 이번에는 유의한 우위가 확인되지 않았다.
- Reach 보정·gain 배분·frontier 세 요소 각각의 효과. 이번에는 묶음으로 비교했다.
- 실제 node 수를 동일하게 맞춘 tree 비교, 최종 정책에서의 모델별 K1/K2/exit/threshold 재튜닝.
- 70B 논문 조건, 다른 온도·다중 seed/process에서의 일반화 및 확정적 TPS 개선.
- 다른 서버에서 진행 중인 알고리즘과의 병합·대조. 그 코드는 이번에 합치지 않았다.

원시 자료는 ARCHIVES.json의 파일별 SHA-256과 함께 보존했다. 성공·실패·제외·초기 검사 기록을 모두 구분하고, 과거 raw 파일을 덮어쓰지 않았다. GPU 4–7의 이번 실험 process는 모두 종료됐다.
