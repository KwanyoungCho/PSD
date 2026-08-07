# DUET confidence tree 최종 검토 (2026-08-07)

## 1. 결론

P2의 네 번 draft forward 사이에 있던 CPU 왕복은 제거됐다. 현재 실행기는
`forward -> sample -> select/fanout -> mask update` 네 라운드를 하나의 CUDA
graph로 실행한다. 최종 profile에서 이 구간은 `p2_graph_replay = 12.45 ms`
한 블록이며 라운드 사이 host 호출, `plan()`, D2H readback은 없다.

그러나 고정된 40노드 예산에서 tree가 chain보다 시스템 전체로 빠르지는
않다. warmup이 production RNG를 소비하지 않도록 수정한 뒤의 최종
3-seed 측정에서 P2 hit당 accepted length는 1.717에서 1.923으로 12.0%
늘었지만 P2 hit는 0.273에서 0.247로 줄었다. P1AL도 3.733에서 3.463으로
감소했다. target은 chain의 5행 대신 최대 9행을 검증한다. 최종적으로
tokens/step은 3.743에서 3.593으로 4.0%, TPS는 69.91에서 60.91으로 12.9%
감소했다. production champion은 여전히 chain이다.

따라서 현재 결과의 정확한 판정은 다음과 같다.

- tree 구성과 실행기의 핵심 오류는 수정됐다.
- 조건부 P2AL 증가는 재현됐다.
- 같은 40노드 예산을 재배치하는 것만으로는 P2 coverage와 depth를 동시에
  높일 수 없다.
- P2 실행기의 forward 사이 공백과 첫-hit capture는 해결됐지만, P1
  coverage와 넓어진 target 검증량은 해결되지 않았다.
- 현재 tree는 연구/비교 arm으로 유효하지만 chain을 대체할 production
  champion은 아니다.

## 2. 최종 confidence 정책

EAGLE-2의 핵심 아이디어와 동일하게 노드 점수는 누적 경로 확률을 쓴다.

```text
score(node) = log P_iv(root) + sum_path log q(child | parent)
```

다만 DUET는 하나의 root에서 tree를 만드는 EAGLE-2와 달리, proxy가 보낸
여러 root를 동시에 덮는 forest다. 또한 temperature sampling과
without-replacement verifier의 무손실 규약 때문에, 샘플된 자식의 정체를 본
뒤 proposal 예산을 소급해 바꾸지 않는다.

최종 `confidence` 모드는 다음 규칙 하나로 고정했다.

1. proxy root를 `P_iv` 순으로 정렬한다.
2. 활성 root 수를
   `R = floor((W * K2) / (K2 + 2))`로 자동 계산한다.
3. 각 활성 root에 K2-deep first-child backbone을 먼저 보장한다.
4. 남은 두 노드/root 상당의 예산을 누적 경로 confidence가 높은 형제에
   배분한다.
5. beta는 0.5로 고정한다. `root_count`와 beta sweep은 canonical 모드에서
   필요 없다.

현재 W=10, K2=4이면 `R=floor(40/6)=6`이다. 이는 수정된 feasibility의
명시적 `backbone + rescue` budget-6 동작점과 일치한다. 10개 root를 모두
유지한 arm도 측정했지만, 각 root에 평균 4노드뿐이라 사실상 chain
backbone만 만들고 유용한 branch를 거의 만들지 못했다.

## 3. P1 품질 보존

tree hit에서 P1 후보를 모든 tree context에 균등 분배하던 방식을 없앴다.
현재 allocator는 다음 순서를 따른다.

1. first-child backbone의 각 context에 short-chain과 같은 후보 수를 먼저
   할당한다.
2. 남은 lane만 sibling context coverage에 한 개씩 배분한다.
3. 추가 lane은 `cumulative confidence / (already_assigned + 1)` 순으로
   배분한다.

allocator 단위에서는 chain backbone의 후보 수를 줄이지 않는다. 하지만
이 규칙만으로 end-to-end P1 품질이 보존되지는 않았다. production RNG를
warmup 전 상태로 복원한 최신 3-seed 결과의 P1AL은 chain 3.733, tree
3.463이고 P1 hit도 0.522에서 0.508로 낮다. P2 tree hit 뒤에는 여러
terminal context를 16개 P1 lane으로 함께 덮어야 하므로, backbone floor와
별개로 실제 다음 context coverage가 분산될 수 있다. 따라서 이 절의 구조
규칙은 유지하되 P1 품질 보존은 미해결 항목으로 판정한다.

## 4. 비교 실험

조건: 25x384에 대응하는 3-seed 순서 회전, PROFILE=0, 동일 모델/temperature,
chain과 세 tree arm을 같은 코드에서 교대 실행했다.

| arm | TPS | tok/step | P1 hit | P1AL | P2 hit | P2AL | target ms | draft ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | 66.63 | 3.73 | 0.531 | 3.73 | 0.281 | 1.70 | 60.42 | 50.80 |
| all-root, Nv6 | 59.21 | 3.79 | 0.562 | 3.55 | 0.237 | 1.79 | 67.88 | 63.65 |
| all-root, Nv8 | 58.54 | 3.86 | 0.551 | 3.71 | 0.206 | 1.93 | 69.95 | 65.42 |
| **confidence auto-R6, Nv8** | **58.83** | **3.88** | **0.554** | **3.74** | **0.200** | **2.10** | **70.08** | **65.50** |

조건부 AL만 보면 auto-R6가 가장 좋다. 그러나 P2의 평균 step 기여를
`hit * (AL + 1)`로 계산하면 chain 0.758, auto-R6 0.622다. P2AL 증가보다
root coverage 손실이 더 크다. 반면 P1 기여는 2.512에서 2.632로 증가해
최종 tok/step을 소폭 끌어올렸다.

원 로그와 집계는
`experiments/proxy_async_overlap/tree_sweep/confidence_final/`에 있다.

## 5. timeline 및 proxy 병목

최종 profile은
`experiments/proxy_async_overlap/tree_sweep/confidence_timeline_final_20260806/`
에 있다.

### `target ms` 지표 정정

`Avg target time per full step`은 target model 실행시간이 아니다.
`LLMEngine.generate()`의 `self.step()` 전체 벽시계로, speculation request를
보낸 뒤 draft 응답을 기다리는 시간까지 포함한다. 3-seed 평균은
chain 60.42ms, tree 70.08ms로 +9.66ms지만, 실제 verifier 구간인
`Avg target verify time`은 50.71ms에서 53.29ms로 **+2.58ms**다.
나머지를 단순히 "느려진 draft를 기다린 시간"이라고 부르면 인과관계가
뒤집힌다. clean profile에서 draft의 `proxy_wait`은 chain K2 0.68ms에서
tree K2 7.82ms로 **+7.14ms** 증가했다. 즉 이 구간은 target의 넓어진
`graph_pre`와 tree proxy 계산 때문에 메시지가 늦게 도착해 draft가
기다린 시간이다. 늦어진 draft 완료가 다음 target step의 spec-wait으로
다시 보일 뿐, 최초 원인을 draft model forward로 귀속하면 안 된다.

최종 profile의 상태별 label 합으로도 target 계산은 chain 48.61ms,
tree 51.66ms로 +3.05ms다. 따라서 “tree가 target 계산을 모든 step에서
10ms 늘렸다”는 해석은 틀리다.

### target p50 (ms)

| arm/status | verify setup | graph pre | exit logits | proxy send | graph post |
|---|---:|---:|---:|---:|---:|
| chain K1 | 0.352 | 32.638 | 3.669 | 0.029 | 12.204 |
| tree K1 | 0.340 | 32.720 | 4.485 | 0.030 | 12.217 |
| chain K2 | 0.335 | 26.550 | 0.003 | 0.003 | 10.334 |
| tree K2 | 2.020 | 34.212 | 5.998 | 0.025 | 12.061 |

K1의 target 본체는 chain과 tree가 같다. K2에서 보이는 큰 증가는 proxy
메시지 전송이 아니다.

- `graph pre +7.66 ms`: chain 5행 대신 tree 최대 9행을 모델 앞부분으로
  검증하는 비용이다.
- `exit logits +6.00 ms`: 가능한 tree terminal별 다음-step proxy 확률을
  계산하는 비용이다. 한 accepted row만 계산하면 이 값은 줄지만 다음
  cache-key coverage를 잃으므로 현재 알고리즘과 동등하지 않다.
- `verify setup +1.69 ms`: 동적 ancestor mask와 FlashInfer plan 비용이다.
- 실제 proxy enqueue는 0.025 ms다. persistent send ring의 누적 대기는
  호출당 약 0.047 ms로 병목이 아니다.

코드상 tree의 `exit_logits` span에는 lm-head뿐 아니라
`_compute_and_send_proxy_tree()`의 full-vocab 작업도 들어간다. 구체적으로
`[valid+1,V]` p softmax, `[valid,V]` q softmax, sibling ladder의 반복
R/D 정규화, `[valid+1,V]` P_iv 구성과 global top-k다. 따라서 label 이름은
`proxy logits`처럼 보이지만 실제 병목은 전송이 아니라 dense terminal-mass
DP다. 단순 topology H2D 위치 변경은 clean 동등 A/B 이득이 확인되지 않아
채택하지 않았다. 다음 최적화는 ladder+P_iv kernel fusion 또는 sparse
candidate 근사이며, 후자는 seed/hit 동등성 검증이 필요하다.

`SSD_PROFILE_DUET_DETAIL=1`로 tree callback을 다시 나눈 10-hit 짧은
진단의 steady p50은 다음과 같다. 이 런은 노드 audit D2H도 켰으므로 전체
TPS에는 쓰지 않고, target GPU 내부 귀속에만 쓴다.

| graph_pre 종료 뒤 작업 | p50 (ms) |
|---|---:|
| exit norm + replica lm-head 완료까지 | 0.67 |
| p/q softmax | 0.04 |
| sibling accept/residual ladder | 2.39 |
| terminal P_iv 구성 + global candidate top-k | 1.15 |
| async send ring enqueue/전송 작업 | 0.53 |
| **proxy send GPU event 완료까지** | **4.81** |

따라서 6.1ms의 주범은 네트워크 전송이 아니다. 약 3.5ms가 tree 전용
dense ladder와 global 후보 선택이고, 동적 Python/PyTorch callback이 이
작업들을 enqueue하는 동안 target도 `graph_post`를 즉시 enqueue하지
못한다. side stream은 graph_post와 GPU 실행을 겹칠 수는 있지만, 현재
callback dispatch 자체를 없애지는 못한다.

당시 첫 tree hit 한 번은 candidate rank kernel의 lazy 초기화 때문에
send-ready가 33.51ms까지 튀었다. 이 cold outlier는 후속 작업에서 target
proxy graph를 init 때 캡처하고 P2 executor bucket도 미리 캡처해 제거했다.
최종 warmup 결과는 8절에 기록한다.

가장 직접적인 다음 최적화는 (a) topology를 고정-shape 입력 버퍼로 만들고
tree proxy 전체를 CUDA graph replay로 바꿔 host dispatch를 제거한 뒤,
(b) ladder와 P_iv/top-k를 1--2개 GPU kernel로 합치는 것이다. 더 공격적인
대안은 full target rows를 draft GPU로 보내 이 계산을 현재 7.8ms의 draft
idle 구간에서 수행하는 것이다. 후자는 통신 schema 변경이므로 exact
seed/hit 패리티 게이트 전에는 production 경로에 넣지 않는다.

과거 문서의 “tree graph_pre가 chain과 동률” 비교는 tree K2의 9행을
chain K1의 10행에 가깝게 비교한 결과였다. 정확한 status-matched 비교는
tree K2 9행 대 chain K2 5행이며, 현재 +7.66ms는 설계 당시 측정한
행당 약 1.9ms × 추가 4행과 정확히 일치한다.

### 실제 topology 및 hit 수 (2026-08-07 audit)

executor가 cache에 넣은 최종 root view와 target이 실제 보행한 tree를
직접 기록했다. 진단 D2H/file I/O가 있으므로 이 런의 TPS는 성능 판정에
쓰지 않는다.

- 생성: 263 P2 forests × 6 roots = 1,578 root trees
- 실제 P2 cache hit/serve: 50 trees (50/263 step = 19.0%)
- serve 직전 topology == target walk topology: 50/50 exact
- hit tree 크기: valid 8 = 34, 7 = 6, 6 = 7, 5 = 2, 4 = 1
- 따라서 valid 7/8의 40개(80%)가 9-row target bucket을 사용
- accepted path 평균: 2.08 (P2AL과 일치)
- sibling branch가 실제 accepted path에 들어간 hit: 9/50 (18%)
- hit root rank: `{0:20, 1:10, 2:3, 3:3, 4:5, 5:9}`

가장 흔한 8-node topology는 아래와 같다. 숫자는 node-local id다.

```text
root context
├─ 0
│  ├─ 3
│  │  └─ 6
│  │     └─ 7       # depth-4 backbone
│  └─ 4             # confidence rescue
├─ 1
│  └─ 5             # confidence rescue continuation
└─ 2                 # confidence rescue
```

원 기록은
`experiments/proxy_async_overlap/tree_sweep/topology_audit_20260807/`
에 있다.

### P1/P2가 실제 모든 tree node를 draft하는가

`SSD_TREE_NODE_AUDIT` 진단을 추가해 50회 P2 생성과 실제 P2 hit 10회를
노드별로 검사했다. 결과는 다음과 같다.

- P2 50/50 통과: 총 300 root view에서 모든 내부 노드는 실제 draft
  forward cell을 가졌고, 모든 leaf token은 정확한 부모 cell의 logits에서
  샘플됐으며, target으로 간 `parent_q_ref`도 300/300 exact였다.
- P2의 round별 실제 forward lane p50은 `[6, 10, 10, 6]`이다. CUDA graph
  안에서 네 round가 연속 실행되며 각 round 사이 host 개입은 없다.
- leaf는 자식이 없으므로 leaf 자체를 다시 forward하지 않는다. 이것은
  누락이 아니다. leaf token 검증에 필요한 q는 그 token을 생성한 부모
  forward의 분포다.
- 실제 tree-hit P1 10/10에서 모든 terminal context가 최소 한 lane을
  받았다. context 수는 5/6/9였고, 0-lane context는 없었다. 흔한 9-context
  배분 예시는 `[3,3,1,1,2,1,1,2,2]`로 합 16이다.
- 별도 topology trace에서 draft가 보낸 parent/sibling 배열과 target이
  보행한 배열도 50/50 exact였다.

진단 원본은
`experiments/proxy_async_overlap/tree_sweep/target_proxy_node_audit_20260807_v2/`
에 있다. 이 계측은 기본 OFF이며 production 성능에는 비용을 추가하지
않는다.

### P2 hit 하락에서 root count의 비중

root를 10개에서 6개로 줄이면 하위 4개 cache key가 없어지므로 hit 하락에
기여하는 것은 맞다. 그러나 이번 3-seed 비교에서 같은 Nv8의 all-root
tree hit는 0.206, auto-R6은 0.200으로 차이가 0.006p뿐이었다. chain
0.281과의 전체 차이 0.081p 대부분은 root count만으로 설명되지 않는다.

남은 차이에는 tree terminal-node key namespace, view truncation, tree-hit
후 terminal-mass proxy DP가 만드는 다음 seed 순위, 그리고 서로 달라진
generation trajectory가 함께 들어간다. 따라서 P2 hit 하락 전체를
“R=6의 의도된 교환”으로 확정해서는 안 된다. production 채택 전에는
동일 상태에서 R10 shadow keys를 함께 만들어, 실제 miss request가
discarded rank 6--9에 있었는지와 terminal/recovery mismatch였는지를
직접 분해해야 한다.

### draft p50 (tree, ms)

| status | P1 build | P1 replay | proxy wait | P2 select/build | P2 prepare | P2 graph |
|---|---:|---:|---:|---:|---:|---:|
| K1 hit | 1.326 | 2.410 | 5.416 | 2.642 | 0.756 | 12.450 |
| K2 hit | 3.269 | 2.388 | 7.876 | 3.421 | 0.839 | 12.450 |

P2 graph 앞의 3.4+0.84ms는 root 선택과 고정 입력 버퍼 준비다. 이는 네
forward **사이**의 공백이 아니며, graph 내부에는 CPU가 들어가지 않는다.
후처리는 GPU-native view로 바뀌어 `p2_output_convert` p50이 0.002ms다.

## 6. 이번에 수정한 구현 문제

- P1 backbone 후보 수를 chain floor보다 줄이지 않는 allocator
- canonical confidence root 수 자동 계산 및 executor/selector 단일화
- tree proxy도 chain과 같은 persistent asynchronous send ring 사용
- target tree mask CPU pre-pack으로 반복 GPU packbits 제거
- P1 topology CPU readback 1회 및 mask 생성 vectorization
- graph finalization의 위험한 custom gather 제거, bounded index-select 사용
- selector의 범위 밖 `chosen_pos`가 `python -O`에서 CUDA indexing으로
  들어가던 assert-only 가드를 runtime-safe clamp/mask로 교체
- graph capture 밖 GPU-native final view 생성으로 반복 D2H/Python 변환 제거

관련 핵심 tree allocator/proxy/CUDA graph 회귀 테스트는 88개가 통과했다.

## 7. 다음 성능 장벽

같은 40노드 안에서 root를 10개 유지하면 branch가 사라지고, root를 6개로
줄이면 P2 hit가 떨어진다. 다음 연구는 parameter sweep이 아니라 아래의
두 구조적 질문 중 하나를 검증해야 한다.

1. **coverage-preserving 추가 예산**: 10개 root의 K2 backbone 40노드를
   그대로 둔 채 branch 노드를 추가하고, 현재 target/draft overlap의
   여유 안에 추가 계산이 들어오는지 확인한다.
2. **target tree verification 축소**: AL을 보존하면서 실제 target이
   처리하는 행 수나 terminal proxy logits 수를 줄이는 수학적으로 동등한
   표현을 찾는다.

현재 데이터에서는 root/Nv/beta sweep을 더 돌릴 근거가 없다. tree를
production default로 전환하지 않고 `--duet_tree_policy confidence` 뒤에
두며, overall champion은 계속 chain으로 유지하는 것이 맞다.

## 8. 2026-08-07 실행 최적화 및 최종 재검증

이 절은 위의 2026-08-06 시간 수치를 대체한다. 알고리즘 파라미터는
`W10/R6/Nv8/F4/confidence`로 고정하고 실행 경로만 바꿨다.

### target proxy와 준비 비용

- tree terminal별 proxy 계산 전체를 valid 폭 1--8의 고정 입력 CUDA
  graph로 바꿨다. topology 값만 미리 채우고 callback에서는 graph 한 번을
  replay한다.
- steady proxy graph는 0.469ms, send enqueue는 0.017ms다. 이전의
  ladder 3.76ms + 후보 선택 1.38ms와 첫 hit 33.51ms 튐이 없어졌다.
- target tree verify 준비는 CPU list를 다시 tensor로 만들던 왕복과
  FlashInfer plan의 작은 GPU readback을 제거해 2.245ms에서 1.729ms로
  줄였다. 최신 무동기화 timeline의 K2 p50은 1.511ms다.

### 프로파일 방식 정정

`SSD_PROFILE=1`은 `torch.cuda.synchronize()`로 모든 stream을 기다린다.
tree glue를 시작하기 전에 이미 proxy `irecv`가 걸려 있으므로, 이 옵션을
켜면 아직 도착하지 않은 target proxy까지 기다리게 된다. 그 결과
`phase2_prep`이 39--42ms로 보였지만 이는 실제 실행에는 없는 계측
간섭이었다.

실제 서빙과 같은 `SSD_PROFILE=0, SSD_PROFILE_DUET=1`에서 다시 잰 K2
tree p50은 다음과 같다.

| 구간 | p50 (ms) |
|---|---:|
| tree glue 전체 | 4.717 |
| 그 안의 attention 준비 | 1.290 |
| 그 안의 draft model replay | 2.149 |
| P1 tree 후보 준비 | 2.458 |
| target proxy 대기 | 5.050 |
| P2 root 선택/입력 준비 | 2.542 + 0.638 |
| P2 4-round 전체 CUDA graph | 12.424 |

따라서 P2 네 forward 사이의 CPU 공백은 제거된 것이 맞다. 한 개의
12.424ms CUDA graph 블록 안에서 네 round가 연속 실행된다. 남은 비용은
forward **사이**의 idle이 아니라 graph 안의 실제 tree 선택/mask kernel,
graph 앞의 root 선택, 그리고 target 앞부분이 9행을 처리해 proxy가 늦게
도착하는 시간이다.

상태별 target p50도 다음처럼 정정한다.

| arm/status | verify setup | graph pre | exit/proxy dispatch | graph post |
|---|---:|---:|---:|---:|
| chain K1 | 0.394 | 32.356 | 4.468 | 12.245 |
| tree K1 | 0.327 | 31.896 | 3.782 | 12.217 |
| chain K2 | 0.386 | 26.242 | 0.003 | 10.421 |
| tree K2 | 1.511 | 32.589 | 0.003 | 12.728 |

K1 target 경로에는 구조적 회귀가 없다. K2의 `graph pre` +6.35ms와
`graph post` +2.31ms는 5행 chain 대신 `recovery + 최대 8 tree node`를
검증하는 실제 계산량이다. 과거 6ms였던 별도 exit/proxy 계산은 CUDA
graph화 후 dispatch label 0.003ms로 사라졌다.

원 timeline과 그림은
`experiments/proxy_async_overlap/tree_sweep/proxy_graph_final_timeline/`의
`chain/`과 `tree/`에 있다. 두 arm 모두 강제 상세 동기화를 끈 동일한
계측 조건이다.

### 첫 hit warmup

P2 executor가 첫 P2 hit에서 page bucket graph를 만들던 1--2초 지연도
실행 경로의 일부로 남아 있었다. 초기화 때 도달 가능한 bucket 1--7을
모두 캡처하도록 바꿨다. 현재 측정에서 5.9--10.8초와 약 602MiB가 서버
시작 비용으로 추가되지만, 엔진은 이 작업과 split layout capture가 모두
끝난 뒤에만 ready를 알린다. `SSD_TREE_EXEC_WARMUP=0`으로 진단용 lazy
동작을 복원하거나 comma-separated bucket만 준비할 수 있다.

CUDA graph capture는 graph 전용 generator를 실제로 전진시킨다. 처음
구현에서는 7개 bucket warmup이 production RNG를 미리 소비해 quality
비교 자체를 바꿨다. 현재는 warmup 직전 generator state를 저장하고 모든
capture 뒤 정확히 복원한다. 아래 최종 결과는 이 수정 뒤의 결과만 쓴다.

warmup smoke에서 실제 요청 중 executor 통계는 `replay=70, capture=0`이었고,
P2 graph replay는 min/median/p99/max =
12.386/12.432/12.552/19.186ms였다. 100ms 이상 outlier는 0회다.

### 하위 root와 최종 품질

같은 trajectory에서 rank 6--9의 key도 그림자로 기록한 1,487-request
진단 결과, 실제 miss를 추가로 맞힐 수 있었던 lower-root exact key는
19개, 즉 1.28%p였다. 하위 root 제거는 P2 hit 하락에 실제로 기여하지만,
3-seed chain/tree 격차 8.23%p 전체를 설명하지는 못한다.

최신 clean 3-seed 결과는 다음과 같다. chain은 동일 warmup 캠페인의
3 seed, tree는 RNG 복원 수정 뒤 같은 3 seed다.

| arm | TPS | tok/step | P1 hit | P1AL | P2 hit | P2AL | target verify ms | draft ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | 69.91 | 3.743 | 0.522 | 3.733 | 0.273 | 1.717 | 48.503 | 48.290 |
| tree | 60.91 | 3.593 | 0.508 | 3.463 | 0.247 | 1.923 | 51.213 | 58.543 |

P2AL은 +0.207(+12.0%)로 조건부 깊이 이득이 있다. 하지만 P2의 step당
평균 기여 `hit * (AL + 1)`은 0.742에서 0.723으로 줄었다. 더 큰 손실은
P1 기여가 2.476에서 2.269로 내려간 것이다. 모든 tree node가 실제로
forward되고 target과 topology가 exact라는 audit는 통과했으므로, 현재
증거는 누락 연산보다 제한된 root/P1 lane 예산의 coverage 문제를 가리킨다.
다만 quality gate를 통과하지 못했으므로 tree를 production default로
채택해서는 안 된다.

최종 원 로그는
`experiments/proxy_async_overlap/tree_sweep/proxy_graph_final_gate_warm_20260807/`
및
`experiments/proxy_async_overlap/tree_sweep/proxy_graph_final_gate_rngrestore_20260807/`
에 있다.
