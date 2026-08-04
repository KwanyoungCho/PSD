# 22 — T6: P2 draft 추가 22ms 제거 (GPU 상주 트리 상태) — v2

2026-08-04 v2 (리뷰4 수용 — v1의 모호성·오류 정정). 근거 수치는 전부
405794e 프로파일 쌍 재계산으로 검증 완료.

## 문제 (쉬운 말로)

트리는 체인보다 스텝당 +23.5ms 느리다. draft 모델 forward 속도는
완전히 같다 (4회 합 9.43 vs 9.44ms). 차이는 **P2 구간의 앞·사이·뒤에
붙은 CPU 왕복**이다:

| P2 구간 분해 (p50) | 체인 | 트리 | 추가 |
|---|---|---|---|
| build 시작→첫 forward | 2.1 | 7.4 | **+5.3** (root piv .cpu(), 예산·선택 파이썬) |
| forward 사이 간격 합 | 0.27 | 9.21 | **+9.2** (샘플 .cpu() 대기 + 파이썬 장부 + mask 재생성 + plan 동기화) |
| 마지막 forward→merge | — | — | **+7.5** (view/wire 파이썬 조립) |
| **build→merge 전체** | **14.7** | **36.7** | **+22.0** |

주의: 간격 9.2ms 전체가 "제거 가능한 낭비"는 아니다 — 그 안에 필수
GPU 샘플링(WOR/softmax/topk ~1.5ms)이 있다. 회수량은 구현 후
kernel-level 재측정으로만 확정.

**target 추가비용은 두 번째 문제**: hit_k2(트리 캐시를 실제로 맞힌
스텝)에서만 +15ms (graph_pre +7.4, exit/seed +5.9); hit_k1·miss는
≈0. 스텝 비중 반영 시 평균 +2~3ms — P2의 22ms를 먼저 없앤 뒤 판단.

**인과 정리 (v1 오류 정정)**: 실행 순서는 P1 → proxy_wait → P2 →
view/merge. 따라서 **P2 절감은 proxy_wait로 전가되지 않고 그대로
스텝에서 빠진다** (전가 논리는 P1 절감에만 해당). P2를 충분히 줄이면
그때 target 완료/다음 요청이 새 한계가 되는 것뿐.

## 목표 TPS (새 기준으로 재산정 — v1 산술 폐기)

기준: 트리 56.59 TPS·스텝 81.33ms (체인 78.89). 의미·AL 유지 가정:
`TPS ≈ 56.59 × 81.33 / (81.33 − 회수량)`

| 실제 critical-path 회수 | 예상 상한 |
|---|---|
| 간격만 ~9ms | ~63.6 |
| P2 전체 17ms | ~71.5 |
| draft-only 최대 ~19.9ms | ~74.9 |

전부 **상한**이다 (마지막 ~2ms는 요청 경계에 가려질 수 있음) — 확정은
wall-time A/B로만.

## 설계 (정책 불변 — 알고리즘은 그대로, 계산 장소만 CPU→GPU)

전제 명확화 (v1 정정):
- **R_phys=10 / R_active=6 구분**: 물리 seed 10개 유지, 상위 6개만
  예산 (#24 — 하위는 piv=0). arena·view·cache-key는 R_phys 기준,
  예산·tip lane은 R_active 기준.
- arena 필드에 **state(0=미평가/1=평가완료)** 명시 (v1 "active" 모호
  — 유효성은 n_nodes 미만 여부, 평가상태는 state).
- **ancestor bitset은 노드 전체 capacity 기준** [cap, F·W]: 이번에
  선택 안 된 후보도 나중에 선택될 수 있으므로 lane 폭 [W]로는 조상
  정보가 유실된다 (v1 오류). 자식 삽입 시 parent bitset | 자기 셀.
- 표기 정정: W≤10, R_phys≤10 (v1 "W·R≤10"은 오기).
- 동등성 게이트를 위해 **priority/예산 산술은 GPU float64** (스칼라
  급 텐서라 비용 무시 가능; vocab-폭 텐서만 float32 유지). stable
  argsort 동률 규약·RNG 소비 순서(현행 tree_sample_wor [W,V] 호출
  형상)를 그대로 보존.

### 구현 순서 (리뷰4 권장 순서 채택)

1. **P2 전체를 GPU 상주로 이전** (정책·예산·형상 불변): root budget →
   select(+tip lane) → fanout → mask → 샘플 삽입 → view/wire까지.
   fixed-topology·새 예산 정책 절대 혼입 금지.
2. **중간 CPU readback 0회 검사**: `.cpu()/.item()/.tolist()`/NumPy
   mask 재생성 전수 grep + 런타임 카운터.
3. **동작 동일성 게이트 (합격 조건 — "+18% 유지"는 여기 통과 전엔
   주장 금지)**: 같은 입력·시드에서 round별 비교 —
   선택 노드 / root별 fanout / token·raw_q / parent·sibling 순서 /
   priority / ancestor mask / 최종 root view / cache key / RNG 소비
   순서. 이후 E2E 스모크(무크래시·불변량) → 인터리브 A/B.
4. **남은 간격 재측정·분리**: GPU 샘플링 자체 vs FlashInfer plan
   동기화(`_plan_event.synchronize()` — GPU 이식만으로는 forward마다
   남는다). 필요 시 forward별 plan을 **rollout 전에 미리 준비**
   (기하는 요청 시점에 이미 확정 — context_len/DBT). 단 **마지막
   forward용 plan 하나를 4회 재사용하는 방식(plan-once)은 금지**
   (#20 붕괴 전례).
5. **P2만 바꾼 3회+ 인터리브 A/B 후 target 판단**: 그래도 평균
   +2~3ms target 병목이 남으면 그때 hit_k2 경로(graph_pre·exit)
   최적화.

## correctness 부채 (병행)

- [x] WOR support(#38)·temp0 게이트·assert 경화(#40)·requested 3값(#39)
- [ ] wire epoch 상수 1 → seq 재진입 카운터 (#35 staging 가드와 통합)
- [ ] SHM read-ACK: B=1·순차·단일 stream 가정에서만 안전 — stream
  추가 시 15번 ACK 계약으로 승격
- [ ] W10-top6 chain 대조군 knob (시간축 엄밀 분리)
- [ ] #36/#37 재도전 arm (인터리브 게이트에서만)


## 1a 실측 결과 (2026-08-04, eslab18 4×192 — 부정적, 원인 확정)

| 지표 (p50) | chain | CPU rollout | arena v1 | arena v2(to_pool 벡터화) |
|---|---|---|---|---|
| P2 창 | 12.35 | 21.49 | 26.71 | 25.95 |
| forward 간격 합 | 0.27 | 9.21 | 14.29 | 13.58 |
| pre | 2.14 | 7.40 | 11.44 | 11.41 |
| build→merge | 14.70 | 36.66 | 53.02 | 47.45 |
| 스텝 평균 | 57.8 | 81.3 | 97.5 | 93.1 |

정확성은 통과 (동등성 게이트 68/68 + E2E 무크래시·P2AL 2.28) —
성능은 **역행**. 원인 두 가지가 실측으로 확정됐다:

1. **eager 커널 dispatch가 대체한 파이썬보다 비싸다**: 장부를 GPU로
   옮기며 forward당 ~60-80개의 초소형 커널이 생겼고, 그 런치 비용이
   제거한 파이썬(pool 0.7 + mask 0.07ms/fwd)을 초과. 예산 계산도
   동일 (GPU 무동기판 ~80런치 > CPU 0.3ms — pre 7.4→11.4).
2. **간격 9.2ms의 상당분은 애초에 host 장부가 아니었다**: gap-prof의
   cpu_sync 2ms/fwd는 GPU(replay+샘플링) **완료 대기**를 포함 — 즉
   장부를 어디서 계산하든 남는 GPU 작업 + FlashInfer plan 동기화가
   간격의 바닥. 체인이 0.27ms인 진짜 이유는 forward+샘플링 루프가
   **CUDA graph 안에 통째로 캡처**되어 host가 사이에 없기 때문.

~~다음 지렛대 = 통째 캡처~~ — **리뷰5로 사실오류 정정 (v3)**:

1. **체인은 forward+샘플링을 통째로 캡처하지 않는다** (코드 확인:
   CG 캡처 범위 = model+logits뿐, cudagraph_helpers ~1051; 샘플링·
   depth 루프는 밖). 체인이 0.27ms인 이유는 "CPU가 없어서"가 아니라
   **replay 후 작업이 readback·가변 mask·plan 없이 연속 비동기
   enqueue 가능**하기 때문. 트리도 model×4 monolithic 캡처가 필수가
   아니다 (기존 graph를 바깥 graph로 감싸기는 PyTorch 2.8에서 불가
   확인 — "Cannot prepare for replay during capturing" — raw model
   재캡처는 최후 선택지).
2. **arena v1-2의 "readback 0회"는 거짓이었다**: boolean indexing이
   내부 nonzero→DtoH 동기화 (격리 프로파일: nonzero 24·DtoH 24·
   streamSync 34회/rollout), to_pool도 다중 DtoH, 진입 전 seed
   int()×10. 라운드당 kernel/copy 이벤트는 ~296 (내 "60-80"은
   표현식 수 — 부정확).
3. 수치 라벨 정정: v1 step 평균 99.07(warmup 제외; 96.9는 p50 근사),
   "후처리 53→47.5"는 build→merge 전체 (마지막 replay 이후 순수
   구간은 14.82→10.13), P2AL 2.28은 v1·v2는 2.07 (버전 구분).

**확정 방향 (리뷰5 채택)**: 기존 model graph 유지 + 트리 갱신부를
고정 형상·무동기 연산으로 만들어 **같은 stream에 교대로 enqueue**:
`plan×4 선준비 → [model graph replay f] → [트리갱신 f] → ...`
- 고정 mask 캔버스 = **주소·용량만 고정, 논리 kv_len은 forward별
  실제값 유지** (mask_indptr = 실제 qo×kv — 최대폭 강제는 plan-once
  의미 변경 재발).
- plan-ahead = forward별 wrapper/plan state 분리 (한 wrapper에 4
  plan 연속은 마지막만 남음; plan은 CG 내부 실행 불가 — FlashInfer
  명세).
- 순서: 문서·수치 정정 → CUDA parity CI → **plan-ahead 단독을 CPU
  tree에 A/B** → persistent arena·동적할당 제거 → boolean indexing
  제거(고정 W×C dummy-slot) → actual-cols mask-pack kernel byte
  parity → (필요시) 갱신부만 Triton/CG → view/wire GPU화.
- 성능 게이트 (단계): ① build→merge ≤36.7·step ≤81.3 (CPU tree
  동률 — 현재 +11.8ms 부채) → ② step ≤72.4 (63.6 TPS) → 그 후에만
  재예측. 67-70 TPS 재예측은 이르다.

**1a 후속 반영 (arena v3, 이번 배치)**: boolean indexing 전폐
(고정 W×C + 말단 scratch 슬롯 라우팅), to_pool 동기화 3회로 통합
(int/float 필드 stack 2 DtoH + n 스칼라; 압축·재매김은 CPU), 정렬
키 f32 정합 (CPU 비교 키와 동률 거동 일치), 진입 전 seed int()×10
제거, CUDA 실기 parity + zero-q support-소진 테스트 CI 추가.
