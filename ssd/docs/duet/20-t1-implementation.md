# 20 — P2-tree 구현 로드맵·진행 (T1 → 최종 산출물까지)

**전권 위임 (사용자 2026-08-03)**: 끊김 없이 최종 구현까지 연속 진행.
모든 이슈는 본 문서(+17/19번)에 기록. 완료 후: timeline 기반 파라미터
sweep → AR/async-SD(C)/기존 DUET 대비 종합 지표 (cache hit, L_p1/L_p2,
TPS 등) → **정확한 timeline 그래프** (정식 도구
`bench/plot_duet_aligned_timeline.py`, 수 step 구간) → 논문/발표용
"기법 동작 원리 + 결과" 절 정리.

## 전체 로드맵

| 국면 | 내용 | 산출물 |
|---|---|---|
| T1 (진행 중) | draft rollout: 노브/예산/샘플러/루프/캐시 | 아래 단계표 |
| T2 | 트리 응답: wire 스키마(max-padded+헤더) + 캐시 뷰 + outcome(node id) + score 관통 | wire 왕복 테스트 |
| T3 | target: attention mode 계약 + rank별 wrapper + tree-verify CG(bucket) + 잔차 사다리 보행 + TP4 commit-ack + glue 트리화 + P1 fork 일반화 + tree Policy B | **작은 vocab 전수 분포-일치(하드 게이트)** + 체인 퇴화 bit-identical |
| T4 | E2E 통합 + **timeline 기반 sweep** (R/N_v/C_tensor/β/정책 × 균형 2조건 확인) | sweep 결과표 (CPU 한가 시간대 실측) |
| T5 | 비교분석: AR / async-SD(C-opt) / DUET-체인(champion) / DUET-tree — 5-rep 인터리브, 종합 지표 + timeline 그래프 (몇 step 정밀) | 비교표 + PNG + 논문용 절 |
| 마감 | NCCL A/B (연기분), 논문/발표용 기법 설명 + 결과 정리 | 21번(가칭) 최종 리포트 |

# T1 상세 (draft 측)

**전제**: 설계 v6 (15번) 확정 결정 ①~⑤ 준수. T1 범위 = draft 측
rollout(정책 스위치) + 비복원 샘플(D8/D11) + 사전 예산(⑤v2) + 형제
순서 기록 + 캐시 구조. **응답 wire(T2)·tree verify(T3)는 범위 밖** —
따라서 T1 동안 tree ON은 E2E로 돌지 않으며, 검증은 unit-수준(CPU 참조
동일성)과 tree OFF 회귀(bit-identical)로 한다.

**게이트 설계 (안전 원칙)**: `--duet_tree_policy {off, level, frontier}`
— 기본 **off** = 기존 경로 코드 그대로 (분기 1회). off에서 기존과
bit-identical이 T1의 hard 게이트.

## 단계 분해 (단계별 커밋 + 테스트)

| 단계 | 내용 | 검증 |
|---|---|---|
| T1.1 | config/CLI 노브 (`duet_tree_policy`, `duet_tree_c_tensor`(=C_tensor, 기본 3), `duet_tree_nv`(=N_v, 기본 8)) + OFF 게이트 배선 | config 동등성 (off ≡ 현행 필드), 기존 44개 회귀 |
| T1.2 | 사전 예산 b_x: root별 P_iv-비례 (β 지수 노브, 기본 0.5 — E1 근거) + 같은-forward 부모 priority-정렬 prefix 배분 + root 예산 소진 관리 — **pure 함수** | CPU 유닛: 합=예산, D10 준수(정체 무관), 몰빵/균등 경계 사례 |
| T1.3 | 비복원 샘플러: C_tensor 일괄 exponential-race + 앞 b_x만 valid + **뽑힌 순서 기록**; temp==0 게이트(체인 폴백); q_eff는 verify.py 분포 빌더 공유(`build_sampling_probs` 추출) | CPU 유닛: 분포 정합(카이제곱), 순서 재현성, fanout=1이 기존 단일 샘플과 **RNG 소비까지 동일** |
| T1.4 | rollout 루프: pool 텐서(고정 용량) + 정책 스위치 선택(level=최신 depth 마스크 / frontier=전체 topk) + 동적 조상 mask 주입(기존 draft packed-bitmask 기계에 내용만) + 셀 주소 규칙(§6 v6) | CPU 참조 구현(파이썬 트리) 대비 topology 동일성; mask 정합(조상 집합) |
| T1.5 | 캐시 구조: root별 서브트리 뷰 (tok/parent/순서/logits, U_max=N_v pad) 저장 — 응답은 아직 체인(기존) | 뷰 스키마 유닛; tree OFF 회귀 44개 + champion 스모크(OFF, 정확성만) |

## 진행 로그

**T1.1 완료**: config 필드 4종 (`duet_tree_policy` off|level|frontier
기본 off / `duet_tree_c_tensor` 3 / `duet_tree_nv` 8 /
`duet_tree_beta` 0.5) + -O 생존 검증 + CLI 4종. off 기본이라 기존
경로 무변경 — 동등성 6/6 + 회귀 44/44 green.

**T1.2 완료**: `ssd/engine/helpers/p2_tree.py` — `alloc_root_budgets`
(P_iv^β largest-remainder, cap=N_v, 결정론) + `alloc_fanouts` (같은
forward 부모들 priority-정렬 prefix 배분 — 리뷰 4차 초과 반례 방지).
**D10을 시그니처로 보장** (자식 정체 입력 없음 — 테스트로 고정).
유닛 11/11 (합 보존/cap/균등·비례 경계/결정론/무변이).

**T1.3 완료**: `tree_sample_wor` — exponential-race top-k (내림차순 =
비복원 추출 순서; 형제 순서 = 열 인덱스). **c=1이면 현행 Sampler와
op 시퀀스·RNG 소비·결과 bit-identical** (RNG state 동일성까지 테스트
고정 — fast-path 게이트의 핵심 근거). raw_q는 재정규화 전 원본
q_eff (결정② c_raw 규약; 독립 재계산 일치 + 형제합≤1 테스트).
temp==0은 명시 raise (v6 게이트 — 폴백은 rollout 호출자 책임).
유닛 누적 15/15.

**T2.0 완료 (순서 최적화 — T1.4의 priority가 라이브 P_iv에 의존하므로
선행)**: P_iv wire 비트-pack — `pack_piv`/`unpack_piv` (log10∈[-6,0]
16bit @ bits 15-30, 버전 bit 31, 양자화 오차 ≤ 9.2e-5 데케이드),
verifier 송신측 pack + draft `_unpack_duet_proxy`에서 **dedup 이전
단일 지점** unpack (D2 함정 회피). `duet_tree_policy != off` 게이트로
OFF wire 바이트 보존. config에 vocab≤32768 -O 생존 raise. 유닛 4종
(라운드트립 정밀도/dedup 안전/버전 비트/극단값 clamp) + 회귀 44/44
(m6 스텁에 tree_policy 필드 추가).

**이슈 #6**: 신설 config 필드 참조가 기존 테스트 스텁(SimpleNamespace)
을 깨는 패턴 반복 (이슈 #2와 동형) — 스텁에 필드 추가로 해결. 신규
config 필드 도입 시 스텁 갱신을 체크리스트화.

**T1.4a 완료**: rollout 알고리즘 골격 pure 구현 — `TreePool`(고정 용량
장부: tok/parent/depth/root/형제순서/logpri/raw_q/state/cell),
`select_nodes`(D1 정책 스위치 — level=depth==f, frontier=priority
상위 W; depth_cap 제외), `rollout_reference`(예산·선택 확정 → 그 후
정체 관측 — D10 순서를 코드 구조로 강제; 셀 주소 = f·W+k; D11
single-shot; priority = log π̂ + Σ log c_raw). 유닛 26/26: level
depth-동기 / **frontier의 depth 혼합 선택 (level은 불가능함을 동일
설정에서 대조)** / D11 재평가 없음 / root별 예산 보존 / priority
누적식 / 조상 셀 체인 / 셀 주소. 이 참조 구현이 T1.4b(엔진 배선)의
정답지가 된다.

**T1.4b-1 완료**: `build_tree_mask_packed` — 기존 chain 빌더의 packed
mask 기하([prefix|glue|spec 블록], packbits little)를 정확 복제한 트리
버전 (조상 셀 비트 + 자기 셀). **핵심 게이트 통과: fanout=1 퇴화에서
기존 체인 대각 mask와 비트 단위 일치** (step 0-2 전수) + 교차-행
조상 비트 정합 테스트. 유닛 누적 28/28.

**T1.4b-2 통합 노트 (다음 조각 — 발견된 정확한 접점)**:
- 진입: `_build_tree_batch_split_k1k2`의 P2 빌드 지점 (`policy != off`
  && B==1 && temps>0 분기; 아니면 기존 체인 경로). T1 단계에서 응답
  경로는 명시 raise (T2 전 — 검증은 stub 하네스).
- forward 재사용: `_decode_tree_step` 그대로 (slot/context는 기존
  `_compute_step_positions_and_slot_maps`의 (f,행) 셀 그리드 재사용).
  **동적 3요소만 교체**: ① input_ids[f] = 선택 노드 토큰, ②
  rope[f] = rope_base[root] + depth(노드) (레이아웃 정적 오프셋 무시),
  ③ mask[f] = `build_tree_mask_packed` 산출을 layout 캐시
  `cache["cpu_packed_masks"][f]`에 주입 (run_fi가 그 슬롯을 읽음 —
  cudagraph_helpers :441-456 확인). 글루 행 = root의 원 seed-행 글루
  패턴을 선택 순서로 재배열.
- 순차 의존: mask(f)는 f-1 샘플 결과에 의존 → F-루프 안에서 선택→
  mask 주입→forward→샘플 순서 (사전 일괄 불가 — 설계 §6 예고대로).
- root piv: T2.1(selector score 관통) 전까지는 rollout 진입부에서
  retained (pos,tok)를 wire에 역매칭해 chosen_piv gather (B=1 trivial).
- 검증(T1.4b-3): m2-패턴 stub 하네스 (run_model/sampler stub) —
  rollout_reference와 topology/장부 완전 일치 + fanout=1 퇴화에서
  기존 체인 경로와 spec_tokens 동일.

**T1.4b-a 완료**: `run_rollout` — forward_fn 주입형 rollout 코어
(per-forward 동적 3요소 [input_ids/rope=base+depth/packed mask] 구성 →
forward → 비복원 샘플 → pool 갱신; pad 행은 fanout 0 + RNG 소비 유지
로 고정 shape). **stub forward에서 rollout_reference와 topology 장부
전 필드 일치** + rope가 forward 인덱스가 아닌 depth 기반임을 검증.
유닛 누적 30/30. 남은 배선: 실엔진 어댑터(forward_fn = _decode_tree_
step + mask 캐시 주입) — T1.4b-b.

**T1.4b-b 완료 (배선)**: ① cudagraph_helpers에 mask override 훅
(`cache["_tree_mask_override"]` — 키 부재 시 체인 경로 무영향 3줄),
② `DraftRunner._p2tree_rollout` — run_rollout 코어에 실엔진 forward_fn
주입 (기존 셀 그리드 slot/context 재사용, input/rope/mask만 동적;
piv 역매칭 브릿지; finally로 override 정리). 라이브 진입 게이트는
T2(응답 조립) 전까지 미연결 — 검증은 stub 하네스(T1.4b-c)와 OFF 회귀.
회귀 44/44 + 유닛 30/30 유지.

**T1.5 완료**: `build_root_views` — root별 [nv] 고정 pad 뷰 (tok/
parent_local/sib_order/raw_q/valid). ⑤v2 생성-시점 캡 덕에 **절단
자체가 소멸** (assert 고정). 생성 순서 뷰 → parent_local < 자기
인덱스 (verify 보행 invariant — 리뷰4 row/slot 규약 충족). 유닛
누적 31/31, 회귀 44/44.

**T1 상태**: 알고리즘·배선·뷰 완료. 남은 것: T1.4b-c stub 하네스
(어댑터 vs 참조 topology — GPU/실런은 T2 응답 후 E2E에서), T2로 진행.

**T2.1 완료**: selector P_iv 관통 — 토큰과 **동일한 dedup/take/정렬
인덱싱**으로 taken_piv 반환 (chosen_piv 있을 때만 3-tuple — 기존 호출
2-tuple 불변). `_p2tree_rollout`의 역매칭 브릿지 제거. 유닛 33/33 +
회귀 44/44.

**T2.2 완료**: run_rollout이 **셀별 logits [F·W, V] 보존** (verify
q_eff의 원천 — 각 노드의 제안분포 = 부모 셀의 draft logits) +
`build_root_views`에 parent_q 참조 추가 (root별 고유 부모 셀 첫-등장
순 U 인덱싱, U_max=N_v assert; v6 §7.1 스키마 [parent_q_ref[nv] +
parent_q_logits[U,V] + u_valid]). 노드↔부모셀 logits 일치 테스트.
유닛 34/34 + 회귀 44/44.

**T3.3 완료 — 무손실 하드 게이트 통과 (v6 §10 go/no-go ①)**:
`tree_verify_walk` (잔차 사다리 + 컨텍스트 리셋 + 잎 bonus/전원기각
recovery + 수치 규약 [D_j=0 raise, 잔차소진 처리]) 참조 구현.
**전수 검증: 비복원 순서 전수 열거 × 사다리 해석해 합성 = target p
(30 케이스 × fanout 1-3, p=q·반희소 포함, 오차 < 1e-9)** + walk
구현의 MC 정합 (20만 시행 ±0.005). 트리 채택의 수학적 전제가 코드로
증명됨 — 남은 것은 이 참조를 엔진(T3 GPU 보행)이 재현하는지.

**T3.1 완료 (부분)**: target rank-로컬 tree-verify wrapper 인프라 —
`_init_tree_verify_wrappers` (N_v bucket {4,6,nv}별 CUDA-graph용
FlashInfer prefill wrapper + **rank당 128MB 공유 workspace** [리뷰4:
draft의 512MB는 그쪽 선택값], KV 할당 전 확보, B=1 고정 버퍼). tree
policy 게이트 — off면 무할당 (OFF 불변, 회귀 44/44). 남은 T3.1b:
attention.py의 명시 TREE_VERIFY mode 배선 (context 플래그 → wrapper
경로, 오진입 raise).

**T3.1b 완료**: attention에 명시 TREE_VERIFY 분기 —
`context.tree_verify_wrapper` 설정 시에만 FlashInfer wrapper 경로
(공통 꼬리와 동일한 [-1, H·D] 반환), draft에서 설정되면 명시 raise
(mode 오배선 방지 — 리뷰4). 미설정이면 기존 경로 완전 불변 (회귀
44/44). 남은 T3: T3.2 bucket capture(packed mask plan/run) → T3.4
엔진 보행+통합 → T3.5 TP4 commit-ack → live E2E.

**T3.2-a/T3.4-a/T3.5-a 완료 (pure 3종)**: target verify 행 조립
(depth 복원·rope·scratch slot·조상 mask — 리뷰4 row 계약) / 프로덕션
텐서 보행 (**참조 dict-보행과 동일-코인 50시행 완전 동등**) / 수락
경로 복사 계획 (겹침 대비). 유닛 40/40.

**T4 사전 준비**: sweep 하네스 골격 `tree_sweep/run_tree_sweep.sh` —
그리드 (정책 2 × R{8,10} × N_v{6,8} × β{0.5,1.0}) + **오염 가드**
(load<24 && GPU 유휴 확인 후 실행 — 19번 이슈의 공유-박스 교훈).

**남은 통합 (T3.4-b, Explore 맵 대기 중)**: SHM 명령 확장(rank1-3에
트리 메타 전달) + verify() 트리 분기(행 조립→wrapper plan→eager
forward→보행) + respond/wire 스플라이스 + glue 수락경로 재실체화 +
commit 실행부. 이 한 방이 live E2E 스위치.

**T3.4-b1/b2 완료 (통합 맵 기반)**: Context에 `tree_verify_wrapper`
정식 필드 + set_context 인자 / **wire 4곳 대칭 스플라이스** —
SpeculateResult에 tree_ints·parent_q_logits 필드, draft 송신에 트리
블록 concat + parent_q logits 추가 send (hit 없으면 zero 블록 —
max-padded로 크기·호출순서 불변), target 수신 버퍼/파싱/조립. 전부
tree policy 게이트 (off는 wire 바이트 불변). 구성자 위치인자 호출
전수 확인 (valid_k 이후는 전부 keyword ✓). 회귀 44/44 + 유닛 40/40.
다음: b3 draft 라이브 경로 (rollout→뷰 저장→respond 채움).

**T3.4-b3 완료 (draft 라이브 경로)**: ① selector 3-tuple 수용 (tree
ON일 때 taken_piv 관통), ② P2 빌드 지점에 tree 분기 — `policy != off
&& B==1 && temps>0`이면 `_decode_tree` 대신 `_p2tree_rollout` 호출
(step slot/context는 chain과 동일한 `_compute_step_positions_and_slot_
maps` 산출 재사용, metadata_ints 기반), `build_root_views`로 뷰 저장
(`self._tree_views`), populate에는 root별 **backbone(맏이 사슬) [R,K2]
투영**을 공급 — 캐시 키/valid_k/serving 형식은 체인과 동일 유지 (실제
응답 내용은 뷰가 담당; root는 pool[0..R-1] seed 순서 — run_rollout
확인), ③ respond 서빙 — P2 hit && B==1이면 out_tokens[:nv] = 뷰 토큰
(생성 순서), valid_k = 유효 노드 수 → **기존 extend/prepare 기계가
트리 행을 그대로 나른다** (창 = [rec]+뷰, scratch 셀 선형 — 리뷰4 row
계약과 정확 일치, 이것이 b4를 rope override+mask+plan으로 최소화하는
핵심), topology/parent_q는 b2 wire 블록에 pack_tree_ints로 채움. wire
필드는 respond 진입 시 step-국소 리셋 (이전 step 잔재 방지). config에
`nv <= max(K1,K2)` 검증 추가 (응답 wire 폭 계약). 유닛 40/40 + 회귀
44/44.

**b3 유의 (다음 조각 전 live ON 금지)**: 트리 hit 서빙 시 이 step의
glue가 뷰를 **선형으로** 실체화하려 들면 폭 nv+1 버킷이 없어 크래시
— TREE_GLUE(b3-3)가 선행돼야 live ON 가능. OFF 경로는 바이트 불변.
