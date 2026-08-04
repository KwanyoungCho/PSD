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

**T3.4-b3-3 완료 (TREE_GLUE — 결정④ ⓐ)**: `_tree_glue_decode` —
트리-hit step의 글루를 split_k2 tree-decode CG(step 0) + packed mask
override 한 번의 W-폭 MQ forward로 대체. 입력 = [rec]+뷰 노드(생성
순서; make_glue가 이미 그 형태), KV = 체인과 동일 canonical scratch
셀 pos0+1+j (뷰 순서 — **target verify row 계약과 일치**), rope만
pos0+1+depth(j) 덮어쓰기, mask = [prefix 1s | rec+조상+self] 직접
구성 (packbits little, B=1 세그먼트 — 기존 규약 복제). pad 행 slot
-1 + prefix-only mask (NaN 방지). respond에 `_tree_hit_root` 스태시
추가. 진입 분기: 트리 step이면 TREE_GLUE 후 **명시 raise** (P1/P2
노드-fork ⓑⓒ가 다음 조각 — live ON 중단 지점을 프런티어에 유지).
유닛 40/40 + 회귀 44/44 (OFF 불변).

**b3-4 계획 (노드-fork P1 — 결정④ ⓑ v1)**: fork 컨텍스트 =
[rec(root-종단), 노드 0..n_valid-1] = 종단 노드 id 네임스페이스
(p=0 → root-종단, p=1+j → 노드 j — populate의 fan_idx가 그대로 종단
id가 됨). CG 불변 유지를 위해 P1 폭은 기존 레이아웃(MQ 16) 유지,
컨텍스트별 fanout은 (2,2,2,2,2,2,1,1,1)+filler 재배분 (F7 예산
재검토 전 v1 — report 명시). fork 행 mask = 조상 비트맵 (split_k1
CG 전 step mask override), rope = depth 기반.

**T3.4-b3-4/-5 완료 (트리 step P1/P2 — 결정④ ⓑⓒ v1)**:
`_tree_step_p1p2` — 트리-hit step의 전용 경로 (TREE_GLUE 후 진입,
체인 경로와 완전 분리). P1(ⓑ): `_select_tree_fork_tokens` (컨텍스트
c의 제외 집합 = 뷰-내 c의 자식 토큰 — 체인 '다음 토큰 제외'의 일반화;
컨텍스트별 fanout = MQ 16의 균등-우선 배분), fork 행 rope = pos0+1+
depth(컨텍스트), 전 K1 step packed mask 직접 구성 ([prefix+rec | 조상
노드 셀 | 자기 행 체인 셀] — gap 셀 자동 배제), populate에
`draft_fan_idx_override` = 컨텍스트 id (= 종단 노드 id 키). P2(ⓒ v1):
선택기 위치축 = 노드 id 축 (chosen_pos = 노드 id — wire 형식 불변),
K_rank = n_valid로 in-place 레이아웃 갱신 (proxy fan_idx = 노드 id
자동), seed rope = 컨텍스트 depth, 롤아웃 어댑터에
`glue_rows_override`/`K_glue_override` (조상 비트맵) 추가, 새 뷰/
backbone populate은 체인 branch와 공용 헬퍼 `_tree_backbone_project`.
유닛 40/40 + 회귀 44/44 (OFF 불변).

**이슈 #7 (b3-5 중 발견, 프리-존재)**: 롤아웃 어댑터의 체인-step 글루
가시성이 `min(p, K2)+1` — vk=K1 step에서는 글루 블록이 K_rank+1 폭
이어야 하는데 K2+1로 고정되어 (a) prefix 1s가 초반 spec 셀을 전 행
공개 (seed p=2가 s3·s4를 봄 — 미래 누출), (b) 글루 비트가 마지막
K2+1 셀로 오정렬. stub 테스트가 vk=K2 구성이라 통과했던 것. 수정:
글루 폭 = 위치축 길이(len(fan_out)), `:p+1` (클램프 제거). 트리
step은 override 경로라 무영향.

**ⓒ 정확성 유의 (v6 §7.5ⓒ 그대로)**: 트리-step Policy B의 α̂/corr은
맏이 형제에만 정확 — 사다리(R̂,D̂) 반영 ĥ-DP 일반화는 verifier 트리
분기(b5)에서 다룸. E1의 체인-ĥ 예측력 검증은 그 시점에 재확인.

**T3.4-b4 완료 (target 트리 verify)**: ① run()에 `tree_meta` 인자 —
SHM pickle로 전 rank 동기 (step_lookahead와 동일 패턴), 매 호출
무조건 대입(리셋). ② verifier가 tree_ints valid>0이면 리스트로 전달.
③ `_run_tree_verify`: 창 = [rec|뷰 노드] (extend/prepare의 선형
slot 그대로), N_v bucket으로 행 패딩 (slot -1, prefix-only mask —
이후 CG capture와 shape 동일), rope = pos0+1+depth 덮어쓰기,
FlashInfer plan (조상 custom mask, head 수는 num_tp_gpus 분모 —
world_size는 draft GPU 포함이라 부적합), eager 분할 forward
pre(end_layer)→exit-proxy(mid 게이트 replica/topm/plain 3종 미러 —
행 축=노드 축이라 proxy가 자연히 노드별 p^E)→post(start_layer+
init_hidden/residual)→compute_logits, pad 행 절단.

**T3.4-b5 완료 (verifier 트리 보행)**: ① `q_probs_from_logits` 공유
헬퍼 추출 — 샘플측(tree_sample_wor)과 verify측이 동일 함수로 q 빌드
(수락 보존 전제; c=1 RNG bit-parity 테스트가 op-동일성 고정). ②
proxy용 q 교체: 트리 step은 logits_q = parent_q_logits를
parent_q_ref로 gather (backbone 캐시 행은 노드의 q가 아님 — α̂ =
p^E/q_parent). 위치축 정렬 확인: 체인 chosen_pos=p ↔ 트리 종단 노드
id p (rec=0, 노드 j=1+j) — 수치 axes가 정확히 일치, ĥ 수식은 맏이-
정확 v1 (DP 일반화 후속). ③ `_tree_verify_walk`:
tree_verify_walk_tensor (동일-코인 동등성으로 고정된 프로덕션 보행)
호출, suffix = [rec]+경로 토큰 / recovery 별도 — postprocess가
내용-무관 append라 **seq 롤백이 자동으로 트리-정확** (step.py는 전체
복원 후 suffix append). 종단 노드 id는 `self._tree_terminal_node`에
스태시 (b6 wire). v1 비용: 보행 CPU 사다리 (~2.3MB p/q 복사) —
GPU화 후속. 유닛 40/40 + 회귀 44/44.

**b6 남은 것**: outcome wire에 종단 노드 id 동승 → speculator 다음
요청 키 k_idx = 종단 노드 id, D14 수락경로 재실체화 (draft, 다음
step 글루 앞 aux forward), target TP4 commit (path cells gather→
scatter + ack — commit_copy_plan 실행부).

**T3.4-b6 완료 (outcome/commit — live 루프 폐합)**: 핵심 단순화 발견
— **재실체화는 forward가 아니라 KV 셀 복사로 충분**: 트리 rope
(pos0+1+depth)가 수락 경로에서는 canonical 위치와 정확히 일치하므로
셀 자리만 뷰-순서 → 경로-순서로 옮기면 D14가 완성된다 (aux forward
불요 — 설계 §7.5ⓐ의 "glue가 다시 만들어준다"보다 싼 등가 실행).
- **b6-1**: verifier가 종단 노드 id를 seq에 스태시 → speculator 다음
  요청 k_idx = 종단 id (트리-step populate의 fan_idx 네임스페이스와
  일치). 절단 시에도 경로-prefix == 수락-prefix라 id 유효 (키 miss
  가능 — 무해).
- **b6-2 (draft)**: respond에서 서빙 스냅샷 double-buffer
  (_tree_served_ints/_numtok), 다음 요청 수신 직후 parent_local
  사슬로 경로 복원 → a_eff = num_tokens 델타 (EOS/max 절단 자동
  반영) → commit_tree_kv 로컬 복사. one-shot 소거.
- **b6-3 (target)**: `ModelRunner.commit_tree_kv(src,dst)` — kv_cache
  flat 뷰 gather→scatter (겹침 대비 tmp). verifier가 보행 직후
  call()로 전 TP rank 브로드캐스트. **ack 생략 근거 (B=1)**: SHM
  명령은 rank별 순차 실행 — commit(cmd n)이 다음 run(cmd n+1)보다
  항상 선행. B>1 tree는 게이트 OFF; ack는 그때 신설 (설계 §7.5 원안
  그대로 명시).
- 보행이 경로를 직접 반환하도록 확장 (_walk_path → src/dst 슬롯).
회귀 44/44 + 유닛 40/40. **이로써 draft 서빙→target 트리 verify→
보행→commit→다음 요청 키→재실체화→노드-fork 캐시빌드의 라이브
루프가 코드 상 폐합** — 남은 것은 E2E 검증 (OFF 회귀 → 퇴화 ON →
실런).

## E2E 검증 (T3.4-b 이후)

측정 환경 유의: 박스 CPU load ~82 (타 사용자 작업) — 19번 이슈
방침대로 **correctness 검증만 진행**, 이 구간의 TPS는 해석 금지.
T4 sweep은 하네스의 wait_clean_box 가드 (load<24)로 한산 시 실행.

**이슈 #8 (E2E-1 1차에서 발견)**: b2의 tree wire 게이트가
`SpeculatorAsync.self.config`를 참조하나 그 클래스는 config를 보유하지
않음 (개별 필드만 복사) → 첫 spec 요청에서 AttributeError. 유닛
스텁은 SimpleNamespace 기반이라 은폐 (이슈 #6과 동류 — 실엔진 생성자
경계는 스텁이 못 잡는다). 수정: 생성자 keyword `config=None` 주입
(스텁은 getattr 기본값 "off"로 안전) + llm_engine 전달. 부수 발견:
백그라운드 셸에서 `pkill -f "bench.py"`가 자기 래퍼 명령줄에 매치되어
자살하는 함정 — E2E 스크립트에서 pkill 분리.

**이슈 #9 (E2E-1 2차)**: b2의 `_tree_ints`가 recv 헬퍼 로컬 변수인데
speculate()의 SpeculateResult 조립이 참조 (NameError — #8과 동류의
함수-스코프 단절). self._tree_ints_step 스태시로 수정.

**E2E-1 통과 (r3)**: tree_policy=off 라이브 런 완주 — Decode 52.13
tok/s (CPU load ~82 오염 하; 절대값 해석 금지), AL 3.53, cache hit
0.78, hit-AL 3.80 — champion-류 정상 지표. OFF 경로 무결 확인.
같은 시각, 트리-ON 경로의 #8/#9 동류 결함을 실행 전 정적으로 걷어내는
5-finder + 적대검증 감사 병행 (아래).

**정적 통합 감사 (E2E-2 전, 5-finder + 발견별 적대검증 워크플로)**:
finder 18건 → 검증 17건 확정 → 근본 결함 7개로 수렴, 전부 수정:
- **이슈 #10 (champion 즉발 크래시)**: run_fi step-0의 MQ-변경
  `cache.clear()`가 `_tree_mask_override`를 삭제 — P1(16)↔P2(10) 폭이
  달라 **모든 트리 빌드**에서 f=0 무음 체인-마스크 + f=1 KeyError.
  pop/복원으로 보존.
- **이슈 #11**: override 활성 진입의 체인 mask 재빌드가 트리 글루
  기하(음수 prefix)로 깨질 수 있고 결과물도 안 읽힘 — override 존재
  시 통째로 생략 (체인 호출은 자기 step-0에서 재빌드).
- **이슈 #12 (즉발 크래시)**: `_run_tree_verify`의 N_v bucket 패딩
  행이 exit-proxy에 유출 — `view(B, vk+1, V)` shape 크래시. 세 게이트
  분기 모두 n_rows 절단 (collective 참여 shape 전 rank 동일 유지).
- **이슈 #13 (즉발 크래시)**: backbone `_bl` fp32 — fp16 draft_logits
  와 `torch.cat` dtype 불일치. hf dtype으로 할당.
- **이슈 #14 (무손실성)**: valid==0 root(예산 0)의 zero-backbone 행이
  체인 응답으로 서빙되면 q가 실제 제안분포가 아니게 됨 — populate 후
  해당 키 -1 무효화 (miss → JIT).
- **이슈 #15**: raw-proxy(SSD_DUET_PROXY_ON_DRAFT)·topm_gather와 tree
  ON 조합이 config-합법이었으나 경로 미구현 (KeyError/piv 소실) —
  v1 상호배제 raise (champion은 무영향).
- **이슈 #16**: 트리 valid 하한이 1이라 P_iv 후보가 wire_N을 못 덮는
  구성 가능 — proxy_top_k 자동 상향에 트리 바닥 ceil(wire_N/2) 추가.

**PROFILE 스팬 분해 (prof_level, load~82 하 — 상대 비교용)**: 트리
스텝(hit_k2) target graph_pre 693ms / graph_post 194ms (체인 33/13ms
— eager 80층의 ~21×), 보행 96ms; draft 측 proxy_wait 619ms는 그
거울상. → **eager 트리 verify가 지배 항 확정**, sweep 전 CG capture
필수 (아니면 sweep이 트리에 구조적 불리 편향).

**최적화 적용 (274877b)**: ① 롤아웃 logits GPU 상주 (per-forward
1.3MB CPU 왕복 제거 — frontier 런에서 level 대비 4× 관측), ② 보행
p/q GPU 상주 (2.3MB/step 제거), ③ **T3.2 트리 verify CG capture** —
N_v bucket별 pre/post 분할 CG (duet_verify 구조 + tree wrapper +
rope 입력 버퍼, duet_pool 공유), replay 전 매 step public plan
(shape 고정 계약), SSD_TREE_VERIFY_EAGER=1 폴백 유지.

**이슈 #17 (CG 검증 1차)**: capture 함수를 `def capture_duet_verify`
앵커 앞에 삽입하면서 그 함수의 `@torch.inference_mode()` 데코레이터를
가로챔 — capture_duet_verify가 grad-모드로 워밍업되어 dynamo가
autograd 경로 컴파일 → inplace-op RuntimeError로 전 rank 사망.
데코레이터 양쪽 명시로 수정. 교훈: 함수-단위 삽입 앵커는 데코레이터
포함 여부를 반드시 확인.

**이슈 #18 (CG 검증 2·3차 — 연속 트리 스텝 크래시의 진짜 원인)**:
CG replay 분기가 prepare_decode의 체인-verify context(cu_seqlens_q
설정)를 교체하지 않아 lm_head의 mq-decode 분기가 [1, rows, V] 3-D를
반환 → `shape[0]==r_b` pad-절단 조건이 무력화되어 r_b행 전체가
verifier로 유출. 증상이 스텝 데이터에 따라 view 크래시(나눠떨어지지
않음) 또는 보행 내 R-D 크기 불일치(나눠떨어짐 — last dim 36000)로
변장해 이틀치 가설을 소모함. **진단 방법론이 결정적이었음**: 양측
wire 불변 가드 + 반환 행수 하드 불변(진단값 동봉)을 심은 재현 런
1회로 rows=1(=3-D) 포착 → lm_head의 context 의존 확인. 수정: CG
분기도 eager와 동일하게 cu_seqlens_q 없는 context로 교체. 가드
3점은 상시 유지 (비용 무시 가능, 이 클래스 재발 시 즉시 국소화).
**CG 효과 확정: 트리 스텝 graph_pre 693ms → 31.5ms (22×), 체인
수준 도달.**

**이슈 #20 (plan-once 회귀 — 롤백)**: 시간축 최적화로 rollout의
per-forward FlashInfer plan을 최종-기하 1회로 대체(64052f6)했으나,
같은 박스·시드 대조에서 P2 히트당 수락 2.14(29d4a2d) → 1.1~1.3으로
붕괴 (reject_all 0.12→0.40; 트리 형상은 무사 dmax4=78%). plan-skip
여부(po0/po1)와 무관하게 깨져 상시-켜진 마스크 최종-폭 확장
(cols_override)이 fa2 packed-mask 소비와 불일치하는 것으로 추정 —
정확 기전 미상. **전체 revert** (검증된 29d4a2d 동작 복원). 교훈:
attention 마스크/plan 기하 변경은 GPU 단위 A/B(동일 시드 수락률
대조) 없이 랜딩 금지. 시간축 재도전은 저위험 항목(pool 장부 텐서화,
packbits 제거, plan-ahead 별도 wrapper)부터.

## 외부 리뷰 수용 (2026-08-04, 사용자 전달) — 전 항목 코드-대조 검증

리뷰 요지: "현 구현은 설계가 의도한 최적 트리가 아니다 — topology·
Policy-B·KV lifecycle 수정 전 sweep/채택 판단 금지." 항목별 검증
결과 **전부 타당** 판정, 이슈 채번 후 수정 착수. 제 이전 판단 3건
정정: ① "SHM 순서가 ack 대체" 철회 (write_shm은 소비-대기 없는 단일
버퍼 — 레이스 실재), ② Policy-B의 p^E row 페어링은 '문서화된 근사'가
아니라 오배열 (q만 parent로 고치고 분자는 체인 그대로였음), ③ E1
+3.8%는 동적-트리 가정의 수치 — 명시 트리 상한 아님.

| 이슈 | 내용 (리뷰 항목) | 검증 | 수정 |
|---|---|---|---|
| #21 | draft 수락경로 KV 지연-copy가 해제된 scratch 재참조 가능 (2D) | scheduler가 매 step excess draft block 해제 — 새 dbt로 src 재계산하는 b6-2는 경계-초과 step에서 오염 가능 | TREE_GLUE 직후 뷰 KV를 staging buffer로 gather, 다음 요청에서 accepted path만 canonical로 scatter |
| #22 | TP commit SHM ACK 부재 (2E) | write_shm이 소비 확인 없이 버퍼 덮어씀 — 스텝당 call 2회(commit+run)가 되며 노출 | write_shm에 소비-대기 (worker read 후 event clear를 기다림) — 전 명령 공통의 클로버 방지 |
| #23 | root budget 조용한 유실 (2A) | 재현: β=1 cap=8 → 32.45/40 (최저 18); +1 루프가 root당 1회뿐 | capped water-filling (소진 보장) + `sum == min(total, R·cap)` 테스트 |
| #24 | R=W 결합 — R8은 다변수 동시 변경 (2B) | 기확인 (root당 예산=K2 상수) | `duet_tree_root_count` — W/CG/예약 불변, 상위-R root만 예산 (나머지는 #14 키 무효화 경로) |
| #25 | Policy-B 체인 수식 (2C) | p^E gather가 row j (체인) — 트리는 parent+1 row가 정답; ĥ cumprod도 체인 전제 | 트리 전용 proxy: p^E row=parent+1 페어링 + terminal-mass DP (reach×앞형제기각×전원기각) |
| #26 | E1 tree_L이 동적-트리 가정 (3) | tree_L은 레벨당 fanout 공유 가정 — 명시 [2,2,2,2]는 30노드 필요 | 명시-트리 terminal DP로 상한 재산정 (분석 스크립트) |

리뷰 4절(무죄 항목: mask/RoPE/보행/dup)은 자체 진단(§4.5 — C=1이
체인 수락 재현)과 합치. 리뷰 인용 체인 P2AL 1.76은 측정창 차이
(우리 1.69~2.14 변동 범위 내). 수정 순서 = 리뷰 5절 채택:
correctness(#21·#22) → 예산(#23) → R분리(#24) → spine+rescue 정련 →
Policy-B DP(#25) → E1 재산정(#26) → C=1 byte-parity 게이트 →
동일-시드 인터리브 재실험. **진행 중이던 v2 sweep은 중단** (판정
부적격 상태 측정 방지).

**#21·#22 완료 (correctness 쌍)**: ① KV staging — TREE_GLUE가 쓴
[rec+뷰] 셀의 KV를 staging 텐서로 gather (tinyllama 기준 ~360KB),
다음 요청의 수락경로 재실체화는 staging→canonical scatter (물리 블록
생존 무의존; 종전의 dbt-재계산 src 제거, same-slot skip도 제거 —
staging이 권위 소스). ② write_shm 소비-대기 — 모든 worker의 event
clear(=버퍼 복사 완료) 확인 후 기록 (전 명령 공통 클로버 방지; 평시
즉시 통과). 회귀 44/44 + 유닛 45/45.

**#23 완료**: alloc_root_budgets를 capped water-filling으로 (frac
내림차순 라운드-로빈, 소진 보장). 검증 강화: 스큐 piv 300케이스
전수에서 `sum == min(total, R·cap)` + cap-바인딩 케이스. 유닛 47/47.

**#24 완료**: `duet_tree_root_count` — W/CG/스케줄러 예약/키 폭 전부
불변, P_iv 상위 R root만 예산 (adapter에서 하위 piv를 0 sentinel로;
alloc은 piv≤0을 water-filling 포화 후에도 명시 배제). 무예산 root는
뷰 0 → #14 키 무효화 = 명시적 miss. 이제 R ablation이 단일 변수.
유닛 48/48 + 회귀 44/44.

**#25 완료**: `_compute_and_send_proxy_tree` — 트리 step 전용
Policy-B. ① α̂의 p^E를 **부모 컨텍스트 row**(parent_local+1)에서
gather (종전 row-j 페어링 오배열 수정), ② ĥ를 `terminal_mass_dp`
(pure 함수)로: reach = 경로곱×앞형제기각, terminal = reach×자식전원
기각 — **체인-퇴화에서 chain first-reject 분포와 정확 일치 + 총질량
1을 유닛으로 고정** (+형제 케이스), ③ residual: 내부 ctx =
(p^E−q_ctx)+에 자식 토큰 제외, 잎 ctx = p^E 그대로 (잎 보너스 정합),
④ P_iv = terminal(ctx)·r̂ → (ctx,tok) 전역 topk — wire/pack 형식
불변 (chosen_pos = ctx id = fork 네임스페이스). 유닛 50/50 + 회귀
44/44.

**#26 완료**: E1 상한 재산정 (e1_explicit_tree.py) — 구모델은 공유-
트리 가정(명시 30노드)이라 무효; 명시 backbone+rescue DP 상한 =
budget8에서 2.137 (+19.1% per-hit) — **control 실측 2.14와 일치**
(backbone이 이미 상한 실현). AL 환산 +2%대 = 교정된 토큰 이득 상한.
상세 21번 §4.6.

**시간축 안전-슬림화 (gap-prof 계측 → 표적 제거)**: forward 사이
구간을 5분할 계측(SSD_TREE_GAP_PROF=1)해 분해 — sample단 9.8ms/step
의 정체는 tree_sample_wor 진입 가드(GPU temps .any())의 조기 동기,
pool 6.1ms는 파이썬 캐스팅. 수정: ① 가드를 호출자-보증
(assume_pos_temps)으로 생략, ② pool 장부 파이썬 미러(root/depth/
logpri/tip_depth — 텐서 캐스팅 제거, 결과 불변·topology 테스트
통과). **비-fwd 오버헤드 16.4 → 12.2ms/step (−4.2)**; 잔여 12.2 중
~7.2는 .cpu() 동기 = GPU 실행 대기(순차 의존 하한). 추가 회수는
plan 오버랩(이중-그래프) 등 리스크 항목 — T5 판정 후 결정.

## 외부 리뷰 2차 수용 (2026-08-04) — 7주장 전수 테스트 확정

검증 배터리(재현 런·반례 계산·전수 열거·프로파일)로 전 항목 확정.
**1차 결론 2건 철회**: "예산 보존 상쇄"(→진범은 P1 악화: P2 축
+0.048 vs P1 축 −0.054), "+0.6pt DP 이득"(top-6=90.7% off-by-one
정정 후 실측 0.217 < 예측 0.220).

| 이슈 | 내용 | 검증 | 수정 방향 |
|---|---|---|---|
| #27 | select_nodes top-W 컷이 backbone tip을 탈락시켜 예산 소실 (재현: 40 중 34 생성, 약root dmax=1 영구정지; trace 평균 32.15/40) | 재현 ✓ | tip 우선 예약 lane + 잔여만 priority rescue + generated==allocated 불변 기록 |
| #28 | 트리 Policy-B 형제 α가 원본 p/q 독립곱 — 실제 사다리(R/D 갱신)와 상이 (반례: 0 vs 0.28125) | 반례 ✓ | 정확 사다리 C≤3 unroll, GPU 벡터화 |
| #29 | R6 손실 서술 off-by-one (top-5 87% ≠ top-6 90.7%) | 재계산 ✓ | 분석·서술 정정 (완료) |
| #30 | AL 상쇄 회계 오류 (잃은 hit은 0이 아니라 miss 2.72로 이동) | 재계산 ✓ | 21번 판정 정정 (완료) — P1 악화가 주범 |
| #31 | 트리-스텝 P1 균등배분 (terminal mass 비대칭 무시) — P1 악화의 유력 원인 | 회계 ✓ | terminal-mass 가중 P1 배분 (empirical prior/2단 배분) |
| #32 | 2.137은 상한 아님 — 전수 열거(24,364) 최적 2.2167 @ [-1,-1,-1,0,0,1,3,6] | 열거 ✓ | "한 형상의 대체 예측값"으로 정정 + 열거-기반 template 설계 입력 |
| #33 | water-filling이 cap 후 비례 재계산 없이 라운드로빈 균등화 ([8,5,3] vs [8,7,1]) | 예제 ✓ | active-set 비례 재정규화 |
| #34 | #25 구현이 exit_logits 0.3→17.6ms (GPU 스칼라 추출·CPU full-vocab 행) | 프로파일 ✓ | Policy-B 전면 GPU화 (사다리와 함께) |
| #35 | pending state 가드 부재 (staging에 seq_id/epoch 없음·preempt/prefill 미청소·tree_terminal_node 비정식 필드) + SHM event는 read-ACK | 코드 ✓ | seq_id/epoch 가드 + prefill/preempt clear + 정식 필드화 |

방법론 노트(수용): T5 스크립트 주석 5회↔실제 3회 정정, 고정 팔-순서,
stale-log skip 위험 — 이후 verdict는 라벨에 코드 rev 포함. **필수
대조군에 chain-R6 추가** (P2AL +12.8%에서 상위-root 선택편향 분리).
현 T5 런의 tree 팔은 "현행-구현" 데이터점으로 캐비앗.

**구현 완료 (2026-08-04, 28b7c21 + e868a71)**:
- #27: select_nodes에 tip 의무 lane (tip_idx/root_remaining 옵션 인자
  — 잔여예산 root의 tip은 top-W 탈락 불가; backbone 정책 R>W는
  rollout ValueError + config `root_count<=p2_budget` 검증). 재현
  픽스처 [7,7,7,7,6,6]·W10·F4·C3: 34/40·약root dmax=1 → **40/40·
  전root dmax=4**. pool.alloc_stats(allocated/generated/per-root
  dmax) 상시 부착, SSD_TREE_ALLOC_CHECK=1이면 불일치 로그.
- #28+#34: `tree_policy_b_ladder` — verify 보행의 기각 갱신
  (R←norm((R−D)+), D[t]=0 renorm)을 그대로 미러하는 조건부 α̂·
  종단질량·잔차 **일괄 텐서 계산** (.item()/.cpu() 0회). verifier
  `_compute_and_send_proxy_tree` 전면 교체 — residual도 사다리 최종
  R (보행 recovery 원천과 동일 분포). MC-vs-walk 3만회 게이트: 전
  ctx 종단빈도 오차 <1.2%. terminal_mass_dp는 조건부-α 규약 명시
  후 분석 도구 전용 강등.
- #33: alloc_root_budgets — active-set 비례 water-filling (포화 root
  제외 후 남은 예산을 남은 가중치 비율로 재계산, ≤R회 수렴; 소진
  보장 유지). [0.9,.09,.01]·total16·cap8: [8,5,3]→[8,7,1].
- #31: 트리-hit step P1 fork 배분 — 균등(W1//n_rows) → 종단질량
  prior ∝ a^depth·(1−a)^자식수 (a=0.52), 바닥 1 lane, largest-
  remainder (합=W1 — CG 폭 불변).
- #35: b6-2 staging 소비에 seq 정체 대조 (`_tree_served_seq` — 다른
  seq면 staged 폐기); `Sequence.tree_terminal_node` 정식 필드
  (_ATTRIBUTES 포함); `scheduler.preempt`에서 소거. _tree_hit_root/
  _tree_wire_*는 hit_cache_and_respond 첫머리 매-스텝 리셋으로 이미
  step-국소임을 확인 (구멍 없음).
- 유닛 58/58 (신규 8: 비례배분·W경합 tip·R>W·사다리 반례·체인퇴화
  패리티·질량보존·MC골드·GPU패리티).

**수정 후 E2E 스모크 (eslab18 유휴, 8×256 + 4×192, 8455b1e)**:
EXIT:0 무크래시, #27 불변 위반 0건 (전 rollout generated==allocated).
P2AL **2.13** (구-코드 verdict 2.06), P2 hit **0.230** (구 0.217) —
hit·AL 동시 개선 방향 (소표본 — 판정은 reverdict로). exit_logits
스팬(hit_k2): 17.6ms(파이썬 구현) → 9.4(1차 GPU화) → **6.3ms**
(dense-padded 단일-H2D; 8455b1e). 잔여 ~6ms는 사다리 커널 dispatch
— 다음 지렛대는 rollout 정적 템플릿 (리뷰2 설계 방향, 별도 트랙).

**AR 기준선 교정 (T5 AR 팔 크래시 수정 후, eslab17 클린박스)**:
c1_ar = **33.48 tok/s** (70B AWQ TP4 확인). 종전 "6.7×"는 오염박스
(61/64 core 점유) 비율 — 클린박스 기준 체인 78.6은 **~2.35×** vs
AR, async-SD C(78.2)와는 parity. c2/c3 완료 후 3-cycle 확정.

## 외부 리뷰 3차 검토 (2026-08-04) — 전수 재계산 판정 (무조건 수용 금지 지침)

raw 프로파일(405794e 쌍)·반례 런·수계산으로 주장별 확정/기각:

| 주장 | 판정 | 근거 |
|---|---|---|
| topology는 budget-only가 아님 (raw_q-적응) | **확정** | 실형상(W10·R6·C3) q 열회전 → 40중 18 노드 재배치. 단 리뷰의 최소반례 수치 자체는 미재현 (해당 파라미터에선 rescue avail=0) — 원리 성립, 예시 부정확 |
| 스텝격차: 동일가중 +25.8 / 실비중 +23.5 | 확정 | 재계산 일치 (첫 20스텝 제외 p50: +18.9/+38.2/+20.2) |
| P2 4-replay GPU 합은 동일, 전체 GPU는 비동일 | 확정 | replay 9.44 vs 9.96; target graph_pre +7.37·exit+send +5.85·graph_post +2.26 |
| core-입증 회수 ≈8.8ms (16→3 과대) | 확정 | P2 창 12.35→21.49 (idle 0.27→9.21) — Δ9.1; build→merge 전체 Δ21.96은 stretch 상한 |
| P1build+proxy_wait 이중계상 | 확정 | critical path = max(draft P1 ready, target proxy ready); target측 +15ms가 게이트 |
| TPS 산술 (65-72·−7~15% 오류) | 확정 | 17-20 회수 시 67.0-70.5 (−12.4~−16.8%); parity bar ≈26ms+AL2% |
| "Nv8만 이득" 미입증 | 확정 | Nv는 root cap 겸용 — W10/Nv4 requested 40 중 allocated 24 (이용률 60%) 교란; nv6@W10 미측정 (sweep 중단). **nv8 우위 자체(재판정 2.12)는 유지** |
| P1 prior에 앞형제-기각 인자 누락 | 확정(모델) → **원복(실측)** | 수학상 둘째 형제 2.08× 과대는 맞으나, 동일-시드 A/B(8×256)에서 presib 보정판이 P1AL 4.13→3.88 회귀 — 실제 형제 조건부 수락은 λ-할인(18번)으로 고정-A 예측보다 높아 미보정형이 보상. #31 형태 유지, calibrated prior는 T6 부채 |
| Policy-B가 temp/sampler_x 미적용 | 확정(모델) → **원복(실측)** | 미러판이 hit +0.011에 P2AL 2.13→1.94 (tok/step 4.55→4.36 순손실) — 날카로운 p^E가 wire 후보를 얕은 ctx로 집중. plain softmax(체인 일관)가 경험적 동작점; 재도전은 P_iv 랭킹·β·prior 공동 recalibration으로 (T6 부채). temp0 명시 게이트는 유지 |
| WOR support 소진 → D[t]=0 fail | 확정 (위험) | #38 수정 (raw_q≤0 자식 배제 — 기존 sync 편승) |
| mask "포인터 교체" 불가 | 확정 | captured _custom_mask_buf in-place copy 필요; plan-once 재도전 금지 (#20 교훈 유지), plan-ahead(기하 불변)만 |
| chainR6=budget6는 CG폭 변경 | 부분수용 | 토큰축(P2AL 귀속) 결론은 유효 (사석행은 hit 불가 — 등가); 시간축 비교엔 W10-top6 knob 필요 (후속) |
| SHM read-ACK / epoch 상수 / assert -O 소거 | 확정 (부채) | assert 3곳 경화(#40); epoch·ACK는 T6 correctness 배치 |

**이번 배치 최종 상태**: #38 WOR support 가드, #39 requested 3값,
#40 assert 경화, #41 픽스처 정합, temp0 게이트 — 채택 (62/62
green). #36·#37은 **이분 탐색 A/B로 원복** (위 표 — "모델상 옳음 ≠
실측 개선"의 교훈; #20 plan-once와 동일 패턴: 라이브 경로 변경은
반드시 동일-시드 A/B 게이트).

**T6 방향 수정 (리뷰 권고 수용)**: budget-only 정적 템플릿은 현
정책의 drop-in이 아니라 **신규 fixed-topology 정책** — P2AL 보존
보장 없음. 1차 구현은 **T6-dynamic: 고정 크기 GPU arena + 동적
GPU select/fanout/WOR/pool** (현 raw_q-적응 의미 보존, 중간 CPU
readback 0회; per-forward plan 유지 채 parity 먼저) → view/wire
GPU화 → plan-ahead·TREE_GLUE·P1 mask 독립 A/B. **T6-static은 별도
정책 arm**으로 compute-matched P2AL 보존 검증 후에만 채택. 성능
판정은 component 합산이 아니라 3연속 구간 wall (target proxy gate /
P2 prep→last replay / gate→merge).

**v1 근사/후속 목록 (T4 전 확정 사항)**:
- P1 컨텍스트별 fanout = 균등-우선 (F7 예산 설계 대기).
- 트리-step Policy B ĥ = 체인 수식의 노드-축 재해석 (맏이-정확;
  사다리 DP 일반화 후속).
- 보행 CPU 실행 (p/q ~2.3MB 복사/step; GPU화 후속).
- 트리 verify eager (CG capture 후속 — bucket shape은 이미 고정).
- parent_q wire fp16 반올림 → 수락식 q가 샘플측과 ~1e-3 상대 오차
  (기존 체인 logits_q wire와 동일한 관례 — 신규 gap 아님).
