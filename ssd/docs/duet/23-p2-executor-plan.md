# 23 — P2 연속 실행 구조 (리뷰7 지시문 — 표준 계획)

2026-08-05. 리뷰7 판정 수용: 22번의 "T6 트랙 완결"은 **GPU 장부
개선 한 사이클의 완결**이지 P2 시간 문제의 근본 해결이 아니다 —
현재도 트리는 체인보다 ~21% 느리고, forward 사이 ~6.6ms와
GPU→CPU 포장 경로(to_pool→python view)가 남아 있다.

## 단계 0 — 동결 (지금부터 적용)

- **sweep 전면 중단**: R/Nv/β, level/frontier, F_total>K2, exit 재탐색,
  target hit_k2 최적화 — 아래 단계 완료 전 금지.
- **canonical 고정: W10 / R6 / Nv8 / F4 / C3 / level.**
- 성공 기준 (P2 executor):
  - 네 forward 사이 CPU 대기·host sync·`.cpu()/.item()/.tolist()`·
    파이썬 트리 갱신 **0회**
  - GPU idle 공백 체인 수준 (필수 C=3 샘플링 GPU 시간은 별도 계측
    — 0이 될 수 없는 계산)
  - 트리 포장(응답/wire/cache key)까지 GPU에서 완료
  - P2AL·tok/step이 현 동적 트리 대비 유의 악화 없음

## 단계 1 — R=6 / W=10 / overfetch 분리

현행: 28개 송신 → dedup 후 상위 10 선택 → 하위 4 piv=0 → 키 무효화.
변경: **24개 고정 송신 (R6 + dedup 여유 18) → dedup 후 6개만 선택 →
W10 실행 버퍼에 6 root 배치, 나머지 4행은 고정 padding** (root/키
생성 안 함 — 무효화 경로 소멸).

동등성 게이트 (단독 성능 캠페인 아님 — executor 입력 정리):
선택 위치·토큰·P_iv·hit 가능 cache key·P2AL이 기존 top-6와 일치.

**단계 1 최종 판정 (리뷰10 표현으로 정정)**: "토큰축 무회귀 통과,
TPS는 정규화 −2.5% 미확정이나 진행 허용" — 게이트는 8ce10ba 기준
(P2AL 2.10·hit 0.226·tok/step 4.44; 집계 지표 기준 무회귀이지
요청별 선택 동일성의 증명은 아님). 게이트 이후 변경 2건 처리:
① selector 고정-shape — exact 단위 테스트로 고정 (boolean 참조
구현과 random dedup/최대 dedup 경계/동률·0토큰/B>1 행독립 전부
동일; a35cb72 + 테스트 커밋). ② **top_k R기준 변경은 원복** —
verifier가 top_k 절단 후 재정규화하므로 14→12는 P_iv 값·순위를
바꾸는 정책 변경 (동결 위반; wire_N=24·선택 R=6만 유지, top_k는
W기준 14). top_k 축소는 추후 별도 정책 arm. __debug__ .item()
가드는 -O 밖 실행에서 sync — 전체-graph 경로에선 GPU 오류 플래그
또는 graph-밖 진단으로 이전 예정.

**[v3 상태 정정 — 리뷰9]** 완료: 송신 24 · 선택 6 · rollout root 6 ·
selector 고정-shape(boolean indexing 제거) · top_k 자동보정 R 기준 ·
R≤W 검사 무조건화. **미완**: cache/view/key 행은 아직 10 (pad 4키
populate 후 #14 무효화 유지 — layout 계약 보존을 위한 과도기),
"키 생성 안 함/무효화 소멸"은 목표 상태이지 현 상태가 아님. 최종
분리 목표: root_token/pos/piv/rope [6] 입력, W=10은 forward 전용,
cache root 행 R=6 — **insertion-시점 root-local index 부여로
to_pool/build_root_views 자체를 제거하는 단계 5와 함께 완결.**

## [v2 개정 — 리뷰8] 주력 = 동적-내용·고정-틀 P2 전체 CUDA graph

리뷰8 (SGLang EAGLE 현행 구현 확인): 동적 트리를 포기하지 않고도
여러 draft forward + 후보 선택을 **하나의 CUDA graph**에 캡처 가능
— 고정해야 하는 것은 반복 횟수·최대 폭·shape·주소·연산 순서뿐이고,
토큰·점수·부모·mask 내용은 실행마다 달라도 된다. 따라서 **정적
트리는 주력이 아니라 보조**(latency 하한 측정·capture PoC·동적
이득 대조)로 강등하고, 주력은:

1. **P2 전용 CUDA graph 실행기**: [round1 forward → 샘플 → 선택/
   배분 → mask 갱신 → round2 ... → round4 → 최종 출력] 전체 캡처.
   기존 model graph의 replay를 감싸지 않는다 (리뷰5 확증: nested
   불가) — 실행기 안에서 **raw draft forward**를 호출해 통째 캡처.
2. **트리 갱신 kernel 통합 (v3 범위 정정 — 리뷰9-7)**: 샘플링
   (WOR softmax/top-k/RNG)은 **kernel에 합치지 않는다** — 기존 GPU
   연산 그대로 graph에 포함 (RNG 순서·verifier q 일치 보존; 병목
   확인 후에만 별도). 합치는 것은 장부만: 커널1 = 선택+fanout+다음
   입력/rope, 커널2 = 자식 삽입+parent/root/local index+다음 mask.
3. **round별 attention 사전 준비 (v3 정밀화 — 리뷰9-3/4/5)**:
   - plan-ahead의 이득 조건: 단순 전진이 아니라 **proxy_wait/P1과
     겹칠 때만** wall 감소 (page/slot 구조는 proxy 도착 전 기지 —
     P1 후·wait 진입 전이 준비 지점).
   - wrapper 상태는 _plan_info 하나가 아니라 int workspace/qo·KV
     indptr/page indices/last-page len/mask buf 전부 — round별
     분리. **캡처된 graph는 캡처 시점 buffer 주소를 기억** — 파이썬
     wrapper 교체는 무효; 주력은 전체-P2 raw-forward graph 안에서
     wrapper 4개 사용. float workspace는 공유 가능성 검토.
   - **선행 검증 구현 (완성 아님) — 통과 기준 (리뷰10-6)**:
     ⓐ round별 독립 metadata (int workspace/qo·KV indptr/page
     indices/last-page len/KV len/mask buf·indptr/plan launch 정보);
     ⓑ page 경계: last-page len ∈ {1, bs−1, bs} + round 중 새 page
     진입 + page ID 변경 케이스; ⓒ page-end canvas: 실KV 밖 슬롯에
     고의로 stale/큰 KV 기록 + mask=0 → 정확-길이 결과와 허용오차
     일치; ⓓ preplanned vs 현행: attention 출력·최종 logits·KV
     write·4-round 최종 topology 비교; ⓔ 시간: P2 직전 배치 vs
     P1/proxy_wait 겹침 — **전체 step wall + GPU timeline**으로
     판정 (proxy_wait span 축소만은 증거 아님 — 계측 구간 밖 이동
     착시); ⓕ 메모리: wrapper 4개 독립 생성 후, 공유 가능한 float
     workspace vs round별 보존 필요한 plan metadata/int workspace
     분리 평가.
   - replay 중 plan()/sync/신규할당/.cpu()/.item()/nonzero/파이썬
     분기 0회.
4. **최종 출력 (v3 의미 정정 — 리뷰9-8)**: P2 시점엔 hit root를
   모르므로 "단일 wire 완성"이 아니라 **root별 [R,Nv] 고정 출력**
   (tok/parent_local/sib/raw_q/pq_ref/valid) + root별 사전 포장
   wire block — hit 확정 후 해당 행만 gather. 최적형: 삽입 시점에
   root-local index를 부여해 [R,Nv]에 직접 기록 (to_pool·
   build_root_views 소멸).

캡처 shape는 요청 context 길이에 의존 — "page bucket"의 정확한
정의 (v3, 리뷰9-5): **mask canvas를 page 끝까지 고정 폭으로 잡고
실길이 밖은 항상 0** (예: kv 173 → canvas 176, 173..175 = 0).
같은 page-수 버킷 안에서 shape 불변 → 버킷당 1 graph. glue 폭
(K_rank+1)·backend/dtype도 capture key. **canvas-패딩이 FlashInfer
attention에서 정확함을 먼저 검증** (전제 미검증 상태로 캡처 금지).
eager는 진단 A/B 전용 (arena v1 실측: eager 커널 비용 > 파이썬).

## 단계 2(보조) — 고정 트리 1개로 latency 하한 확인

- 기존 **동적 실행 로그에서 대표 부모·자식 구조 1개** 추출 (root
  순위별 평균 예산·깊이별 선택 빈도·형제 기여 위치 — 오프라인 설계,
  sweep 아님).
- round별 부모 인덱스/예산/슬롯/깊이/형제/조상 mask/최종 view 배치
  전부 사전 계산. round별 독립 입출력 버퍼.
- **round별 CUDA graph**: [draft forward → C=3 샘플 → 다음 round
  token 버퍼 기록] ×4 — CPU는 completion wait 없이 같은 stream에
  연속 enqueue (GPU stream이 데이터 의존성 보장).
- attention 준비(plan)는 이 구조의 일부로: round별 wrapper/graph
  분리 또는 plan 템플릿 D2D 복사. **plan-once(최대 길이 재사용)
  금지** — 실제 context length/KV page/마지막 page 길이/mask 길이
  유지 + capture-vs-runtime plan 정보 일치 확인 + 불일치 fallback.
- 목적: "트리 구조가 느린가, 동적 파이썬/PyTorch 구현이 느린가"에
  가장 빨리 답하는 기준 구현.

## 단계 3(보조) — 3-arm 판단 (sweep 금지)

chain / 현 동적 tree / 고정 tree 1개. 고정이 이득 대부분 유지 →
채택. 체인 수준으로 AL 하락 → 정적은 최종안 아님 (round-graph
틀만 재사용). 템플릿 추가는 이때도 금지 (동적 통합으로 이동).

## 단계 4 — 동적 정책의 kernel 통합 (v2: 주력 트랙의 2단계로 승격)

select/fanout/다음 입력/mask/장부를 round당 1~2개 Triton/CUDA
kernel로 — round graph에 포함. 동적 AL 유지 + CPU 개입 0 + 소형
커널 제거. 장기 상한은 이쪽이 최고.

## 단계 5 — GPU에서 최종 응답까지

`to_pool()`·python `build_root_views()` 제거 — 유효 노드 정리/부모
변환/root별 Nv 배치/wire 정수 블록/cache key/hit-root parent-q까지
GPU 직접 생성 (정적이면 대부분 사전 계산).

## 단계 6 — 그 후에만

target hit_k2 (tree attention 준비·exit proxy·verify row·mask) →
파라미터 재탐색 재개.

## 주의 (리뷰7 §4)

- P2AL +15%는 hit step(~22%)에만 적용 — 전체 tok/step은 +3~4.4%.
  P2 시간이 체인과 같아져도 TPS 이득은 그 수준.
- target도 트리에서 더 많은 노드를 검증 (hit_k2 +15ms, 평균
  +2~3ms) — draft 공백 제거 후 남는 몫.
- 체인 측정 근거: 체인 CG도 model+logits만 캡처 — 빠른 이유는
  "전부 캡처"가 아니라 **readback·가변 mask·plan 없는 연속 비동기
  enqueue** (리뷰5). round-graph는 여기에 캡처를 더하는 것.


## PoC 결과 (2026-08-05 — 캡처 실증)

**전체-P2 단일 CUDA graph 캡처 성공** (기존 arena 텐서 연산 그대로,
kernel 융합 전): [reset → GPU 예산 → (select → fanout → mask 기록 →
FlashInfer attention(round별 preplanned wrapper) → WOR 샘플 → 자식
삽입) ×4]. 검증: 재실행 무오류·**replay마다 동적 트리** (RNG 전진 —
동적 정책 보존의 캡처 증명)·토폴로지 불변량 유지.
**시간: eager 4-round 13.92ms → captured replay 1.87ms (×7.4)** —
host 공백·런치 비용의 붕괴 (PoC 스케일; 실모델은 forward가 지배).

발견된 캡처 차단 패턴 (프로덕션 실행기 체크리스트):
1. `torch.tensor(스칼라, device)` = pageable H2D → `torch.full()`
   (alloc_root_budgets_gpu 프로덕션 수정 완료)
2. advanced-index 대입의 파이썬-스칼라 RHS → 사전 할당 텐서 RHS
3. 캡처 그래프 생존 중 기본 CUDA RNG는 graph-모드 — eager RNG와
   교차 사용 금지 (그래프 파기 후 manual_seed 재설정 규약)

**갭 폐쇄 (동일 날)**: PoC의 mask 기록은 bool 바이트였음 (커널은
packed bit 해석 — 토폴로지 검증엔 무관했으나 의미론 미검증) →
in-graph GPU packbits→_custom_mask_buf 기록이 JIT-plan 참조와 정확
일치함을 별도 테스트로 확정 (buffer zero 후 replay 재생성 포함).

~~전제 검증 전체 완료~~ **[리뷰11 정정]: "핵심 capture 가능성 검증
완료 — 프로덕션 blocker 4종 잔존"이 정확한 표현.** blocker 처리:

1. **전용 graph RNG — 해소**: teardown 규약(파기+reseed)은 프로덕션
   불가 (graph 상주 + eager 교차 필요). 해법: P2 전용
   torch.Generator(cuda) + CUDAGraph.register_generator_state +
   tree_sample_wor(generator=) — eager↔graphA↔graphB 교차·전진·
   무오염 테스트 통과.
2. **버킷 정의 — 확정**: 버킷 수는 config.max_blocks 유도 (8 하드
   코딩 금지). round간 page 전환은 **p+1 고정 canvas** (예비 1 page
   전체 mask=0; F4·W10 총확장 40 ≤ PAGE=256이라 최대 1 page 추가)
   — 전체-예비-page 오염 + 비연속 page ID 검증 통과 (기존 arange
   한계 지적 해소). glue 폭 2종(K1+1/K2+1)은 capture key 또는
   최대-폭 canvas 통합 — 실행기에서 결정.
3. **메모리 — 해소**: 실측 wrapper당 auto 72.2MB (vector sparse)
   vs **fa2 명시 8.1MB** → 8버킷×4 = 2.26GB → **0.25GB**. 실행기는
   backend="fa2" 명시 (+pinned 8MB/wrapper 별도 계상).
4. **실모델 KV 경로 — 미해소 (다음 블록)**: PoC는 K/V를 attention
   '후' 기록 (self-slot stale — 캡처 역학 검증용). 실 transformer는
   현재 토큰 KV 기록 후 self 포함 attention. **단일 버킷 실모델
   PoC** (엔진 배선 前): layer별 KV slot 기록/판독, round간 KV 의존,
   slot/page 교체 replay, fallback↔graph 캐시 무오염 + **결정적
   debug parity** (고정 noise buffer 주입으로 eager==graph 전항목
   exact 비교 — budget/sel/fanout/rope/mask byte/logits/token/
   장부/[R,Nv]/KV) → 통과 후에만 전 버킷 + SSD_TREE_EXEC 배선.

설계 단순화 (리뷰11-6): plan은 **init/capture 시 버킷별 1회, runtime
plan 0회** — "proxy_wait 겹침" 요구는 이 설계에선 소멸 (문서상 두
설계 병존 금지; overlap 검증 불요).


## 리뷰12 연속-실행 지침 채택 (2026-08-05) + 단계 1 완성

**지침 요지**: 중간 허가 요청 없이 구현→검증→실험→채택 판정까지
연속 진행. 고정: W10/R6/Nv8/F4/C3/level/wire24/seed6/top_k14.
중단 조건 4가지(수락 규약 변경 필요·캡처 불가 증거·공유 후에도
OOM·반복 캐시 오염)에서만 질문. 채택 기준: 결정적 parity 전통과·
4-forward 사이 sync/readback/plan 0·미설명 idle ≤ chain+1ms 또는
arena 20% 이하·P2 시간 3/3 단축·TPS 3/3 우세(95% CI>0)·P2AL 하락
≤0.05·tok/step 하락 ≤0.03·hit 하락 ≤1%p. 비교 대상은 **arena**
(chain 동률은 요구 아님 — target 검증 비용은 별도 문제).

**단계 1 완성 (프로덕션 조건 — "3종 해소 과장" 정정 후 재검증)**:
- RNG: 기본 수열 state-복원 기준 **정확 보존** (e1≠e2 수준 아님) +
  동일-seed 재구축 replay 수열 재현 ✓.
- 버킷: **결정 실험 통과** — fa2·use_cuda_graph·PAGE=256 실치수,
  plan 캡처-전 1회, replay 사이 page-ID 버퍼 A→B→A 교체 = 매번
  fresh-plan eager 일치 + runtime plan 0회 계수 → **indices는
  런타임에 버퍼에서 읽힘 = runtime-plan-0 설계 성립**.
- glue 폭: 최대-폭 canvas 0-mask == 좁은 glue 정확 → **버킷 키 =
  page-count 단독** (wrapper 수 ×2 회피).
- 메모리: 실측은 wrapper 생성분 기준 (fa2 8.1MB) — **전체 상주
  측정(모델+graph+pinned)은 단계 2 실모델 실행기에서** (0.25GB는
  wrapper 추정치로만 표기).

**다음 (연속)**: 단계 2 — 실모델 단일 버킷 실행기
(p2_tree_executor.py; raw self.model+compute_logits 직접 캡처,
실 KV 순서: KV 기록→self 포함 attention; round별 wrapper·set_context
캡처-시 1회; [R,Nv] 직접 기록) → 단계 3 결정적 parity (고정 noise
주입) → 단계 4 전버킷+SSD_TREE_EXEC → 단계 5 성능 (마이크로→스모크
→타임라인→eslab17 3-arm 회전 인터리브) → 단계 6 채택 판정.


## 단계 2 진입 노트 (실모델 실행기 구현 사실 — 코드 확인)

- **KV 순서 호재**: `layers/attention.py` forward가 이미
  `store_kvcache(k,v,slot_mapping) → attention` 순서 — 리뷰12가
  요구한 실순서와 동일. PoC의 후-기록 문제는 실모델 캡처에선
  자동 해소 (mask의 self-cell도 방금 기록된 KV를 읽음).
- **context 소비 방식**: attention은 get_context()를 trace 시점에
  읽음 — 캡처 중 set_context는 라운드당 1회(캡처 시)만 실행되고,
  slot_mapping/wrapper 버퍼는 **주소가 박히고 내용은 replay 가변**
  (page-ID 교체 실험과 동일 원리) → 라운드별 고정 버퍼 설계 그대로.
- 실행기 모듈: `ssd/ssd/engine/helpers/p2_tree_executor.py` (신규) —
  raw `model(input_ids, positions)` + `compute_logits` 직접 캡처
  (run_model/run_fi_tree_decode_cudagraph 우회; 기존 graph 재생 금지).
- 지원 범위: 비-EAGLE Llama draft만 1차 (그 외 arena fallback).


## 단계 2 진행 (2026-08-05 연속) — 실행기 모듈 + 모듈 parity 통과

- `p2_tree_executor.py` v1: raw forward 캡처 구조, round별 fa2
  wrapper(plan 1회), 전용 RNG, **[R,Nv] 삽입-시점 직접 기록**
  (root-local index — to_pool/build_root_views 소멸 경로),
  **mask 열 전부 버퍼-구동** (python-int 슬라이싱의 캡처-박힘 결함
  자체 발견·수정 — prefix 경계는 요청별 '내용').
- parity-noise 모드 (리뷰12 §3): tree_sample_wor(noise=) — 고정
  exponential noise를 eager/graph 동일 주입.
- **모듈 결정적 parity 통과**: 동일 입력 버퍼·동일 noise에서
  eager run_once == captured replay — [R,Nv] 정수 전항목 exact,
  logits/raw_q/KV allclose, 불변량 유지. 미니모델은 실 attention
  계약(KV 기록→attention, get_context의 slot/wrapper 소비)을 미러.
- 남은 것: draft_runner 배선 (SSD_TREE_EXEC=1 + 미지원 fallback +
  계수), 실모델 스모크, arena-vs-executor 의미 parity (동일 noise),
  전버킷 lazy capture, 3-arm 인터리브 → 채택 판정.

## 판별 parity 결론 (2026-08-05) — 기록기 버그 1건 수정 + 게이트 재정의

배경: 실모델 스모크 v5에서 실행기 경로 fallback 0회 완주했으나
P2AL 1.33 붕괴 → 같은 미니모델·같은 noise로 arena(JIT-plan 경유
fwd) vs 실행기(preplanned fa2)를 비교하는 판별 테스트 구축
(`TestExecutorVsArenaSemantics`).

### 발견 1 — 기록기 중복-scatter 충돌 (실버그, 수정 완료)
[R,Nv] 직접 기록에서 비활성 lane의 목적지를 dst=0으로 뭉개면
같은 replay 안에서 slot 0에 다수 lane이 scatter되어 승자-미정
(키메라 레코드, tok[0]=0). **수정: flat R·Nv+1 버퍼 + 더미슬롯
R·Nv 라우팅** (`dst_safe = where(wmask, dst, R*Nv)`), 소비자는
`view_*` [R,Nv] 뷰 사용. v5의 P2AL 붕괴 주범으로 추정 — v6
스모크로 확인.

### 발견 2 — 커널 비결정성에 의한 트리 분기 (버그 아님, 원리적)
round별 logits는 두 경로가 fp16 허용오차 내 일치 (max ~2e-3;
mask/slot/rope 정합 증거). 그러나 참조(auto backend JIT-plan)와
실행기(fa2 preplanned)는 **커널이 달라 bit-동일이 아니고**,
~1e-3 logits 차가 WOR raw_q의 근접-동률 priority를 뒤집어
f=1부터 rescue 순서(8,7 vs 7,8)·fanout 배분이 갈림 — 토폴로지
분기는 18/40 민감성(기실측)과 동일 기전. CPU/GPU 예산은 동일
확인 ([8,8,8,8,5,3]). **결론: 커널이 다른 두 경로 간 결정적
트리-동일성은 원리적으로 비보장.**

### 게이트 재정의 (판별 테스트 최종형, 통과)
- ① logits-경로 검증: round별 exec vs ref logits fp16 허용오차
  (mask/slot/rope/KV 순서의 정합 게이트).
- ② 기록기 게이트: 실행기 [R,Nv] 직접 기록 == **자기 arena**의
  build_root_views — 같은 트리의 두 서술이므로 커널 비결정성과
  무관하게 exact 요구. (valid/tok/par/sib exact, raw_q 1e-5.)
- arena-vs-exec 의미 판정은 분포 지표(인터리브 AL/hit)로 —
  리뷰12 §6 채택 기준 그대로 (P2AL 하락 ≤0.05 등).

### 실모델 스모크 v6/v7 (2026-08-05, eslab18 상대비교 — 기록기 수정 후)
동일 명령(70B AWQ TP4 + 1B, E9K24 exit56, W10/Nv8/level/β0.5,
numseqs 10, temp 0.7 seed 42) A/B:
- **실행기 실가동 확정**: p2exec stats `{capture: 2, replay: 1550}`,
  fallback 0회.
- **P2AL 회복**: arena 1.74 vs exec 1.72 — Δ0.02 (기준 ≤0.05 통과).
  v5의 1.33 붕괴 = 기록기 중복-scatter 충돌이 원인이었음 확정.
- Decode TPS: exec 52.58 vs arena 50.88 (+3.3%) — 소표본·옆 GPU
  부하 있는 상대 참고치. 절대 판정은 eslab17 3-arm에서.
- tok/step(회복 포함) 3.74 vs 3.36, P1 hit 0.508 vs 0.471 —
  소표본 변동 범위; P2 지표(AL/AR/hit)는 사실상 동일.
