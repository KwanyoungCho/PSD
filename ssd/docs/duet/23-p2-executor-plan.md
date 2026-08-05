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

**동등성 게이트 통과 (2026-08-05, eslab17 인터리브 3-cycle, 8ce10ba)**:
tree 60.81 (60.7-61.1) vs chain 78.69 — 토큰축 완전 일치 (P2AL
2.10·hit 0.226·tok/step 4.44 = 전일 2.08/0.223/4.49 밴드), TPS는
전일 스프레드(60.9-64.1) 하단 (−2.2% — 편차 범위, 회귀 근거 없음).
selector 고정-shape판(a35cb72)은 로컬 스모크 통과 (P2AL 2.14).

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
   - **선행 검증 구현 (완성 아님)**: round별 독립 wrapper ×4로
     page-경계 포함 케이스에서 preplanned == 매-round plan 결과
     일치 + proxy_wait 중 준비 시에만 wall 감소함을 확인.
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
