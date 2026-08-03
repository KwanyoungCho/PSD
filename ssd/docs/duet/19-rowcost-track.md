# 19 — verify 행당-비용 엔지니어링 트랙 (G0 후속, 사용자 승인 2026-08-03)

**동기** (18번 Step 6): 트리의 AL 이득(+3.5~3.8%)은 실재하나 verify
행당 비용이 병목 — 현 엔진 실측 **2.36ms/행** vs 계산 필연 **~0.1ms/행**.
행당 ~0.5ms 달성 시 트리 +2.1~2.6%, 그리고 **트리 없이도 현행 체인
(P1 10행 verify)이 빨라지는 독립 가치**. 미해결 graph_pre 이상
(+19%/layer)과 동근원 가능성.

**행당 2.36ms의 분해** (짝런 span, hit_k1(10행) vs hit_k2(5행) 차):

| 성분 | Δ (5행) | 행당 | per-layer/행 |
|---|---:|---:|---:|
| graph_pre (층 0-55) | +6.25ms | 1.25ms | 22.3µs |
| graph_post (층 56-79) | +2.58ms | 0.52ms | 21.5µs |
| proxy 경로 (exit_logits + policy B) | +2.99ms | 0.60ms | — |
| **합** | **+11.82ms** | **2.36ms** | |

주목: pre/post의 **per-layer 행당 비용이 ~22µs로 균일** — 즉 "레이어
안의 무언가"가 행당 22µs를 먹는다. standalone fp16 GEMM+sdpa는
~1.3µs/layer/행 → **범인은 셋 중 하나**: ① Marlin(AWQ) 커널의 소형-M
스케일링 (fp16 mm과 다를 수 있음), ② flash_attn_with_kvcache의 q_len
스케일링, ③ TP4 PCIe allreduce의 payload 스케일링 (층당 2회).

## Phase A — 진단 (standalone, 엔진 무수정)

- A1: Marlin gptq_marlin_gemm M-sweep — **실제 70B rank-shard artifact**
  로드 (ssd.quant 공개 API), M ∈ {1..16}. (진행)
- A2: flash_attn_with_kvcache q_len sweep (엔진 실제 커널·shape).
- A3: 4-GPU NCCL allreduce [M, 8192] fp16 M-sweep (PCIe 실측).
- 판정: 세 실측의 합이 22µs/layer/행을 설명하는가 → 범인 확정.

## Phase B — 개선안 (Phase A 후 설계, 사용자 보고 후 구현)

후보 (범인에 따라): Marlin 소형-M 특성이면 M-패딩 전략/커널 파라미터,
allreduce면 통신-계산 겹침/양자화-allreduce, proxy 경로면 연산 융합.
graph_pre 이상(+19%)과의 관계도 이 진단에서 판별.

## 진행 로그

**A1 — Marlin 소형-M (실제 70B rank0 artifact, 층0의 4개 선형)**:
M 5→13 한계비용 **+0.1 µs/layer/행** — 완전 평평 (M=1만 gate_up 커널
전환으로 2배 비쌈 — M≥2에서 소멸). **무혐의.**

**A3 — 4-GPU PCIe allreduce [M,8192] fp16**: M 5→13 **+2.1 µs/layer/행**
(런-투-런 지터 ±20µs로 비단조 — 상한으로도 소폭). 절대 지연
~220-310µs/회는 standalone 런치 오버헤드 포함 — 엔진 CG 내부와 다른
값이라 참고만. **주범 아님.**

**A2 — flash_attn_with_kvcache (sgl_kernel, 엔진 실제 커널)**: q_len
5→13 **−2.4 µs/layer/행** (노이즈). **무혐의.**

**Phase A 중간 판정**: 표준 성분(GEMM·attention·allreduce)의 행당
한계비용 합은 **수 µs/layer/행** — 엔진 실측 22µs/layer/행을 설명하지
못한다. → 범인은 엔진 구조 내부 (CG 캡처된 커널의 실제 구성/그리드,
버킷별 차이). **다음: nsys `--cuda-graph-trace=node`로 실제 그래프의
커널-수준 diff** (10행 vs 5행 버킷을 grid 크기로 구분해 어느 커널이
행당 얼마를 먹는지 직접 측정 — 코드 무수정, 진행 중).

**A4 — 판별 실험: 8B TP1 (GPU 0-1, 같은 K1=9/K2=4 구조; 70B nsys는
타 사용자 GPU 점유로 대기)**: hit_k1(10행) vs hit_k2(5행) —
graph_pre Δ+0.22ms / graph_post Δ+0.10ms → **행당 0.065ms =
2.1µs/layer/행** (standalone 성분 합과 일치 — TP1에서는 행이 사실상
공짜).

**Phase A 판정: 범인은 TP 경로다.** 같은 엔진·같은 K 구조에서 TP1은
2.1µs, TP4는 22µs/layer/행. allreduce의 payload 스케일링은 +2µs뿐
(A3)이므로, 유력 기전은 **in-graph NCCL 동기화의 straggler 누적**
(층당 2회 × 80층의 rank 동기화 지점에서 최슬로우 rank 대기; M이
커질수록 rank 간 커널시간 편차 증가) — 미해결 graph_pre +19% 이상
(DUET의 pre/post 그래프 분할 = 동기화 지점 증가)과 같은 계열로 보임.
확증: A5(8B TP2 — TP를 켜는 순간 행당 비용이 뛰는지) 진행 중; 70B
nsys 커널-diff는 GPU 확보 시.

**A5 — 8B TP2 (GPU 0,1,4)**: graph_pre Δ+1.15ms / graph_post Δ**−1.87ms**
(음수!) — TP≥2에서는 span에 **rank 간 대기가 혼입**되어 폭 효과가 분리
되지 않는다 (음수 marginal은 물리적 비용일 수 없음 — 동기화 잡음이
지배한다는 방증). **span 계측의 한계 확인 → 커널-수준(nsys) 계측이
필수** (진행 중, GPU 확보됨).

**A6 — nsys 커널-diff (70B TP4, --cuda-graph-trace=node): 범인 확정.**
10행/5행 버킷(분류: silu grid 280 vs 560 = M 5 vs 10; 창 26/13)의
커널별 diff — **`ncclDevKernel_AllReduce_Sum_bf16_TREE_LL` 가 차이의
사실상 전부** (+239ms/replay vs 2위 Marlin +2.0ms; nsys 노드-추적
왜곡으로 절대값은 5-8× 부풀려짐 — 구조 판정용). NCCL 커널 시간 =
rank 대기이므로: **행당 22µs/layer의 정체 = 층당 2회×80층 = 160개
동기화 지점에서의 rank-대기가 M과 함께 증가** (PCIe TP4, TREE_LL
프로토콜 선택 관측). 8B TP1 2.1µs(대기 없음)·표준성분 무혐의와 완전
정합. graph_pre +19% 이상도 같은 기전(pre/post 분할 = 랑데부 추가)일
개연성.

## Phase B — 개선 후보 (우선순위)

1. **NCCL env 튜닝 (코드 무수정, 즉시 A/B 가능)**: NCCL_ALGO
   (Tree→Ring), NCCL_PROTO (LL→LL128/Simple), NCHANNELS — 대기 구조가
   프로토콜 의존이므로 1순위 시험. (진행)
2. proxy 경로 0.60ms/행 (전체의 25%): Policy B 연산 융합 — 엔진 코드
   최적화 (승인 후).
3. 구조적: 동기화 지점 축소/통신-계산 겹침 — 대공사, 1·2 결과 후 판단.
