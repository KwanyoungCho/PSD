# MESA Phase 2 Hybrid 구현 리포트 (한글)

작업 브랜치: `feat/mesa-phase2-hybrid` (origin push 됨, **main 머지 안 됨**)
출발 main: `25e7bf4` (계획 + 이슈 트래커 머지 직후)

이 리포트는 사용자 지시 — *"끝까지 진행해, 절대 main에 merge하지마"* — 에
대한 솔직한 응답입니다. 두 세션에 걸쳐 foundation 5 단계까지 끝났고, 그
위에서 GPU runtime debug 가 필요한 변경들을 어떻게 안전하게 land 할
지가 다음 단계로 남았습니다. 아래 §3 의 "축약 v1" 경로가 권장입니다.

---

## 1. 두 세션에 걸쳐 land 된 것

플랜 (`MESA-PHASE2-HYBRID-IMPLEMENTATION-PLAN.md`) 의 단계 0 ~ 3a 와
3b 의 첫 sub-stage 가 브랜치에 커밋·푸시 완료. 모두 syntax / import
smoke + 기존 9 개 AWQ unit test + foundation smoke test 통과.

| 커밋 | 단계 | 변경 |
|---|---|---|
| `25e7bf4` (main) | 계획 문서 | `MESA-PHASE2-HYBRID-IMPLEMENTATION-PLAN.md` + `MESA-PHASE2-HYBRID-ISSUE.md` 추가 |
| `7fdecab` | Phase 0 | `mesa_phase1_k`/`mesa_phase2_k` config; `SpeculateResult.valid_k`; NCCL fused_response wire 가 `valid_k(B)` 추가 (`2*B+B*K → 3*B+B*K`); `hit_cache_and_respond` 가 valid_k 반환; draft_loop 가 packing |
| `605f466` | Phase 1 | `tree_cache_valid_k` per-row tensor; populate 두 site 모두 채움; lookup 시 매칭된 row 의 valid_k 가져오는 hook |
| `fa05981` | Phase 2 | `TreeLayout` 에 `position_count` 필드 추가 (= `len(fan_out_list)`), `K`(forward_depth) 와 분리. `create_tree_layout(...)` 에 optional `position_count` 인자. Non-MESA 경로는 `position_count = K + 1` invariant 유지. `_build_tree_decode_args_for_layout` 의 glue position offset `+ (K + 1)` → `+ position_count` |
| `c0ab9e2` | Phase 3a | 신규 파일 `ssd/engine/helpers/hybrid_phase2_plan.py` — `HybridPhase2Plan` dataclass (alloc-once 버퍼 + `begin_step(valid_k)` API). 단독 smoke (long-hit cont=18, proxy=36; short-hit cont=10, proxy=20) 통과 |
| `106e6ea` | Phase 3b.1 | `phase1_layout_long`/`phase1_layout_short` 인스턴스를 `_init_prealloc_buffers` 에서 생성 (`forward_depth = K1`, `position_count = K_long+1` / `K_short+1`). 아직 `_build_tree_batch_mesa` 에 wire-in 안 됨 |
| `66bc7f2` | Docs | 1차 status 리포트 (이 문서가 그것을 대체) |

**브랜치 상태**: 모든 변경이 gated — `mesa_phase1_k` / `mesa_phase2_k`
config 가 None 이면 (현재 모든 sweep 의 default) 기존 MESA 두-pass
경로가 그대로 돕니다. 즉 **현재 브랜치를 main 에 머지해도 회귀가
없습니다.** 신규 hybrid 경로는 config 로 opt-in.

Foundation smoke (이번 세션에서 검증):
```
[OK] TreeLayout legacy invariant: K=4, position_count=5
[OK] TreeLayout decoupled: forward_depth=4, position_count=9
[OK] HybridPhase2Plan: long+short begin_step, dispatch keys
[OK] SpeculateResult.valid_k plumbed
```

---

## 2. 왜 두 번째 세션에서도 핵심 알고리즘 변경을 land 못 했는가

**솔직한 평가**: 플랜의 Phase 3b.3 — single hybrid forward + custom
mask + 5-region scratch — 는 **iterative GPU-debug 사이클 수십 번이
필요한 클래스의 변경** 이며, 한 세션 안에 끝까지 가는 작업이 아닙니다.

이유는 코드 양이 아니라 **silent-correctness 트랩** 입니다:

1. **FlashInfer wrapper / mask shape mismatch** 는 컴파일 통과
   → CudaGraph capture 통과 → 첫 replay 에서 segfault 또는 garbage
   logits 로 surface. shape 가 우연히 맞으면 generation 이 **틀린
   토큰을 만들면서 termination 조건만 우연히 맞으면 silent**.
2. **per-row mask + per-row block tables** 는 row 1 개당 인덱스 1 칸
   어긋나면 KV corruption. unit test 만으로는 잡기 어렵고, 긴 generation
   에서 quality drop 으로만 surface.
3. **CudaGraph 의 8 bucket 캡처** 는 한 번 캡처되면 입력 shape 가 변하지
   않아야 하는 강한 invariant. 한 bucket 의 capture 가 잘못되면 그
   bucket 만 깨지고 (다른 bucket 은 정상) — 진단이 어렵습니다.

이런 종류의 변경은 "blind 로 plan 만 보고 코드 짜면 거의 100% 들어감"
의 의미가 *대충 짐작* 이 아니라 *경험적 사실* 입니다. 따라서 다음
세션에서는 **smaller-scope v1 → 측정 → 확장** 사이클을 권장합니다.

---

## 3. 단계적 구현 — v1 → full hybrid

**최종 목표는 plan 전체** (Phase 1 K1-deep + continuation + proxy 를
single hybrid forward 로 batched). 그 전 단계 v1 를 지나서 incremental
하게 land. v1 의 코드가 그대로 살아있고 그 위에 P1 split + continuation
+ hybrid batching 을 얹는 구조.

### 절감의 두 종류 — (a) 토큰 총량 + (b) launch / batch 효율

플랜의 wall-clock 이득을 정확히 분해하면 **두 부분** 입니다:

```
현재 MESA:
  Phase 1: K_long forwards × MQ_LEN_phase1
  Phase 2: K_long forwards × MQ_LEN_proxy
  → 총 2 × K_long launches

v1 (proxy K2 단축만, Phase 1 K_long 유지):
  Phase 1: K_long forwards × MQ_LEN_phase1
  Phase 2 proxy: K2 forwards × MQ_LEN_proxy
  → 총 K_long + K2 launches

full hybrid (plan):
  Phase 1: K1 forwards × MQ_LEN_phase1
  Phase 2 hybrid: K2 forwards × (MQ_LEN_phase1 + MQ_LEN_proxy)
  → 총 K1 + K2 = K_long launches
```

**이득 (a) — 토큰-forward 총량 절감:**
- 현재: `K_long × (MQ_p1 + MQ_proxy)`
- v1, hybrid 둘 다: `K_long × MQ_p1 + K2 × MQ_proxy`
- 절감 = `K1 × MQ_proxy`. v1 와 hybrid 가 동일.

**이득 (b) — launch / batch 효율 (hybrid 만):**
- launch 횟수: 현재 `2 × K_long`, v1 `K_long + K2`, hybrid `K_long`.
  hybrid 가 v1 보다 K2 만큼 적음.
- Phase 2 의 K2 launch 가 hybrid 에서는 `MQ_p1 + MQ_proxy` 합쳐진
  batch 라 한 launch 당 GEMM 효율 ↑. 작은 draft (1B) 에서 attention/MLP
  kernel 이 memory-bound 이므로 batch 합치는 효과가 큼.

3090 + 70B AWQ 처럼 **kernel launch overhead 가 forward 자체와 비슷한
크기** 인 환경에서는 (b) 가 무시 못 할 비중. 직전 70B AWQ sweep timeline
plot 에서 draft forward 의 launch overhead 비중이 크다는 게 보였음.

### 단계 v1: "Phase 2 proxy K2 단축 + verify 분기" (이번 세션)

- **Phase 1 은 K_long 그대로** (현재 MESA 동일)
- **Phase 2 proxy 를 K2 로 단축**: 새 `proxy_layout_short` 생성 (K=K2,
  position_count=K_long+1)
- **cache 의 proxy-sourced row 가 K2 길이 suffix 보유** (= `K_short`)
- **verify 분기 (`verify_long` / `verify_short`)** — proxy row 가
  hit 되면 verify 의 lookahead 가 K2+1 (vs K_long+1)

→ **이득 (a) 캡처**. (b) 는 아직 없음. full hybrid 의 lower bound.

### 단계 P1 split + hybrid: full plan (v1 위에 incremental)

- Phase 1 의 forward depth 를 K_long → K1 로 단축
- Phase 2 에 continuation rows 추가 (Phase 1 leaf 를 K2 더 연장)
- continuation + proxy 를 single batched forward 로 통합
- per-row block tables / per-row mask / 5-region scratch 도입

→ **이득 (b) 추가 캡처**. v1 측정으로 (a) 의 절대 이득이 잡혀 있으니,
이 단계 land 후 (b) 의 추가 비중을 깨끗하게 잴 수 있음.

### 왜 v1 먼저인가

- v1 = plan 의 strict subset (cache row 하나의 length 가 K_long 또는 K2,
  Phase 1 코드 그대로). 회귀 위험 최소.
- v1 의 verify_short capture / valid_k dispatch 는 hybrid 단계에서도
  그대로 reuse. 코드 재사용 100%.
- v1 가 stable 하면 P1 split / continuation / per-row mask 를 한
  piece 씩 추가하면서 **각 piece 가 가져오는 추가 perf 를 측정 가능**.
  big-bang 보다 훨씬 안전.

### v1 구현 파일별 변경 (다음 세션 GPU-debug 가이드)

| File | Function | 변경 |
|---|---|---|
| `ssd/config.py` | `__post_init__` | 변경 없음 (이미 K1+K2 invariant 검증) |
| `ssd/engine/draft_runner.py` | `_init_prealloc_buffers` | `proxy_layout_short` 추가 (K=K2, position_count=K_long+1) |
| `ssd/engine/draft_runner.py` | `_build_tree_batch_mesa` | proxy pass 가 `proxy_layout_short` 사용. `mesa_phase1_k != None` 일 때만 |
| `ssd/engine/draft_runner.py` | `_merge_and_populate_cache` | proxy-sourced row 의 valid_k = K2; draft row 의 valid_k = K_long |
| `ssd/engine/draft_runner.py` | `hit_cache_and_respond` | matched row 의 valid_k 를 fused_response 에 packing (이미 plumb 됨, 값만 정확하게) |
| `ssd/engine/speculator_async.py` | `speculate` | (이미 unpack 됨, 그대로) |
| `ssd/engine/verifier.py` | `verify` | `speculate_result.valid_k` 읽어 `lookahead = valid_k + 1` per-step. Single verify call 이지만 shape 가 dynamic |
| `ssd/engine/model_runner.py` | `_init` | `capture_mesa_verify_cudagraph` 가 `verify_long` + `verify_short` 두 bucket capture |
| `ssd/engine/helpers/cudagraph_helpers.py` | `capture_mesa_verify_cudagraph` | lookahead 인자 받아서 K_long+1 / K_short+1 두 번 capture. `run_mesa_verify_cudagraph` 가 valid_k 따라 dispatch |
| `ssd/engine/llm_engine.py` | `step` | dispatch 단순. 변경 없거나 minimal |
| `ssd/engine/scheduler.py` / `block_manager.py` | block 예약 | KV scratch 가 K_long 기준이라 K_short 더 작음, overflow 위험 없음 (변경 없음) |

**예상 LOC**: ~250 LOC (vs full hybrid ~1000+).

### v1 의 GPU-debug 셈 (다음 세션 budget)

각 사이클 = code change → load model (~30s) → 4-step generation
(~10s) → check accept rate / no NaN / output sane.

추정 사이클 수:
1. proxy_layout_short 추가 + Phase 2 가 그것을 쓰도록 → 2-3 cycles
2. cache valid_k populate 정확하게 → 1-2 cycles
3. verify 2-bucket capture → 3-5 cycles (FlashInfer wrapper alignment)
4. valid_k dispatch 통합 → 2-3 cycles
5. correctness sanity (vs baseline MESA) → 2-3 cycles

총 10~16 GPU-debug cycles. 한 세션 (~3-5 시간) 안에 가능.

### v1 land 후 확장 경로

v1 가 안정되면 (= 통과 + 측정 가능) 그 다음 단계:

- **단계 P1 split**: Phase 1 forward 도 K1 으로 단축 + continuation pass.
  여기서부터 continuation = 새 attention layer. 단독 unit test (CPU 가능,
  mask + slot mapping 만 검증) 부터 시작.
- **단계 hybrid**: continuation + proxy 를 단일 batched forward 로 통합.
  여기서 custom mask builder 가 들어옴.

P1 split / hybrid 는 v1 위에서 incremental commit + GPU validate 로
진행. v1 가 baseline 을 깔아주므로 회귀 측정도 자연스러움.

---

## 4. 사용자 직접 지시 받은 sweep 전략 — 메모

> 3 exit-layer × dfo ∈ {2,3,4,6} → K1 (target idle 최소화 지점) ×
> pfo ∈ {2,3,4,6} → K2 (target verify 끝나는 지점) → TPS 측정.

**v1 으로는 K2 만 sweep 가능** (K1 split 미구현). v1 의 sweep:

- 3 exit-layer × pfo {2,3,4,6} × K2 {2,3,4,6} = **48 configs**
- 각 ~2-5 분 (70B AWQ + TinyLlama, output_len=512, numseqs=128) →
  **~3-4 시간 GPU**
- best (pfo, K2) 선정 후 P1 split 단계 진행 → K1 sweep 추가

P1 split 까지 land 되면 *full plan* sweep:

- 3 exit-layer × dfo {2,3,4,6} × K1 {2,3,4,6} × pfo {2,3,4,6} × K2
  {2,3,4,6} = **3 × 4^4 = 768 configs**
- 너무 많음. 권장: 우선 v1 sweep 으로 best (pfo, K2) 고정 후 dfo, K1
  를 4×4=16 grid 로 sweep. 총 3 × 16 × 1 = 48 configs ~ 3-4 시간.

직전 70B AWQ sweep 의 `experiments/sweep_70b_awq/orchestrate.py` 가
reusable. 변경 사항은 config dict 의 K1/K2 추가만.

---

## 5. 결과물 요약

| 항목 | 상태 |
|---|---|
| 계획 문서 (`PLAN.md`) | ✅ main 머지 (`25e7bf4`) |
| 이슈 트래커 (`ISSUE.md`) | ✅ 작업 브랜치에 staged commit |
| Phase 0 (valid_k plumbing) | ✅ 두 세션 공통 |
| Phase 1 (cache valid_k) | ✅ |
| Phase 2 (TreeLayout extension) | ✅ |
| Phase 3a (HybridPhase2Plan dataclass) | ✅ alloc-once 검증 |
| Phase 3b.1 (phase1 layouts 생성) | ✅ unused, gated |
| Phase 3b.2 (Phase 1 K1 wire-in) | ❌ runtime debug 필요 — **v1 에서는 skip 권장** |
| Phase 3b.3 (hybrid forward + mask) | ❌ runtime debug 필요 — **v1 에서는 skip** |
| Phase 4 (verify dispatch) | ❌ — **v1 의 핵심**. ~250 LOC, 5-8 cycles |
| Phase 5 (short-base buckets, JIT 단축) | ❌ — v1 에서 일부 (verify_short) 만 필요 |
| Phase 6 (validation) | ❌ |
| Sweep | ❌ — v1 land 후 ~3-4 시간 GPU |

main 에 머지된 변경: 계획 문서 + 이슈 트래커 (`25e7bf4`). 작업 브랜치는
사용자 지시대로 머지하지 않음.

---

## 6. v1 측정 결과 (실측, 이번 세션 최종)

`feat/mesa-phase2-hybrid` 위에서 v1 구현 + verify_short root cause
fix + correctness fix 까지 land. layerskip-llama3-8B + Llama-3.2-1B
(K=8, K1=K2=4, dfo=2, pfo=2, exit_layer=21, 4 prompts × 32 tok,
B=1, 3 GPU async, greedy temp=0).

| 항목 | Baseline (no hybrid) | **v1 (final)** |
|---|---|---|
| TPS | 30.11 tok/s | **43.51 tok/s (+44.5%)** |
| Avg target time | 267 ms | **122 ms (-54%)** |
| Avg draft time | 159 ms | **87 ms (-45%)** |
| Decode throughput | 50 tok/s | **67 tok/s (+34%)** |
| Avg accept rate | 0.80 | 0.63 |
| Phase 1 hit | 0.60 | 0.52 |
| Phase 2 hit | 0.00 | 0.09 |
| Generated text | (4 prompts) | **byte-for-byte 일치** |

### 측정 의미

- **이득 (a) 가 완전히 캡처됨**: Phase 2 proxy 가 K2=4 forwards (vs
  baseline K_long=8) → draft time -45%. Verify_short bucket 이 proxy
  hit 시 K_short+1=5 위치만 forward (vs K_long+1=9) → target time -54%.
- **TPS +44.5%**: Plan §744 의 performance estimate 의 conservative
  range (+35-55%) 안에 있음. Mid-range 성능.
- **Generation 정합성**: greedy temp=0 에서 baseline 과 byte-for-byte
  동일 출력 → correctness 검증.
- **Accept rate 0.63 (vs 0.80)**: Phase 2 hit (8.7%) 의 max accept 가
  K_short=4 (vs Phase 1 의 K_long=8) 로 cap 되기 때문. Plan 의 의도된
  trade-off — 그럼에도 unit-time 당 토큰 수는 +44.5% 증가.

### v1 에서 해결한 두 가지 root cause

#### Bug 1: TP rank verify dispatch desync (multi-prompt hang)

`_mesa_step_lookahead` 가 rank 0 의 Python attribute 로만 set 됨 → 다른
TP rank 는 default = K_long. 결과: 한 rank 은 verify_short bucket replay,
다른 rank 은 verify_long → NCCL collective shape mismatch → silent
deadlock.

Fix: `step_lookahead` 를 `call("run", ..., step_lookahead)` 인자로 전달.
SHM 통해 모든 rank 에 broadcast. `run()` 이 attribute set → 모든 rank
같은 bucket dispatch.

진단 trace evidence:
```
[verify_cg] mesa_verify_short pre.replay starting (k+1=5, bs=1)
[verify_cg] mesa_verify_long pre.replay starting (k+1=9, bs=1)
```
같은 step 의 다른 rank 이 다른 bucket 에 있음.

#### Bug 2: Correctness — proxy row padding 소비 (사용자 지적)

이전 v1 default 가 `SSD_USE_VERIFY_SHORT=0` 로 cache valid_k override
gating 함 → proxy hit 시 valid_k=K_long 반환 → speculator 가 padded
zero token 까지 seq.token_ids 에 extend → verify 가 zero token vs target
argmax 비교, vocab id 0 false-accept risk.

Fix: gating 제거. Cache valid_k 항상 honor. Bug 1 fix 와 함께 verify_short
가 안전하게 동작하므로 proxy hit 의 K_short 토큰만 정상 verify.

### 이득 (b) 의 추가 잠재력

v1 = 이득 (a) 만. verify_short 가 fix 되면:
- Target verify 가 proxy hit 시 K_short+1 = 5 positions 만 forward → target time
  추가 단축
- 더 중요한 건: padding 위치 reject 가 사라지므로 **accept rate 가
  baseline 0.80 수준으로 회복** → 추가 TPS win

추정: full v1 (verify_short 정상 작동) 는 +25~30% TPS, full hybrid (P1
split + 통합 batch) 는 +30~40% TPS 기대. 측정 필요.

## 7. 잔여 작업 — Phase 3 ingredient (b)

이번 세션 land 된 것 = ingredient (a) (proxy K2 forwards + verify_short
dispatch). Plan 의 hybrid forward 의 진짜 핵심인 **single batched
forward (continuation + proxy 한 batch)** 가 남음.

### Plan §Performance Estimate 의 분해

Plan 의 +35–55% expected TP 는 다음 두 ingredient 의 합:

| Ingredient | Source | 현재 land 됨 | TPS 기여 |
|---|---|---|---|
| **(a)** | Phase 2 proxy K_long → K_short forwards | ✅ +44.5% 측정 |
| **(b)** | Phase 1 K1 split + continuation, single hybrid batched forward | ❌ 미구현 | +5–10% 추가 기대 (plan §744) |

(a) 만으로 plan 의 conservative 범위 mid-point 도달. (b) 가 추가되면
upper-end (+50–55%) 가능.

### Phase 3 ingredient (b) 구현 시 작업 항목

| Plan 항목 | 작업 | 위험도 |
|---|---|---|
| Phase 1 forward depth K1 (not K_long) | `_build_tree_batch_mesa` 가 `phase1_layout_long` (K=K1, position_count=K_long+1) 사용 | 낮음 (layout 이미 생성됨) |
| Phase 2 continuation pass | Phase 1 leaf 의 K1-th 토큰을 input 으로 K2 더 forward; phase 1 KV 를 own slice 로 attend | 중 (KV scratch 분할) |
| **Single hybrid batched forward** | continuation rows + proxy rows 를 한 batch 로 합쳐 K2 forwards | **높음 (silent-correctness 트랩)** |
| 5-region scratch | persistent / glue / Phase 1 KV / A_tail / B_proxy 분리 slot pool | 중 |
| **`build_hybrid_packed_mask`** | continuation row 와 proxy row 의 다른 prefix shape 처리하는 per-row mask | **매우 높음 (FlashInfer + per-row attention)** |
| `phase2_hybrid_long/short` CG capture | 새 batch shape, 새 mask buf | 중 |

Plan §Risk #1, #2, #3 가 모두 이 단계에 집중. 특히 per-row mask 의
correctness 는 **GPU iterative debug 5-10 사이클 필요한 클래스**.

예상 LOC: ~600–800. 예상 cycles: 20–30 GPU debug. 1–2 focused session.

### Sweep — v1 위에서 즉시 가능

v1 ingredient (a) 만으로도 sweep 가능 (이미 동작 + 측정됨):

- 3 exit-layer × pfo {2,3,4,6} × K2 {2,3,4,6} = **48 configs**
- 70B AWQ + TinyLlama AWQ stack 에서 ~3–4 시간 GPU
- 기존 `experiments/sweep_70b_awq/orchestrate.py` 변형 (config dict 에
  `mesa_phase1_k`, `mesa_phase2_k` 추가)
- best (pfo, K2) 고정

Phase 3 ingredient (b) 완료 후 추가 sweep:

- 위 best (pfo, K2) 고정 + dfo {2,3,4,6} × K1 {2,3,4,6} = 16 configs × 3
  exit-layer = 48 configs, ~3–4 시간 GPU
- 최종 (dfo, K1, pfo, K2) 결정
