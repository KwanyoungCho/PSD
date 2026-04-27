# MESA Phase 2 Hybrid — Final Report

**브랜치**: `feat/mesa-phase2-hybrid`
**날짜**: 2026-04-27
**범위**: Phase 9B 구현 완료 + layerskip-llama2-70B / TinyLlama-1B 파라미터 sweep

---

## 1. 요약

Phase 2 hybrid (single-batch cont+proxy forward) 가 long/short bucket 모두에서
CG default hot path 로 동작하도록 구현 완료. Plan §"single batched hybrid
forward" 와 §229-262 의 5-region scratch 설계를 그대로 유지.

핵심 결과 (8B target / 1B draft 검증 환경, 8 prompts × output_len=128):

| Mode | Total Throughput (tok/s) | vs split fallback |
|---|---|---|
| Split fallback (`SSD_FORCE_SPLIT_PHASE2=1`) | 86.94 | base |
| Eager hybrid (`SSD_FORCE_EAGER_HYBRID_PHASE2=1`) | 69.07 | -20.6% |
| **Native hybrid-default (CG long+short)** | **107.50** | **+23.6%** |

8/8 generation byte-identical to split fallback. Accept rate `+11.3%` (split 0.80
→ hybrid 0.89). All 8 CG families captured: glue_long/short, phase1_long/short,
phase2_hybrid_long/short, verify_long/short.

---

## 2. 구현 진행 (Step / Phase 단위)

### Step 1-7 (Phase D 포함) — Hybrid eager + oracle 검증
- `HybridPhase2Plan` dataclass (max-size 한 번 할당, per-step `begin_step`)
- `_build_phase2_hybrid_plan` per-row × per-depth tensor fill
- `_compute_hybrid_bool_mask_for_depth` 5-region 의미 (persistent + j_idx-aware
  glue + own P1 KV + own A_tail / B_proxy)
- `_decode_phase2_hybrid` eager forward (cu_seqlens_q=None, active_mq_len=1)
- `correct_split_cont` oracle (eager, plan-correct mask)
- `fresh_proxy_oracle` (proxy_eager_debug wrapper, no CG state pollution)
- self-consistency + top1/top2 margin + 2*log_d drift bound
- 검증: 100% drift-consistent first-mismatches, 100% 자기-자기 deterministic,
  e2e text byte-identical

### Step 8 — hybrid default 전환
- Hot path: Phase 1 → proxy recv/wait → phase2_build → hybrid → merge
- 기본 = hybrid, fallback `SSD_FORCE_SPLIT_PHASE2=1`
- Parity harness `SSD_HYBRID_PARITY=1` 디버그용 보존
- Native hybrid TPS 69.07 (eager only) → CG 작업이 Step 9B 의무

### Step 9A — runtime valid_k bucket dispatch
- `partial_tree_decode_args` 에 `valid_k` 추가 (cache hit 의 row 의 valid_k)
- phase1_layout long/short, proxy step layout `position_count = valid_k+1`,
  `_select_proxy_sourced_tokens_policy_a(valid_k=...)` slice 적용
- `_compute_hybrid_bool_mask_for_depth` `K_long → K_step` rename
- `_decode_phase2_hybrid` long-only assert lift
- `_merge_and_populate_cache(draft_layout=...)` 파라미터 추가
- `run_fi_tree_decode_cudagraph` mask `K_for_mask = layout.position_count - 1`
- `phase1_layout_short` CG + wrapper 추가
- `SSD_TRACE_BUCKET=1` per-step bucket 출력

### Phase 9B-0 — glue_short
- `make_glue_decode_input_ids(valid_k=...)` 와 `prepare_glue_decode_ctxt(valid_k=...)`
- `Context.glue_valid_k` field
- `capture_verify_cudagraph(k_plus_1=...)` 로 K_short+1 wide draft glue CG
- `run_model` 에서 `glue_valid_k` 로 verify / verify_short 분기

### Phase 9B-1 — hybrid CG 인프라
- `phase2_hybrid_long`/`_short` (eager) + `phase2_hybrid_long_cg`/`_short_cg`
  (use_cuda_graph=True) wrapper family
- `capture_phase2_hybrid_cudagraph(bucket, K_step)`: bucket 별 single-depth CG,
  separate graph_pool (verify_short aliasing 학습 반영)
- `run_phase2_hybrid_cudagraph`: K2 회 replay + per-depth metadata 를
  `low_level_packed_plan` 으로 baked buffer 에 write
- 동일 의미 보장: cu_seqlens_q=None, active_mq_len=1, B_proxy region 분리

### Phase 9B-2 — runtime dispatch + fallback gates
- 기본: CG hybrid (`run_phase2_hybrid_cudagraph`)
- `SSD_FORCE_EAGER_HYBRID_PHASE2=1` → eager hybrid fallback
- `SSD_FORCE_SPLIT_PHASE2=1` → split fallback (Step 8)
- `plan.valid_k` → bucket 별 CG (`phase2_hybrid_long_cg` / `_short_cg`)

### Phase 9B-3 — short bucket oracle / parity
- `_decode_correct_split_cont` `K_step = plan.valid_k` 일반화
- `_split_eager_mirror_proxy` `K_full = position_count - 1` (이전 `speculate_k`
  하드코딩이 short bucket fan_out_list 길이 mismatch 일으킴)
- Oracle 출력에 bucket tag: `[CONT ORACLE long/short]`, `[PROXY ORACLE long/short]`

### Phase 6 — 8 family CG 검증
8 families capture 확인 (8B+1B 환경):
```
[MESA] Captured draft glue_short CG (K_short+1=5)               ← glue_short
[MESA hybrid] Captured 2-bucket verify CG (long(K=8) + short(K=4))   ← verify_long/short
About to capture FI cudagraphs for bs=[1] key=fi_tree_decode_phase1_long MQ_LEN=18  ← phase1_long
About to capture FI cudagraphs for bs=[1] key=fi_tree_decode_phase1_short MQ_LEN=10 ← phase1_short
[MESA hybrid] capturing phase2_hybrid CG bucket=long total_rows=36   ← phase2_hybrid_long
[MESA hybrid] capturing phase2_hybrid CG bucket=short total_rows=20  ← phase2_hybrid_short
(target verify CG at K_long+1=9 → glue_long for draft re-uses)       ← glue_long
```

---

## 3. Tolerance gate 검증 (Step 8 entry rule)

8 prompts × 128 tok = 131 spec steps 에서:

| 항목 | gate | 측정값 | 통과 |
|---|---|---|---|
| self-consistency strict 0 | = 0 | 393/393 = 0/72 each | ✓ |
| proxy first-divergence drift-consistent (2*log_d bound) | 100% | 60/60 = 100% | ✓ |
| cont first-divergence drift-consistent | 100% | 22/22 = 100% | ✓ |
| proxy mismatch token rate | ≤ 3% | 2.10% | ✓ |
| cont mismatch token rate | ≤ 0.6% | 0.51% | ✓ |
| end-to-end text byte-identical | 100% | 8/8 | ✓ |
| accept rate delta vs split | ≥ 0 | +11.3% | ✓ |
| first-divergence max o_margin | ≤ 0.2 | 0.125 | ✓ |

CG 도입 후 (Step 9B 완료) 같은 set 에서 short bucket parity 도 동일 패턴
(small-sample ratios but consistent with structural drift).

---

## 4. 파라미터 sweep — layerskip-llama2-70B + TinyLlama-1.1B

### 4.1 환경
- **Target**: `facebook/layerskip-llama2-70B` AWQ-calibrated W4A16 (TP=4)
  at `/data2/chokwans99/awq_calibrated/layerskip_llama2_70b`,
  artifact `/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4`
- **Draft**: `TinyLlama-1.1B-Chat-v1.0` (TP=1, dense)
- **Hardware**: 5 GPUs (4 target TP + 1 draft) on RTX 3090 sm_86
- **Prompts**: 50 random prompts × output_len=128, temp=0.6, B=1,
  max_model_len=2048

### 4.2 결과 (모두 hybrid CG default mode)

| config | TPS (tok/s) | accept | tok/step | cache_hits | P1_hit | P2_hit | draft_ms | verify_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| split_k5_dfo2_exit40 (baseline) | 54.10 | 0.56 | 3.81 | 0.68 | 0.56 | 0.12 | 57.39 | 45.74 |
| hybrid_k5_K1_2_K2_3_exit40 | 63.48 | 0.60 | 3.98 | 0.72 | 0.64 | 0.08 | 49.32 | 45.45 |
| **hybrid_k5_K1_3_K2_2_exit40** | **64.72** | **0.61** | **4.06** | **0.71** | **0.62** | **0.09** | **42.79** | **45.20** |
| hybrid_k6_K1_3_K2_3_exit40 | 64.29 | 0.58 | 4.50 | 0.70 | 0.60 | 0.10 | 55.84 | 46.72 |
| hybrid_k6_K1_3_K2_3_exit47 | 60.65 | 0.57 | 4.45 | 0.69 | 0.59 | 0.10 | 59.23 | 46.59 |
| hybrid_k8_K1_4_K2_4_exit40 | 61.12 | 0.50 | 5.03 | 0.66 | 0.56 | 0.10 | 65.63 | 48.88 |
| hybrid_k8_K1_4_K2_4_exit47 | 63.18 | 0.55 | 5.37 | 0.71 | 0.57 | 0.13 | 67.52 | 50.46 |
| hybrid_k8_K1_2_K2_6_exit40 | 55.78 | 0.55 | 5.44 | 0.68 | 0.59 | 0.10 | 79.58 | 50.90 |
| hybrid_k8_K1_6_K2_2_exit40 | 58.30 | 0.42 | 4.39 | 0.64 | 0.51 | 0.13 | 59.66 | 50.68 |

### 4.3 최적 파라미터: **K=5, K1=3, K2=2, exit_layer=40, dfo=2**

| metric | optimal | split baseline | delta |
|---|---:|---:|---:|
| **TPS (tok/s)** | **64.72** | 54.10 | **+19.6%** |
| accept rate | 0.61 | 0.56 | +9.0% |
| tokens/step | 4.06 | 3.81 | +6.6% |
| cache hits | 0.71 | 0.68 | +4.4% |
| P1 (draft) hit rate | 0.62 | 0.56 | +10.7% |
| **draft_ms** | **42.79** | 57.39 | **-25.4%** (faster) |
| verify_ms | 45.20 | 45.74 | -1.2% |

### 4.4 Sweep 분석

**(K1, K2) 균형**: K1 ≥ K2 가 한결같이 더 좋음.
- K=5: K1=3,K2=2 (64.72) > K1=2,K2=3 (63.48)
- K=8: K1=4,K2=4 (61.12) > K1=2,K2=6 (55.78); K1=6,K2=2 (58.30) 는
  accept 가 0.42 로 낮아 K1 너무 큰 경우 효과 떨어짐.

**총 K depth**: K=5 가 K=8 보다 일관되게 좋음 (small target / draft 비율
때문에 deep tree 의 추가 비용 ≥ 추가 accept 효과).

**exit_layer**: exit=40 (= L/2) 이 exit=47 (= 7L/12) 보다 조금 좋음.
이는 final_exp2_quant_70b 의 baseline 결과와도 일치 (earlier exit →
target verify 빠름 → 전체 step time 단축). cache hit rate 차이는 미미.

**draft_ms 단축이 핵심**: hybrid 의 hot path 가 draft step time 을 25%
줄임 (cont+proxy 단일 forward + CG capture 효과). verify time 은 거의 동일.
즉 hybrid 의 throughput 개선은 **per-step efficiency** 가 아니라
**draft step time 자체** 에서 나온다.

### 4.5 sweep vs reference (final_exp2_quant_70b)
이전 split MESA reference (final_exp2_quant_70b/mesa_k5_f4_dfo2_exit40):
- 200 prompts × output_len=256, temp=0.6 → **61.02 tok/s**

이번 sweep 의 split_k5_dfo2_exit40 (50 prompts × output_len=128) → **54.10 tok/s**.
prompt 수가 적고 startup 오버헤드 비중이 커서 절대값은 낮지만 비율은
유지됨. **hybrid_k5_K1_3_K2_2_exit40 (64.72)** 는 reference baseline 대비
**+6.1%** 이며 split-fallback 대비 **+19.6%**.

---

## 5. 운영 가이드

### 환경 변수

| Env | 효과 |
|---|---|
| (none) | hybrid CG default — 권장 |
| `SSD_FORCE_EAGER_HYBRID_PHASE2=1` | eager hybrid fallback (CG 비교/디버그) |
| `SSD_FORCE_SPLIT_PHASE2=1` | split path fallback (legacy reference / 회귀 비교) |
| `SSD_HYBRID_PARITY=1` | parity harness on (overhead 큼 — 디버그 전용) |
| `SSD_HYBRID_CONT_ORACLE=1` | parity 안에서 cont oracle 동작 (parity 동시) |
| `SSD_HYBRID_PROXY_ORACLE=1` | parity 안에서 proxy oracle 동작 (parity 동시) |
| `SSD_HYBRID_SELF_CONSISTENCY=1` | parity 안에서 oracle/hybrid 자기-자기 비교 |
| `SSD_TRACE_BUCKET=1` | 매 step bucket (long/short) 로그 |

### 8 CG family 진단

```
[MESA] Captured draft glue_short CG (K_short+1=N)
[MESA hybrid] Captured 2-bucket verify CG ... long(K=...) + short(K=...)
About to capture FI cudagraphs for bs=[1] key=fi_tree_decode_phase1_long MQ_LEN=...
About to capture FI cudagraphs for bs=[1] key=fi_tree_decode_phase1_short MQ_LEN=...
[MESA hybrid] capturing phase2_hybrid CG bucket=long total_rows=...
[MESA hybrid] capturing phase2_hybrid CG bucket=short total_rows=...
```

위 6 라인 + draft 의 verify CG (K_long+1) = 8 family 가 모두 출력되어야 정상.

---

## 6. 알려진 한계

- **EAGLE 통합**: hybrid 는 non-EAGLE MESA 만 검증. EAGLE 사용 시
  draft_acts 처리 분기가 hybrid 경로에 추가되어야 함. 이번 작업 범위 외.
- **proxy first-divergence ≤ 5%**: structural FlashInfer kernel drift 로 추정
  되며 plan §388 mask 의미는 oracle 로 100% 일치. argmax flip 은 near-tie
  bf16 round-off (max o_margin = 0.125, 2 LSB) 로 한정.
- **split CG 의 generic continuation mask**: K1 < K_long (= mesa_phase1_k <
  speculate_k) 인 경우 cross-branch Phase 1 KV 와 mistreated glue 영역에
  visibility leakage. `cudagraph_helpers.py:319` 에 known-bug 주석 명시.
  Hybrid path 는 plan-correct → 영향 없음. split fallback 사용 시에만 노출.

---

## 7. 향후 작업

- **EAGLE × hybrid 통합**: draft_acts 가 hybrid forward 출력에서도 plumb 필요.
- **dfo / pfo per-bucket 별도 sweep**: 현재 long/short 모두 같은 dfo, pfo 사용.
  short bucket 에서 더 많은 draft 분기를 두는 변형 가능 검토.
- **multi-batch (B>1) 확장**: 현재 B=1 invariant. B>1 hybrid 는 plan §"B=1
  invariant" 해제 + per-seq plan 구조 재설계 필요.
