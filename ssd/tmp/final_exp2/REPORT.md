# MESA Rev1 Final Experiment (Post-Fix)

## Context

Rev1 수정 사항 (`MESA-rev1-problems.md` 기반) 적용 후 재실험.

**수정 완료 항목**:
1. **#3** `B=1 only` assert (config.py) — MESA 경로 correctness 강제
2. **#4** `mesa_proxy_top_k` 자동 확대 + **draft fallback 제거**
   - `proxy_top_k = pfo × (K+1) + draft_fan_out + 2` (e.g., K=5, pfo=2: 3→16)
   - draft fallback 분기 완전 삭제 — Phase-2 tree 전체가 target-informed
3. **#D** `_decode_tree` spec 버퍼 pre-allocation (2-slot round-robin)
   - 매 step 8 MB 재할당 + zero-fill 제거
   - **2 슬롯 필수**: Phase 1과 Phase 2 결과를 merge까지 동시 보관 (1-slot은 버그 유발)
4. **#1** Policy A proxy selection **fully vectorized** (`.tolist()` 제거, Python loop 제거)
5. **#8** `get_forked_recovery_tokens_from_logits`의 dead `mesa_proxy` 분기 삭제

## 1-slot 버그 (구현 중 발견 및 수정)

최초 구현에서 `_decode_tree`를 **단일 pre-alloc 버퍼**로 만들었다가 MESA에서 accept 0.22로 붕괴 (이전 0.48의 절반). 원인: Phase 2의 `_decode_tree` 호출이 **Phase 1이 반환한 view를 덮어씌움** → merge 시점에 Phase 1 데이터가 Phase 2로 오염 → tree 절반이 잘못된 토큰.

**Fix**: `_spec_tokens_bufs`, `_spec_logits_bufs` 를 **n_slots=2 리스트**로 만들고 `_spec_buf_counter`로 round-robin. Phase 1 → slot 0, Phase 2 → slot 1. 다음 step은 다시 slot 0 (merge 후 `torch.cat`으로 복사된 덕에 안전).

## Experiment

- **Target**: `facebook/layerskip-codellama-34B` (48 layers, TP=4)
- **Draft**: `TinyLlama-1.1B-Chat-v1.0` (TP=1)
- **Prompts**: 200 (humaneval/alpaca/gsm8k/ultrafeedback × 50)
- `output_len=256`, `temp=0.6`, B=1

### Configs

| # | Name | Flags | MQ_LEN |
|:--:|------|-------|:------:|
| 1 | `ar` | no spec (TP=4 only) | — |
| 2 | `baseline_k7_uniform` | `--k 7 --f 3` | 24 |
| 3 | `baseline_k7_geo` | `--k 7 --flh 5 4 4 3 3 2 2 1` | 24 |
| 4 | `mesa_k5_f4_dfo2_exit24` | `--k 5 --f 4 --mesa --exit 24 --dfo 2` | 24 |
| 5 | `mesa_k5_f4_dfo2_exit28` | 동일, exit=28 | 24 |
| 6 | `mesa_k5_f4_dfo2_exit32` | 동일, exit=32 | 24 |

**모든 config MQ_LEN=24로 통일** (tree 예산 매치).

## Results

| Config | TP (tok/s) | Speedup vs AR | Accept | **CacheHit** | Tok/Step | Draft (ms) | Verify (ms) |
|--------|-----------:|--------------:|:------:|:------------:|:--------:|-----------:|------------:|
| AR (TP=4, no spec) | **28.26** | 1.00× | — | — | — | — | — |
| Baseline K=7 uniform | 67.99 | 2.41× | 0.44 | 0.66 | 4.06 | 46.5 | 47.2 |
| **Baseline K=7 geo** | **68.45** | **2.42×** | 0.44 | 0.66 | 4.07 | 46.1 | 47.2 |
| **MESA K=5 exit=24** | **60.73** | **2.15×** | **0.52** | 0.83 | 3.62 | 54.0 | 45.7 |
| MESA K=5 exit=28 | 58.48 | 2.07× | 0.52 | 0.85 | 3.60 | 56.1 | 45.5 |
| MESA K=5 exit=32 | 58.22 | 2.06× | 0.52 | **0.87** | 3.58 | 55.9 | 45.5 |

### 이전 실험 (final_exp, pre-fix) 대비 변화

| 지표 | 이전 MESA (K=6, exit=24) | 이번 MESA (K=5, exit=24) | Δ |
|------|:---:|:---:|:---:|
| Throughput | 55.09 | **60.73** | **+10.2%** |
| Accept | 0.48 | 0.52 | +0.04 |
| Draft step (ms) | 64.6 | 54.0 | -10.6 ms |
| baseline 대비 gap | -17% | **-11%** | +6pp |

## 분석

### Rev1 수정 개별 기여
| 수정 | 효과 |
|------|------|
| #4 fallback 제거 + proxy_top_k 확대 | Phase 2 branch 전부 target-informed → **accept +0.04** |
| #D 2-slot pre-alloc | `_decode_tree` 진입 overhead 제거, **draft step -5 ms** |
| #1 벡터화 | `phase2_build` GPU sync 제거 |

### Baseline K=7 uniform vs geo
- 67.99 vs 68.45 — 거의 동일 (0.7% 차)
- Accept/cache hit도 동일 → geometric fanout이 이 workload에선 uniform보다 유의미한 이득 없음
- 이유: K=7로 충분히 깊고 MQ_LEN=24 예산이 uniform에서 자연스럽게 잘 분산됨

### Baseline이 여전히 이기는 이유
- **draft step baseline 46 vs MESA 54 (+17%)**
- MESA의 Phase 2 replay 5번 = ~20 ms가 고정 overhead
- Accept rate 개선 (+0.04) × step 수는 draft step 증가를 상쇄 못 함

### Exit layer 민감도 (MESA)
| exit | TP | CH | 해석 |
|:---:|:---:|:---:|---|
| 24 (1/2) | **60.73** | 0.83 | graph_pre 짧아 proxy 빨리 도착 → Phase 1-2 wait 최소 |
| 28 (7/12) | 58.48 | 0.85 | 중간 |
| 32 (2/3) | 58.22 | **0.87** | proxy 품질 최고, 하지만 target graph_pre 길어 wait 증가 |

**이전 K=6 실험과 반대 경향** (이전엔 exit=32가 근소 우세). K=5에선 graph_pre 시간 단축 효과가 cache hit 미세 차이보다 중요.

## Per-Phase Breakdown (ms per spec step, post-warmup mean)

### Target side

| phase | baseline uniform | baseline geo | MESA exit=24 | MESA exit=28 | MESA exit=32 |
|-------|:---:|:---:|:---:|:---:|:---:|
| verify_setup | — | — | 0.24 | 0.23 | 0.23 |
| graph_pre | — | — | 22.28 | 25.68 | **28.98** |
| exit_logits | — | — | 0.40 | 0.43 | 0.40 |
| proxy_compute_send | — | — | 0.76 | 0.67 | 0.66 |
| graph_post | — | — | 18.48 | 15.24 | 11.97 |
| final_logits | — | — | 0.38 | 0.37 | 0.43 |
| **verify_replay** (monolithic) | **43.50** | **43.78** | — | — | — |
| verify_sample_accept | 2.27 | 2.09 | 2.24 | 2.13 | 2.12 |
| target_spec_wait | 10.18 | 9.95 | 11.95 | 14.07 | 14.06 |
| target_postprocess | 0.04 | 0.03 | 0.04 | 0.03 | 0.03 |
| **target total (est.)** | **56.0** | **55.9** | **56.7** | **58.9** | **58.9** |

- Baseline verify_replay = 43.5 ms (단일 graph, 48 layers 전체)
- MESA는 split: graph_pre (exit layer 수만큼) + graph_post (나머지) + proxy — 합은 baseline과 거의 같음
- MESA target_spec_wait가 baseline 대비 **+2-4 ms 증가**: Phase 2로 draft가 더 오래 걸려 target이 대기

### Draft side

| phase | baseline uniform | baseline geo | MESA exit=24 | MESA exit=28 | MESA exit=32 |
|-------|:---:|:---:|:---:|:---:|:---:|
| glue (draft forward) | 3.97 (+replay 3.52) | 3.96 (+3.51) | 4.44 (+3.54) | 4.65 (+3.56) | 4.43 (+3.55) |
| hit_cache_respond | 9.17 | 8.98 | 3.64 | 3.43 | 2.99 |
| draft_recv_cmd | 6.75 | 6.85 | 0.22 | 0.22 | 0.24 |
| **tree_prep × K** | 1.42 (0.35×4) | 1.36 | — | — | — |
| **tree_replay × K** | **18.55** (4.64×4) | **18.55** | — | — | — |
| phase1_build | — | — | 0.21 | 0.21 | 0.23 |
| phase1_prep × K | — | — | 1.60 (0.40×4) | 2.06 (0.51×4) | 1.64 |
| **phase1_replay × K** | — | — | **16.76** (4.19×4) | **16.77** | **16.80** |
| proxy_wait | — | — | 0.02 | 0.05 | 2.36 |
| phase2_build | — | — | 1.42 | 1.77 | 1.37 |
| phase2_prep × K | — | — | 1.43 | 1.87 | 1.45 |
| **phase2_replay × K** | — | — | **16.76** (4.19×4) | **16.77** | **16.80** |
| merge_cache | — | — | 0.04 | 0.04 | 0.04 |
| draft_send_response | 0.14 | 0.14 | 0.16 | 0.19 | 0.16 |
| **draft total (measured)** | **42.1** | **42.0** | **46.7** | **47.9** | **49.1** |

- Baseline: `hit_cache_respond 9ms + recv_cmd 7ms + glue 4ms + 4×tree_replay 18.5ms = 42ms`
- MESA: `hit_cache_respond 3.6ms + glue 4.4 + 4×phase1_replay 16.8 + 4×phase2_replay 16.8 + merge/build 3ms ≈ 47ms`
- **MESA의 추가 ~5 ms는 Phase 2 replay의 구조적 overhead** (sum_phase_replays = 33.6 vs baseline's 18.5)

### Exit layer 효과

| exit | graph_pre | graph_post | sum(pre+post) | proxy_wait (draft) | target_spec_wait |
|:----:|:---:|:---:|:---:|:---:|:---:|
| 24 (1/2) | 22.28 | 18.48 | 40.76 | **0.02** | 11.95 |
| 28 (7/12) | 25.68 | 15.24 | 40.92 | 0.05 | 14.07 |
| 32 (2/3) | **28.98** | 11.97 | 40.95 | 2.36 | 14.06 |

- Sum target (pre+post) 거의 동일 (41 ms) — 같은 모델 full forward 소요 시간
- exit=32는 graph_pre 길어짐 → proxy_wait 2.4 ms 생김 (draft가 target proxy 기다림)
- exit=24가 perf best: proxy 빨리 보내 draft 기다림 0

## Artifacts

```
tmp/final_exp2/
├── REPORT.md                             — 이 문서
├── run_all.sh                            — 전체 실행 스크립트
├── rerun_mesa.sh                         — 2-slot fix 후 MESA 재실행 스크립트
├── SUMMARY.txt                           — 메트릭 표
├── compare_breakdown.png                 — **baseline vs MESA side-by-side 스택 bar**
├── ar/run.log
├── baseline_k7_uniform/
│   ├── run.log
│   ├── mesa_profile_{draft,target_rank0}_*.json    — raw events
│   ├── mesa_breakdown.png                           — per-phase bar chart
│   ├── mesa_breakdown_over_time.png                 — 시계열
│   ├── mesa_timeline_step100.png                    — single-step Gantt
│   ├── mesa_breakdown_summary.csv
│   └── mesa_per_step_contribution.csv
├── baseline_k7_geo/ (동일 plot 세트)
├── mesa_k5_f4_dfo2_exit24/ (동일 plot 세트)
├── mesa_k5_f4_dfo2_exit28/ (동일 plot 세트)
└── mesa_k5_f4_dfo2_exit32/ (동일 plot 세트)
```

### 핵심 plot
- `compare_breakdown.png` — 4 config (baseline geo + MESA 3 exit)의 target/draft 스택 bar. 한눈에 MESA가 어디서 baseline보다 길어지는지 보여줌
- 각 subdir의 `mesa_timeline_step100.png` — step 100 단일 step Gantt (handshake 경계 정렬)
- 각 subdir의 `mesa_breakdown.png` — phase별 평균 ms bar

재생성:
```bash
python bench/plot_mesa_breakdown.py tmp/final_exp2/<config>
python bench/plot_mesa_timeline.py tmp/final_exp2/<config> --step 100 --warmup 0
python bench/plot_compare_breakdown.py tmp/final_exp2
```

## 다음 단계

사용자가 언급한 **"Phase 2가 Phase 1 sequence 이어서 depth 확장"** 구조 설계 및 구현.

현재 MESA의 Phase 2 replay × 5 = 20 ms가 draft step의 가장 큰 단일 기여.
Phase 2를 "Phase 1 tree의 leaf에서 K2 depth 더 확장"하는 구조로 바꾸면:
- Phase 2 replay 수 줄어듦 (K2 < K)
- 전체 tree depth K1 + K2 증가 → target이 verify할 토큰 수 증가 (per-step 개선)
- Draft step 단축 → throughput 직접 상승

이번 Rev1 마무리로 기반 마련 완료.

## Reproduce

```bash
# 전체 실행 (AR + baseline 2 + MESA 3)
bash tmp/final_exp2/run_all.sh

# MESA만 재실행 (baseline 결과 유지)
bash tmp/final_exp2/rerun_mesa.sh
```

환경:
- `SSD_HF_CACHE=/data2/chokwans99/models`
- `SSD_DATASET_DIR=/data2/chokwans99/datasets`
- `TORCH_CUDA_ARCH_LIST=8.6` (RTX 3090)
- 5 GPUs (0-4), B=1, 프로세스별 `SSD_DIST_PORT=12295`
