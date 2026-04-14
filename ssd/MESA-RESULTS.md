# MESA-SSD 구현 결과 리포트

## 구현 상태

### 완료
1. Config 확장 (mesa_enabled, mesa_exit_layer, mesa_proxy_top_k, mesa_draft_fan_out + gating)
2. TreeLayout 추상화 (full/draft/proxy layout + backward compat)
3. Layout 일반화 (_decode_tree, _compute_step_positions, cudagraph_helpers)
4. LlamaModel split forward (start_layer/end_layer/init_hidden_states/init_residual)
5. Split CudaGraph target verify (capture_mesa_verify + run_mesa_verify)
6. Proxy 계산 + 전송 (_compute_and_send_proxy: accept_probs + residual top-k)
7. Proxy token swap (irecv → _build_tree_batch → proxy swap fork tokens → decode)
8. Attention layout-aware (context.active_mq_len/active_wrappers)
9. FlashInfer layout wrappers (prefill_wrappers_by_layout)
10. bench.py (--mesa, --mesa_exit_layer, --model_path, --draft_path)
11. Llama2 tokenizer fix (sentencepiece + use_fast fallback)

### 미완료 (IMPL_ISSUE.md 참조)
- ISSUE-003: Draft/proxy layout별 CudaGraph 캡처 hang → budget split 불가
- ISSUE-004: Proxy token swap이 cache hit rate 하락 유발
- ISSUE-005: Llama2-13B triton 에러 (SSD 자체 호환성 문제, non-power-of-2 관련)

---

## 실험 결과

### 환경
- GPU: RTX 3090 (24GB) x 2
- CUDA: 12.8, compute capability 8.6
- Settings: k=4, f=3, temp=0.6 (stochastic), random prompts, output_len=128, numseqs=3

### A. LayerSkip-Llama3-8B (target) + Llama-3.2-1B-Instruct (draft)

| 설정 | Throughput | Accept Rate | Cache Hit | Tok/Step (Hit) | Tok/Step (Miss) |
|------|-----------|-------------|-----------|----------------|-----------------|
| **Baseline SSD** | 87.61 tok/s | 0.58 | 0.69 | 3.74 | 2.39 |
| **MESA-SSD** (exit=21) | 56.51 tok/s | 0.53 | 0.47 | 3.93 | 2.44 |
| **차이** | **-35.5%** | -0.05 | **-0.22** | +0.19 | +0.05 |

### B. LayerSkip-Llama2-7B (target) + TinyLlama-1.1B (draft)

| 설정 | Throughput | Accept Rate | Cache Hit | Tok/Step (Hit) | Tok/Step (Miss) |
|------|-----------|-------------|-----------|----------------|-----------------|
| **Baseline SSD** | 80.59 tok/s | 0.60 | 0.58 | 3.82 | 2.79 |
| **MESA-SSD** (exit=21) | 51.11 tok/s | 0.53 | 0.35 | 3.73 | 2.77 |
| **차이** | **-36.6%** | -0.07 | **-0.23** | -0.09 | -0.02 |

### C. TP=2 동작 검증 (LayerSkip-Llama3-8B, 3 GPU)

| 설정 | 상태 |
|------|------|
| MESA-SSD TP=2 | **정상 동작** (데드락 없음, 두 TP rank 모두 split CudaGraph 캡처) |

### D. Split CudaGraph 오버헤드

| 측정 | 결과 |
|------|------|
| graph_pre + mid-forward + graph_post vs 단일 graph | **<0.1%** 차이 (측정 오차 수준) |

---

## 분석

### Throughput 하락 원인 (-35~37%)
1. **Proxy 대기 idle time (~10ms)**: Draft가 proxy를 기다리며 idle. Budget split 미구현 (ISSUE-003)으로 인해 draft decode를 proxy 도착 전에 시작할 수 없음.
2. **Cache hit rate 하락**: Proxy token swap이 기존 draft top-k와 다른 correction tokens를 cache에 넣어, 다음 step에서 cache key 매칭이 달라짐.

### Cache hit rate 하락 원인 (-0.22~0.23)
- Proxy의 residual `[p_E - p_D]_+` 기반 correction tokens이 실제 target의 recovery token과 반드시 일치하지 않음
- Early-exit proxy (layer 21/32 ≈ 66%)의 prediction quality가 final layer와 차이가 있음
- Cache key는 (seq_id, k_idx, **rec_token**)으로 구성되는데, proxy가 다른 rec_token을 예측하면 miss

### 긍정적 발견
- Split CudaGraph 오버헤드가 무시 가능 (~0.04ms)
- TP=2 정상 동작 (NCCL collective 정합)
- Proxy send/recv가 deadlock 없이 동작
- **Tok/Step on Hit (Llama3)**: 3.74 → 3.93 (+5%) — cache hit 시 proxy tokens가 더 나은 continuation 제공

---

## 미해결 이슈

| # | 이슈 | 영향 | 해결 방향 |
|---|------|------|----------|
| 003 | Layout별 CudaGraph hang | Budget split 불가, idle time 발생 | FlashInfer wrapper 바인딩 재설계 |
| 004 | Proxy swap이 cache hit 하락 | 전체 throughput 감소 | Exit layer 최적화, swap 비율 조절, proxy quality 향상 |
| 005 | Llama2-13B triton 에러 | 13B 테스트 불가 | SSD 자체 수정 필요 (non-MESA) |

## 변경된 파일

| 파일 | 변경 |
|------|------|
| `ssd/config.py` | MESA params + validation |
| `ssd/models/llama3.py` | Split forward |
| `ssd/engine/model_runner.py` | Split CudaGraph 캡처/실행 + layout wrappers + tokenizer fallback |
| `ssd/engine/verifier.py` | Proxy fn + _compute_and_send_proxy |
| `ssd/engine/draft_runner.py` | TreeLayout + irecv + proxy swap + _build_tree_batch_mesa |
| `ssd/engine/helpers/tree_layout.py` | **신규** TreeLayout dataclass |
| `ssd/engine/helpers/cudagraph_helpers.py` | capture/run_mesa_verify + layout param |
| `ssd/layers/attention.py` | Layout-aware wrapper selection |
| `ssd/utils/context.py` | active_mq_len/active_wrappers |
| `ssd/utils/async_helpers/async_spec_helpers.py` | mesa_proxy token swap |
| `bench/bench.py` | --mesa args + --model_path/--draft_path |
