# MESA-SSD Rev1 (Policy A) 구현 결과

## Rev1 변경 사항
1. **Glue decode 분리**: `_glue_decode()` 추출 → ~2ms/step 절약
2. **tolist() 최적화**: GPU sync B×K×3회 → 3회로 감소
3. **Target 측 h_i + fan_out_list 계산**: Draft critical path에서 제거
4. **Runtime TreeLayout**: 동적 fan_out_list로 매 step TreeLayout 생성
5. **Policy A token selection**: h_i 기반 position별 proxy budget 배분
6. **Runtime layout 전달**: Context.active_layout → run_model → mask/plan 반영
7. **Cache key 수정**: _merge_and_populate_cache에 runtime proxy_layout 전달

## 추가 Guard
- `assert jit_speculate` (MESA 사용 시 강제)
- bench.py에서 `--mesa` → `jit_speculate=True` 자동 설정

---

## 실험 결과

### 환경
- GPU: RTX 3090 (24GB) × 2 (비어있는 GPU 사용)
- Settings: k=4, f=3, temp=0.6, random prompts, 10 seqs × 256 tokens = 2560 total

### A. LayerSkip-Llama3-8B + Llama-3.2-1B

| 메트릭 | Baseline SSD | MESA v1 (고정) | MESA Rev1 (Policy A) |
|--------|-------------|----------------|---------------------|
| **Throughput** | 151.20 tok/s | 84.31 tok/s (-44%) | 84.00 tok/s (-44%) |
| **Accept Rate** | 0.87 | 0.83 | 0.79 |
| **Cache Hit** | 0.90 | 0.87 | 0.88 |
| **Tok/Step** | 4.47 | 4.31 | 4.15 |
| **Tok/Step (Hit)** | 4.72 | 4.57 | 4.42 |
| **Tok/Step (Miss)** | 2.14 | 2.56 | 2.19 |

### B. LayerSkip-Llama2-7B + TinyLlama-1.1B

| 메트릭 | Baseline SSD | MESA Rev1 (Policy A) |
|--------|-------------|---------------------|
| **Throughput** | 97.03 tok/s | 68.85 tok/s (-29%) |
| **Accept Rate** | 0.58 | 0.61 |
| **Cache Hit** | 0.61 | **0.80** (+31%) |
| **Tok/Step** | 3.31 | **3.42** (+3%) |
| **Tok/Step (Hit)** | 3.91 | 3.71 |
| **Tok/Step (Miss)** | 2.34 | 2.32 |

---

## 분석

### Llama3-8B

Policy A가 v1(고정 fan_out)보다 약간 떨어짐:
- Accept rate: 0.83 → 0.79 (-4.8%)
- Tok/Step: 4.31 → 4.15 (-3.7%)

**원인 추정**: h_i 기반 budget 배분이 특정 position에 과도하게 집중 → 다른 position coverage 감소. Llama3-8B에서는 early-exit proxy의 h_i 예측이 실제 reject 패턴과 완벽히 일치하지 않을 수 있음.

### Llama2-7B

Policy A가 baseline보다 cache hit 크게 개선:
- Cache hit: 0.61 → **0.80** (+31%)
- Accept rate: 0.58 → **0.61** (+5%)
- Tok/Step: 3.31 → **3.42** (+3%)

**분석**: LayerSkip-Llama2-7B의 early-exit이 reject 위치를 더 정확히 예측 → h_i 기반 budget 배분 효과적.

### Throughput

두 모델 모두 throughput은 2-pass 구조적 비용으로 -29~44% 하락. Policy A 자체의 추가 overhead(runtime layout 생성 + h_i 기반 token selection)는 무시 가능 수준 (v1의 84.31 vs Rev1의 84.00).

---

## IMPL_ISSUE 업데이트

Rev1 구현 중 발견된 이슈:
- `_select_proxy_sourced_tokens`에서 `mesa_proxy["accept_probs"].shape` 참조 → `fan_out_list`로 변경 후 제거 필요. B, K를 logits.shape과 config에서 직접 획득으로 수정.
- Runtime TreeLayout의 `graph_key`가 "fi_tree_decode_proxy"인데, CudaGraph 선택은 정적 proxy graph를 사용 (MQ_LEN 동일). `active_layout.graph_key`로 정상 dispatch 확인.
