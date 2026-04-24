# MESA-SSD 구현 이슈 트래커

MESA-IMPL-PLAN.md 계획 대비 달라진 부분과 발견된 이슈를 기록.

---

## 해결된 이슈

### ISSUE-001: attention.py + context.py layout-aware — **해결**
context.py에 `active_mq_len`/`active_wrappers` 추가. attention.py의 tree_decode 분기에서 context 기반 wrapper selection 구현. 계획 3단계에서 예정했으나 7단계 구현 시 함께 적용.

### ISSUE-002: Budget split 2-pass → **해결**
초기에는 proxy token swap (full_layout 단일 decode)으로 우회했으나, ISSUE-003 해결 후 계획대로 draft_layout + proxy_layout 2-pass decode로 구현 완료.

### ISSUE-003: Layout CudaGraph 캡처 hang — **해결**
4가지 원인을 모두 수정:
1. `capture_fi_tree_decode_cudagraph`에서 `prefill_wrappers[bs]` → layout별 wrapper 선택
2. `set_context`에 `active_mq_len`/`active_wrappers` 전달 (캡처 시에도)
3. `outputs`/`logits` 텐서의 dtype 미명시 → `dtype=hf_config.torch_dtype` 추가
4. 모듈 전역 `cache` dict가 layout 간 공유 → `cache.clear()` on MQ_LEN change

---

## 미해결 이슈

### ISSUE-004: Throughput 하락 (-43%)
**실측 결과** (비어있는 GPU 기준, 10 seqs × 256 tokens):
- Baseline: 149 tok/s, MESA: 84 tok/s (-43%)

**주 원인**: 2-pass tree decode의 구조적 비용:
1. 2× CudaGraph replay (K steps × 2 passes): 각 replay마다 고정 overhead
2. 2× FlashInfer wrapper.plan() (K steps × 2 passes): batch-dependent KV metadata 재구성
3. 2× batch-dependent packed mask 재구성: context_lens, block_tables, cache_hits에 의존하여 매 pass 필수
4. `_build_tree_batch()` full layout tree args 중복 생성 (~2ms)

**이전 분석 정정**: mask cache clear가 주 원인이라고 했으나, layout-independent한 것은 `glue_hit_np`/`glue_miss_np` (~0.1ms)뿐. 나머지 precompute는 batch 의존이라 layout 분리로 절약 불가.

### ISSUE-005: Llama2-13B/70B triton 에러
SSD KV cache copy 커널에서 `tl.arange(0, D)` — D가 non-power-of-2이면 에러. Llama2-13B (hidden=5120, heads=40)이 TP 분할 후 해당. MESA와 무관한 SSD 자체 이슈. Llama2-7B (hidden=4096, heads=32)로 대체.

---

## 계획 대비 달라진 점

### 1. Token swap → 2-pass (계획 실현)
계획은 처음부터 2-pass였으나 구현 과정에서 ISSUE-003으로 인해 일시적으로 token swap (full_layout 단일 decode)으로 우회. ISSUE-003 해결 후 계획대로 2-pass 구현.

### 2. irecv 위치
계획: `_build_tree_batch` 내부에서 glue decode와 fork token 사이에 irecv.
실제: `_build_tree_batch_mesa` 시작 시 irecv를 가장 먼저 post (line 1030). Target send와의 overlap 최대화.
**결과**: proxy_wait = 0.0ms (완벽한 overlap).

### 3. _build_tree_batch 리팩토링 불완전
계획: glue decode 로직을 분리하여 MESA 전용 경로로 재사용.
실제: `_build_tree_batch`를 그대로 호출하고 `_mesa_glue_logits`/`_mesa_gd_for_fork`를 tree_decode_args에 추가하여 반환. Full layout tree_decode_args가 불필요하게 구축됨 (~2ms 낭비).

### 4. Global cache 문제 발견 및 해결
계획에 없던 이슈: `cudagraph_helpers.py`의 `cache = {}` (모듈 전역)가 draft/proxy pass 간 공유. Layout 변경 시 `cache.clear()` 추가로 해결.

### 5. FlashInfer wrapper 초기화 순서
계획: TreeLayout 생성 후 wrapper 생성.
실제: `_init_flashinfer_wrappers()`가 `ModelRunner.__init__()` 안에서 호출되므로, config 값 (`mesa_draft_fan_out`, `mesa_proxy_fan_out`)으로 직접 MQ_LEN 계산하여 wrapper 생성. Layout 객체는 이후 `_init_prealloc_buffers()`에서 생성.

### 6. run_model() tree decode dispatch
계획: `_decode_tree`에 layout을 전달하면 자동으로 올바른 CudaGraph 사용.
실제: `_decode_tree` → `_decode_tree_step` → `set_context(active_mq_len=...)` → `run_model` → context에서 layout 읽어 `graph_vars[layout.graph_key]` dispatch. 간접적이지만 동작.

### 7. Llama2 지원
계획: Llama-only assert 후 Llama2 자동 지원.
실제: sentencepiece 설치 필요 + `use_fast=False` fallback 추가 + 13B는 triton 에러로 7B 사용.

### 8. mesa_budget_mode 파라미터 미구현
계획: `token_swap` / `h_redistribute` / `outcome_posterior` 모드.
실제: 2-pass budget split만 구현. 모드 선택 파라미터 없음. 향후 추가.

### 9. _select_proxy_sourced_tokens Python loop
계획: vectorized 구현.
실제: `for b in range(B): for pos in range(K):` Python loop. B=1에서는 무시 가능 (~0.9ms).

### 10. Hot path overhead 수정
`mesa_enabled=False`일 때 `_decode_tree_step`과 `run_model` tree decode 분기에서 불필요한 layout 체크 코드가 매 step 실행됨. `if self.config.mesa_enabled:` 가드를 추가하여 기존 경로 보호.

### 11. 타이밍 코드 sync overhead
`_build_tree_batch_mesa`의 `torch.cuda.synchronize()` 8개가 GPU 파이프라인을 깨뜨림. 제거 완료.

---

## Rev1 변경사항

### Rev1-1: Glue decode 분리
`_glue_decode()` 함수 추출. `_build_tree_batch_mesa`가 full tree args 구축 없이 glue decode만 호출.

### Rev1-2: Target 측 h_i + fan_out_list 계산
verifier의 `_compute_and_send_proxy`에서 accept_probs → h_i → fan_out_list 계산 후 전송. Draft에서 h_i 계산 불필요.
전송 포맷: `[fan_out_list(K+1), topk_ids(B*K*top_k), topk_probs(B*K*top_k)]`

### Rev1-3: Policy A 동적 fan_out
- `_select_proxy_sourced_tokens_policy_a()`: fan_out_list 기반 position별 가변 token 수
- Runtime TreeLayout 생성: `create_tree_layout(fan_out_list=...)`
- Context.active_layout으로 동적 layout 전달
- `_merge_and_populate_cache`에 proxy_layout 파라미터 추가

### Rev1 실험 결과
- Llama3-8B: Policy A가 v1(고정)보다 약간 하락 (accept 0.83→0.79)
- Llama2-7B: Cache hit 0.61→0.80 (+31%), accept 0.58→0.61 (+5%)
- Throughput: 두 모델 모두 -29~44% (2-pass 구조적 비용, v1과 동일)
