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

### ISSUE-004: Throughput 하락 (-24~32%)
**실측 원인 분석** (타이밍 계측 기반):
- Draft step: baseline 33ms → MESA 54.5ms (+65%)
- 구성: glue+select 5.5ms + draft_decode 18.8ms + proxy_wait 0.0ms + proxy_select 0.9ms + proxy_decode 19.1ms
- 2× decode 합(38ms) > 1× full decode(28ms) → **CudaGraph replay당 고정 오버헤드 ~9ms**
- Target step: 65ms (baseline과 동일, split CudaGraph 오버헤드 무시)
- 이론적으로 max(65, 54.5) = 65ms → throughput 동일해야 하지만, step 간 sync/첫 step recompile/cache clear로 추가 지연

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
