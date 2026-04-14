# MESA-SSD 구현 이슈 트래커

구현 중 MESA-IMPL-PLAN.md 계획과 달라지는 부분을 기록.

---

### ISSUE-001: attention.py + context.py layout-aware 변경 미완료
계획 3단계에서 attention.py의 `mq_len = self.F * (self.K+1)` 변경과 context.py의 `active_mq_len`/`active_wrappers` 추가가 필요하지만, 실제 draft/proxy layout별 CudaGraph 캡처 + wrapper 생성이 선행되어야 함. 현재 7단계에서 full_layout 단일 decode로 MESA를 구현하고, 이 이슈는 budget split 최적화 시 함께 해결.

### ISSUE-002: Budget split → proxy token swap으로 변경
계획의 2-pass tree decode (draft_layout → proxy_layout)는 CudaGraph 캡처/FlashInfer wrapper를 layout별로 3세트 관리해야 하고, 캡처 시 hang이 발생 (DraftRunner 초기화 순서 문제 + wrapper 바인딩). 
**실용적 대안**: full_layout 단일 decode를 유지하되, proxy 수신 후 fork tokens를 proxy 기반으로 교체. Budget split (idle time 제거)은 후속 최적화.
- Draft idle time ~10ms 발생 (proxy 대기)
- 하지만 proxy-sourced correction tokens의 cache hit 개선 효과 측정 가능

### ISSUE-003: Draft/proxy layout별 CudaGraph 캡처 hang
DraftRunner.__init__에서 layout CudaGraph 캡처 시 hang 발생. 원인: capture_fi_tree_decode_cudagraph 내부에서 FlashInfer wrapper 접근 시 full layout wrapper와 draft/proxy layout MQ_LEN 불일치. 해결: layout별 wrapper를 캡처 전에 active하게 바인딩해야 함. Budget split 최적화 시 해결.

### ISSUE-004: MESA proxy token swap이 throughput과 cache hit rate 모두 하락시킴
v1 실험 결과:
- **Throughput**: 87.61 → 56.51 tok/s (-35%) — proxy 대기 시간 (~10ms idle)이 주 원인
- **Cache hit rate**: 0.69 → 0.47 — proxy가 기존 draft top-k와 다른 correction tokens를 cache에 넣기 때문에, 다음 step에서 cache key (seq_id, k_idx, rec_token) 매칭이 달라짐
- **근본 원인 분석**: 현재 proxy swap은 모든 position에서 draft top-k를 proxy correction으로 교체하지만, correction token이 실제 recovery에서 선택될 확률이 draft top-k보다 반드시 높지 않을 수 있음. Early-exit proxy의 quality가 중요한 factor.
- **해결 방향**: (1) Budget split으로 idle time 제거, (2) proxy swap 비율 조절 (일부만 교체), (3) exit layer 최적화, (4) early-exit proxy quality 향상

### ISSUE-005: LayerSkip-Llama2-13B triton 컴파일 에러
SSD의 KV cache copy triton 커널에서 `tl.arange(0, D)` 호출 시 D가 non-power-of-2이면 에러. Llama2-13B의 hidden_size=5120, num_heads=40, head_dim=128은 power of 2이지만, KV cache 커널의 D 계산이 TP 분할 후 non-power-of-2가 되는 것으로 추정. 이건 SSD 자체의 Llama2 호환성 문제이며 MESA와 무관. `--eager` 모드에서도 동일 에러 발생.
