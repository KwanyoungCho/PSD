# MESA-SSD 구현 결과 리포트 (v2 — 2-pass budget split)

## 구현 상태: 모든 핵심 이슈 해결

- ISSUE-001: attention/context layout-aware → **해결**
- ISSUE-002: Budget split 2-pass tree decode → **해결** (proxy token swap → 실제 2-pass)
- ISSUE-003: Layout CudaGraph 캡처 hang → **해결** (dtype fix + wrapper 바인딩 + cache 분리)
- ISSUE-005: Llama2-13B triton 에러 → Llama2-7B로 대체 (SSD 자체 이슈)

---

## 실험 결과

### 환경
- GPU: RTX 3090 (24GB) × 2
- CUDA: 12.8, compute capability 8.6
- Settings: k=4, f=3, temp=0.6 (stochastic), random prompts, output_len=128, numseqs=5

### A. LayerSkip-Llama3-8B (target) + Llama-3.2-1B-Instruct (draft)

| 메트릭 | Baseline SSD | MESA 2-pass | 차이 |
|--------|-------------|-------------|------|
| **Throughput** | 111.16 tok/s | 75.22 tok/s | -32.3% |
| **Accept Rate** | 0.74 | **0.80** | **+8.1%** |
| **Cache Hit** | 0.77 | **0.78** | +1.3% |
| **Tok/Step** | 3.98 | **4.22** | **+6.0%** |
| **Tok/Step (Hit)** | 4.68 | 4.69 | ~동일 |
| **Tok/Step (Miss)** | 1.66 | **2.48** | **+49.4%** |

### B. LayerSkip-Llama2-7B (target) + TinyLlama-1.1B (draft)

| 메트릭 | Baseline SSD | MESA 2-pass | 차이 |
|--------|-------------|-------------|------|
| **Throughput** | 80.74 tok/s | 61.71 tok/s | -23.6% |
| **Accept Rate** | 0.57 | **0.73** | **+28.1%** |
| **Cache Hit** | 0.56 | **0.79** | **+41.1%** |
| **Tok/Step** | 3.27 | **3.90** | **+19.3%** |
| **Tok/Step (Hit)** | 3.79 | **4.43** | **+16.9%** |
| **Tok/Step (Miss)** | 2.60 | 1.91 | -26.5% |

---

## 핵심 분석

### 긍정적 결과
1. **Accept rate 대폭 개선**: Llama3 +8%, Llama2 +28%. Early-exit proxy가 correction token 예측에 효과적.
2. **Cache hit rate 개선**: Llama2에서 0.56 → 0.79 (+41%). 2-pass budget split으로 draft+proxy tokens가 함께 cache 커버리지 확대.
3. **Tok/Step 개선**: Llama2에서 3.27 → 3.90 (+19%). Step당 더 많은 토큰 생성.
4. **Split CudaGraph 오버헤드 무시 가능** (<0.1%).

### 남은 과제: Throughput 하락
- Llama3: -32%, Llama2: -24%
- **원인**: Proxy 대기 idle time (~10ms per step). Draft가 proxy를 기다리며 idle.
- **해결 방향**: irecv를 draft decode 시작 전에 걸어 target send와 병렬화 (이미 구현됨). 실제 idle time은 `graph_pre` 시간 - draft glue decode 시간에 의존.

### Throughput vs Token Efficiency 트레이드오프
MESA는 step당 더 많은 토큰을 생성하지만 (Tok/Step +6~19%), step 자체가 느려짐 (proxy 대기). Budget split 최적화(idle time 제거)가 완성되면 throughput 하락이 크게 줄어들 전망.

---

## 변경된 파일

| 파일 | 변경 |
|------|------|
| `ssd/config.py` | MESA params + validation |
| `ssd/models/llama3.py` | Split forward |
| `ssd/engine/model_runner.py` | Split CudaGraph + layout wrappers + layout-aware tree decode dispatch |
| `ssd/engine/verifier.py` | Proxy fn + _compute_and_send_proxy |
| `ssd/engine/draft_runner.py` | TreeLayout + 2-pass decode + irecv + token selection + merge cache |
| `ssd/engine/helpers/tree_layout.py` | **신규** TreeLayout dataclass |
| `ssd/engine/helpers/cudagraph_helpers.py` | capture/run mesa_verify + layout-aware fi_tree_decode + cache invalidation |
| `ssd/layers/attention.py` | Layout-aware wrapper selection |
| `ssd/utils/context.py` | active_mq_len/active_wrappers |
| `ssd/utils/async_helpers/async_spec_helpers.py` | mesa_proxy token swap |
| `bench/bench.py` | --mesa args + --model_path/--draft_path |
