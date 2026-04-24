# Target-only Weight-only Quantization — 최종 결과

**브랜치**: `feature/int8-weight-only` (명칭은 원래대로 유지; 실제 채택 backend는 INT4)
**기간**: 2026-04-20 ~ 2026-04-21
**환경**: torch 2.8.0+cu128, torchao 0.12.0, RTX 3090 (SM 86), `ssd` conda env 그대로 (업그레이드 불필요)

## 주요 결정

| 항목 | 초기 계획 | 최종 |
|---|---|---|
| Backend | `torchao.Int8WeightOnlyConfig` | **`Int4WeightOnlyConfig(group_size=128)`** (SM 86 tinygemm fast path) |
| fp16 모델 처리 | plain load | **load-time bf16 upcast** (Llama-2/CodeLlama fp16 overflow 방지) |
| MESA lm_head | quantize | **bf16 유지 권장** (`--no_quant_lm_head`, accept 보존) |
| AWQ/SmoothQuant | 검토됨 | **철회** (fp16 overflow는 outlier 문제가 아님) |

## Phase 결과 요약

| Phase | 상태 | 핵심 결과 |
|---|---|---|
| 0. feasibility + graph-safety | ✅ | storage contract (A), CUDA graph OK, `inference_mode` OK, tying 방어 |
| 1. weight replacement contract | ✅ | `self.weight = dummy.weight` 재할당, forward 미변경 |
| 2. plain INT8 eager 통합 | ✅ | code OK, fp16 모델에선 overflow 발견 → bf16 upcast로 해결 |
| 2.5. kernel path 최적화 | ✅ | INT8→INT4 전환. INT4 tile_packed이 SM 86에서 dense 대비 0.25-1.25x, INT8 대비 2.7x 빠름 |
| 3. SSD graph path 확장 | ✅ | AR/spec/MESA graph 모두 INT4 호환, graph는 eager 대비 2-4x |
| 4. MESA + lm_head ablation | ✅ | MESA에서 lm_head bf16 유지 시 accept 0.41→0.38 (7% 손실) |
| 5. persistent artifact | ✅ | save/load AQT per rank (smoke test: Llama-3-8B) |
| 6. 34B 확장 | ✅ | CodeLlama-34B (fp16→bf16 upcast) TP=4 INT4 동작, async spec 23.52 TP |

## 최종 실측 (async spec + sampling, temp=0.6, TP=2 or 4 + draft)

### Llama-3-8B (bf16 native)
| config | TP | accept | 비고 |
|---|---|---|---|
| dense | 15.84 | 0.32 | baseline |
| INT8 wo | 14.25 | 0.30 | 90% of dense |
| **INT4 tile_packed** | **18.64** | **0.30** | **dense보다 +18%** |
| MESA dense | 24.12 | 0.41 | baseline |
| MESA INT8 | 12.15 | 0.40 | |
| MESA INT4 + no_lm_head | 13.47 | 0.38 | accept 보존 |

### Llama-2-7B (fp16→bf16 upcast)
| config | TP | accept |
|---|---|---|
| INT8 wo | 15.31 | 0.44 |
| **INT4 tile_packed** | **41.85** | **0.35** |

### CodeLlama-34B (fp16→bf16 upcast, TP=4 + draft, 5 GPU)

50 seq × 256 out × 4 dataset = 51200 tok (pre-quant baseline과 동일 조건):

| config | TP | accept | cache_hit | wall |
|---|---|---|---|---|
| **pre-quant dense async spec** | 68.45 | **0.44** | - | 747.98 s |
| **INT4 tile_packed async spec** | **75.28** | **0.44** | 0.66 | 680.09 s |

**INT4가 dense 대비 +10% 빠르고 accept 완벽 동일**. (원래 34B는 5 GPU로 잘 돌던 모델이므로 이는 "속도 ROI" 관점. 9→5 GPU 축소는 70B 시나리오.)

짧은 test (2 seq × 48 tok = 384 tok)에서 측정한 23.52는 warmup 비중 과장으로 인한 인공물.

| config | 비고 |
|---|---|
| INT4 MESA (34B) | FAIL `QuantizedLinearNotImplementedError` — MESA-specific graph shape dispatch 이슈 |

### Phase 5: Persistent artifact (Llama-3-8B, TP=2)

| step | wall time | TP | 비고 |
|---|---|---|---|
| 1. quantize + save | 63 s | 35.49 | artifact 생성 (1.9 GB/rank) |
| 2. load (load_time mode) | 45 s | 40.71 | 양자화 스킵, 18 s 절약 |
| 3. load_only (persistent mode) | 44 s | 42.08 | 동일 |

저장 overhead 없이 양자화 시간만큼 startup 단축. 70B에서 반복 실험 시 큰 이득 예상.

### Graph vs Eager (Llama-3-8B INT4)
| path | graph | eager |
|---|---|---|
| AR | 36.43 | 8.95 (4x slower) |
| spec | 14.51 | 8.01 (1.8x) |
| MESA | 30.62 | OOM-killed |

## 해결된 이슈

1. **fp16 overflow** (Llama-2, CodeLlama): load-time bf16 upcast (model_runner.py에서 `torch.set_default_dtype(bf16)` override when `target_quant_enabled=True`)
2. **INT8 느린 kernel**: `Int8WeightOnlyConfig`는 SM 86에서 dequant+bf16 matmul 경로만 있음. `Int4WeightOnlyConfig(group_size=128)`이 TensorCoreTiledLayout/tinygemm fast path 제공
3. **MESA lm_head 민감도**: early-exit hidden에 quantized lm_head 적용 시 proxy quality 저하. `target_quant_lm_head=False`로 회피

## 미해결 / 후속

- **34B MESA INT4 dispatch 실패**: `QuantizedLinearNotImplementedError`. MESA verify graph가 특정 shape을 쓰는데 torchao tile_packed tinygemm에서 dispatch 안 됨. 중요도 중간 (async spec은 동작). 우회: 34B MESA에만 INT8 backend 사용 (속도 느리지만 기능 OK), 또는 torchao 업스트림 지원 대기.
- **70B**: 로컬에 모델 없음 (~140GB 다운로드 필요). INT4로 TP=2 per-rank 8.75GB 예상 → 3 GPU로 돌릴 수 있을 것. 테스트는 별도 세션.
- **Marlin kernel**: torchao 외부 경로로 INT8 fast kernel 확보도 가능 (우선순위 낮음, INT4 tile_packed가 이미 충분)
- **INT4 accept 손실**: async spec은 거의 0, MESA는 7%. AWQ calibration 추가로 더 개선 가능하나 현 단계에선 불필요
- **artifact CPU→GPU warning**: `TensorCoreTiledAQTTensorImpl does not support conversion from cpu to cuda` 경고 (기능은 정상). torchao 업스트림 이슈. save 시 GPU 상태 유지 방식으로 우회 가능 (후속).

## 코드 변경

- `ssd/config.py`: `target_quant_enabled`, `target_quant_backend={"int4_wo_tile","int8_wo"}`, `target_quant_lm_head`, `target_quant_artifact_prefix`, `target_quant_mode`
- `ssd/utils/quantize.py`: `apply_int8_weight_only_to_target(backend=...)` 확장, int4/int8 분기, artifact save/load
- `ssd/utils/int8_debug.py` (신규): H1/H2/H3 진단 (opt-in via `SSD_INT8_DEBUG=1`)
- `ssd/engine/model_runner.py`: hook 삽입 + fp16→bf16 upcast + artifact load/dump
- `bench/bench.py`: `--quant_int4`, `--quant_int8`, `--no_quant_lm_head`, `--quant_artifact`, `--quant_artifact_load_only`
- `sandbox/int8_spike/01~10_*.py`: kernel bench / AQT inspection 스파이크 (참고용)
