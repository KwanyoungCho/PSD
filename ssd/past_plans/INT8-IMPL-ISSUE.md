# INT8 Weight-Only 구현 이슈 로그

브랜치: `feature/int8-weight-only`
계획 문서: `INT8-WEIGHT-ONLY-PLAN.md`

구현 중 발생하는 이슈, 결정, 해결 과정을 시간순으로 기록한다.
자체 해결된 항목은 [resolved], 사용자 판단 필요한 항목은 [open]으로 표기.

---

## 환경

- `torch`: 2.8.0+cu128
- `torchao`: **0.12.0** (0.17.0은 torch 2.8 cpp extension 비호환으로 경고 발생 → 0.12로 downgrade)
- `cuda`: 12.8
- `cudnn`: 91002
- GPU: RTX 3090 × 8 (24 GB each)
- conda env: `/home/chokwans99/anaconda3/envs/ssd`

---

## Phase 0

### [setup] torchao 0.17 → 0.12 downgrade
- torchao 0.17.0은 `torch>=2.11.0`에서만 cpp extension 로드. torch 2.8 환경에선 "Skipping import of cpp extensions" 경고 + 일부 최적 kernel 경로 비활성
- 0.12.0으로 내려서 해결. Int8WeightOnlyConfig, quantize_ 모두 정상 import 확인

### [resolved] Phase 0 모든 체크 통과 (sandbox/int8_spike/)

**01_dispatch_and_shapes.py** (checks 1-3):
- `quantize_()`는 `nn.Linear`만 변환 (SSDLikeLinear 미변환 확인)
- 저장 계약 (A): `dummy.weight → self.weight` 재할당 성공, `F.linear` 정상 dispatch
- 저장 계약 (B): `del _parameters["weight"]` + plain attribute 할당도 성공
- SSD-sized shards (qkv_packed 6144×4096, gate_up 11008×4096, o_proj row, down_proj row) 전부 성공
- cosine similarity 전체 ≥ 0.9999, dense 대비 max abs diff 작음

**02_graph_and_compile.py** (checks 4, 8):
- CUDA graph capture + 10회 replay: max diff 0.0 ✓
- `@torch.inference_mode()` 하에서 dispatch 정상, diff 0.0 ✓
- `@torch.compile` 모듈 + AQT linear 같은 forward 공존 정상 ✓
- **graph capture under inference_mode** (SSD 실제 패턴): diff 0.0 ✓
- 즉 가장 큰 리스크였던 graph + tensor subclass dispatch 완전 해결

**03_scale_tying_loader.py** (checks 5, 6, 7):
- scale shape: `(out_f,)` → per-output-channel (dim=0) 확정
- block_size `(1, 128)` — 각 output row 안에서 input axis를 128 chunk로 grouped (Int8WeightOnly default)
- **Column shard**: local scale = global scale의 row prefix (max diff 0) → TP 안전
- **Row shard**: local scale ≤ global scale 모든 row (mean ratio 0.9311) → Review B 실증: local quantize가 finer-grained이며 수치 정밀도 유리
- **tying 관찰**: `quantize_(lm_head)`는 attribute 교체 방식이라 원본 float 저장소를 수정하지 않음 → tied embed는 그대로 float 유지, F.embedding 정상. 단 메모리 공유 효과는 잃음
- **loader 순서**: float `param.data.copy_()` → quantize 순서 정상. AQT 상태에서 재로드 시 `ValueError: Not supported args for copy_` (loud failure, 안전)

### Phase 0 결정

- 저장 계약: **(A)** 채택 — `self.weight = dummy.weight`로 Param(AQT) 재할당. Forward 코드 변경 없음
- 양자화 API: **dummy `nn.Linear` + `quantize_(Int8WeightOnlyConfig())`** 경로 (내부 API 직접 호출 안 함)
- 교체 시점: float load 후, warmup 전
- tying 방어: 방어 분기를 두되 naive 케이스도 torchao가 자동 안전 (embed data ptr 유지)

---

## Phase 2

### [done] 통합 구현 완료

- `ssd/config.py`: `target_quant_enabled`, `target_quant_lm_head`, `target_quant_backend`, `target_quant_mode` 플래그 추가 (모두 기본 off)
- `ssd/utils/quantize.py`: `apply_quantization_to_target()` hook (이전 이름 `apply_int8_weight_only_to_target`) — Phase 0 계약 (A) 그대로 구현
  - dummy `nn.Linear` 생성 → `quantize_(Int8WeightOnlyConfig())` → `mod.weight = dummy.weight`
  - `tie_word_embeddings` 방어 (untie)
  - `target_quant_lm_head` flag
  - `SSD_INT8_SKIP` 환경변수로 module name substring 기반 exclude 가능
- `ssd/engine/model_runner.py`: `load_model(...)` 직후, `warmup_model()` 직전에 hook 삽입 (§3.5). `is_draft` 검사로 draft 완전 제외
- `bench/bench.py`: `--quant_int4` / `--quant_int8` 플래그. 이후 `--quant_force_bf16_runtime` 추가. `--no_quant_lm_head`는 **deprecated (no-op)** — config default가 `target_quant_lm_head=False`로 바뀌면서 더 이상 의미 없음. 반대로 `--quant_lm_head`로 opt-in

### [done] Dense flag-off 회귀 테스트 통과

- layerskip-llama2-7B TP=2 + TinyLlama draft, async spec K=7 geo
- 기존 (pre-quant): TP=52-53 → 변경 후 flag off: TP=59.09
- 유사 범위, run-to-run 변동 내. 기존 MESA/SSD 경로 변경 없음 확인.

### [done] INT8 path 동작하는 케이스

| config | 결과 | 비고 |
|---|---|---|
| AR (target only) + int8, layerskip-7B | TP=30.44 | 5 seqs × 48 tok OK |
| AR (target only) + int8, **codellama-34B** TP=4 | TP=9.43 | 16.7 GB → 8.4 GB per rank |
| Greedy spec (temp=0) + int8, 7B | TP=12.18 | 동작하지만 accept rate 낮아질 수 있음 |
| Greedy spec (temp=0) + int8, 34B TP=4 | TP=8.53 | **accept=0.03** (dense 대비 급락) |

### [correction 2026-04-21] 34B 기존 성능 비교 정정

이전 최종 보고에서 "CodeLlama-34B dense는 안 돌았고 INT4가 첫 성공"으로 잘못 적음. 실제로는:

- **34B TP=4 + draft (5 GPU)**은 원래 pre-quant 상태에서도 정상 동작함 (`tmp/final_exp2/baseline_k7_geo` 기록)
- 9 GPU 부족 시나리오는 **70B MESA+async**이지 34B가 아님

pre-quant 34B 실제 baseline (`tmp/final_exp2/baseline_k7_geo`, 50 seqs × 256 tok = 51200 tok):
- async spec dense: **TP=68.45, accept=0.44**
- MESA dense: **TP=58.48, accept=0.52**

내 INT4 짧은 test (2 seqs × 48 tok = 384 tok): TP=23.52 — **warmup 비중 과장되어 낮게 측정됨**. 34B는 load/warmup에 10+s 걸리는데 384 tok은 6-7s면 끝남.

공정 비교 long run (50 seq × 256 out × 4 dataset = 51200 tok, 동일 조건):

| | TP (tok/s) | accept | cache_hit | wall time |
|---|---|---|---|---|
| pre-quant dense | 68.45 | 0.44 | - | 747.98 s |
| **INT4 tile_packed** | **75.28** | **0.44** | 0.66 | 680.09 s |

**INT4가 dense 대비 +10% 빠르고 accept 완벽 동일 (0.44)**. 짧은 test의 23.52는 warmup 인공물.

### [completed 2026-04-21] Phase 4: MESA target verify + lm_head ablation

Phase 2.5 실측 데이터로 MESA 동작 + lm_head 영향 정량 확인 완료. 별도 Phase로 분리할 필요 없을 만큼 동일 setup에서 ablation 수행됨.

**결론**:
- MESA target verify가 INT4 target model 위에서 NaN/inf 없이 정상 동작
- `target_quant_lm_head` flag는 MESA에서 accept rate에 민감
- `lm_head=on` (quantize 포함): accept 0.41 → 0.33 (20% 손실)
- `lm_head=off` (bf16 유지): accept 0.41 → 0.38 (7% 손실, 허용 범위)

**MESA 실험 기본값 권장**: lm_head는 dense 유지. 현재 config default `target_quant_lm_head=False`가 이를 반영. (이전 `--no_quant_lm_head` 플래그는 deprecated no-op이며, 반대로 `--quant_lm_head`로 opt-in해야 양자화됨.) 메모리 이득은 lm_head 크기 만큼만 포기 (Llama-3-8B 기준 128256×4096×2=1GB 정도) — target weight 대부분(32 layer의 qkv/o/gate_up/down)은 여전히 int4 유지.

### [resolved 2026-04-21] Phase 2.5: INT8 kernel 느림 → INT4 tile_packed로 전환

torchao `Int8WeightOnlyConfig`는 SM 86에서 fused kernel이 없어 decode/verify에서 3x 느림 확인. `Int4WeightOnlyConfig(group_size=128)` (기본 `TensorCoreTiledLayout`, tinygemm 경로)가 **SM 86에서 실질 fast kernel**. 원래 `ssd` env (torch 2.8 + torchao 0.12) 그대로 사용 (업그레이드 불필요).

**Microbench (Llama-3-8B shapes, SM 86)**:

| shape | dense bf16 | int8 wo | **int4 tile_packed** |
|---|---|---|---|
| AR decode | 0.11ms | 0.36ms (3.3x↓) | **0.07ms (0.63x ↑)** |
| verify gate_up | 0.11ms | 0.36ms (3.3x↓) | **0.07ms (0.62x ↑)** |
| verify down_proj | 0.06ms | 0.19ms (3.2x↓) | 0.08ms (1.25x) |
| prefill gate_up | 0.62ms | 0.88ms (1.4x↓) | 1.84ms (3.0x↓) |

Prefill은 느리지만 one-shot/seq. Verify는 48×K+1 호출 → verify 가중치 큼.

**End-to-end (async spec + sampling, TP=2+draft)**:

| 모델 | backend | TP | accept |
|---|---|---|---|
| Llama-3-8B | dense | 15.84 | 0.32 |
| Llama-3-8B | INT8 wo | 14.25 | 0.30 |
| Llama-3-8B | **INT4 tile_packed** | **18.64** | **0.30** ✓ |
| Llama-2-7B (fp16→bf16 upcast) | INT8 wo | 15.31 | 0.44 |
| Llama-2-7B (upcast) | **INT4 tile_packed** | **41.85** | **0.35** ✓ |

**MESA 추가 발견**: MESA에서 INT4 `target_quant_lm_head=True`이면 accept 0.41 → 0.33 급락. 기본값 False (lm_head dense 유지) 시 **0.38 회복** — lm_head는 dense가 안전 (MESA early-exit logit 정밀도 이슈). 이 관찰이 config default를 False로 바꾼 근거.

| MESA config | TP | accept |
|---|---|---|
| dense | 24.12 | 0.41 |
| INT8 wo | 12.15 | 0.40 |
| INT4 (lm_head on) | 13.15 | 0.33 |
| **INT4 (lm_head off)** | **13.47** | **0.38** |

**구현 변경**:
- `target_quant_backend: str = "int4_wo_tile"` (기본값) / `"int8_wo"` (비교용)
- `--quant_int4` / `--quant_int8` CLI flag 분리
- `_quantize_weight_to_int4_wo()`: `Int4WeightOnlyConfig(group_size=128)` 사용
- MESA 사용 시 lm_head는 dense 유지 권장 (현재 config default `target_quant_lm_head=False`가 이미 이 동작. `--no_quant_lm_head` 플래그는 deprecated no-op)

---

### [root cause 재정정 2026-04-21] fp16 overflow 주장은 틀렸음 — torchao WO backend의 fp16 activation 미지원이 실체

이전 결론(`"fp16 overflow → bf16 upcast"`)은 **증거에 맞지 않는 오진단**이었다. 재측정 결과:

**반증 증거**:
- Llama-2-7B + **dense fp16** async spec: layer 1 hidden `shape=(921, 4096) dtype=fp16 nan=0 inf=0 **finite_absmax=1597**` — fp16 max 65504의 1/40. overflow 아님. 정상 완주 (TP=56.40)
- Llama-2-7B + **int4 WO + fp16 (no upcast)**: `ValueError: Expected Tensor argument zeros to have dtype torch.float16, but got torch.bfloat16` — torchao API level dtype assert
- Llama-2-7B + **int8 WO + fp16 (no upcast)**: layer 1 hidden `nan=0 **inf=22** finite_absmax=440.75` — 22개 inf + 유한부는 오히려 dense보다 작음. 수치 불안정

**실체 (정정)**:
현재 선택한 torchao weight-only 경로 (`Int4WeightOnlyConfig`, `Int8WeightOnlyConfig`)는 공식 문서가 **bf16 activation workflow**로 명시하는 backend임 (https://docs.pytorch.org/ao/stable/workflows/inference.html). fp16 activation은:
- Int4: scale/zero를 bf16으로 고정 생성 → fp16 activation과 matmul kernel 레벨 dtype assert fail
- Int8: API는 통과하지만 수치적으로 불안정 (원인 세부는 fp16 accum 가설 등 있으나 **미확정**)

→ torchao 전체가 fp16 미지원은 아니고, 다른 backend (예: `GemliteUIntXWeightOnlyConfig`는 fp16-only로 문서화됨, Marlin 등)는 fp16 native. 단 **우리가 선택한 backend는 fp16 runtime을 신뢰할 수 없음**.

### 지원 매트릭스 (정정)

| checkpoint dtype | 선택 backend | 현재 상태 |
|---|---|---|
| bf16 (Llama-3 family) | int4_wo_tile / int8_wo | **정상 지원** |
| fp16 (Llama-2, CodeLlama) | int4_wo_tile / int8_wo | **미지원** — 기본 동작은 `ValueError`, 명시적 `target_quant_force_bf16_runtime=True` opt-in 시 bf16 runtime으로 우회 (이 경우 "fp16 runtime"이라는 계약은 깨진다) |
| fp16 | GemliteUIntXWeightOnly / Marlin | **미통합** (별도 과제) |

### 정책 변경 (2026-04-21)

1. fp16 checkpoint + 현재 backend 조합은 **기본적으로 loud ValueError** (`model_runner.py`). 사용자가 명시적으로 `target_quant_force_bf16_runtime=True` 설정할 때만 bf16 runtime 우회 경로 허용
2. bf16 runtime override는 "fp16 체크포인트를 bf16 런타임에서 돌린다"는 workaround임을 로그/문서에 명시 — "fp16 runtime 지원"과는 다르다
3. Artifact schema v2: `effective_runtime_dtype` + `original_checkpoint_dtype` 저장/검증. fp16-checkpoint-but-bf16-runtime artifact와 bf16-native artifact를 구분

### 이전 실측 수치 (backend = int4, runtime = bf16, 여전히 유효)

| 모델 | 경로 | TP | accept |
|---|---|---|---|
| Llama-3-8B (bf16 native) | async spec int4 sampling | 18.64 | 0.30 |
| Llama-3-8B (bf16 native) | MESA int4 (no lm_head) long | 58.96 | 0.52 |
| Llama-2-7B (fp16 → bf16 override) | async spec int4 sampling | 41.85 | 0.35 |
| CodeLlama-34B (fp16 → bf16 override) long | async spec int4 sampling | 75.28 | 0.44 |

수치 자체는 여전히 valid. 다만 **"fp16 checkpoint가 fp16 runtime으로 도는 건 아님"**을 명확히 밝혀야 함. Llama-2/CodeLlama 수치는 모두 **bf16 runtime 우회** 조건이다.

### [archived 2026-04-21] 초기 오진단 기록

**오진단 경로 1**: "Llama outlier activation → bf16 overflow → AWQ 필요"
- 틀림. 대부분 모델 dtype mismatch 방향을 잘못 파악

**오진단 경로 2**: "fp16 모델이 원래 overflow 경계에 있음 → quant가 tip over"
- 틀림. dense fp16 absmax=1597로 65504와 거리 멂. quant에서만 inf 나오는 이유는 torchao backend 자체의 fp16 activation 제약

양쪽 다 **현재 torchao `Int4WeightOnlyConfig` / `Int8WeightOnlyConfig`의 bf16-activation 전제**라는 동일한 실체를 서로 다르게 오해한 결과.

### 진단 sandbox 로그 (당시)

- `tmp/int8_smoke/dbg_full/` — 층별 inf 탐지 결과
- `tmp/int8_smoke/dbg_fine/` — layer 내부 sub-step NaN 전파 추적
- `tmp/int8_smoke/34b_ar_int8/` — 34B AR 정상 동작 확인
- `tmp/int8_smoke/34b_spec_int8*/` — 34B spec 실패 재현 (fp16 원본 + int8, upcast 전)
