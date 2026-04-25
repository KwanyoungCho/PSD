# 양자화 구현 이슈 트래커 (v1 + v2 통합)

이 문서는 양자화 통합 작업 중 발생한 이슈, 결정, 해결 과정을 시간순으로
정리한다. v1 (torchao) 단계와 v2 (AWQ Marlin) 단계가 모두 들어있다.

---

## v1 환경

- `torch`: 2.8.0+cu128
- `torchao`: **0.12.0** (0.17.0 은 torch 2.8 cpp extension 비호환 → 0.12 로 downgrade)
- `cuda`: 12.8, `cudnn`: 91002
- GPU: RTX 3090 × 8 (24 GB each), conda env `ssd`

### [setup] torchao 0.17 → 0.12 downgrade

- torchao 0.17.0 은 `torch>=2.11.0` 에서만 cpp extension 로드. torch 2.8
  환경에선 "Skipping import of cpp extensions" 경고 + 일부 최적 kernel 경로
  비활성
- 0.12.0 으로 내려서 해결. `Int8WeightOnlyConfig`, `quantize_` 모두 정상
  import 확인

---

## v1 Phase 0 — feasibility / graph-safety

### [resolved] Phase 0 모든 체크 통과 (sandbox/int8_spike/)

**01_dispatch_and_shapes.py** (checks 1-3):
- `quantize_()` 는 `nn.Linear` 만 변환 (SSDLikeLinear 미변환 확인)
- 저장 계약 (A): `dummy.weight → self.weight` 재할당, `F.linear` 정상 dispatch
- 저장 계약 (B): `del _parameters["weight"]` + plain attribute 할당도 성공
- SSD-sized shards (qkv_packed 6144×4096, gate_up 11008×4096, o_proj row,
  down_proj row) 전부 성공
- cosine similarity 전체 ≥ 0.9999

**02_graph_and_compile.py** (checks 4, 8):
- CUDA graph capture + 10회 replay: max diff 0.0 ✓
- `@torch.inference_mode()` 하에서 dispatch 정상, diff 0.0 ✓
- `@torch.compile` 모듈 + AQT linear 같은 forward 공존 정상 ✓
- **graph capture under inference_mode** (SSD 실제 패턴): diff 0.0 ✓
- 가장 큰 리스크였던 graph + tensor subclass dispatch 완전 해결

**03_scale_tying_loader.py** (checks 5, 6, 7):
- scale shape: `(out_f,)` per-output-channel (dim=0)
- block_size `(1, 128)` — 각 output row 안에서 input axis 를 128 chunk 로
  grouped (Int8WeightOnly default)
- **Column shard**: local scale = global scale 의 row prefix (max diff 0)
  → TP 안전
- **Row shard**: local scale ≤ global scale 모든 row (mean ratio 0.9311)
  → local quantize 가 finer-grained 이며 수치 정밀도 유리
- **tying 관찰**: `quantize_(lm_head)` 는 attribute 교체 방식이라 원본 float
  저장소 미수정 → tied embed 그대로 float 유지, `F.embedding` 정상. 단
  메모리 공유 효과는 잃음
- **loader 순서**: float `param.data.copy_()` → quantize 순서 정상. AQT
  상태에서 재로드 시 `ValueError: Not supported args for copy_` (loud
  failure, 안전)

### Phase 0 결정

- 저장 계약: **(A)** 채택 — `self.weight = dummy.weight` 로 Param(AQT)
  재할당. Forward 코드 변경 없음
- 양자화 API: **dummy `nn.Linear` + `quantize_(Int8WeightOnlyConfig())`**
  경로 (내부 API 직접 호출 안 함)
- 교체 시점: float load 후, warmup 전
- tying 방어: 방어 분기는 두되 naive 케이스도 torchao 가 자동 안전

---

## v1 Phase 2 — eager 통합

### [done] 통합 구현 완료

- `ssd/config.py`: `target_quant_enabled`, `target_quant_lm_head`,
  `target_quant_backend`, `target_quant_mode` 플래그 추가 (모두 기본 off)
- `ssd/utils/quantize.py`: `apply_quantization_to_target()` hook
  - dummy `nn.Linear` 생성 → `quantize_(Int8WeightOnlyConfig())` →
    `mod.weight = dummy.weight`
  - `tie_word_embeddings` 방어 (untie)
  - `target_quant_lm_head` flag
  - `SSD_INT8_SKIP` 환경변수로 module name substring 기반 exclude
- `ssd/engine/model_runner.py`: `load_model(...)` 직후, `warmup_model()`
  직전에 hook 삽입. `is_draft` 검사로 draft 완전 제외
- `bench/bench.py`: `--quant_int4` / `--quant_int8` 플래그.
  `--quant_force_bf16_runtime` 추가. `--no_quant_lm_head` 는 deprecated
  (no-op) — config default 가 `target_quant_lm_head=False` 로 바뀌면서
  의미 없음. 반대로 `--quant_lm_head` 로 opt-in

### [done] Dense flag-off 회귀 테스트 통과

layerskip-llama2-7B TP=2 + TinyLlama draft, async spec K=7 geo:
- 기존 (pre-quant): TP=52-53 → 변경 후 flag off: TP=59.09
- 유사 범위, run-to-run 변동 내. 기존 MESA/SSD 경로 변경 없음 확인.

### [done] INT8 path 동작 케이스

| config | 결과 | 비고 |
|---|---|---|
| AR (target only) + int8, layerskip-7B | TP=30.44 | OK |
| AR + int8, codellama-34B TP=4 | TP=9.43 | 16.7 GB → 8.4 GB per rank |
| Greedy spec + int8, 7B | TP=12.18 | 동작하지만 accept rate 낮아질 수 있음 |
| Greedy spec + int8, 34B TP=4 | TP=8.53 | **accept=0.03** (dense 대비 급락) |

### [correction 2026-04-21] 34B 기존 성능 비교 정정

이전 보고에서 "CodeLlama-34B dense 는 안 돌았고 INT4 가 첫 성공" 이라
잘못 기록. 실제로는:
- 34B TP=4 + draft (5 GPU) 는 pre-quant 상태에서도 정상 동작
- 9 GPU 부족 시나리오는 70B MESA+async, 34B 가 아님

pre-quant 34B 실제 baseline (`tmp/final_exp2/baseline_k7_geo`, 50 seqs ×
256 tok = 51200 tok):
- async spec dense: TP=68.45, accept=0.44
- MESA dense: TP=58.48, accept=0.52

내 INT4 짧은 test (2 seqs × 48 tok) 에서 측정한 23.52 는 warmup 인공물.
공정 비교 long run:

| | TP | accept | wall |
|---|---|---|---|
| pre-quant dense | 68.45 | 0.44 | 747.98 s |
| **INT4 tile_packed** | **75.28** | **0.44** | 680.09 s |

**INT4 가 dense 대비 +10% 빠르고 accept 완벽 동일 (0.44).**

### [completed] Phase 4 — MESA target verify + lm_head ablation

MESA target verify 가 INT4 target model 위에서 NaN/inf 없이 정상 동작.

`target_quant_lm_head` flag 영향:
- `lm_head=on`: accept 0.41 → 0.33 (20% 손실)
- `lm_head=off`: accept 0.41 → 0.38 (7% 손실, 허용 범위)

**MESA 권장 default**: lm_head dense 유지. 현재 config default
`target_quant_lm_head=False` 가 이 동작.

### [resolved] Phase 2.5 — INT8 kernel 느림 → INT4 tile_packed 전환

torchao `Int8WeightOnlyConfig` 는 SM 86 에서 fused kernel 부재 →
decode/verify 에서 3× 느림. `Int4WeightOnlyConfig(group_size=128)` (기본
TensorCoreTiledLayout, tinygemm 경로) 가 SM 86 fast kernel.

**Microbench (Llama-3-8B shapes, SM 86)**:

| shape | dense bf16 | int8 wo | int4 tile_packed |
|---|---|---|---|
| AR decode | 0.11ms | 0.36ms (3.3×↓) | **0.07ms (1.6× ↑)** |
| verify gate_up | 0.11ms | 0.36ms | **0.07ms** |
| verify down_proj | 0.06ms | 0.19ms | 0.08ms (1.25× 느림) |
| prefill gate_up | 0.62ms | 0.88ms | 1.84ms (3× 느림) |

Prefill 은 느리지만 one-shot. Verify 는 매 step → INT4 가 큰 이득.

**MESA 추가 발견**: MESA 에서 INT4 + `target_quant_lm_head=True` 면
accept 0.41 → 0.33 급락. lm_head dense 시 0.38 회복. 이 관찰이 config
default 를 `target_quant_lm_head=False` 로 바꾼 근거.

| MESA config | TP | accept |
|---|---|---|
| dense | 24.12 | 0.41 |
| INT8 wo | 12.15 | 0.40 |
| INT4 (lm_head on) | 13.15 | 0.33 |
| **INT4 (lm_head off)** | **13.47** | **0.38** |

**구현 변경**:
- `target_quant_backend: str = "int4_wo_tile"` (당시 기본값) / `"int8_wo"`
- `--quant_int4` / `--quant_int8` CLI 분리
- `_quantize_weight_to_int4_wo()`: `Int4WeightOnlyConfig(group_size=128)`

---

## v1 root cause 재정정 — fp16 overflow 가설 폐기

이전 결론 (`fp16 overflow → bf16 upcast`) 은 증거에 맞지 않는 오진단이었다.

**반증 증거**:
- Llama-2-7B + dense fp16 async spec: layer 1 hidden `shape=(921, 4096)
  dtype=fp16 nan=0 inf=0 finite_absmax=1597`. fp16 max 65504 의 1/40.
  overflow 아님. 정상 완주 (TP=56.40)
- Llama-2-7B + int4 WO + fp16 (no upcast): `ValueError: Expected Tensor
  argument zeros to have dtype torch.float16, but got torch.bfloat16` —
  torchao API level dtype assert
- Llama-2-7B + int8 WO + fp16 (no upcast): layer 1 hidden `nan=0 inf=22
  finite_absmax=440.75` — 22 개 inf + 유한부 dense 보다 작음. 수치 불안정

**실체 (정정)**:
선택한 torchao weight-only 경로 (`Int4WeightOnlyConfig`,
`Int8WeightOnlyConfig`) 는 공식 문서가 **bf16 activation workflow** 로 명시.
fp16 activation 은:
- Int4: scale/zero 가 bf16 으로 고정 → fp16 activation 과 matmul kernel
  dtype assert fail
- Int8: API 통과하지만 수치 불안정 (원인 미확정)

**지원 매트릭스 (정정)**:

| checkpoint dtype | 선택 backend | 상태 |
|---|---|---|
| bf16 (Llama-3 family) | int4_wo_tile / int8_wo | **정상 지원** |
| fp16 (Llama-2, CodeLlama) | int4_wo_tile / int8_wo | **미지원** — `ValueError`, `target_quant_force_bf16_runtime=True` opt-in 시 bf16 runtime 우회 |
| fp16 | GemliteUIntXWeightOnly / Marlin | **미통합** (별도 과제) |

### [archived] 초기 오진단 기록

- **오진단 1**: "Llama outlier activation → bf16 overflow → AWQ 필요" (틀림)
- **오진단 2**: "fp16 모델이 원래 overflow 경계에 있음 → quant 가 tip over"
  (틀림. dense fp16 absmax=1597 로 65504 와 거리 멂)

양쪽 다 **현재 torchao backend 의 bf16-activation 전제** 라는 동일한 실체를
서로 다르게 오해한 결과.

### v1 진단 sandbox 로그

- `tmp/int8_smoke/dbg_full/` — 층별 inf 탐지
- `tmp/int8_smoke/dbg_fine/` — layer 내부 sub-step NaN 전파 추적
- `tmp/int8_smoke/34b_ar_int8/` — 34B AR 정상 동작 확인
- `tmp/int8_smoke/34b_spec_int8*/` — 34B spec 실패 재현 (fp16 원본 + int8,
  upcast 전)

---

## v2 — AWQ Marlin 통합

v1 의 한계 (fp16 미지원, INT8 fast kernel 부재) 로 backend 방향 전환.

### [Phase 0] Backend choice — sgl-kernel Marlin W4A16

- **Finding**: `sgl_kernel` 0.3.17.post1 (이미 `ssd` env 에 있음) 이
  `gptq_marlin_gemm` + `awq_marlin_repack` + `awq_dequantize` 노출. CUDA
  graph capture, fp16 + bf16 activation, decode-M (1,4,8) 와 prefill-M
  (256, 1024) shape 모두 RTX 3090 sm_86 에서 성공
- **Chosen backend**: `gptq_marlin_gemm(b_q_type=scalar_types.uint4,
  is_zp_float=False)`. AWQ-format 입력 텐서를 load time 에 Marlin layout
  으로 repack
- **신규 의존성 불필요**: 모든 작업이 기존 `ssd` conda env 에서
- **Fallback**: torchao `int4_wo_tile` / `int8_wo` 는 internal fallback 으로
  유지

### [Phase 2] Marlin scales + qzeros 사전 permutation 필요

- **증상**: 첫 Phase 2 round-trip 이 모든 shape 와 dtype 에서 dequantize-
  then-matmul reference 대비 rel-err ≈ 0.48
- **원인**: sgl-kernel 의 `awq_marlin_repack` 은 **weight 텐서만** 처리.
  Marlin 의 `b_scales` 와 `b_zeros` 인자도 특정 permutation (vLLM 의
  `marlin_permute_scales`, `marlin_zero_points`) 을 기대. AutoAWQ-native
  `scales`/`qzeros` 그대로 넘기면 64-chunk 내 column-order 가 틀려 Marlin
  silent garbage 생성 (shape 맞아서 assert 안 걸림)
- **해결**: `ssd/quant/marlin_utils.py` 에 `marlin_permute_scales` +
  `marlin_zero_points_from_awq` 추가 (vLLM 에서 포팅, Apache-2.0).
  `build_awq_state` 가 세 변환 모두 적용. Round-trip rel-err fp16 ≈ 5e-4,
  bf16 ≈ 4e-3 (pure kernel roundoff)

### [Phase 4/5] TP=2 8B 가 쓰레기 출력 — GQA QKV shard 슬라이싱 오류

- **증상**: `layerskip-llama3-8B` (GQA: 32 q-heads, 8 kv-heads) TP=2 AR 에서
  반복 노이즈 ("ongoongoongo... iriiriiri"). TP=1 동일 모델에선 정상 →
  TP path 에 버그
- **원인**: `shard_awq_column_parallel` 이 전체 QKV packed tensor 를 하나의
  균일 블록으로 보고 `out_features / tp_size` 에서 슬라이스. GQA 에서는 q
  (32 heads) 와 k/v (각 8 heads) 가 한 rank-slice 에 같이 들어가 head 경계
  어긋남
- **해결**: `RawAwqTensors` 에 `part_out_features` 필드 추가.
  `shard_awq_column_parallel` 이 각 sub-projection 을 `part // tp_size` 로
  자르고 per-rank concat. dense `QKVParallelLinear.weight_loader` 규약과
  정확히 일치. 동일 크기 파트 (gate_up) 는 기존 균일 슬라이스와 결과 같음
  → QKV 만 눈에 띄게 깨졌었음
- **계획 영향**: plan v2 §9.3.2 "concat first, then TP shard" 는 동일
  크기 파트엔 맞지만 GQA QKV 에선 silent wrong. 구현은 여전히 concat-
  then-shard 이지만 shard 단계가 sub-part 인식

### [Post-review High #1] External AutoAWQ → SSD artifact → runtime 경로 깨짐

- **증상**: 광고된 external-AutoAWQ flow 가 artifact 생성에서 끝남:
  (a) `importer.py` 가 `model_id` 를 AutoAWQ dir 로 stamp 했으나 runtime 은
  `config.model` 과 비교 (전형적으로 dense path)
  (b) 양쪽 맞춰도 `config.model` 이 AutoAWQ dir 면 dense loader 가
  `qkv_proj.qweight` 같이 SSD 모델에 없는 이름으로 `get_parameter` 시도 →
  크래시
- **해결**:
  1. `load_safetensors_model` 이 `.qweight / .qzeros / .scales / .g_idx` 로
     끝나는 키 silent skip. dense loader 는 AutoAWQ hf dir 에서 embeddings,
     `lm_head`, norms 정상 로드
  2. importer 에 `--base-model` flag. `model_id` 기본값 `--model`.
- **테스트**: `sandbox/awq_spike/09_fake_autoawq_roundtrip.py` — 합성
  AutoAWQ-format hf dir 만들어 전 구간 검증. RTN-direct 와 같은 token IDs
  생성 (greedy 결정론)

### [Post-review High #2] Artifact completeness check (load-time hard fail)

- **증상**: loader 가 artifact 가 모델에 없는 모듈을 참조하는 경우만 거부.
  반대 (모델에 있는 quant-mode TP linear 가 artifact 에 없음) 는 silently
  meta + `quant_state=None` 으로 남고 first forward 에서야 크래시
- **해결**: `apply_ssd_awq_artifact` 가 attach 후 모든 `LinearBase`
  서브클래스 스캔. meta + quant state 없는 모듈 전부 나열하여
  `RuntimeError`. warmup / CUDA-graph capture 전 발생
- **테스트**: `sandbox/awq_spike/10_negative_checks.py::test_missing_module`

### [Post-review Medium #1] QuantConfig 가 runtime source of truth

- **증상**: plan §13.3 은 "LLM/runner 경계에서 legacy flat 필드로부터
  `QuantConfig` 파생" 을 요구. 우리 구현은 dataclass 만 있고 runner 는
  여전히 flat 필드를 읽음
- **해결**: `model_runner.__init__` 이 `quant_config_from_legacy_flags(config)`
  호출하여 `self.quant_config` 에 저장. AWQ branch 는 이 구조화 객체만 read.
  Legacy flat 필드는 CLI compat shim 으로 `Config` 에 유지

### [Post-review Medium #2] AutoAWQ `zero_point` / `w_bit` 엄격 검증

- **증상**: importer 가 `quantize_config.json` 에서 group size 만 읽음.
  `zero_point`, `w_bit` 검증 없어 unsupported AWQ 체크포인트 silently 통과
- **해결**: `quantize_config.json` 부재, `zero_point != True`, `w_bit != 4`
  각각에서 hard-fail
- **테스트**: `sandbox/awq_spike/10_negative_checks.py`

### [Post-review Open-Q #2] Runtime `--quant_group_size` 의미

- **증상**: CLI `--quant_group_size` 가 runtime 에서 받아들여지지만 효과
  없음 (loader 는 항상 artifact 메타데이터 사용)
- **해결**: `apply_ssd_awq_artifact` 가 `expected_group_size` 가짐.
  `model_runner` 는 사용자가 default 가 아닌 값 제공 시에만 전달. 그 경우
  runtime 값과 artifact 값 일치해야 하고 아니면 raise

### [Phase 7] 같은 프로세스에서 두 번째 `LLM(...)` 생성 시 mp semaphore rebuild 실패

- **증상**: 한 스크립트에서 dense 다음 AWQ 돌리면 두 번째 모델의 worker
  spawn 중 `multiprocessing/synchronize.py:_rebuild` 에서
  `FileNotFoundError: No such file or directory`. 첫 실행 worker process 는
  내려가지만 "ssd" 이름의 shared-memory handle 과 semaphore 파일이 잠시
  남음
- **해결**: dense / AWQ 변형을 별개 프로세스로 실행. `bench/run_*.sh`
  패턴과 일치
- **계획 영향**: 없음, quant 와 독립

### [Phase 2] Meta-tensor placeholder 가 `.cuda()` 깨뜨림

- **증상**: `quant_init_context()` 안에서 생성된 TP linear 에 `.cuda()`
  호출 시 `NotImplementedError: Cannot copy out of meta tensor; no data!`
- **원인**: §6.3.1 옵션 (2) (meta placeholder) 는 dense `weight` 를 실제
  device 로 "이동" 하지 않아야 함. 실제 SSD flow 에서는 모델 생성 전 이미
  `torch.set_default_device("cuda")` 설정 → 이 문제는 sandbox 테스트에만
- **해결**: sandbox 테스트에서 `.cuda()` 제거. `attach_quant_state()` 가
  이후 실제 CUDA state 공급

---

## v2 — Draft AWQ 확장

### [Draft-AWQ extension] Role-aware AWQ — runner generalization + artifact v2

target AWQ 가 출고된 뒤 async SSD / MESA step time 이 dense draft 에 의해
지배 → draft 도 같은 방식으로 quantize.

**변경 요약**:
1. `ssd/config.py`: `draft_quant_enabled` / `draft_quant_backend` /
   `draft_quant_awq_artifact` / `draft_quant_external_awq_path` /
   `draft_quant_group_size` flat 필드 추가
2. `ssd/quant/config.py`: `QuantConfig.role: "target" | "draft"`.
   `quant_config_from_legacy_flags(cfg, role)` 가 `{role}_quant_*` 필드 읽음.
   draft lm_head / embeddings quant 는 v1 에서 미구현
3. `ssd/quant/io.py`: artifact schema v2, `model_role` 필수 필드. v1 artifact
   는 backward compat 위해 `role="target"` 처리. `load_awq_artifact` 에
   `expected_role`
4. `ssd/quant/loader.py::apply_ssd_awq_artifact`: `expected_role`. Completeness
   check 가 draft modules 에도 적용
5. `ssd/quant/importer.py` + `scripts/awq_import.py`: `--role target|draft`
6. `ssd/engine/model_runner.py`: role-specific `QuantConfig` 파생. Draft
   model 도 `draft_quant_enabled=True` 시 `quant_init_context()` 에서 build.
   Eagle draft AWQ hard-fail. Draft AWQ 는 `tp_size==1` 요구
7. `ssd/bench/bench.py`: `--quant_awq_draft`, `--quant_awq_draft_artifact`,
   `--quant_awq_draft_external`, `--quant_awq_draft_group_size`

**보존**: `ssd/layers/linear.py`, `ssd/quant/marlin.py`, `ssd/quant/build.py`,
scheduler/verifier/FlashInfer wrappers 모두 변경 없음.
`LinearBase.attach_quant_state` 계약은 두 role 에 동일.

**검증** (`sandbox/awq_spike/13b_negative_fast.py`):
- wrong-role (target→draft) 가 load 시 `ValueError`
- correct-role (draft→draft) 정상 (schema v2)
- correct-role (target→target) 정상

**Smoke** (`sandbox/awq_spike/13_draft_awq_smoke.py sync`): sync spec decode
(layerskip-llama3-8B dense target + TinyLlama-1.1B AWQ draft) 정상 출력.
Accept rate 0.07 은 TinyLlama-Chat 이 Llama3-8B 의 자연 draft 가 아니라
(tokenizer / 분포 mismatch) — draft AWQ wiring 과 무관.

**Scope (v1)**:
- 지원: Llama-family non-EAGLE draft, tp_size=1 draft, SSD-native artifact,
  external AutoAWQ direct-load, `awq_marlin` only
- 미지원: Eagle draft AWQ (hard-fails), draft `lm_head` / embeddings, draft
  TP > 1, INT8 / GPTQ 등 다른 backend

### [Draft-AWQ operational] 동시 calibration GPU-conflict

- **증상**: 70B AutoAWQ calibration 이 GPU 0 에서 OOM. 같은 GPU 에서
  TinyLlama calibration 이 동시 실행 중
- **해결**: calibration 직렬화 또는 `CUDA_VISIBLE_DEVICES` 를 disjoint set
  으로 명시. orchestrator script 는 이미 명시적으로 `CUDA_VISIBLE_DEVICES`
  세팅

---

## v2 Cleanup pass — Legacy torchao 격리

### [Cleanup] Legacy torchao path 를 internal/deprecated 로 이동

- **결정**: AWQ Marlin 이 유일한 public quantization path. plan v2 §12.3 은
  "torchao 를 임시 fallback 으로" 라고 했지만, 실제로 public surface 가 두
  경로 (target-only torchao vs role-aware AWQ) 를 보여주면서 사용자 혼란
- **변경 (이번 revision)**:
  1. `bench/bench.py` — `--quant_int4`, `--quant_int8`, `--quant_artifact`,
     `--quant_artifact_load_only`, `--quant_force_bf16_runtime`,
     `--no_quant_lm_head` 제거. `--quant_awq*` family 만 남음.
     `--quant_lm_head` 는 AWQ 의 no-op 으로 유지 (lm_head 는 v1 에서 dense)
  2. `ssd/utils/quantize.py` — top header `[LEGACY / DEPRECATED]`.
     internal fallback 으로만. trigger 하려면 caller 가 `Config` 의
     `target_quant_backend` 를 torchao value 로 직접 설정해야 함
  3. `ssd/engine/model_runner.py` — legacy torchao branch 진입 시
     `[quant][LEGACY]` runtime warning
  4. `CLAUDE.md` — quant 섹션 재작성. AWQ 가 supported, torchao 는
     deprecated/internal
- **Deferred** — `ssd/utils/quantize.py` + `ssd/utils/int8_debug.py` 의
  실제 삭제와 runner branch 삭제는 release cycle 한 번 후 next PR
- **계획 영향**: 미미. plan §12.3 "임시 bf16 fallback" 그대로. plan §17
  "AWQ 검증 끝나기 전 torchao 코드 삭제 안 함" — AWQ 검증 완료
  (target+draft, 34B+70B), deletion 은 sequencing 만 남음

---

## v2 Pre-merge review hardening

External code review (두 reviewer) 가 main merge 전 6 개 항목 지적. 해결
완료:

1. **Default backend trap** — `Config.target_quant_backend` 가 legacy
   `"int4_wo_tile"` 이었음. 프로그램 caller 가 `target_quant_enabled=True`
   + `target_quant_awq_artifact=...` (backend 재설정 없이) 만 주면 silently
   torchao branch 로 떨어짐
   **Fix**:
   - default 를 `"awq_marlin"` 으로 (`ssd/config.py`)
   - `quant_config_from_legacy_flags` 가 AWQ artifact / external path 가 있으면
     backend 를 `"awq_marlin"` 으로 auto-route (stderr notice)
   - 테스트: `tests/test_awq_load_validation.py::TestConfigAutoRoute`

2. **External AutoAWQ direct-load 검증 부족** — `adapter.py` 가
   `q_group_size` 만 read. `w_bit`/`zero_point` check 없음,
   `config.json["quantization_config"]` fallback 없음, post-attach
   completeness check 없음
   **Fix**: `adapter.py` rewrite — importer 의 full validation 공유
   (mandatory quant config, `w_bit==4`, `zero_point==True`, hard-fail).
   `LinearBase` completeness scan 추가 (SSD-native loader 와 parity)

3. **`bench.py` 에서 `--quant_awq` 가 artifact 없이 실행 가능** — 깊은 곳에서
   fail 했음
   **Fix**: `--quant_awq` (와 draft 변형) 가 argparse 단계에서 `_artifact`
   나 `_external` 없으면 `SystemExit`

4. **`QuantConfig` unused 필드** — `method`, `artifact_mode`,
   `runtime_backend`, `use_zero_point`, `quantize_lm_head`,
   `quantize_embeddings` 가 default 만 있고 live consumer 없음
   **Fix**: runner 가 실제 read 하는 필드만 trim (`role`, `enabled`,
   `artifact_path`, `quant_source`, `external_quant_path`, `group_size`,
   `expected_runtime_dtype`)

5. **Dead `_QUANT_WEIGHT_SUFFIXES`** in `ssd/utils/loader.py` — 참조 없음.
   **Fix**: 제거

6. **Custom calibrator framing** — `scripts/awq_calibrate.py` 가 primary
   calibration path 처럼 보였음
   **Fix**: docstring 을 `[EXPERIMENTAL]` 로. `scripts/awq_calibrate_autoawq.py`
   를 production 으로 가이드

**New tests** (`ssd/tests/test_awq_load_validation.py`, 9 cases): role
validation (target↔draft cross-load fails), model_id mismatch, tp_size
mismatch, runtime_dtype mismatch, stale legacy backend auto-routing.
모두 CPU only, pass.

**Reviewer 추가 정리 (D 항목)**:

- **D1**: `quant_config_from_legacy_flags` 의 torchao passthrough 분기. 현재
  `quantize.py` 를 internal fallback 으로 keep 중이라 이 return-None 경로가
  legacy runner branch 진입점 — 코멘트 명확화. quantize.py 완전 삭제 시 D1
  분기도 함께 삭제 예정
- **D2**: "(target only)" 주석 → "AWQ W4A16 quantization (target + draft,
  role-aware)" 로 수정
- **D3**: `import sys` 3 중복 → 파일 상단 1 개로 통합
- **D4**: `quantize_lm_head/embeddings` dead read → `Config` 에서
  `draft_quant_lm_head`/`draft_quant_embeddings` 필드 삭제, `quant/config.py`
  에서 read+warn 코드 제거. lm_head/embeddings 는 AWQ 에서 항상 dense

**Deferred to next cleanup PR**:
- `ssd/utils/quantize.py` + `ssd/utils/int8_debug.py` 완전 삭제
- legacy `Config` 필드 (`target_quant_mode`,
  `target_quant_force_bf16_runtime`, `target_quant_artifact_prefix`) 삭제
- `tmp/final_exp2_quant*/` 를 tracking 대신 `.gitignore`

---

## v2 코드 변경 종합

**추가된 파일** (13 개, `ssd/quant/`, ~1,512 lines):

```
ssd/quant/
├── __init__.py              — public exports
├── config.py                — QuantConfig + legacy flat 변환
├── state.py                 — AwqQuantState (Marlin-packed, per-rank)
├── pack.py                  — AutoAWQ pack/unpack + RTN W4A16 quantizer
├── marlin.py                — sgl-kernel Marlin wrapping awq_matmul
├── marlin_utils.py          — marlin_permute_scales + marlin_zero_points_from_awq
│                              (vLLM 포팅, Apache-2.0)
├── build.py                 — concat-packed + TP-shard + build_awq_state
├── init_context.py          — quant_init_context — meta-device placeholder
├── naming.py                — HF → SSD packed module name map
├── importer.py              — CPU 오프라인 importer (rtn / autoawq mode)
├── adapter.py               — Phase 3a external-AutoAWQ thin loader
├── loader.py                — Phase 4 SSD-native artifact loader
└── io.py                    — SSD-native artifact save/load + 메타 schema
```

`scripts/awq_import.py`, `scripts/awq_calibrate.py`,
`scripts/awq_calibrate_autoawq.py`, `tests/test_awq_load_validation.py`,
`sandbox/awq_spike/` (10+ smoke / diagnosis / perf / negative scripts)

**수정된 파일**:

```
ssd/layers/linear.py          — meta-device placeholder + quant dispatch
ssd/utils/loader.py           — dense load 중 meta param skip + AWQ key skip
ssd/engine/model_runner.py    — AWQ backend wiring + meta construction + QuantConfig
ssd/config.py                 — AWQ flat fields (target + draft, plan §13.3)
bench/bench.py                — --quant_awq* CLI
```

**Validation tests**: `sandbox/awq_spike/01_tp_linear_roundtrip.py` (수치
정확성), `02_diagnose_layout.py` (Marlin layout), `03_ar_smoke.py` (1B AR),
`04_tp2_8b_ar.py` (TP=2 AR), `05_tp1_8b_ar.py` (TP=1 8B), `06_spec_8b.py`
(sync spec), `07_mesa_awq.py` (MESA), `08_perf_bench.py` (micro+E2E),
`09_fake_autoawq_roundtrip.py` (external path), `10_negative_checks.py`
(role/group/w_bit/missing module 4 case), `11_awq_calibrated_smoke.py`
(custom AWQ output), `12_autoawq_smoke.py` (AutoAWQ output),
`13_draft_awq_smoke.py` (draft AWQ), `13b_negative_fast.py` (draft role
hard-fail).

`tests/test_awq_load_validation.py` (CPU-only unittest, 9 case): role
validation × 4, model_id mismatch, tp_size mismatch, runtime_dtype
mismatch, auto-route.
