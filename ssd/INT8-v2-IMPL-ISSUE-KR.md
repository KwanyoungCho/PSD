# AWQ 통합 v2 — 구현 이슈 로그 (한국어)

`INT8-WEIGHT-ONLY-PLAN-v2.md` phase를 따라 구현 중. 이슈, 이탈, 수정
내역을 발생 시점에 기록. 계획 자체는 수정하지 않음.

## 포맷

각 이슈는 `### [Phase X.Y] <짧은 제목>` 하에:
- **증상** — 무엇이 깨졌거나 모호했는지
- **원인** — 왜
- **해결** — 무엇을 했는지
- **계획 영향** — 있다면

---

### [Phase 0] 백엔드 선택: sgl-kernel Marlin W4A16

- **조사 결과** — `sgl_kernel` 0.3.17.post1 (이미 `ssd` env에 있음)이
  `gptq_marlin_gemm` + `awq_marlin_repack` + `awq_dequantize`를 노출.
  RTX 3090 sm_86에서 CUDA graph capture, fp16 + bf16 activation,
  decode-M (1,4,8) 및 prefill-M (256, 1024) shape 모두 성공 (계획
  §5.1–5.4 gate 통과).
- **선정 백엔드** — `gptq_marlin_gemm(b_q_type=scalar_types.uint4,
  is_zp_float=False)`. AWQ-format 입력 텐서는 load time에 Marlin이 기대하는
  레이아웃으로 repack (forward마다 하지 않음).
- **신규 의존성 불필요** — 모든 작업이 기존 `ssd` conda env 내부에서
  수행. 추가 `pip install` 없음.
- **Fallback** — `ssd/utils/quantize.py`의 torchao int4_wo_tile / int8_wo
  경로는 bf16-native fallback용으로 유지 (계획 §12.3).

### [Phase 2] Marlin scales + qzeros는 weight repack뿐 아니라 pre-permutation 필요

- **증상** — Phase 2 round-trip 초기 결과가 모든 shape과 dtype에서
  dequantize-then-matmul reference 대비 rel-err ≈ 0.48.
- **원인** — sgl-kernel의 `awq_marlin_repack`은 **weight** 텐서만
  처리. Marlin의 `b_scales`와 `b_zeros` 인자도 특정 permutation을
  기대 (vLLM에서 `marlin_permute_scales`, `marlin_zero_points`라고 부름).
  AutoAWQ-native `scales` / `qzeros`를 그대로 넘기면 64-chunk 내
  column-order가 틀려 Marlin이 silently 쓰레기 결과 생성 (shape은
  맞아서 assert 안 걸림).
- **해결** — `ssd/ssd/quant/marlin_utils.py`에 `marlin_permute_scales`와
  `marlin_zero_points_from_awq` 추가 (vLLM의 `marlin_utils.py`에서 포팅,
  Apache-2.0). `build_awq_state`가 이제 세 변환 모두 적용. Round-trip
  rel-err이 fp16 ≈ 5e-4, bf16 ≈ 4e-3로 감소 (순수 커널 roundoff,
  dense-matmul-on-dequantized-weight과 동일).
- **계획 영향** — 없음; 계획 §3.1/§3.3 "AWQ runtime adapter"의 내부
  구현 디테일.

### [Phase 4/5] TP=2 8B가 쓰레기 출력; GQA QKV shard 슬라이싱 오류

- **증상** — `layerskip-llama3-8B` (GQA: 32 q-heads, 8 kv-heads)로 TP=2
  AR 돌렸을 때 출력이 반복 노이즈: "ongoongoongo... iriiriiri". 같은
  모델로 TP=1에서는 정상 ("Paris. The country is divided into 27 regions
  and 96 departments...")이라 버그가 TP 경로에 있음을 확인.
- **원인** — `shard_awq_column_parallel`이 전체 QKV packed 텐서를 하나의
  균일 블록으로 보고 `out_features / tp_size`에서 슬라이스. GQA에서는
  q (32 heads)와 k/v (각 8 heads)가 한 rank-slice에 같이 들어가 head
  경계가 어긋남: rank 1이 "q의 후반부 + k/v의 후반부"를 받는 꼴이 되어
  dense `QKVParallelLinear.weight_loader`가 내는 "내 local q heads +
  내 local k/v heads" 형태와 다름.
- **해결** — `RawAwqTensors`에 packed module용 `part_out_features`
  필드를 추가; `shard_awq_column_parallel`이 각 sub-projection을
  `part // tp_size`로 자르고 per-rank concat. dense TP 규약과 정확히
  일치. 동일 크기 파트(gate_up)는 기존 균일 슬라이스와 결과가 같아
  QKV만 눈에 띄게 깨졌던 것.
- **계획 영향** — 계획 §9.3.2는 "concat first, then TP shard"를 권장.
  동일 크기 파트에 대해선 맞지만 GQA QKV에서는 silent wrong. 우리
  구현은 여전히 "concat-then-shard"이지만 shard 단계가 sub-part 인식을
  하므로 계획 문구를 수정할 필요는 없음; GQA 관련 구현 노트는 여기에
  기록.

### [Post-review Fix High #1] External AutoAWQ → SSD artifact → runtime 경로가 깨져 있었음

- **증상 (리뷰)** — 광고된 external-AutoAWQ 흐름이 artifact 생성에서
  끝남:
  (a) `importer.py`가 `model_id`를 AutoAWQ 디렉토리로 찍었는데, runtime은
      이를 `config.model`과 비교하는데 전형적으로 `config.model`은
      (dense) 다른 경로를 가리킴;
  (b) 양쪽을 맞춘다 해도 `config.model`이 AutoAWQ 디렉토리를 가리키면
      dense loader가 `qkv_proj.qweight` 같이 SSD 모델에 없는 이름으로
      `get_parameter`를 시도하다 크래시.
- **해결** — 두 변경:
  1. `load_safetensors_model`이 `.qweight / .qzeros / .scales / .g_idx`로
     끝나는 키를 silently skip. dense loader는 AutoAWQ hf 디렉토리에서
     여전히 embeddings, `lm_head`, norms를 가져오고, quant state는
     AWQ loader가 담당.
  2. importer에 `--base-model` 플래그 추가. artifact에 찍히는 `model_id`는
     `--model`이 기본값; 별도의 dense base가 필요하면 override. 전형적인
     흐름(AutoAWQ 디렉토리가 dense embed/lm_head/norms와 quant linear를
     모두 포함)에서는 `config.model` == `--model`이라 추가 플래그 없이
     검증 통과.
- **테스트** — `sandbox/awq_spike/09_fake_autoawq_roundtrip.py`가
  Llama-3.2-1B의 모든 linear `.weight`를 RTN 기반 AutoAWQ trio로 바꾼
  합성 AutoAWQ-format hf 디렉토리를 만들어 실행. importer가 이를
  ingest하고 runtime은 **RTN-direct 경로와 같은 토큰 ID**를 생성
  (greedy decode 결정론성 활용), 전 구간 검증:
  `external safetensors → autoawq importer → SSD-native artifact →
  dense loader (AWQ-key skip) → AWQ artifact loader → Marlin forward`.
- **계획 영향** — 없음; 계획은 두 흐름 모두 기술하고 있고 수정 내용과
  일관.

### [Post-review Fix High #2] Artifact 완전성 체크 (load-time hard fail)

- **증상 (리뷰)** — loader는 artifact가 모델에 없는 모듈을 참조하는 경우만
  거부했음; 반대로 모델에 있는 quant-mode TP linear가 artifact에 없는
  경우는 silently meta device + `quant_state=None`으로 남고 first
  forward에서야 크래시.
- **해결** — `apply_ssd_awq_artifact`가 attach 후 모든 `LinearBase`
  서브클래스를 스캔하고, meta weight이면서 quant state가 없는 모듈을
  전부 나열하여 `RuntimeError`. warmup이나 CUDA-graph capture 전에
  발생.
- **테스트** — `sandbox/awq_spike/10_negative_checks.py::test_missing_module`
  이 작동 중인 artifact에서 한 모듈을 지우고, loader가 forward가 아닌
  load 시점에 `"did not provide quant state"`로 raise 하는지 확인.

### [Post-review Fix Medium #1] QuantConfig를 runtime source of truth로 배선

- **증상 (리뷰)** — 계획 §13.3은 "LLM/runner 경계에서 legacy flat
  필드로부터 `QuantConfig` 파생"을 요구; 우리 구현은 데이터클래스만
  있고 runner는 여전히 flat 필드를 읽고 있었음.
- **해결** — `model_runner.__init__`이 이제
  `quant_config_from_legacy_flags(config)`를 한 번 호출하여
  `self.quant_config`에 저장. AWQ branch가 이를 통해 동작 분기
  (`ssd_artifact` vs `external_awq`, artifact path, external path,
  expected runtime dtype, group size override). Legacy flat 필드는
  CLI 호환 shim으로 `Config`에 유지 (계획 §13.3); 구조화된 객체가
  runtime contract.

### [Post-review Fix Medium #2] AutoAWQ `zero_point`/`w_bit` 엄격 검증

- **증상 (리뷰)** — importer가 `quantize_config.json`에서 group size만
  읽고 `zero_point`와 `w_bit`는 무시. Marlin `uint4` 경로가 쓸 수
  없는 지원 불가 체크포인트(예: symmetric GPTQ-style)를 silently 통과
  시킴.
- **해결** — importer가 `quantize_config.json` 부재, `zero_point !=
  True`, `w_bit != 4` 각각에서 hard-fail. 에러 메시지가 제약 조건을
  명시하고 AWQ vs GPTQ 분기를 안내.
- **테스트** — `sandbox/awq_spike/10_negative_checks.py::test_*_rejects_*`
  가 `zero_point=False`와 `w_bit=8` 케이스를 각각 실행. 둘 다 raise.

### [Post-review Open-Q #2] Runtime `--quant_group_size`는 sanity check

- **증상 (리뷰)** — CLI `--quant_group_size`가 runtime에서 받아들여지지만
  실제 효과 없음 (loader는 항상 artifact 메타데이터 사용).
- **해결** — `apply_ssd_awq_artifact`에 `expected_group_size` 추가.
  `model_runner`는 사용자가 기본값이 아닌 `target_quant_group_size`를
  제공한 경우에만 전달. 그 경우 runtime 값과 artifact 값이 일치해야
  하며, 아니면 warmup 전에 loader가 raise. 미지정 = 체크 안 함. 플래그
  의미를 유지하면서 artifact와의 괴리를 차단.
- **테스트** — `sandbox/awq_spike/10_negative_checks.py::test_runtime_group_size_mismatch`.

### [Phase 7] 같은 프로세스에서 두 번째 `LLM(...)` 생성 시 mp semaphore rebuild 실패

- **증상** — 한 스크립트에서 dense 다음에 AWQ를 돌리면 두 번째 모델의
  worker spawn 중에 `multiprocessing/synchronize.py:_rebuild`에서
  `FileNotFoundError: No such file or directory` 발생. 첫 실행의
  worker process는 내려가지만 "ssd"라는 이름의 shared-memory handle과
  semaphore 파일이 잠시 남아있음.
- **해결** — dense와 AWQ 변형을 **별개 프로세스**로 실행. perf 스크립트가
  이제 `dense` 또는 `awq`를 단일 CLI 인자로 받음. 기존 `bench/run_*.sh`
  패턴(설정당 하나의 bench 프로세스 스폰)과 일치.
- **계획 영향** — 없음; quant 작업과 독립.

### [Phase 2] Meta-tensor placeholder가 `.cuda()`를 깨뜨림

- **증상** — `quant_init_context()` 안에서 생성된 TP linear에 `.cuda()`를
  호출하면 `NotImplementedError: Cannot copy out of meta tensor; no data!`.
- **원인** — 계획 §6.3.1 옵션 (2) (meta placeholder)은 dense `weight`이
  실제 device로 "이동"되지 않아야 함을 요구. 실제 SSD 흐름에서는 모델
  생성 전에 이미 `torch.set_default_device("cuda")`가 설정되므로 이
  문제는 `.cuda()`를 호출하는 standalone sandbox 테스트에서만 발생.
- **해결** — sandbox 테스트에서 `.cuda()` 호출 제거; `attach_quant_state()`가
  이후에 실제 CUDA state를 공급하므로 그대로 동작. 실제 model_runner
  흐름과 일치하며 비용 없음.
- **계획 영향** — 없음.
