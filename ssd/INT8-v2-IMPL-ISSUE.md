# AWQ Integration v2 — Implementation Issues Log

Working through `INT8-WEIGHT-ONLY-PLAN-v2.md` phases. Issues, deviations, and
fixes are recorded here as they happen. Plan itself is not modified.

## Format

Each issue follows: `### [Phase X.Y] <short title>` with:
- **Symptom** — what broke / was ambiguous
- **Root cause** — why
- **Resolution** — what we did
- **Impact on plan** — if any

---

### [Phase 0] Backend choice: sgl-kernel Marlin W4A16

- **Finding** — `sgl_kernel` 0.3.17.post1 (already in `ssd` env) exposes
  `gptq_marlin_gemm` + `awq_marlin_repack` + `awq_dequantize`. CUDA graph
  capture, fp16 + bf16 activation, decode-M (1,4,8) and prefill-M (256, 1024)
  shapes all succeed on RTX 3090 sm_86 (plan §5.1–5.4 gates pass).
- **Chosen backend** — `gptq_marlin_gemm(b_q_type=scalar_types.uint4,
  is_zp_float=False)`, with AWQ-format input tensors repacked into Marlin's
  expected layout at load time (not at every forward).
- **No new dependency needed** — all work happens inside the existing `ssd`
  conda env. No extra `pip install`.
- **Fallback** — torchao int4_wo_tile / int8_wo paths in `ssd/utils/quantize.py`
  stay in tree for bf16-native fallback (plan §12.3).

### [Phase 2] Marlin scales + qzeros need pre-permutation, not just weight repack

- **Symptom** — first-cut Phase 2 roundtrip showed rel-err ≈ 0.48 vs a
  dequantize-then-matmul reference, at every shape and dtype.
- **Root cause** — `awq_marlin_repack` in sgl-kernel only handles the **weight**
  tensor. Marlin's `b_scales` and `b_zeros` arguments also expect a specific
  permutation (vLLM calls these `marlin_permute_scales` and `marlin_zero_points`).
  Passing AutoAWQ-native `scales` / `qzeros` straight through leaves them in
  the wrong column-within-64-chunk order, and Marlin silently produces garbage
  (no assert fires because shapes still match).
- **Resolution** — added `ssd/ssd/quant/marlin_utils.py` with
  `marlin_permute_scales` and `marlin_zero_points_from_awq` (ported from
  vLLM `marlin_utils.py`, Apache-2.0). `build_awq_state` now applies all
  three transforms. Roundtrip rel-err collapses to fp16 ≈ 5e-4, bf16 ≈ 4e-3
  (pure kernel roundoff, matches dense-matmul-on-dequantized-weight).
- **Impact on plan** — none; this is an internal implementation detail of
  the §3.1/§3.3 "AWQ runtime adapter" referenced in the plan.

### [Phase 4/5] TP=2 8B produces garbage; GQA QKV shard had wrong slicing

- **Symptom** — with TP=2 on `layerskip-llama3-8B` (GQA: 32 q-heads, 8 kv-heads),
  AR output was a repeating noise string: "ongoongoongo... iriiriiri". TP=1 on
  the same model produced coherent text ("Paris. The country is divided into
  27 regions and 96 departments..."), isolating the bug to the TP path.
- **Root cause** — `shard_awq_column_parallel` treated the full QKV packed
  tensor as one uniform block and sliced at `out_features / tp_size`. For
  GQA this splits q (32 heads) and k/v (8 heads each) together into a
  single slice per rank, which mis-aligns head boundaries: rank 1 ends up
  with "the second half of q concatenated with the second half of k/v",
  not "my local q heads concatenated with my local k/v heads" as the dense
  `QKVParallelLinear.weight_loader` produces.
- **Resolution** — `RawAwqTensors` now carries `part_out_features` for
  packed modules; `shard_awq_column_parallel` slices each sub-projection
  by `part // tp_size` and concatenates per-rank. Matches the dense TP
  convention exactly. Equal-sized parts (gate_up) happen to coincide with
  the old uniform slice, so only QKV was visibly broken.
- **Impact on plan** — plan §9.3.2 recommended "concat first, then TP
  shard". That recipe is correct for equal-sized parts but silently wrong
  for GQA QKV. Our implementation is still "concat-then-shard" but the
  shard step is now sub-part aware, so the plan language doesn't need to
  change; the implementation note about GQA is recorded here.

### [Post-review Fix High #1] External AutoAWQ → SSD artifact → runtime path was broken

- **Symptom (review)** — the advertised external-AutoAWQ flow ended at
  artifact creation:
  (a) `importer.py` stamped `model_id` as the AutoAWQ dir, but runtime
      validated it against `config.model`, which typically points to a
      different (dense) path;
  (b) even if both were made to match, pointing `config.model` at an
      AutoAWQ dir crashed the dense loader because it tried to
      `get_parameter` on names like `qkv_proj.qweight` that don't exist
      on the SSD model.
- **Resolution** — two changes:
  1. `load_safetensors_model` now silently skips keys ending in
     `.qweight / .qzeros / .scales / .g_idx`. The dense loader still
     picks up embeddings, `lm_head`, and norms from an AutoAWQ hf dir,
     and the AWQ loader owns the quant state.
  2. The importer grew a `--base-model` flag. `model_id` stamped on the
     artifact defaults to `--model`; users who want a separate dense
     base can override it. For the typical flow (AutoAWQ dir has both
     dense embed/lm_head/norms and quant linears), `config.model` ==
     `--model` and validation passes without extra flags.
- **Test** — `sandbox/awq_spike/09_fake_autoawq_roundtrip.py` builds a
  synthetic AutoAWQ-format hf dir from Llama-3.2-1B by replacing every
  linear's `.weight` with RTN-derived AutoAWQ trio. The importer ingests
  it and runtime produces **the same token IDs as the RTN-direct path**
  (greedy decode determinism), validating every link:
  `external safetensors → autoawq importer → SSD-native artifact →
  dense loader (AWQ-key skip) → AWQ artifact loader → Marlin forward`.
- **Impact on plan** — none; the plan describes both flows and is
  consistent with the fix.

### [Post-review Fix High #2] Artifact completeness check (load-time hard fail)

- **Symptom (review)** — the loader only rejected artifacts that named
  modules not present on the model; the reverse case (model has
  quant-mode TP linears not in the artifact) silently left them on the
  meta device with `quant_state=None`, and crashed only on first forward.
- **Resolution** — `apply_ssd_awq_artifact` now scans every `LinearBase`
  subclass after attaching and raises `RuntimeError` listing every
  module that still has a meta weight and no quant state. This fires at
  load time, before warmup or CUDA-graph capture.
- **Test** — `sandbox/awq_spike/10_negative_checks.py::test_missing_module`
  drops a single module from a working artifact and confirms the
  loader raises `"did not provide quant state"` at load, not at forward.

### [Post-review Fix Medium #1] QuantConfig wired as runtime source of truth

- **Symptom (review)** — plan §13.3 says "derive `QuantConfig` from the
  legacy flat fields at the LLM/runner boundary"; our implementation
  had the dataclass but the runner was still reading flat fields.
- **Resolution** — `model_runner.__init__` now calls
  `quant_config_from_legacy_flags(config)` once and stores the result
  on `self.quant_config`. The AWQ branch reads from it (dispatch between
  `ssd_artifact` vs `external_awq`, artifact path, external path,
  expected runtime dtype, group size override). Legacy flat fields stay
  on `Config` as the CLI compat shim (plan §13.3); the structured
  object is the runtime contract.

### [Post-review Fix Medium #2] AutoAWQ `zero_point`/`w_bit` strictly validated

- **Symptom (review)** — importer read `quantize_config.json` for
  group size but ignored `zero_point` and `w_bit`, silently accepting
  unsupported checkpoints (e.g. symmetric GPTQ-style) that the Marlin
  `uint4` path can't use.
- **Resolution** — importer now hard-fails if `quantize_config.json`
  is missing, or if `zero_point != True`, or if `w_bit != 4`. The error
  message names the constraint and points at the AWQ-vs-GPTQ split.
- **Test** — `sandbox/awq_spike/10_negative_checks.py::test_*_rejects_*`
  exercise both `zero_point=False` and `w_bit=8`. Both raise.

### [Post-review Open-Q #2] Runtime `--quant_group_size` is a sanity check

- **Symptom (review)** — CLI `--quant_group_size` was accepted at
  runtime but had no effect (loader always used the artifact metadata).
- **Resolution** — `apply_ssd_awq_artifact` gained `expected_group_size`.
  `model_runner` passes it only when the user provided a non-default
  `target_quant_group_size`. In that case the runtime and artifact
  values must match, else the loader raises before warmup. Unset =
  no check. This keeps the flag useful without letting it diverge
  from the artifact.
- **Test** — `sandbox/awq_spike/10_negative_checks.py::test_runtime_group_size_mismatch`.

### [Phase 7] Second `LLM(...)` in same process can't rebuild mp semaphores

- **Symptom** — running dense-then-AWQ in one script raises
  `FileNotFoundError: No such file or directory` from
  `multiprocessing/synchronize.py:_rebuild` during the second model's
  worker spawn. Worker processes from the first run get torn down but
  the shared-memory handle named "ssd" and semaphore files briefly
  remain.
- **Resolution** — run the dense and AWQ variants as **separate process
  invocations**. The perf script now accepts `dense` or `awq` as a single
  CLI argument. This matches the existing `bench/run_*.sh` pattern which
  always spawns one bench process per configuration.
- **Impact on plan** — none; orthogonal to the quant work.

### [Phase 2] Meta-tensor placeholder breaks `.cuda()`

- **Symptom** — calling `.cuda()` on a TP linear constructed inside
  `quant_init_context()` raises `NotImplementedError: Cannot copy out of
  meta tensor; no data!`.
- **Root cause** — plan §6.3.1 option (2) (meta placeholder) requires that
  the dense `weight` never be "moved" to a real device. In real SSD flow
  `torch.set_default_device("cuda")` is already set before model construction,
  so the issue only surfaces in standalone sandbox tests that call `.cuda()`.
- **Resolution** — drop `.cuda()` in the sandbox test; the module works as-is
  because `attach_quant_state()` supplies real CUDA state afterwards. This
  matches the real model_runner flow and costs nothing.
- **Impact on plan** — none.
