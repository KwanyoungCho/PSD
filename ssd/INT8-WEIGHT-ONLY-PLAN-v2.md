# AWQ Integration Plan v2

> File name is legacy. This document is no longer an "INT8-first" plan.
> The new goal is an **AWQ-style optimized weight-only backend** that fits SSD's
> current inference architecture and supports the checkpoint/runtime mix we
> actually need.

## 1. Goal

### 1.1 Final objective

Integrate a **fp16/bf16-friendly optimized weight-only backend** into SSD so that:

- large target models fit in VRAM,
- existing SSD optimized paths remain intact,
- autoregressive decode, speculative verify, and MESA target verify continue to work,
- we do **not** first load full-precision weights onto GPU and only then quantize.

The backend direction for this plan is **AWQ-style W4A16**, not torchao INT8.

### 1.2 Why we are changing direction

The current torchao path is useful as a fallback for some bf16-native models, but
it is not the long-term answer for this repository:

- current selected torchao WO backends are not a good fit for fp16 runtime,
- the INT8 path is not an optimized fast path on our hardware,
- the current implementation still materializes dense GPU weights before replacing
  them, so startup/peak-memory savings are incomplete,
- "optimized inference" in SSD depends on keeping the existing TP / PagedAttention /
  CUDA graph / prefix-cache architecture intact, and only swapping the local linear
  backend.

### 1.3 Core design choice

We are **not** going to:

- rewrite SSD scheduling,
- rewrite PagedAttention / KV cache / attention kernels,
- replace the engine with a Hugging Face runtime,
- build a quantized GEMM kernel from scratch as the first step.

We **are** going to:

- keep SSD's current optimized engine,
- add an **offline AWQ artifact pipeline**,
- add an **AWQ runtime adapter for SSD local TP linear layers**,
- keep quantization scope narrow and explicit,
- integrate at the **local linear matmul** boundary only.


## 2. What Stays the Same

These parts of SSD are to be preserved unless a hard blocker is found:

- tensor-parallel wrappers and semantics in `ssd/ssd/layers/linear.py`
- attention path and PagedAttention / FlashInfer wrappers
- KV cache layout and block-table handling
- speculative decoding control flow
- MESA orchestration and split verify structure
- CUDA graph capture/replay structure
- `@torch.compile` norm / activation / rope path
- prefix caching and scheduler behavior

This is the most important architectural constraint of the whole plan.


## 3. What Changes

Only the **storage and execution of heavy linear weights** change.

### 3.1 Quantized by default

By default, quantize only the large target-side projection weights:

- `q_proj / k_proj / v_proj` -> SSD packed module `qkv_proj`
- `o_proj`
- `gate_proj / up_proj` -> SSD packed module `gate_up_proj`
- `down_proj`

### 3.2 Dense by default

Keep these dense unless explicitly enabled later:

- embeddings
- `lm_head`
- norms
- rope
- attention core
- KV cache
- draft model

### 3.3 Why `lm_head` stays dense by default

For SSD specifically, `lm_head` is a bad first quantization target:

- it is a hot per-step path,
- `ParallelLMHead` already has TP gather/cat overhead,
- MESA uses exit-layer logits and is more sensitive to `lm_head` quality.

So the default policy is:

- **target linear projections: quantized**
- **target `lm_head`: dense**
- **draft: dense**


## 4. Why AWQ, Not bitsandbytes, Not Current torchao

### 4.1 Why not bitsandbytes

bitsandbytes is a poor fit for SSD as the main backend because it is most natural
in a Hugging Face / `nn.Linear` replacement flow.

SSD is not built that way:

- it has custom TP linear modules,
- packed loader rules for QKV / gate-up,
- graph-heavy execution,
- custom runner orchestration.

bitsandbytes would force us to fight the SSD architecture instead of fitting into it.

### 4.2 Why not stay on current torchao as the primary path

torchao stays useful as a temporary fallback for some bf16-native cases, but it is
not the primary direction anymore because:

- current selected torchao WO paths do not give us the fp16-runtime story we want,
- current INT8 is not the optimized path we want to bet on,
- the implementation model is convenient but not enough for the final goal.

### 4.3 Why AWQ is the better direction

AWQ-style backends are a better match because they are typically:

- offline quantization first,
- low-bit weight-only,
- activation-friendly for fp16/fp8/bf16 inference depending on backend,
- tied to optimized inference kernels rather than generic dequantized `F.linear`.

For SSD, the important point is not the AWQ algorithm alone. It is this pair:

1. **offline AWQ artifact**
2. **optimized runtime backend for local linear matmul**

### 4.4 AWQ calibration vs AWQ runtime backend

These two concepts are separable and should not be conflated:

- **AWQ calibration algorithm**: the offline procedure that decides per-channel
  scaling factors using activation-aware importance weighting. This produces a
  quantized checkpoint (artifact). It is a *quality* decision.
- **AWQ runtime backend**: the optimized kernel path that executes packed low-bit
  weight × fp16/bf16 activation matmul at inference time. Examples include Marlin,
  AutoAWQ GEMM, ExLlamaV2, etc. This is a *performance* decision.

The calibration algorithm and the runtime backend are chosen independently:

- a Marlin kernel can execute both AWQ-calibrated and GPTQ-calibrated artifacts,
- a different calibration method could produce artifacts compatible with the same
  runtime backend.

Throughout this document:

- "AWQ artifact" refers to the **calibration output** (quantized weights + scales +
  zero-points in a specific packed format).
- "AWQ runtime" or "AWQ backend" refers to the **kernel path** used at inference.
- When the distinction matters, it will be stated explicitly.


## 5. Hard Constraints for the New Backend

Any AWQ runtime candidate must satisfy these gates before we integrate it.

### 5.1 Runtime dtype support

The chosen backend must support the runtime dtypes we actually care about:

- fp16 activation runtime
- bf16 activation runtime, or a clearly scoped fallback strategy

If a candidate only supports fp16 well:

- it may still be used for fp16 checkpoints first,
- but bf16-native models must either keep using the existing torchao fallback or
  wait for a second backend.

This is acceptable as an intermediate state, but it must be explicit.

### 5.2 Shape support

The backend must cover SSD-relevant regimes:

- **decode / verify / MESA**: very small-M, GEMV-like or tiny-GEMM regime
- **prefill**: larger-M GEMM regime

If a backend only benchmarks well for large GEMM and collapses for decode-sized
 matmuls, it is not acceptable as the primary SSD quant backend.

### 5.3 Graph compatibility

The backend must be compatible with:

- `torch.inference_mode()`
- existing CUDA graph capture/replay
- current TP wrappers

If graph compatibility fails, the backend is rejected or scoped to eager-only
debugging until fixed.

### 5.4 Storage contract

The backend must allow us to keep quantized storage on GPU:

- packed low-bit weights
- scales
- zero-points / metadata if required

A fallback that expands full fp16 weights on GPU and keeps them there is not
acceptable.


## 6. Final Architecture

### 6.1 High-level structure

The final structure should look like this:

1. **Offline AWQ producer**
   - external tool or import script
   - creates quantized artifact from original checkpoint

2. **SSD AWQ importer**
   - reads the external AWQ artifact
   - converts it into an SSD-friendly, rank-local artifact
   - resolves packed modules (`qkv_proj`, `gate_up_proj`)
   - pre-shards by TP rank

3. **SSD runtime adapter**
   - loads rank-local quantized weights directly
   - existing SSD TP wrapper calls an AWQ runtime op instead of dense `F.linear`

4. **SSD engine**
   - unchanged attention / cache / graph orchestration

### 6.2 Integration point

The integration point is **only** the local linear execution boundary.

That means:

- keep existing TP wrapper classes,
- keep model definitions mostly unchanged,
- keep scheduler / runner / attention structure unchanged,
- branch inside linear forward based on weight kind / quant backend.

We do **not** want a whole second model stack if we can avoid it.

### 6.3 Preferred module strategy

Do **not** start by building a full parallel hierarchy of brand-new model modules.

Preferred approach:

- keep `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`,
  `MergedColumnParallelLinear`,
- add quantized state to them or a small helper object they own,
- make forward dispatch:
  - dense path -> existing `F.linear`
  - AWQ path -> backend-specific local quantized matmul

This keeps:

- TP semantics,
- loader signatures,
- model code,
- graph call sites

stable.

### 6.3.1 Quant-mode module instantiation contract

This must be defined explicitly before implementation starts.

Current SSD model construction is not quant-friendly by default:

- `ModelRunner` sets `torch.set_default_device("cuda")` before model construction,
- TP linear modules allocate dense `nn.Parameter(torch.empty(...))` weights during
  `__init__`,
- therefore "build model first, then load quantized artifact" would still allocate
  dense GPU weight storage at construction time even if loader never copies dense
  checkpoint tensors into them.

For AWQ mode, this is unacceptable. Quant mode must use one of these contracts:

1. **Quant-aware TP module init**
   - in quant mode, TP linear modules do **not** allocate dense GPU `weight`
   - instead they allocate only quant-state placeholders / metadata holders

2. **Meta/CPU placeholder init**
   - TP linear modules allocate placeholder storage on `meta` or CPU
   - loader/runtime later attaches real quantized state

3. **Immediate replacement after construction**
   - dense `weight` exists only transiently in a controlled placeholder form,
   - and is replaced by quant state before any real GPU residency cost is paid

The preferred direction is (1) or (2). The plan should assume that quant mode
**must not create dense GPU weight tensors as part of normal model construction**.

**CUDA graph interaction note**: Python-level forward branches are evaluated once at
graph capture time and then frozen for all subsequent replays. A model captured in
quant mode will always replay the quant branch; a model captured in dense mode will
always replay the dense branch. This is fine as long as the quant op itself is
graph-safe. Phase 0 must verify this for the chosen backend.

### 6.4 When new module classes are allowed

If the runtime backend forces a cleaner separation, use lightweight wrappers like:

- `ColumnParallelAWQLinear`
- `RowParallelAWQLinear`

But only if needed for implementation clarity.

Even in that case:

- forward signatures must match current modules,
- TP semantics must remain unchanged,
- replacement must happen only at module construction / load time.


## 7. Offline Artifact Strategy

### 7.1 AWQ is an offline-first plan

Unlike the current load-time torchao approach, this plan assumes quantization is
done **before** normal SSD serving starts.

This is intentional.

The main runtime should load:

- quantized rank-local artifacts,
- not full dense weights.

### 7.2 External artifact vs SSD-local artifact

We should **not** make the runtime depend directly on raw external AWQ checkpoint
format if we can avoid it.

Preferred flow:

1. original HF checkpoint
2. external AWQ quantization
3. AWQ checkpoint / files
4. **SSD importer**
5. SSD-native rank-local artifact
6. runtime load

This gives us control over:

- TP sharding,
- packed-module naming,
- startup speed,
- version validation,
- exact metadata needed at runtime.

### 7.3 SSD-native artifact contents

The SSD-native AWQ artifact should store, per TP rank:

- quant method: `awq_int4`
- bits
- group size
- zero-point flag
- compute dtype expectation
- source model id / revision
- TP size / rank
- backend kind
- module name list
- per-module:
  - packed quantized weight
  - scales
  - zero-points if used
  - any backend-specific layout metadata needed for direct runtime load

### 7.4 Artifact naming and versioning

Required metadata:

- `artifact_version`
- `quant_scheme`
- `backend`
- `model_id`
- `tp_size`
- `tp_rank`
- `group_size`
- `use_zero_point`
- `expected_runtime_dtype`
- `quantize_lm_head`
- `quantize_embeddings`

We should treat this as a strict runtime contract, not a best-effort cache.


## 8. Loader Plan

### 8.1 Main rule

Do **not** first materialize full dense weights on GPU.

The new loader flow should be:

1. detect AWQ SSD artifact
2. build model modules
3. allocate quantized storage on target device
4. load packed weight/scales/metadata directly
5. skip dense GPU weight load entirely

This implies an additional runtime rule:

- quant mode cannot rely on the current "construct dense GPU parameters first,
  then overwrite them later" behavior.

Loader work alone is not sufficient; the module construction contract in §6.3.1
must also be implemented.

### 8.2 Dense checkpoint load remains only for:

- non-quantized path
- draft path
- unsupported model families during development
- fallback debugging

### 8.3 Import-time CPU work

If we need a one-shot importer path, it can:

- read the external AWQ checkpoint on CPU,
- repack to SSD naming and TP layout on CPU,
- write SSD-native artifact,
- never require dense GPU materialization.


## 9. TP and Packed Module Mapping

### 9.1 Must preserve current SSD semantics

Current SSD uses:

- column-parallel shard rules
- row-parallel shard rules
- packed QKV loading
- packed gate/up loading

These semantics must remain unchanged.

### 9.2 Required mapping work

For Llama-family models:

- `q_proj`, `k_proj`, `v_proj` -> `qkv_proj`
- `gate_proj`, `up_proj` -> `gate_up_proj`

This mapping already exists in dense SSD and must be reused, not reinvented.

### 9.3 AWQ importer responsibility

The importer must define exactly how external AWQ tensors map into SSD packed
module storage. This is the highest implementation-complexity section of the plan.

#### 9.3.1 Concat order for packed modules

SSD packs multiple HF projections into single modules. The importer must concat
external AWQ tensors in this exact order along the **output dimension (dim=0)**:

- `qkv_proj`: `q_proj` → `k_proj` → `v_proj`
- `gate_up_proj`: `gate_proj` → `up_proj`

All associated metadata (scales, zero-points) must be concatenated in the same
order along the same dimension.

#### 9.3.2 Concat-then-shard vs shard-then-concat

Preferred order: **concat first, then TP shard**.

Rationale:

- external AWQ artifacts store per-projection tensors (not pre-packed),
- concatenating first produces the full SSD packed tensor,
- then applying TP shard rules gives the correct per-rank slice.

The alternative (shard each projection, then concat) is also valid but harder to
verify and more error-prone for QKV where q/k/v have different sizes.

#### 9.3.3 TP sharding rules per module type

- **ColumnParallelLinear** (`qkv_proj`, `gate_up_proj`):
  shard along **output dim (dim=0)**. Groups are along input dim, so group
  boundaries are unaffected by this shard.

- **RowParallelLinear** (`o_proj`, `down_proj`):
  shard along **input dim (dim=1)**. Groups are also along input dim, so the
  importer must verify that `shard_size % group_size == 0`. If this fails, the
  importer must reject the configuration.

#### 9.3.4 Group boundary alignment assertion

For every RowParallelLinear module, the importer must assert:

```
input_size_per_partition = input_size // tp_size
assert input_size_per_partition % group_size == 0, \
    f"RowParallel shard size {input_size_per_partition} not divisible by group_size {group_size}"
```

For standard Llama-family models with group_size=128, this holds for all known
configurations (8B, 34B, 70B at TP=1/2/4/8). But the assert must be present.

#### 9.3.5 Scale and zero-point tensor sharding

Scale and zero-point tensors have shape `[out_features, num_groups]` where
`num_groups = in_features // group_size`.

- **ColumnParallel shard**: slice scales along dim=0 (output), keep dim=1 intact.
- **RowParallel shard**: keep dim=0 (output) intact, slice scales along dim=1
  (groups), since groups correspond to input channels which are sharded.

#### 9.3.6 Required unit tests before integration

Before proceeding to runtime integration:

1. round-trip test: concat → shard → load → verify shapes match expected per-rank sizes
2. numerical test: sharded quant matmul == unsharded quant matmul (within expected tolerance)
3. edge-case test: assert failure when group_size does not divide shard size


## 10. Model Scope

### 10.1 Phase-1 model family

First implementation target:

- Llama-family only

Reason:

- the packed module mapping is already clear,
- current SSD quantization/debug work already centered on Llama-family,
- MESA target path also currently matters most there.

### 10.2 Out of scope initially

- Qwen3
- EAGLE draft quantization
- quantized embeddings
- quantized `lm_head`
- cross-family generic importer

Qwen3 can be added after Llama-family integration is stable.

### 10.3 Weight tying note for future optional scope

Both Llama-family and Qwen3 tie `lm_head.weight` to `embed_tokens.weight` when
`tie_word_embeddings=True` in the HF config.

This does not affect the first AWQ integration because:

- embeddings stay dense,
- `lm_head` stays dense,
- quantization targets only the heavy internal projections.

If `lm_head` quantization is revisited later, the implementation must explicitly
choose one of:

- untie `lm_head` before quantization,
- keep both embedding and `lm_head` dense,
- or deliberately support a tied quantized embedding/logit path.


## 11. MESA Policy

### 11.1 Final objective includes MESA

This plan is not complete unless the target quantized path also works in:

- normal autoregressive decode
- verify path
- MESA target verify path

### 11.2 MESA-specific policy

Keep this policy unless data proves otherwise:

- target heavy linear layers: quantized
- `lm_head`: dense by default

This is the safest first policy for MESA proxy quality and accept rate.

### 11.3 MESA validation is not optional

Even if AR and ordinary verify pass, the plan is not done until:

- split MESA verify captures,
- MESA target path correctness,
- accept-rate impact from dense `lm_head` vs quantized `lm_head`

are measured.


## 12. Backend Selection Gate

Before implementation starts, we must choose a concrete runtime backend style.

### 12.1 Candidate classes

We are evaluating **AWQ-style optimized runtimes**, not generic `F.linear` fallback.

Examples of candidate directions:

- AWQ + Marlin-style backend
- AWQ + another optimized runtime backend that supports our dtypes/shapes

### 12.2 Selection criteria

A candidate is accepted only if all of the following hold:

1. works with SSD-relevant small-M decode-like shapes
2. works with SSD prefill-like larger GEMM shapes
3. preserves quantized storage on GPU
4. compatible with TP-local matmul integration
5. compatible with CUDA graph capture or has a clear plan to become so
6. supports at least fp16 runtime; bf16 support strongly preferred

### 12.3 If no candidate passes

If no AWQ runtime candidate passes Phase 0 gates, do **not** pivot to custom
Triton kernel implementation immediately.

Instead:

1. keep current torchao path as temporary bf16 fallback,
2. narrow the backend search further,
3. only consider custom kernel work after a focused backend evaluation document.

Custom kernel work is the last resort, not the default plan.


## 13. Config Design

### 13.1 New config shape

Use a structured config rather than scattering booleans.

Recommended shape:

```python
@dataclass
class QuantConfig:
    enabled: bool = False
    method: str = "none"          # "none" | "awq_int4"
    target: bool = True
    draft: bool = False
    quantize_lm_head: bool = False
    quantize_embeddings: bool = False
    artifact_path: str | None = None
    artifact_mode: str = "load_only"   # "load_only" | "import_then_load"
    runtime_backend: str = "auto"      # concrete backend chosen in Phase 0
    quant_source: str = "ssd_artifact" # "ssd_artifact" | "external_awq"
    external_quant_path: str | None = None
    group_size: int = 128
    use_zero_point: bool = True
```

### 13.2 Default policy

Default policy for initial release:

- `enabled=False`
- `method="none"`
- target-only when enabled
- `quantize_lm_head=False`
- `quantize_embeddings=False`
- offline artifact required

### 13.3 Migration from current flat config

Current SSD code uses flat quantization-related fields such as:

- `target_quant_enabled`
- `target_quant_backend`
- `target_quant_lm_head`
- `target_quant_mode`
- `target_quant_artifact_prefix`

The new structured config should not require a flag-day rewrite of the whole
engine/bench stack.

Migration rule:

1. introduce `QuantConfig`
2. keep existing flat CLI/config fields as a temporary compatibility shim
3. derive `QuantConfig` from those legacy fields at the LLM/runner boundary
4. remove the legacy flat fields only after the AWQ path is stable


## 14. Exact Implementation Phases

## Phase 0. Backend Feasibility Spike

### Objective

Pick the concrete AWQ runtime direction.

### Tasks

1. Measure candidate backend on SSD-relevant local matrix shapes:
   - decode-like small M
   - verify-like small M
   - prefill-like larger M
2. Check fp16 runtime support
3. Check bf16 runtime support
4. Check graph capture safety
5. Check whether local shard shapes are supported

### Deliverable

A short technical note that says:

- chosen backend
- supported runtime dtypes
- known unsupported shapes
- graph compatibility result
- whether SSD should keep torchao as bf16 fallback

### Hard gate

Do not start broad integration before this phase is closed.

### Optional exploratory check

As a low-priority side investigation during Phase 0, check whether AWQ-calibrated
weights can be loaded through the current torchao INT4 runtime path. This requires
matching torchao's internal AQT/tile-packed layout contract, which is non-trivial
and may not be feasible without significant adapter work. Do **not** treat this as a
main simplification path — it is a nice-to-know hypothesis only.


## Phase 1. Runtime Quant State Skeleton

### Objective

Define the in-memory quantized state and the minimum TP-module changes needed so
that an external packed AWQ weight can be attached to a local TP module and
executed.

This phase exists **before** any real loader integration because both the thin
adapter path and the SSD-native artifact path need a runtime destination for
packed weights.

### Tasks

1. Define quant-state ownership for TP linear modules
2. Define dense-vs-quant forward dispatch shape contract
3. Define the quant-mode module construction contract from §6.3.1
4. Define how the existing `weight_loader(param, loaded_weight[, shard_id])`
   contract is preserved or adapted in quant mode
5. Decide whether existing TP classes can be extended in place or need light
   wrappers

### Success criteria

- there is a concrete in-memory representation for packed AWQ weight, scales,
  zero-points, and backend metadata
- TP modules can exist in quant mode without dense GPU weight residency
- the current loader's packed/QKV/merged loader calling convention still has a
  clear quant-mode equivalent
- no loader or importer work is started before this contract is closed


## Phase 2. Runtime Quant State + Local Matmul Adapter

### Objective

Add runtime support for AWQ-backed local linear execution without changing model
topology.

### Tasks

1. Extend TP linear modules to hold quantized state
2. Add a local runtime branch for AWQ-backed matmul
3. Keep existing dense path untouched
4. Preserve row/column/QKV/merged semantics

### Files likely involved

- `ssd/ssd/layers/linear.py`
- possibly `ssd/ssd/layers/embed_head.py` if optional later work touches `lm_head`

### Success criteria

- unit test: local quant path matches dense reference within expected error
- TP semantics still correct
- no dense GPU weight storage required in quant mode


## Phase 3a. Thin Adapter for External AWQ Checkpoint

### Objective

Enable fast backend/runtime validation by loading external AWQ checkpoints directly
into SSD without first building a full SSD-native artifact pipeline.

### Tasks

1. Read external AWQ checkpoint (HF safetensors + `quantize_config.json`)
2. Map HF module names to SSD module names at runtime
3. Repack `qkv_proj` and `gate_up_proj` on CPU
4. Apply TP sharding rules on CPU
5. Load packed weights + scales + zero-points directly into runtime modules

### Files likely involved

- new thin loader/adapter under `ssd/ssd/utils/` or `ssd/ssd/quant/`
- modifications to `ssd/ssd/utils/loader.py` or `ssd/ssd/engine/model_runner.py`

### Success criteria

- external AWQ checkpoint loads into SSD runtime without crash
- no SSD-native artifact file required
- enables end-to-end runtime validation without waiting for Phase 3b


## Phase 3b. SSD-Native Pre-Sharded Artifact Pipeline

### Objective

Build the offline artifact pipeline for production startup speed.

### Tasks

1. Accept an external AWQ checkpoint as input
2. Convert HF module names to SSD module names
3. Repack `qkv_proj` and `gate_up_proj`
4. Apply TP sharding rules
5. Save SSD-native per-rank artifacts
6. Write manifest/version metadata

### Files likely involved

- new importer script under `ssd/scripts/`
- new helper module under `ssd/ssd/utils/` or `ssd/ssd/quant/`

### Success criteria

- importer runs fully on CPU
- produces rank-local artifact files
- artifact can be inspected without SSD runtime boot
- startup time materially faster than Phase 3a thin adapter path


## Phase 4. Loader Integration

### Objective

Load SSD-native AWQ artifacts directly into the runtime.

### Tasks

1. Add artifact detection
2. Add artifact metadata validation
3. Instantiate quant state directly from artifact
4. Skip dense GPU weight materialization
5. Keep dense loader behavior untouched when quant is disabled

### Files likely involved

- `ssd/ssd/utils/loader.py`
- `ssd/ssd/engine/model_runner.py`
- `ssd/ssd/config.py`
- `ssd/bench/bench.py`

### Success criteria

- quantized target loads without first placing dense weights on GPU
- draft still loads via dense path
- startup VRAM materially reduced


## Phase 5. End-to-End Target-Only Validation

### Objective

Make sure the quantized target path works in the real engine.

### Required checks

1. AR decode
2. verify path
3. one speculative path
4. CUDA graph capture/replay
5. prefix cache path
6. TP gather/all_reduce correctness

### Success criteria

- no crash
- correct shapes/dtypes
- stable generation on short prompts
- graph capture succeeds on required hot paths


## Phase 6. MESA Validation

### Objective

Validate quantized target under MESA.

### Required checks

1. split verify capture still works
2. target MESA verify path correctness
3. `lm_head` dense default baseline
4. optional `lm_head` quant ablation if later enabled
5. accept rate / throughput comparison

### Success criteria

- MESA path runs correctly
- no severe accept-rate collapse under default dense `lm_head`


## Phase 7. Performance and Startup Optimization

### Objective

Confirm the backend is actually worth using.

### Required benchmarks

1. decode-like microbench
2. prefill-like microbench
3. end-to-end AR
4. end-to-end spec
5. end-to-end MESA target path

### Compare against

- dense fp16/bf16 baseline
- current torchao fallback where applicable
- AWQ path

### Required analysis

If performance is bad, first check:

- wrong backend selected
- hidden dense materialization
- extra copies
- bad shard packing
- bad small-M behavior
- graph fallback to eager


## 15. Files Expected to Change

Expected primary files:

- `ssd/ssd/config.py`
- `ssd/bench/bench.py`
- `ssd/ssd/utils/loader.py`
- `ssd/ssd/layers/linear.py`
- `ssd/ssd/engine/model_runner.py`

Expected new files:

- AWQ artifact importer script
- AWQ runtime helper module
- validation / smoke / microbench scripts

Possible later files:

- `ssd/ssd/layers/embed_head.py` if `lm_head` quantization is revisited


## 16. Risks

### 16.1 Highest risks

1. chosen AWQ runtime does not support SSD decode-like small-M shapes well
2. graph capture compatibility is weaker than expected
3. external AWQ artifact schema is awkward to map into SSD packed modules
4. bf16 runtime support is weaker than fp16 runtime support
5. AWQ calibration does not yield a meaningful quality improvement over simpler
   round-to-nearest quantization for SSD's use case — in particular, if MESA
   accept rate and generation quality are not materially better than what a
   naïve W4A16 quantization would produce, the complexity of the offline
   calibration pipeline has low ROI

### 16.2 Explicit mitigations

1. Phase 0 must reject weak backends early
2. keep current torchao path as temporary bf16 fallback if needed
3. do not quantize `lm_head` initially
4. do not quantize draft initially
5. measure MESA accept rate with AWQ vs round-to-nearest early in Phase 5; if the
   difference is negligible, consider simplifying the calibration pipeline


## 17. Non-Goals

The following are intentionally out of scope for the first AWQ integration:

- direct bitsandbytes runtime integration
- scratch Triton quant GEMM backend
- draft quantization
- quantized embeddings
- quantized `lm_head` by default
- generic multi-model-family support on day 1
- deleting the current torchao code before AWQ is proven


## 18. Final Recommendation

The correct v2 direction is:

1. **Keep SSD engine architecture intact**
2. **Keep current torchao path only as a temporary fallback**
3. **Add an AWQ-style offline artifact pipeline**
4. **Integrate an optimized AWQ runtime at the local TP linear boundary**
5. **Target-only first**
6. **Llama-family first**
7. **MESA validation is required before calling the work complete**

This is the narrowest plan that still addresses the real problem:

- fp16/bf16 practical support,
- VRAM reduction,
- optimized runtime path,
- SSD architecture preservation.
