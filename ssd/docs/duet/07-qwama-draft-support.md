# Qwama (Qwen2 + Llama-3 vocab) draft support

Status tracker for adding `turboderp/Qwama-0.5B-Instruct` as a usable draft
for Llama-3 family targets in the SSD engine.

## 0. Motivation

Existing draft inventory by target family:

| target family | smallest viable draft | ratio (vs 70B) |
|---|---|---|
| Llama-2 | TinyLlama-1.1B-Chat | 63x |
| Llama-2 | AMD-Llama-135m | 518x |
| Llama-3 | Llama-3.2-1B-Instruct | 70x (vs 3.3-70B), 8x (vs 8B) |
| Qwen3 | Qwen3-0.6B | 53x (vs 32B) |

Llama-3-70B + Llama-3.2-1B already gives a ~70x ratio, but the 1B draft is
still relatively large. The community-released
[`turboderp/Qwama-0.5B-Instruct`](https://huggingface.co/turboderp/Qwama-0.5B-Instruct)
is **Qwen2-0.5B architecture with the original Llama-3 vocabulary transplanted
in and re-finetuned**. The author's reported numbers (greedy, 4-bit):

| target | draft | code | prose |
|---|---|---|---|
| Llama3-70B-instruct | Qwama-0.5B | 3.72x | 1.92x |
| Qwen2-72B-instruct | Qwen2-0.5B | 3.68x | 1.70x |

This gives Llama-3-70B users a 140x-ratio draft option, and offers a clean
cross-architecture experiment in the paper.

## 1. Why the codebase does not run it as-is

`ssd/engine/model_runner.py:488-493` dispatches model construction by
`hf_config.model_type`:

```python
elif hf_config.model_type == 'llama':
    model_class = LlamaForCausalLM
elif hf_config.model_type == 'qwen3':
    model_class = Qwen3ForCausalLM
else:
    raise ValueError(f"Unsupported model type: {hf_config.model_type}")
```

Qwama's `config.json` declares `model_type: "qwen2"`, which falls through to
the raise. Also `ssd/utils/misc.py::infer_model_family()` does Llama/Qwen
substring matching on the path and requires
`target_family == draft_family` (enforced in `llm_engine.py:60`). Llama-3-8B
path matches `"llama"`, `"qwama"` matches neither token cleanly.

Qwen3 and Qwen2 differ in two real ways the loader cares about:

| feature | Qwen3 | Qwen2 |
|---|---|---|
| Attention QKV bias | none | `q_proj.bias`, `k_proj.bias`, `v_proj.bias` |
| Q / K per-head norm | RMSNorm before RoPE | absent |
| `tie_word_embeddings` | usually false | typically true (Qwama: true) |

Forcing Qwama weights through `Qwen3ForCausalLM` would silently produce
garbage (missing q/k norm, dropped qkv bias).

## 2. Design choice

Add a real `Qwen2ForCausalLM` to `ssd/models/qwen2.py`. Use `qwen3.py` as the
template since the only diffs are the two attention features above and the
embedding tying.

Family check: extend `infer_model_family` so `qwama` resolves to the
`qwen` family, and allow `target=llama` × `draft=qwen` to pass **iff the
two configs share `vocab_size`**. Vocab equality is the only structural
requirement for token-id-level speculation; same-family was always a
proxy for it.

KV cache and Triton constraints (already verified):

- Qwama: `num_kv_heads=2`, `head_dim=64` → `D = 128`, power of 2 → existing
  `store_kvcache_kernel` patch is not exercised but the kernel still works.
- `torch_dtype = bfloat16` → matches Llama-3-8B target, no fp16 wrapper
  dtype concerns.

## 3. Implementation phases

Each phase is a single small commit so a parallel session can
revert / inspect cleanly.

### Phase 0 — design doc (this file)
- Add `docs/duet/07-qwama-draft-support.md`.
- Commit: `docs(duet): 07 qwama draft support plan`.

### Phase 1 — add Qwen2 model file
- New `ssd/models/qwen2.py`, copied from `qwen3.py` then:
  - Add `bias=True` on `QKVParallelLinear` in attention.
  - Remove Q-norm / K-norm modules and their forward calls.
  - Make `tie_word_embeddings` honor `config.tie_word_embeddings` if the
    upstream class did not.
  - Keep `Qwen2ForCausalLM.packed_modules_mapping` matching HF Qwen2
    checkpoint key layout
    (`model.layers.{i}.self_attn.q_proj.weight`,
    `model.layers.{i}.self_attn.q_proj.bias`, …).
- No engine wiring yet. Commit: `feat(qwama): add Qwen2 model file`.

### Phase 2 — wire dispatch + relax family check
- `ssd/engine/model_runner.py`: add `elif hf_config.model_type == 'qwen2':
  model_class = Qwen2ForCausalLM`.
- `ssd/utils/misc.py`: extend `infer_model_family` so `qwama` substring
  resolves to `"qwen"`. Add a vocab-based override path in
  `llm_engine.py`: target/draft families differ → check
  `target_hf_config.vocab_size == draft_hf_config.vocab_size` and allow
  if equal (with a one-line log).
- Commit: `feat(qwama): dispatch qwen2 + relax cross-family check by vocab`.

### Phase 3 — download Qwama (no commit, blob is not in repo)
- `HF_HUB_CACHE=/data2/chokwans99/models huggingface-cli download
  turboderp/Qwama-0.5B-Instruct`.
- Verify config keys, vocab_size, dtype.

### Phase 4 — smoke test
- Llama-3.1-8B-Instruct (target) + Qwama-0.5B-Instruct (draft),
  async-only spec, `--gpus 2`, `--numseqs 2 --output_len 32`.
- Validate: model loads, both KV caches allocate, decode completes,
  output is plausible text. Record headline numbers in this doc.

### Phase 5 — paper-config baseline
- Same config as `ssd_dense_7b_amd135m_split` runner but with Llama-3-8B
  target and Qwama draft, no DUET. Compare against the existing
  Llama-3-8B + Llama-3.2-1B baseline.
- Add a `experiments/.../async_sd_llama3_8b_qwama_smoke/` directory with
  the runner script. Commit: `feat(qwama): async SD baseline runner +
  smoke result`.

### Phase 6 — DUET on top
- Reuse split-K1/K2 runner template, swap target + draft. Verify proxy
  send / receive paths work across cross-arch draft (Qwama uses
  different hidden-size; proxy payload is tokens + topk logits, both are
  vocab-indexed so cross-arch should be transparent).

## 4. Validation rules

The user explicitly required parallel-safe execution. Rules followed:

- Each phase is a single commit touching only the files listed above.
- Do not stage unrelated working-tree changes
  (`scheduler.py`, `speculator_async.py`, `step.py`,
  `async_spec_helpers.py`, `plot_duet_aligned_timeline.py`,
  `_archive_20260512_cleanup/` deletions) — those are an in-flight WIP
  on the same branch.
- This doc is updated at the end of each phase with status + commit hash.

## 5. Phase status

| phase | status | commit | notes |
|---|---|---|---|
| 0 — design doc | done | `b093640` | initial plan |
| 1 — qwen2.py | done | `6fdd568` | qkv bias + no q/k norm + Qwen2Config |
| 2 — dispatch + family check | done | `3d9f863` | model_type=qwen2 dispatch, vocab-match cross-family override; +`d089470` head_dim inject on Config.__post_init__ |
| 3 — download | done | (no commit) | `/data2/.../models--turboderp--Qwama-0.5B-Instruct/` (1.0 GB) |
| 4 — smoke | done | (no commit) | Llama-3.1-8B + Qwama-0.5B, --gpus 3 --async --spec --k 5 --f 2 --numseqs 2 --output_len 32 → generations OK, total 31.90 tok/s, decode 103.58 tok/s, cache hit 0.42, cross-family allowed log present. Logs at `experiments/qwama_smoke/run.log`. Smoke required --gpus 3 (target TP=2 + draft) on RTX 3090; --gpus 2 OOMs because Llama-3-8B bf16 + workspaces + KV reservation overruns a single 24 GiB card. |
| 5 — paper-config baseline | pending | | match `ssd_dense_7b_amd135m_split` shape, swap target/draft, compare against Llama-3-8B + Llama-3.2-1B baseline |
| 6 — DUET on top | pending | | reuse split-K1/K2 runner |

## 6. Issues discovered + fixes (Phase 4)

### 6.1 Two head_dim fix sites needed

Symptom: `AttributeError: 'Qwen2Config' object has no attribute 'head_dim'`.

Surface: cudagraph_helpers.capture_fi_tree_decode_cudagraph reads
`config.hf_config.head_dim` (not model_runner.hf_config). Llama and
Qwen3 expose head_dim on the config; Qwen2 does not.

Fix: ``_ensure_head_dim`` helper in `ssd/config.py` called from
`Config.__post_init__` immediately after each `AutoConfig.from_pretrained`.
Covers both target and draft hf_configs. Idempotent — safe for configs
that already have a non-None head_dim.

### 6.2 OOM on --gpus 2 with Llama-3.1-8B target

Symptom: ``torch.OutOfMemoryError`` on Llama-3 init even with clean
GPU state.

Cause: not a code bug. Llama-3.1-8B (bf16) = 16 GiB weights, plus
workspaces (~512 MiB), prefill_wrapper buffers, KV-cache reservation,
and CUDA-graph capture state push a single 24 GiB RTX 3090 past the
limit. Same fits comfortably when target is split across two GPUs
(TP=2, ~8 GiB per GPU).

Workaround: use `--gpus 3` so target TP=2 + draft = 3 GPUs total. This
is the same layout already used elsewhere in the codebase for 8B
targets; not Qwama-specific.

Both fixes / observations apply equally to any same-vocab cross-arch
draft pairing, not just Qwama.
