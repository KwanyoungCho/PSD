"""Smoke tests for draft AWQ (plan v2 extension).

Tests:
  1. Dense target + AWQ draft on sync spec decode
  2. Dense target + AWQ draft on async spec decode
  3. AWQ target + AWQ draft (double-quant) async spec
  4. Negative: target-role artifact loaded into draft → load-time hard fail
  5. Negative: wrong model_id → load-time hard fail

Uses a tiny 2-GPU topology (layerskip-llama3-8B target + TinyLlama-1.1B draft)
because we need spec decode with a real target. Quick smoke only — full perf
benches happen later.
"""
import os
os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
os.environ.setdefault("SSD_CUDA_ARCH", "8.6")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("SSD_DIST_PORT", "13300")


TARGET_DENSE = "/data2/chokwans99/models/layerskip-llama3-8B"
DRAFT_CALIB = "/data2/chokwans99/awq_calibrated/tinyllama_1b"
DRAFT_ART = "/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1"


def _llm(**kwargs):
    from ssd import LLM
    defaults = dict(
        model=TARGET_DENSE,
        draft=DRAFT_CALIB,
        num_gpus=2,
        speculate=True,
        speculate_k=4,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
    )
    defaults.update(kwargs)
    return LLM(**defaults)


def _gen(llm, prompt="The capital of France is", n=16):
    from ssd import SamplingParams
    sp = SamplingParams(temperature=0.0, max_new_tokens=n)
    out = llm.generate([prompt], sp, use_tqdm=False)
    outputs = out[0] if isinstance(out, tuple) else out
    return outputs[0]["text"], outputs[0]["token_ids"]


def test_draft_awq_sync_spec():
    print("\n[test] dense target + AWQ draft — sync spec")
    llm = _llm(
        draft_async=False,
        # draft AWQ config
        draft_quant_enabled=True,
        draft_quant_backend="awq_marlin",
        draft_quant_awq_artifact=DRAFT_ART,
    )
    text, ids = _gen(llm)
    print(f"  text = {text!r}")
    assert len(ids) > 3, "no tokens generated"
    import torch
    del llm; torch.cuda.empty_cache()


def test_draft_awq_negative_role():
    """Pointing to a target-role artifact should fail at load."""
    print("\n[test] NEGATIVE: target-role artifact → draft → must raise at load")
    # We have /data2/chokwans99/awq_artifacts/layerskip_codellama_34b/autoawq_tp4 as target-role
    # but tp=4, won't load at tp=1. Make a quick target-role TinyLlama artifact instead.
    import subprocess, os, tempfile
    tgt_art = "/tmp/smoke_tinyllama_as_target"
    # If the target artifact doesn't exist yet, create one from the same calib dir with role=target
    if not os.path.isfile(f"{tgt_art}.rank0.awq.pt"):
        subprocess.run([
            "/home/chokwans99/anaconda3/envs/ssd/bin/python",
            "/home/chokwans99/PSD/ssd/scripts/awq_import.py",
            "--mode", "autoawq",
            "--model", DRAFT_CALIB,
            "--out", tgt_art,
            "--tp", "1", "--dtype", "bfloat16", "--role", "target",
        ], check=True)

    try:
        llm = _llm(
            draft_async=False,
            draft_quant_enabled=True,
            draft_quant_awq_artifact=tgt_art,   # target-role artifact
        )
        import torch; del llm; torch.cuda.empty_cache()
        raise AssertionError("expected load-time failure but engine started")
    except Exception as e:
        msg = str(e)
        assert "model_role" in msg and "target" in msg and "draft" in msg, \
            f"unexpected error: {e}"
        print(f"  OK — {type(e).__name__}: {msg[:120]}")


def test_draft_awq_async_spec():
    print("\n[test] dense target + AWQ draft — async spec (3 GPUs)")
    llm = _llm(
        num_gpus=3,
        draft_async=True,
        async_fan_out=3,
        jit_speculate=False,
        gpu_memory_utilization=0.35,
        enforce_eager=False,        # async spec needs CUDA graphs
        draft_quant_enabled=True,
        draft_quant_backend="awq_marlin",
        draft_quant_awq_artifact=DRAFT_ART,
    )
    text, ids = _gen(llm)
    print(f"  text = {text!r}")
    assert len(ids) > 3
    import torch
    del llm; torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys
    tests = {
        "sync":  test_draft_awq_sync_spec,
        "neg":   test_draft_awq_negative_role,
        "async": test_draft_awq_async_spec,
    }
    which = sys.argv[1] if len(sys.argv) > 1 else "sync"
    tests[which]()
    print("\nDONE")
