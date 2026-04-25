"""Negative tests for the fix pass.

1. Missing-module artifact → loader must raise at load time, NOT first forward.
2. quantize_config.json with zero_point=False → importer must raise.
3. Runtime group_size assertion mismatch (simulate via manual override).

Run:
    cd ssd/
    python sandbox/awq_spike/10_negative_checks.py
"""
import json
import os
import shutil
import sys
import tempfile
from glob import glob

os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
os.environ.setdefault("SSD_CUDA_ARCH", "8.6")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("SSD_DIST_PORT", "13270")

import torch

FAKE_AWQ_DIR = "/tmp/fake_autoawq_1b"
FAKE_AWQ_ART = "/tmp/awq_artifacts/fake_autoawq_1b"


def _expect_failure(label, fn, *, error_substring):
    print(f"\n[neg-test] {label}")
    try:
        fn()
    except Exception as e:
        if error_substring in str(e):
            print(f"  OK — raised {type(e).__name__}: {str(e)[:150]}")
            return
        print(f"  UNEXPECTED {type(e).__name__}: {e}")
        raise
    raise AssertionError(f"{label}: expected failure with '{error_substring}', got no exception")


def test_missing_module_fails_at_load():
    """Delete one module from artifact → apply_ssd_awq_artifact must raise."""
    assert os.path.isfile(f"{FAKE_AWQ_ART}.rank0.awq.pt"), \
        f"Run 09_fake_autoawq_roundtrip.py first to produce {FAKE_AWQ_ART}"

    # Make a copy of the artifact and drop one module
    damaged = "/tmp/awq_artifacts/fake_autoawq_1b_damaged"
    shutil.copy2(f"{FAKE_AWQ_ART}.rank0.awq.pt", f"{damaged}.rank0.awq.pt")
    art = torch.load(f"{damaged}.rank0.awq.pt", map_location="cpu", weights_only=False)
    # Remove one linear module from the artifact (simulate a truncated import)
    victim = "model.layers.0.self_attn.qkv_proj"
    assert victim in art["modules"], f"missing: {victim}"
    del art["modules"][victim]
    art["ssd_module_names"] = sorted(art["modules"].keys())
    torch.save(art, f"{damaged}.rank0.awq.pt")
    print(f"  dropped {victim} from artifact ({len(art['modules'])} modules left)")

    def _run():
        from ssd import LLM
        LLM(
            model=FAKE_AWQ_DIR,
            num_gpus=1,
            max_model_len=256, max_num_seqs=1, max_num_batched_tokens=256,
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            target_quant_enabled=True,
            target_quant_backend="awq_marlin",
            target_quant_awq_artifact=damaged,
        )

    _expect_failure(
        "missing module in artifact → load-time hard fail",
        _run,
        error_substring="did not provide quant state",
    )


def test_importer_rejects_zero_point_false():
    """Writing quantize_config with zero_point=False must hard-fail the importer."""
    dst = tempfile.mkdtemp(prefix="fake_awq_nozp_")
    # Copy everything from FAKE_AWQ_DIR (already built) and rewrite quantize_config.json
    for name in os.listdir(FAKE_AWQ_DIR):
        src = os.path.join(FAKE_AWQ_DIR, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst, name))
    with open(os.path.join(dst, "quantize_config.json"), "w") as f:
        json.dump({"q_group_size": 128, "zero_point": False, "w_bit": 4}, f)

    def _run():
        from ssd.quant.importer import import_autoawq_to_ssd_artifact
        import_autoawq_to_ssd_artifact(
            model_path=dst,
            out_prefix="/tmp/awq_artifacts/nozp",
            tp_size=1, group_size=128,
            expected_runtime_dtype="bfloat16",
        )

    _expect_failure(
        "quantize_config zero_point=False → importer hard-fails",
        _run,
        error_substring="zero_point=False not supported",
    )
    shutil.rmtree(dst, ignore_errors=True)


def test_importer_rejects_w_bit_mismatch():
    dst = tempfile.mkdtemp(prefix="fake_awq_w8_")
    for name in os.listdir(FAKE_AWQ_DIR):
        src = os.path.join(FAKE_AWQ_DIR, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst, name))
    with open(os.path.join(dst, "quantize_config.json"), "w") as f:
        json.dump({"q_group_size": 128, "zero_point": True, "w_bit": 8}, f)

    def _run():
        from ssd.quant.importer import import_autoawq_to_ssd_artifact
        import_autoawq_to_ssd_artifact(
            model_path=dst,
            out_prefix="/tmp/awq_artifacts/w8",
            tp_size=1, group_size=128,
            expected_runtime_dtype="bfloat16",
        )

    _expect_failure(
        "w_bit != 4 → importer hard-fails",
        _run,
        error_substring="w_bit=8 not supported",
    )
    shutil.rmtree(dst, ignore_errors=True)


def test_runtime_group_size_mismatch():
    """--quant_group_size at runtime that disagrees with the artifact must fail."""
    def _run():
        from ssd import LLM
        LLM(
            model=FAKE_AWQ_DIR,
            num_gpus=1,
            max_model_len=256, max_num_seqs=1, max_num_batched_tokens=256,
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            target_quant_enabled=True,
            target_quant_backend="awq_marlin",
            target_quant_awq_artifact=FAKE_AWQ_ART,
            target_quant_group_size=64,    # artifact has 128
        )

    _expect_failure(
        "runtime --quant_group_size disagrees with artifact → load-time fail",
        _run,
        error_substring="group_size",
    )


if __name__ == "__main__":
    test_importer_rejects_zero_point_false()
    test_importer_rejects_w_bit_mismatch()
    test_missing_module_fails_at_load()
    test_runtime_group_size_mismatch()
    print("\n[ALL NEGATIVE TESTS PASSED]")
