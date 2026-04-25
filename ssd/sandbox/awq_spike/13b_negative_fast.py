"""Fast negative test: load target-role artifact into draft, expect hard fail.

Skips the full engine init — directly calls the loader. Much faster than
spinning up a full LLM process.
"""
import os
os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
os.environ.setdefault("SSD_CUDA_ARCH", "8.6")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

import torch
from ssd.quant.io import load_awq_artifact


def test_wrong_role():
    path = "/tmp/smoke_tinyllama_as_target"  # stamped role=target
    try:
        art = load_awq_artifact(path, tp_rank=0, tp_size=1, expected_role="draft")
    except ValueError as e:
        msg = str(e)
        assert "model_role" in msg and "target" in msg and "draft" in msg, \
            f"unexpected error: {e}"
        print(f"[OK] wrong-role load raised ValueError: {msg[:150]}")
        return
    raise AssertionError("expected ValueError but got artifact")


def test_correct_role():
    path = "/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1"
    art = load_awq_artifact(path, tp_rank=0, tp_size=1, expected_role="draft")
    assert art["model_role"] == "draft"
    print(f"[OK] correct-role (draft→draft) loaded: schema=v{art['schema_version']}")


def test_target_into_target():
    """Sanity: target-role into target must still succeed."""
    path = "/tmp/smoke_tinyllama_as_target"
    art = load_awq_artifact(path, tp_rank=0, tp_size=1, expected_role="target")
    assert art["model_role"] == "target"
    print(f"[OK] target-role (target→target) loaded")


if __name__ == "__main__":
    test_wrong_role()
    test_correct_role()
    test_target_into_target()
    print("\nALL NEG+POS TESTS PASS")
