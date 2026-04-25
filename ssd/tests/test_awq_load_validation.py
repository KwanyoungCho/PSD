"""Promote two key AWQ load-time validations from sandbox into a real test.

Run from repo root with:
    cd ssd/
    pytest tests/ -q
or directly:
    python -m unittest tests.test_awq_load_validation

The tests do NOT need a GPU: they only exercise the artifact I/O + the
schema/role validation code path, which is the highest-risk surface for
silent misconfiguration.
"""
import os
import sys
import tempfile
import unittest

# Ensure the package root is importable when the file is run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

# Required env vars that ssd.paths reads at import time.
os.environ.setdefault("SSD_HF_CACHE", "/dev/null")
os.environ.setdefault("SSD_DATASET_DIR", "/dev/null")
os.environ.setdefault("SSD_CUDA_ARCH", "8.6")

import torch  # noqa: E402

from ssd.quant.io import (  # noqa: E402
    load_awq_artifact,
    save_awq_artifact,
)


def _write_minimal_artifact(prefix: str, *, role: str, model_id: str = "/dev/null/model") -> str:
    """Stamp a single-rank tp=1 artifact with one fake module entry."""
    in_f, out_f, gs = 128, 128, 128
    modules = {
        "model.layers.0.self_attn.qkv_proj": {
            "qweight": torch.zeros(in_f, out_f // 8, dtype=torch.int32),
            "qzeros":  torch.zeros(in_f // gs, out_f // 8, dtype=torch.int32),
            "scales":  torch.zeros(in_f // gs, out_f, dtype=torch.float16),
            "in_features": in_f,
            "out_features": out_f,
            "group_size": gs,
            "bias": None,
        },
    }
    return save_awq_artifact(
        prefix=prefix, tp_rank=0, tp_size=1, modules=modules,
        model_id=model_id,
        group_size=gs, use_zero_point=True,
        expected_runtime_dtype="float16",
        quantize_lm_head=False, quantize_embeddings=False,
        quant_source="rtn", model_role=role,
    )


class TestRoleValidation(unittest.TestCase):
    """Loader must hard-fail when role does not match expectation."""

    def test_target_role_into_target_loads(self):
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "art")
            _write_minimal_artifact(prefix, role="target")
            art = load_awq_artifact(prefix, tp_rank=0, tp_size=1, expected_role="target")
            self.assertEqual(art["model_role"], "target")
            self.assertIn("model.layers.0.self_attn.qkv_proj", art["modules"])

    def test_draft_role_into_draft_loads(self):
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "art")
            _write_minimal_artifact(prefix, role="draft")
            art = load_awq_artifact(prefix, tp_rank=0, tp_size=1, expected_role="draft")
            self.assertEqual(art["model_role"], "draft")

    def test_target_role_loaded_as_draft_raises(self):
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "art")
            _write_minimal_artifact(prefix, role="target")
            with self.assertRaisesRegex(ValueError, r"model_role"):
                load_awq_artifact(prefix, tp_rank=0, tp_size=1, expected_role="draft")

    def test_draft_role_loaded_as_target_raises(self):
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "art")
            _write_minimal_artifact(prefix, role="draft")
            with self.assertRaisesRegex(ValueError, r"model_role"):
                load_awq_artifact(prefix, tp_rank=0, tp_size=1, expected_role="target")


class TestModelIdValidation(unittest.TestCase):
    """`expected_model_id` must match exactly when given."""

    def test_model_id_match_loads(self):
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "art")
            _write_minimal_artifact(prefix, role="target", model_id="/abs/path/A")
            art = load_awq_artifact(
                prefix, tp_rank=0, tp_size=1,
                expected_model_id="/abs/path/A",
            )
            self.assertEqual(art["model_id"], "/abs/path/A")

    def test_model_id_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "art")
            _write_minimal_artifact(prefix, role="target", model_id="/abs/path/A")
            with self.assertRaisesRegex(ValueError, r"model_id"):
                load_awq_artifact(
                    prefix, tp_rank=0, tp_size=1,
                    expected_model_id="/abs/path/DIFFERENT",
                )


class TestTpValidation(unittest.TestCase):
    """tp_size / tp_rank must match exactly."""

    def test_tp_size_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "art")
            _write_minimal_artifact(prefix, role="target")
            with self.assertRaisesRegex(ValueError, r"tp_size"):
                load_awq_artifact(prefix, tp_rank=0, tp_size=2)


class TestRuntimeDtypeValidation(unittest.TestCase):
    def test_dtype_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "art")
            _write_minimal_artifact(prefix, role="target")  # writes float16
            with self.assertRaisesRegex(ValueError, r"runtime_dtype"):
                load_awq_artifact(
                    prefix, tp_rank=0, tp_size=1,
                    expected_runtime_dtype="bfloat16",
                )


class TestConfigAutoRoute(unittest.TestCase):
    """`quant_config_from_legacy_flags` should auto-route a stale legacy
    backend to awq_marlin when an artifact path is supplied."""

    def test_auto_route_target_when_artifact_set(self):
        from dataclasses import dataclass
        from ssd.quant.config import quant_config_from_legacy_flags

        @dataclass
        class FakeHF:
            torch_dtype = torch.float16

        @dataclass
        class FakeCfg:
            target_quant_enabled: bool = True
            target_quant_backend: str = "int4_wo_tile"  # stale legacy
            target_quant_awq_artifact: str = "/dev/null/prefix"
            target_quant_external_awq_path: str = None
            target_quant_group_size: int = 128
            target_quant_lm_head: bool = False
            hf_config = FakeHF
            draft_hf_config = FakeHF

        qc = quant_config_from_legacy_flags(FakeCfg(), "target")
        self.assertIsNotNone(qc, "auto-route should produce a QuantConfig")
        self.assertTrue(qc.enabled)
        self.assertEqual(qc.role, "target")
        self.assertEqual(qc.artifact_path, "/dev/null/prefix")
        self.assertEqual(qc.quant_source, "ssd_artifact")


if __name__ == "__main__":
    unittest.main()
