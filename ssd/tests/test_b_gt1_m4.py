"""M4 unit tests for B>1 support (docs/duet/13-b-gt-1-design.md, stage M4).

Tests the REAL Config.__post_init__ (CPU-only: AutoConfig reads local
config.json, no weights) with the champion DUET args:

(a) Gate lift (design §7, cap 32 since bscale32): max_num_seqs ∈ {1, 2, 8, 32} construct fine;
    9 fails the new <=8 assert.
(b) B==1-only gate guard (design §6): each of SSD_DUET_EXIT_TOPM_GATHER /
    SSD_DUET_EXIT_REPLICA / SSD_DUET_PROXY_ON_DRAFT raises ValueError at
    max_num_seqs=2 but stays constructible at max_num_seqs=1.

Run from project root (/home/chokwans99/PSD/ssd):
    python -m unittest tests.test_b_gt1_m4

Requires the local model dirs (config.json only):
    /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
    /data2/chokwans99/awq_calibrated/tinyllama_1b
"""
import os
import unittest

os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/data2/chokwans99/datasets")
os.environ["SSD_FORCE_SPLIT_K1K2"] = "1"

from ssd.config import Config

TARGET = "/data2/chokwans99/awq_calibrated/layerskip_llama2_70b"
DRAFT = "/data2/chokwans99/awq_calibrated/tinyllama_1b"

GATES = (
    "SSD_DUET_EXIT_TOPM_GATHER",
    "SSD_DUET_EXIT_REPLICA",
    "SSD_DUET_PROXY_ON_DRAFT",
)


def _champion_config(max_num_seqs: int) -> Config:
    """Champion E9K24_jit shape (k=13, K1=9, K2=4, exit 56, dfo=2)."""
    return Config(
        model=TARGET,
        draft=DRAFT,
        max_num_seqs=max_num_seqs,
        num_gpus=5,
        speculate=True,
        draft_async=True,
        jit_speculate=True,
        speculate_k=13,
        duet_enabled=True,
        duet_exit_layer=56,
        duet_phase1_k=9,
        duet_phase2_k=4,
        duet_draft_fan_out=2,
        duet_policy="b",
        # M4 tests the legacy B>1 chain and proxy-gate contract.  Dynamic
        # P2 tree is now the DUET default, so keep this comparison explicit;
        # B>1 tree has its own separately gated implementation roadmap.
        duet_tree_policy="off",
        duet_split_phase1_fan_out_list=[2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
    )


class GateEnvMixin:
    def setUp(self):
        self._saved = {g: os.environ.pop(g, None) for g in GATES}

    def tearDown(self):
        for g, v in self._saved.items():
            if v is None:
                os.environ.pop(g, None)
            else:
                os.environ[g] = v


class TestM4GateLift(GateEnvMixin, unittest.TestCase):
    def test_b1_still_constructs(self):
        cfg = _champion_config(1)
        self.assertEqual(cfg.max_num_seqs, 1)

    def test_b2_constructs(self):
        cfg = _champion_config(2)
        self.assertEqual(cfg.max_num_seqs, 2)

    def test_b8_boundary_constructs(self):
        cfg = _champion_config(8)
        self.assertEqual(cfg.max_num_seqs, 8)

    def test_b32_boundary_constructs(self):
        cfg = _champion_config(32)
        self.assertEqual(cfg.max_num_seqs, 32)

    def test_b33_rejected(self):
        with self.assertRaises(AssertionError):
            _champion_config(33)


class TestM4B1OnlyGateGuard(GateEnvMixin, unittest.TestCase):
    def test_each_gate_rejected_at_b2(self):
        for gate in GATES:
            with self.subTest(gate=gate):
                os.environ[gate] = "1"
                try:
                    with self.assertRaises(ValueError) as cm:
                        _champion_config(2)
                    self.assertIn("B==1-only", str(cm.exception))
                finally:
                    del os.environ[gate]

    def test_gates_still_allowed_at_b1(self):
        # Each gate alone at B=1 must remain constructible (they are
        # mutually exclusive with each other, so test one at a time).
        # PROXY_ON_DRAFT / TOPM_GATHER additionally require TOPM >= the
        # auto-raised top_k (24 >= 14 for champion shapes — default OK).
        for gate in GATES:
            with self.subTest(gate=gate):
                os.environ[gate] = "1"
                try:
                    cfg = _champion_config(1)
                    self.assertEqual(cfg.max_num_seqs, 1)
                finally:
                    del os.environ[gate]


if __name__ == "__main__":
    unittest.main()
