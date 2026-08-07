"""Equivalence tests for the args/config cleanup (docs/duet/16, Tier 1-3).

Contract: the new interface (canonical aliases, env retirement,
duet_p2_budget) must produce a Config identical to the old interface
(env-driven, formula-derived) for the champion shape — field-for-field.

CPU-only; needs the champion model dirs (AutoConfig reads config.json).
Run from project root (/home/chokwans99/PSD/ssd):
    python -m unittest tests.test_args_cleanup_equiv
"""
import dataclasses
import os
import unittest

_TM = "/data2/chokwans99/awq_calibrated/layerskip_llama2_70b"
_DM = "/data2/chokwans99/awq_calibrated/tinyllama_1b"
_HAVE_MODELS = os.path.isdir(_TM) and os.path.isdir(_DM)

_CHAMPION_KW = dict(
    model=_TM, draft=_DM, num_gpus=5, speculate=True, speculate_k=13,
    draft_async=True, async_fan_out=3, duet_enabled=True,
    duet_phase1_k=9, duet_phase2_k=4, duet_exit_layer=56,
    duet_draft_fan_out=2, jit_speculate=True, enforce_eager=False,
    duet_split_phase1_fan_out_list=[2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
)
_SKIP_ATTRS = {"hf_config", "draft_hf_config"}


_LAST_EXPORTS = {}  # env values right after construction (before restore)


def _mk(env, **overrides):
    """Construct a Config under a controlled env snapshot."""
    from ssd.config import Config
    saved = {k: os.environ.get(k)
             for k in ("SSD_FORCE_SPLIT_K1K2", "SSD_DUET_JIT_SHORT")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update(env)
        cfg = Config(**{**_CHAMPION_KW, **overrides})
        _LAST_EXPORTS.clear()
        _LAST_EXPORTS.update({k: os.environ.get(k) for k in saved})
        return cfg
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _diff(a, b):
    return [(k, getattr(a, k), getattr(b, k)) for k in vars(a)
            if k not in _SKIP_ATTRS and getattr(a, k) != getattr(b, k)]


@unittest.skipUnless(_HAVE_MODELS, "champion model dirs not present")
class TestOldNewEquivalence(unittest.TestCase):
    def test_dynamic_tree_is_the_duet_default_and_chain_is_explicit(self):
        dynamic = _mk({})
        self.assertEqual(dynamic.duet_p1_tree_policy, "off")
        self.assertEqual(dynamic.duet_p2_tree_policy, "on")
        self.assertEqual(dynamic.duet_tree_policy, "hybrid")
        self.assertEqual(dynamic.duet_p2_seed_count,
                         dynamic.duet_proxy_total_budget)

        smaller_r = _mk({}, duet_tree_root_count=6)
        self.assertEqual(smaller_r.duet_tree_policy, "hybrid")
        self.assertEqual(smaller_r.duet_p2_active_root_count, 6)

        chain = _mk({}, duet_p2_tree_policy="off")
        self.assertEqual(chain.duet_p2_tree_policy, "off")
        self.assertEqual(chain.duet_tree_policy, "off")

    def test_legacy_tree_names_normalize_to_public_on_off(self):
        dynamic = _mk({}, duet_p2_tree_policy="off",
                      duet_tree_policy="eagle", duet_tree_nv=7)
        self.assertEqual(dynamic.duet_p2_tree_policy, "on")
        self.assertEqual(dynamic.duet_tree_policy, "eagle")
        self.assertEqual(dynamic.duet_p2_tree_max_nodes, 7)
        self.assertEqual(dynamic.duet_tree_nv, 7)

        chain = _mk({}, duet_tree_policy="off")
        self.assertEqual(chain.duet_p2_tree_policy, "off")
        self.assertEqual(chain.duet_tree_policy, "off")

    def test_normalized_config_is_dataclasses_replace_safe(self):
        original = _mk({}, duet_p1_tree_policy="on",
                       duet_p2_tree_policy="on")
        copied = dataclasses.replace(original)
        self.assertEqual(copied.duet_p1_tree_policy, "on")
        self.assertEqual(copied.duet_p2_tree_policy, "on")
        self.assertEqual(copied.duet_tree_policy, "hybrid")
        self.assertEqual(copied.duet_p1_tree_max_nodes, 18)
        self.assertEqual(copied.duet_p2_tree_max_nodes, 8)

    def test_wire_capacity_is_derived_from_phase_maxima(self):
        c = _mk({}, duet_p1_tree_policy="on",
                duet_p1_tree_max_nodes=12,
                duet_p2_tree_max_nodes=8)
        self.assertEqual(c.duet_tree_wire_nodes, 12)
        self.assertEqual(c.duet_response_token_width, 13)

    def test_tree_response_can_exceed_chain_depth(self):
        c = _mk({}, duet_p1_tree_policy="on",
                duet_p1_tree_max_nodes=18,
                duet_p2_tree_max_nodes=8)
        self.assertEqual(c.speculate_k, 13)
        self.assertEqual(c.duet_tree_wire_nodes, 18)
        self.assertEqual(c.duet_response_token_width, 18)

    def test_p1_tree_node_envelope_keeps_a_complete_backbone(self):
        with self.assertRaisesRegex(ValueError, "N1=8, K1=9"):
            _mk({}, duet_p1_tree_policy="on",
                duet_p1_tree_max_nodes=8)
        with self.assertRaisesRegex(ValueError, "N1=28, K1=9, C=3"):
            _mk({}, duet_p1_tree_policy="on",
                duet_p1_tree_max_nodes=28)

    def test_env_free_equals_old_env_style(self):
        # Old style: champion scripts exported both envs.
        old = _mk({"SSD_FORCE_SPLIT_K1K2": "1", "SSD_DUET_JIT_SHORT": "1"})
        new = _mk({})  # Tier-2: --duet implies split; jit_short default ON
        self.assertEqual(_diff(old, new), [])
        self.assertTrue(new.duet_jit_short)

    def test_direct_budget_equals_derived(self):
        old = _mk({"SSD_FORCE_SPLIT_K1K2": "1", "SSD_DUET_JIT_SHORT": "1"})
        direct = _mk({}, duet_p2_budget=10)  # == pfo(1) * (K1+1)(10)
        self.assertEqual(old.duet_proxy_total_budget,
                         direct.duet_proxy_total_budget)
        self.assertEqual(old.duet_proxy_wire_N, direct.duet_proxy_wire_N)
        self.assertEqual(old.duet_proxy_top_k, direct.duet_proxy_top_k)
        for K in (4, 9):  # per-step helper (verifier axis)
            self.assertEqual(old.duet_p2_budget_at(K),
                             direct.duet_p2_budget_at(K))

    def test_budget_at_default_formula(self):
        c = _mk({})
        for K in (1, 4, 9):
            self.assertEqual(c.duet_p2_budget_at(K),
                             c.duet_proxy_fan_out * (K + 1))

    def test_env_jit_short_off_still_wins(self):
        # Old scripts that explicitly disabled jit-short keep their behavior.
        c = _mk({"SSD_DUET_JIT_SHORT": "0"})
        self.assertFalse(c.duet_jit_short)
        self.assertEqual(_LAST_EXPORTS.get("SSD_DUET_JIT_SHORT"), "0")

    def test_k2_gt_k1_early_raise(self):
        with self.assertRaises(ValueError):
            _mk({}, duet_phase1_k=4, duet_phase2_k=9)

    def test_missing_k1k2_early_raise(self):
        with self.assertRaises(ValueError):
            _mk({}, duet_phase1_k=None, duet_phase2_k=None,
                duet_split_phase1_fan_out_list=None)


if __name__ == "__main__":
    unittest.main()
