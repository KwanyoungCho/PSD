"""Regression cases from the independent MLSys-ver0.1 review."""
import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")

import numpy as np
import torch

from ssd.engine.draft_runner import _should_run_p2_tree, _tree_temperatures_supported
from ssd.engine.helpers import p2_tree as PT
from ssd.utils.verify import verify


class ReviewRegressions(unittest.TestCase):
    @staticmethod
    def devices():
        return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    def test_generic_greedy_recovery_uses_real_proposal_end(self):
        for device in self.devices():
            for mixed in (False, True):
                for lengths in ([4, 2], [4, 0]):
                    for jit in (False, True):
                        with self.subTest(device=device, mixed=mixed,
                                          lengths=lengths, jit=jit):
                            p = torch.full((2, 5, 8), -100., device=device)
                            for pos in range(5):
                                p[:, pos, pos + 1] = 100.
                            q = p[:, :4].clone()
                            spec = torch.tensor([[0, 1, 2, 3, 4]] * 2,
                                                device=device)
                            temps = torch.tensor([.7 if mixed else 0., 0.],
                                                 device=device)
                            suffix, rec = verify(
                                p, q, spec, temps, temps,
                                cache_hits=torch.ones(2, device=device),
                                jit_speculate=jit,
                                valid_k=torch.tensor(lengths, device=device))
                            self.assertEqual(suffix[1], list(range(lengths[1] + 1)))
                            self.assertEqual(rec[1], lengths[1] + 1)

    def test_generic_matches_specialized_greedy_with_padding_and_rejection(self):
        for device in self.devices():
            for mismatch in (None, 0, 1, 3):
                for lengths in ([4, 2], [4, 0], [4, 4]):
                    with self.subTest(device=device, mismatch=mismatch,
                                      lengths=lengths):
                        p = torch.zeros(2, 5, 8, device=device)
                        for pos in range(5):
                            p[:, pos, pos + 1] = 10
                        spec = torch.tensor([[0, 1, 2, 3, 4]] * 2,
                                            device=device)
                        if mismatch is not None:
                            spec[1, mismatch + 1] = 7
                        temps = torch.zeros(2, device=device)
                        kwargs = dict(valid_k=torch.tensor(lengths, device=device))
                        expected = verify(p, None, spec, None, None,
                                          all_greedy=True, **kwargs)
                        actual = verify(p, None, spec, temps, temps, **kwargs)
                        self.assertEqual(actual, expected)

    def test_rollout_fallbacks_greedy_first_child_ties_scores_and_rng(self):
        for device in self.devices():
            for arena in (False, True):
                for c in (1, 3):
                    with self.subTest(device=device, arena=arena, c=c):
                        # Includes a tie: torch.topk alone need not return
                        # torch.argmax's first index as the first child.
                        logits = torch.tensor([[3., 3., 1., -1., -2., -3.]],
                                              device=device)
                        samples = []
                        original_sample = PT.tree_sample_wor

                        def capture_sample(*args, **kwargs):
                            output = original_sample(*args, **kwargs)
                            samples.append(output)
                            return output

                        from unittest.mock import patch
                        state = torch.get_rng_state().clone()
                        gpu_state = (torch.cuda.get_rng_state().clone()
                                     if device == "cuda" else None)
                        kwargs = dict(
                            policy="dynamic", W=1, F_total=2, c_tensor=c,
                            nv=2, beta=.5, depth_cap=2,
                            temps=torch.zeros(1, device=device),
                            forward_fn=lambda *args: logits.clone(),
                            glue_rows_by_root=np.ones((1, 1), dtype=np.uint8),
                            rope_base_by_root=[2], K_glue=0, context_len=4,
                            greedy_only=True)
                        rollout = PT.run_rollout_arena if arena else PT.run_rollout
                        if arena:
                            kwargs["device"] = device
                        with patch.object(PT, "tree_sample_wor", capture_sample):
                            rollout([4], torch.tensor([1.]), **kwargs)
                        self.assertEqual(len(samples), 2)
                        for toks, scores in samples:
                            self.assertEqual(int(toks[0, 0]), 0)
                            self.assertEqual(len(set(toks[0].tolist())), c)
                            self.assertTrue(torch.equal(
                                scores, logits.softmax(-1).gather(1, toks)))
                        self.assertTrue(torch.equal(state, torch.get_rng_state()))
                        if gpu_state is not None:
                            self.assertTrue(torch.equal(gpu_state, torch.cuda.get_rng_state()))

    def test_stochastic_dispatch_does_not_read_device_temperature(self):
        class ForbiddenTemperatureRead:
            def __gt__(self, other):
                raise AssertionError("homogeneous contract must avoid device read")

        for greedy in (False, True):
            cfg = SimpleNamespace(duet_p2_tree_policy="on",
                                  greedy_only=greedy, stochastic_only=not greedy)
            self.assertTrue(_tree_temperatures_supported(cfg, ForbiddenTemperatureRead()))
            self.assertTrue(_should_run_p2_tree(cfg, 1, ForbiddenTemperatureRead()))
            self.assertFalse(_should_run_p2_tree(cfg, 2, ForbiddenTemperatureRead()))
            cfg.duet_p2_tree_policy = "off"
            self.assertFalse(_should_run_p2_tree(cfg, 1, ForbiddenTemperatureRead()))

    def test_generic_dispatch_still_checks_temperature(self):
        cfg = SimpleNamespace(duet_p2_tree_policy="on")
        self.assertTrue(_should_run_p2_tree(cfg, 1, torch.tensor([.7])))
        self.assertFalse(_should_run_p2_tree(cfg, 1, torch.tensor([0.])))


if __name__ == "__main__":
    unittest.main()
