"""Target-argmax, ranking-score and request-temperature contracts for the port."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

os.environ.setdefault('SSD_HF_CACHE', '/tmp')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp')
import torch
from ssd import SamplingParams
from ssd.engine.llm_engine import LLMEngine
from ssd.engine.helpers.p2_tree import tree_sample_wor, q_probs_from_logits, tree_verify_walk_greedy
from ssd.layers.sampler import Sampler
from ssd.utils.verify import verify


class PortContracts(unittest.TestCase):
    def test_branching_and_ties(self):
        info = dict(valid=4, u_valid=3, tok=[1, 2, 3, 4],
                    parent_local=[-1, -1, 0, 1], sib_order=[0, 1, 0, 0], parent_q_ref=[0, 0, 1, 2])
        p = torch.zeros(5, 7)
        p[0, 2] = p[2, 4] = p[4, 6] = 9
        self.assertEqual(tree_verify_walk_greedy(info, p), ([1, 3], 6))
        p.zero_()
        self.assertEqual(tree_verify_walk_greedy(info, p), ([], 0))

    def test_greedy_draft_first_child_is_argmax(self):
        logits = torch.tensor([[3., 3., 1., -1.], [0., 0., 0., 0.]])
        original = logits.clone()
        state = torch.get_rng_state().clone()
        tokens, scores = tree_sample_wor(logits, torch.zeros(2), 3, allow_greedy=True, greedy_only=True)
        self.assertTrue(torch.equal(tokens[:, 0], logits.argmax(-1)))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(original, logits))
        self.assertTrue(torch.allclose(scores, logits.softmax(-1).gather(1, tokens)))
        self.assertTrue(all(len(set(row)) == 3 for row in tokens.tolist()))

    def test_ragged_chain_recovery_uses_real_end(self):
        p = torch.zeros(2, 5, 8)
        for row in range(2):
            for pos in range(5): p[row, pos, pos + 1] = 10
        spec = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        suffix, token = verify(p, None, spec, None, None, all_greedy=True, valid_k=torch.tensor([4, 2]))
        self.assertEqual(suffix, [[0, 1, 2, 3, 4], [0, 1, 2]])
        self.assertEqual(token, [5, 3])

    def test_temperature_mode_contract(self):
        engine = SimpleNamespace(config=SimpleNamespace(greedy_only=True, stochastic_only=False),
                                 scheduler=SimpleNamespace(add=lambda _: None))
        with self.assertRaises(ValueError):
            LLMEngine.add_request(engine, [1], SamplingParams(temperature=.7))
        LLMEngine.add_request(engine, [1], SamplingParams(temperature=0))
        engine.config.greedy_only, engine.config.stochastic_only = False, True
        with self.assertRaises(ValueError):
            LLMEngine.add_request(engine, [1], SamplingParams(temperature=0))
        LLMEngine.add_request(engine, [1], SamplingParams(temperature=.7))

    def test_sampler_positive_parity_and_greedy_no_rng(self):
        for dtype in (torch.float16, torch.bfloat16):
            logits = torch.randn(8, 32000, dtype=dtype)
            for temp in (.3, .7, 1.):
                t = torch.full((8,), temp)
                torch.manual_seed(41)
                expected = Sampler()(logits.clone(), t)
                torch.manual_seed(41)
                actual = Sampler(stochastic_only=True)(logits.clone(), t)
                self.assertTrue(torch.equal(actual, expected))
            state = torch.get_rng_state().clone()
            self.assertTrue(torch.equal(Sampler(greedy_only=True)(logits, torch.zeros(8)), logits.argmax(-1)))
            self.assertTrue(torch.equal(state, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
