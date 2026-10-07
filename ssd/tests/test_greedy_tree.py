import unittest
import torch
from ssd.engine.helpers.p2_tree import tree_sample_wor, tree_verify_walk_greedy


class TestGreedyTree(unittest.TestCase):
    def test_branching_target_argmax_never_uses_draft_ratio(self):
        # Target chooses the second root child, then a grandchild. Its next
        # argmax is absent, so that exact token must become recovery.
        tree = dict(valid=4, u_valid=3, tok=[1, 2, 3, 4],
                    parent_local=[-1, -1, 0, 1], sib_order=[0, 1, 0, 0],
                    parent_q_ref=[0, 0, 1, 2])
        p = torch.zeros(5, 7)
        p[0, 2] = p[2, 4] = p[4, 6] = 9
        self.assertEqual(tree_verify_walk_greedy(tree, p), ([1, 3], 6))
        p[0, 5] = 10
        self.assertEqual(tree_verify_walk_greedy(tree, p), ([], 5))

    def test_top_children_unique_finite_and_no_rng_for_greedy(self):
        logits = torch.tensor([[3., 1., 2., -1.], [-3., 0., 5., 2.]])
        temps = torch.zeros(2)
        before = torch.get_rng_state()
        toks, score = tree_sample_wor(logits, temps, 3,
                                     allow_greedy=True, greedy_only=True)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(toks.tolist(), [[0, 2, 1], [2, 3, 1]])
        self.assertTrue(torch.allclose(score, logits.softmax(-1).gather(1, toks)))
        self.assertTrue(torch.equal(logits, torch.tensor([[3.,1.,2.,-1.],[-3.,0.,5.,2.]])))

    def test_mixed_temperature_sampler_preserves_positive_rows(self):
        logits = torch.tensor([[3., 1., 2., -1.], [-3., 0., 5., 2.]])
        noise = torch.tensor([[.1, 4., 1., .8], [.4, .3, 7., .1]])
        toks, _ = tree_sample_wor(logits, torch.tensor([0., .7]), 3,
                                 noise=noise, allow_greedy=True)
        ref, _ = tree_sample_wor(logits[1:], torch.tensor([.7]), 3,
                                noise=noise[1:])
        self.assertEqual(toks[0].tolist(), [0, 2, 1])
        self.assertTrue(torch.equal(toks[1:], ref))


if __name__ == "__main__":
    unittest.main()
