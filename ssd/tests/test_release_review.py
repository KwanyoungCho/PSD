"""Configured sibling bound must be honored at both B1 wire boundaries."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('SSD_HF_CACHE', '/tmp')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp')
import torch
from ssd.engine.helpers import p2_tree as PT
from ssd.engine.model_runner import ModelRunner
from ssd.engine.verifier import Verifier


class Validated(Exception):
    pass


class SiblingBound(unittest.TestCase):
    def check_boundary(self, target):
        real = PT.validate_tree_ints
        for capacity, width in ((2,2), (3,3), (4,4), (2,3)):
            with self.subTest(target=target, capacity=capacity, width=width):
                cfg = SimpleNamespace(duet_enabled=True, duet_phase1_k=2,
                    duet_tree_enabled=True, duet_tree_wire_nodes=width,
                    duet_tree_c_tensor=capacity, hf_config=SimpleNamespace(vocab_size=32))
                wire = [width,1,1]+list(range(width))+[-1]*width+list(range(width))+[0]*width
                runner = SimpleNamespace(config=cfg, _duet_tree_meta=wire,
                                         async_pg=None, draft_rank=1)
                seen = []
                def checked(ti, nv, vocab_size, sibling_capacity):
                    seen.append(sibling_capacity)
                    real(ti, nv, vocab_size, sibling_capacity)
                    raise Validated()
                expected = Validated if width<=capacity else RuntimeError
                with patch.object(PT, 'validate_tree_ints', checked), self.assertRaises(expected):
                    if target:
                        ModelRunner._run_tree_verify(runner, None, None, False)
                    else:
                        verifier = SimpleNamespace(target_model_runner=runner)
                        seq = SimpleNamespace(temperature=.7, draft_temperature=None)
                        result = SimpleNamespace(valid_k=torch.tensor([width]),
                            tree_ints=torch.tensor([wire]))
                        Verifier.verify(verifier, [seq], result)
                self.assertEqual(seen, [capacity])

    def test_target_boundary(self): self.check_boundary(True)

    def test_acceptance_boundary(self): self.check_boundary(False)


if __name__ == '__main__': unittest.main()
