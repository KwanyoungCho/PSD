"""Independent dense-attention tests reused with the fixed-plan query layout."""
import sys
from pathlib import Path
from unittest.mock import patch
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
if __package__:
    from . import test_batch_tree_serving as serving_tests
else:
    import test_batch_tree_serving as serving_tests
from mlsys_tree_candidates import FixedPlanPackedTree


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class FixedPlanAttention(serving_tests.ForwardAttention):
    def test_captured_ancestor_attention_matches_independent_dense(self):
        with patch('ssd.engine.helpers.batch_tree_forward.BatchedTreeForward', FixedPlanPackedTree):
            super().test_captured_ancestor_attention_matches_independent_dense()


if __name__ == '__main__':
    unittest.main()
