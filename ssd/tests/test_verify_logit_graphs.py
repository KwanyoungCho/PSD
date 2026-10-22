"""CPU contract tests, with no CUDA imports or allocations."""
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock

p=Path(__file__).resolve().parents[1]/'ssd/engine/helpers/verify_logit_graphs.py'
spec=importlib.util.spec_from_file_location('verify_heads_cpu',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class VerifyLogitGraphs(unittest.TestCase):
    def test_disabled_by_default(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertFalse(m.enabled(SimpleNamespace(max_num_seqs=1)))

    def test_exact_supported_mode(self):
        with patch.dict(os.environ,{'SSD_VERIFY_GRAPH_LOGITS':'1'},clear=True):
            self.assertTrue(m.enabled(SimpleNamespace(max_num_seqs=1)))
            self.assertFalse(m.enabled(SimpleNamespace(max_num_seqs=1),is_draft=True))
            self.assertFalse(m.enabled(SimpleNamespace(max_num_seqs=2)))
            for key in ('duet_exit_replica','duet_exit_topm_gather'):
                self.assertFalse(m.enabled(SimpleNamespace(max_num_seqs=1,**{key:True})))

    def test_proxy_stream_retains_original_path(self):
        with patch.dict(os.environ,{'SSD_VERIFY_GRAPH_LOGITS':'1','SSD_PROXY_STREAM':'1'},clear=True):
            self.assertFalse(m.enabled(SimpleNamespace(max_num_seqs=1)))

    def test_invalid_configuration_fails(self):
        with patch.dict(os.environ,{'SSD_VERIFY_GRAPH_LOGITS':'yes'},clear=True):
            with self.assertRaises(ValueError):m.enabled(SimpleNamespace(max_num_seqs=1))

    def test_nonmixing_rejects_unsupported_shapes(self):
        with patch.dict(os.environ,{'SSD_VERIFY_GRAPH_LOGITS':'1','NCCL_GRAPH_MIXING_SUPPORT':'0'},clear=True):
            self.assertTrue(m.enabled(SimpleNamespace(max_num_seqs=1)))
            with self.assertRaises(ValueError):m.enabled(SimpleNamespace(max_num_seqs=2))

    def test_nonmixing_is_explicit(self):
        with patch.dict(os.environ,{},clear=True):self.assertFalse(m.no_mixing())
        with patch.dict(os.environ,{'NCCL_GRAPH_MIXING_SUPPORT':'0'},clear=True):self.assertTrue(m.no_mixing())

    def test_eager_boundary_synchronizes_only_when_required(self):
        sync=Mock()
        fake=SimpleNamespace(cuda=SimpleNamespace(synchronize=sync))
        with patch.dict(sys.modules,{'torch':fake}):
            with patch.dict(os.environ,{},clear=True):m.synchronize_eager()
            sync.assert_not_called()
            with patch.dict(os.environ,{'NCCL_GRAPH_MIXING_SUPPORT':'0'},clear=True):m.synchronize_eager()
            sync.assert_called_once_with()

    def test_output_preserves_rank_and_shape_contract(self):
        class Tensor:
            def unsqueeze(self,axis):return ('unsqueeze',axis,self)
        x=Tensor()
        self.assertIs(m.output({'exit':(None,x)},'exit'),x)
        self.assertEqual(m.output({'exit':(None,x)},'exit',chain=True),('unsqueeze',0,x))
        self.assertIsNone(m.output({'exit':(None,None)},'exit',chain=True))


if __name__=='__main__':unittest.main()
