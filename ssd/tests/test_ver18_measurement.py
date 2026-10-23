"""Release contracts: phase fanout shapes, draft reconstruction, clean modes."""
import argparse
import ast
from copy import copy
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

from test_blackwell_review import make_config
from test_tree_policy_defaults import cli
from ssd.engine.helpers.tree_fanout_config import phase_fanout, sibling_capacity
from ssd.presets import duet_7b_measurement_kwargs
sys.path.insert(0,str(Path(__file__).parents[1]/'bench'))
from duet_measurement_mode import configure_measurement, PROFILE_FLAGS


class PhaseFanout(unittest.TestCase):
    def test_legacy_common_fallback_and_independent_overrides(self):
        for common,c1,c2,expected in [(3,None,None,(3,3)),(3,2,3,(2,3)),
                                     (1,4,2,(4,2)),(8,1,1,(1,1))]:
            cfg,draft=make_config(duet_tree_c_tensor=common,
                duet_p1_tree_c_tensor=c1,duet_p2_tree_c_tensor=c2)
            for c in (cfg,draft):
                self.assertEqual((phase_fanout(c,'p1'),phase_fanout(c,'p2')),expected)
                self.assertEqual(sibling_capacity(c),max(expected))

    def test_invalid_fanout_rejected_before_loading_models(self):
        for name in ('duet_tree_c_tensor','duet_p1_tree_c_tensor','duet_p2_tree_c_tensor'):
            for value in (0,9,True,2.5):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                    make_config(**{name:value})

    def test_p2_backbone_capacity_uses_p2_fanout(self):
        # Shared fallback=1 must not reject an explicit P2 C3 backbone.
        make_config(duet_tree_policy='backbone',duet_tree_c_tensor=1,
                    duet_p1_tree_c_tensor=2,duet_p2_tree_c_tensor=3,
                    duet_p2_tree_verify_nodes=12)
        with self.assertRaises(ValueError):
            make_config(duet_tree_policy='backbone',duet_tree_c_tensor=8,
                        duet_p2_tree_c_tensor=1,duet_p2_tree_verify_nodes=8)

    def test_cli_exposes_both_overrides_without_enabling_duet(self):
        a=cli('--duet_tree_c_tensor','4','--duet_p1_tree_c_tensor','2',
              '--duet_p2_tree_c_tensor','3')
        self.assertFalse(a.duet)
        self.assertEqual((a.duet_tree_c_tensor,a.duet_p1_tree_c_tensor,a.duet_p2_tree_c_tensor),(4,2,3))

    def test_explicit_draft_memory_fraction_survives_reconstruction(self):
        cfg,draft=make_config(draft_gpu_memory_utilization=.35)
        self.assertEqual(draft.gpu_memory_utilization,.35)
        for value in (0,-.1,1,float('nan')):
            with self.assertRaises(ValueError):make_config(draft_gpu_memory_utilization=value)

    def test_batched_chain_override_resets_phase_specific_c(self):
        # Exercise the real _forest config preparation, stopping before CUDA.
        from ssd.engine.helpers.batch_tree_draft import BatchedDuetDraft
        cfg,_=make_config(duet_p1_tree_policy='off',duet_p2_tree_policy='off',
                          duet_p1_tree_c_tensor=2,duet_p2_tree_c_tensor=3)
        class Stop(Exception):pass
        for phase,name in [(1,'P1TreeExecutor'),(2,'P2TreeExecutor')]:
            r=NS(config=cfg,hf_config=NS(vocab_size=16,num_attention_heads=1,
                num_key_value_heads=1,head_dim=8,torch_dtype=None),
                model=NS(compute_logits=None),device='cuda',block_size=256)
            service=BatchedDuetDraft.__new__(BatchedDuetDraft)
            service.cfg=cfg;service.r=r;service.forests={}
            observed=[]
            def construct(*args,**kw):
                observed.append(phase_fanout(args[2],f'p{phase}'));raise Stop
            with patch(f'ssd.engine.helpers.batch_tree_draft.{name}',side_effect=construct),self.assertRaises(Stop):
                service._forest(phase,2,1,1)
            self.assertEqual(observed,[1])
            self.assertEqual((cfg.duet_p1_tree_c_tensor,cfg.duet_p2_tree_c_tensor),(2,3))


class MeasurementMode(unittest.TestCase):
    def test_preparation_defaults_and_explicit_overrides(self):
        for mode in ('performance', 'timeline'):
            with patch.dict(os.environ, {}, clear=True):
                configure_measurement(mode)
                self.assertEqual(os.environ['SSD_SCOPED_DEFAULT_DEVICE'], '1')
                self.assertEqual(os.environ['SSD_FUSED_RESPONSE_TOPOLOGY'], '0')
            with patch.dict(os.environ, {'SSD_SCOPED_DEFAULT_DEVICE': '0',
                                        'SSD_FUSED_RESPONSE_TOPOLOGY': '1'}, clear=True):
                configure_measurement(mode)
                self.assertEqual(os.environ['SSD_SCOPED_DEFAULT_DEVICE'], '0')
                self.assertEqual(os.environ['SSD_FUSED_RESPONSE_TOPOLOGY'], '1')

    def test_performance_clears_stale_profile_and_review_settings(self):
        with patch.dict(os.environ,dict.fromkeys(PROFILE_FLAGS,'1')|{
            'DUET_REVIEW_FLAT':'1','DUET_EQUAL_ROOTS':'1'},clear=True):
            configure_measurement()
            self.assertTrue(all(os.environ[k]=='0' for k in PROFILE_FLAGS))
            self.assertNotIn('DUET_EQUAL_ROOTS',os.environ)
            self.assertNotIn('DUET_REVIEW_FLAT',os.environ)
            self.assertEqual(os.environ['SSD_VERIFY_GRAPH_LOGITS'],'1')
            self.assertEqual(os.environ['SSD_ASYNC_PROXY_SEND'],'1')
            self.assertEqual(os.environ['SSD_DUET_EXIT_REPLICA'],'0')

    def test_timeline_is_explicit(self):
        with patch.dict(os.environ,{},clear=True):
            configure_measurement('timeline')
            self.assertEqual([os.environ[k] for k in PROFILE_FLAGS],['1','1','0','0','0'])
        with self.assertRaises(ValueError):configure_measurement('unknown')

    def test_benchmark_parser_defaults_to_performance_and_current_settings(self):
        path=Path(__file__).parents[1]/'bench/duet_measure.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='parse_arguments')
        scope=dict(argparse=argparse,Path=Path)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
        argv=['duet_measure.py','--target','/target','--draft','/draft','--prompts','/inputs','--output','/run.json']
        with patch.object(sys,'argv',argv):a=scope['parse_arguments']()
        self.assertEqual((a.measurement_mode,a.dtype),('performance','float16'))
        self.assertEqual((a.k1,a.k2,a.p1_c,a.p2_c),(10,5,2,3))
        self.assertEqual((a.input_cap,a.output_tokens,a.warm_prompts,a.passes),(0,1024,3,1))

    def test_named_preset_matches_tested_parameters(self):
        k=duet_7b_measurement_kwargs('/draft')
        self.assertEqual((k['dtype'],k['duet_phase1_k'],k['duet_phase2_k']),('float16',10,5))
        self.assertEqual((k['duet_p1_tree_c_tensor'],k['duet_p2_tree_c_tensor']),(2,3))
        self.assertEqual((k['duet_p1_tree_verify_nodes'],k['duet_p2_tree_verify_nodes']),(15,15))
        # make_config's fixture has 80 layers, but all explicit preset fields
        # must survive Config validation and independent draft reconstruction.
        k.pop('draft');cfg,draft=make_config(**k)
        self.assertEqual((phase_fanout(draft,'p1'),phase_fanout(draft,'p2')),(2,3))


if __name__=='__main__':unittest.main()
