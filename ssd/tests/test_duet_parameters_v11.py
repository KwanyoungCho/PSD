"""Parameter migration, root coverage, and target-side candidate ranking."""
import dataclasses
import inspect
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
os.environ.setdefault('SSD_HF_CACHE', '/tmp')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp')
import torch
from test_blackwell_review import make_config
from ssd.config import Config
from ssd.presets import duet_70b_paper_kwargs
from ssd.engine.helpers.duet_parameters import (
    ALIASES, THRESHOLDS, normalize_duet_kwargs, p1_continuation_width)
from ssd.engine.helpers.root_policy import rank


class Parameters(unittest.TestCase):
    def test_public_fields_have_one_node_cap_and_no_derived_knobs(self):
        names={f.name for f in dataclasses.fields(Config)}
        removed=set(ALIASES)|set(THRESHOLDS)|{'duet_p1_tree_forward_scale','duet_proxy_top_k'}
        self.assertFalse(names & removed)
        self.assertTrue({'duet_p1_forward_width','duet_p2_forward_width',
                         'duet_p2_root_count'} <= names)
        self.assertFalse(set(inspect.signature(Config).parameters) & removed)

    def test_preset_shapes_and_reconstruction(self):
        cfg,draft=make_config(**duet_70b_paper_kwargs('/draft'))
        for c in (cfg,draft):
            self.assertEqual((c.duet_p1_tree_max_nodes,c.duet_p2_tree_max_nodes),(12,8))
            self.assertEqual((c.duet_proxy_wire_N,c.duet_proxy_top_k),(51,51))
            self.assertEqual((c.duet_p2_seed_count,c.duet_proxy_total_budget),(10,15))
            for contexts in (5,9,13):
                roots=contexts*3
                self.assertEqual(p1_continuation_width(c,roots),roots)
        for field in ALIASES:
            self.assertNotIn(field,dataclasses.asdict(cfg))

    def test_generation_limit_tracks_the_only_cap(self):
        c,_=make_config(duet_p1_tree_verify_nodes=7,duet_p2_tree_verify_nodes=5)
        self.assertEqual((c.duet_p1_tree_max_nodes,c.duet_p2_tree_max_nodes),(7,5))
        c.duet_p1_tree_verify_nodes=9
        self.assertEqual(c.duet_p1_tree_max_nodes,9)
        with self.assertRaises(AttributeError):c.duet_p1_tree_max_nodes=12

    def test_legacy_equal_caps_migrate_but_unequal_caps_reject(self):
        with self.assertWarns(FutureWarning):
            x=normalize_duet_kwargs(dict(duet_p1_tree_max_nodes=12,
                duet_p1_tree_verify_nodes=12,duet_p2_budget=15,duet_tree_root_count=10))
        self.assertEqual(x,dict(duet_p1_tree_verify_nodes=12,
                              duet_p2_forward_width=15,duet_p2_root_count=10))
        for g,m in ((18,12),(12,18)):
            with self.assertRaisesRegex(ValueError,'G=M'):
                normalize_duet_kwargs(dict(duet_p1_tree_max_nodes=g,duet_p1_tree_verify_nodes=m))

    def test_removed_score_and_scale_settings_are_not_silently_ignored(self):
        for kwargs in ({'duet_proxy_top_k':28},{'duet_p1_tree_forward_scale':1.5},
                       *[{key:.1} for key in THRESHOLDS]):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                normalize_duet_kwargs(kwargs)
        with self.assertWarns(FutureWarning):
            self.assertEqual(normalize_duet_kwargs(dict.fromkeys(THRESHOLDS,0)),{})

    def test_integer_contract(self):
        for key in ('duet_p1_forward_width','duet_p2_forward_width','duet_p2_root_count',
                    'duet_p1_tree_verify_nodes','duet_p2_tree_verify_nodes'):
            for value in (0,-1,True,1.5):
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    make_config(**{key:value})

    def test_ds_continuation_cap_preserves_root_round_and_short_buckets(self):
        c=SimpleNamespace(duet_p1_forward_width=12,duet_p1_tree_allocation_policy='dynamic')
        self.assertEqual(p1_continuation_width(c,18),12)
        self.assertEqual(p1_continuation_width(c,6),6)
        c.duet_p1_tree_allocation_policy='backbone'
        with self.assertRaises(ValueError):p1_continuation_width(c,18)

    def test_ps_roots_cannot_exceed_forward_width(self):
        with self.assertRaises(ValueError):
            make_config(duet_p2_forward_width=15,duet_p2_root_count=16)

    def test_scheduler_reserves_root_round_plus_explicit_continuation_rows(self):
        from ssd.engine.scheduler import Scheduler
        for batch in (1,4):
            c,draft=make_config(**{**duet_70b_paper_kwargs('/draft'),
                'duet_p1_forward_width':12,'max_num_seqs':batch,
                'num_kvcache_blocks':64})
            with patch('ssd.engine.scheduler.AutoTokenizer.from_pretrained'):
                scheduler=Scheduler(c,draft)
            self.assertEqual(scheduler._sk_cells_p1,39+7*12)
            self.assertEqual(scheduler._sk_cells_p2,4*15)

    def test_cli_has_no_removed_tuning_knobs(self):
        import ast
        from pathlib import Path
        tree=ast.parse((Path(__file__).parents[1]/'bench/bench.py').read_text())
        flags={n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument'
               and n.args and isinstance(n.args[0],ast.Constant)}
        for key in set(ALIASES)|set(THRESHOLDS)|{'duet_proxy_top_k','duet_p1_tree_forward_scale'}:
            self.assertNotIn('--'+key,flags)
        self.assertIn('--duet_p1_forward_width',flags)
        self.assertIn('--duet_p2_forward_width',flags)

    def test_local_top_n_recovers_global_top_n_with_fixed_full_scores(self):
        torch.manual_seed(331)
        scores=torch.rand(2,5,100)
        terminal=torch.tensor([[.95,.01,.01,.01,.02],[.1,.2,.3,.2,.2]])
        n=20
        pos,tok,value=rank(scores,terminal,n,n,'full',pack_scores=False)
        joint=(scores/scores.sum(-1,keepdim=True))*terminal[...,None]
        expected,index=joint.flatten(1).topk(n,1)
        self.assertTrue(torch.equal(pos,index//100))
        self.assertTrue(torch.equal(tok,index%100))
        self.assertTrue(torch.allclose(value,expected))
        self.assertEqual(pos.shape,(2,n)) # never sends 5*N entries

    def test_topm_uses_new_local_pool_and_can_change_position_ranking(self):
        scores=torch.tensor([[[9.,8.,7.,6.],[9.,1.,1.,1.]]])
        terminal=torch.tensor([[.5,.5]])
        old=rank(scores,terminal,1,2,'topm',pack_scores=False)
        new=rank(scores,terminal,2,2,'topm',pack_scores=False)
        self.assertFalse(torch.equal(old[2],new[2]))
        self.assertEqual(new[0].shape,(1,2))

    def test_model_api_rejects_removed_and_unknown_duet_fields_before_loading(self):
        from ssd.engine.llm_engine import LLMEngine
        with patch('ssd.engine.llm_engine.Config') as factory:
            with self.assertRaisesRegex(ValueError,'top-L'):
                LLMEngine('/target',duet_proxy_top_k=28)
            factory.assert_not_called()
        with self.assertRaisesRegex(TypeError,'Unknown DUET'):
            LLMEngine('/target',duet_p1_forward_wdth=12)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeConfigGraphs(unittest.TestCase):
    @torch.inference_mode()
    def test_canonical_config_drives_both_phase_graphs_with_equal_caps(self):
        from test_p2_executor_parity import _MiniDraft, TestExecutorModuleParity
        from ssd.engine.helpers.p1_tree import P1TreeExecutor
        from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor
        fixture=TestExecutorModuleParity()
        for greedy in (False,True):
            cfg,_=make_config(duet_phase1_k=3,duet_phase2_k=2,speculate_k=5,
                duet_p1_tree_policy='on',duet_p1_roots_per_position=2,
                duet_p1_forward_width=4,duet_p2_forward_width=6,
                duet_p2_root_count=4,duet_p1_tree_verify_nodes=7,
                duet_p2_tree_verify_nodes=5,greedy_only=greedy,stochastic_only=not greedy)
            for phase in (1,2):
                cache=torch.zeros(8,2,64,2,64,dtype=torch.float16,device='cuda')
                model=_MiniDraft(128,4,2,64,cache,'cuda')
                args=(model,model.logits_fn,cfg,'cuda',64,8,128,4,2,64)
                ex=(P1TreeExecutor(*args,context_bucket=3,materialize_backbone_logits=False)
                    if phase==1 else P2TreeExecutor(*args,materialize_backbone_logits=False))
                self.assertEqual(ex.NV,7 if phase==1 else 5)
                self.assertEqual(ex.round_widths,(6,4,4) if phase==1 else (6,6))
                ex.prepare_bucket(2);fixture._fill_inputs(ex,64,85)
                ex.in_temps.fill_(0 if greedy else .7)
                gen=torch.Generator().manual_seed(448)
                ex.parity_noise=[torch.empty(w,128).exponential_(1,generator=gen).cuda()
                                 for w in ex.round_widths]
                ex.run_once(2)
                names=('view_tok','view_par','view_sib','out_valid','out_pq_ref')
                expected={name:getattr(ex,name).clone() for name in names}
                self.assertTrue((ex.out_valid<=ex.NV).all())
                address=ex.arena.tok.data_ptr()
                cache.zero_();ex.capture(2)
                cache.zero_();ex.replay(2)
                self.assertEqual(address,ex.arena.tok.data_ptr())
                for name,value in expected.items():
                    torch.testing.assert_close(value,getattr(ex,name),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
