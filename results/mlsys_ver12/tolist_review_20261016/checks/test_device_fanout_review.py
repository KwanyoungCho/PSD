"""Candidate-only contract: defer host fan-out on B1 captured P2."""
import os
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.helpers import p2_tree


ENV={'SSD_TREE_EXEC':'1','SSD_TREE_ARENA':'1','SSD_TREE_STAGE1':'0','SSD_TREE_LAB':'0'}


def runner(device='cuda', enabled=True):
    r=object.__new__(DraftRunner);r.device=torch.device(device)
    r.config=types.SimpleNamespace(duet_p2_tree_policy='on' if enabled else 'off',
        use_eagle=False,stochastic_only=True,greedy_only=False,duet_phase2_k=3,
        max_model_len=2048,duet_p2_active_root_count=6,duet_proxy_total_budget=9,
        duet_tree_policy='dynamic',duet_tree_c_tensor=3,duet_tree_nv=40,
        duet_tree_beta=1.,duet_tree_fanout_policy='constant',sampler_x=1,
        async_fan_out=4,duet_tree_proxy_threshold=0.,duet_tree_conf_threshold=0.)
    r.hf_config=types.SimpleNamespace(vocab_size=32)
    r.split_k2_layout=types.SimpleNamespace(MQ_LEN=9,K=3,arange_mq=torch.arange(9,device=device),name='phase2')
    r._try_p2_executor=lambda *a,**k: None
    r.prefill_wrappers_by_layout={};r._tree_backbone_project=lambda *a,**k: ('tokens','logits')
    return r


def args(r,counts):
    r._update_phase2_layout_inplace(counts,counts.shape[1]-1)
    partial=dict(num_tokens=torch.tensor([40],device=r.device),dbt=torch.tensor([[0,1,2,3]],device=r.device,dtype=torch.int32),
        temperatures=torch.tensor([.7],device=r.device),seq_ids=torch.tensor([2],device=r.device),
        cache_hits=torch.tensor([True],device=r.device))
    tok=torch.arange(9,device=r.device).view(1,9)
    a=r._build_tree_decode_args_for_layout(partial,tok,r.split_k2_layout,[1])
    a.update(proxy_piv=torch.ones(1,6,device=r.device),_p2_ctx_len=60)
    return tok,a


@unittest.skipUnless(torch.cuda.is_available(),'requires CUDA')
class DeviceFanout(unittest.TestCase):
    def test_layout_and_arguments_need_no_tolist(self):
        for counts in ([0,0,9],[3,0,2,4],[0,1,0,8],[1,1,1,1,1,1,1,1,1]):
            with self.subTest(counts=counts),patch.dict(os.environ,ENV):
                r=runner();c=torch.tensor([counts],device=r.device)
                expected=torch.tensor([p for p,n in enumerate(counts) for _ in range(n)],device=r.device)
                with patch.object(torch.Tensor,'tolist',side_effect=AssertionError('Unexpected host fan-out')):
                    tok,a=args(r,c)
                self.assertIsNone(r.split_k2_layout.fan_out_list)
                self.assertTrue(torch.equal(r.split_k2_layout.fan_idx_hit,expected))
                self.assertTrue(torch.equal(a['rope_positions'],40+expected))

    def test_graph_gets_exact_mask_and_real_root_count(self):
        for override in (False,True):
            with self.subTest(override=override),patch.dict(os.environ,ENV):
                r=runner();counts=[2,0,3,4];c=torch.tensor([counts],device=r.device)
                tok,a=args(r,c);expected_positions=[0,0,2,2,2,3]
                expected=np.asarray([[int(j<=p) for j in range(4)] for p in expected_positions],dtype=np.uint8)
                if override:
                    expected=np.asarray([[1,0,0,0],[1,1,0,0],[1,0,1,0],[1,1,0,1],[1,1,1,0],[1,0,1,1]],dtype=np.uint8)
                    a['glue_rows_override']=torch.as_tensor(expected,device=r.device);a['K_glue_override']=3
                def execute(toks,piv,glue,rope,ctx,gw,slots,ctxs,dbt,temps,seeds):
                    self.assertEqual(len(seeds),6);self.assertEqual(gw,3)
                    self.assertTrue(torch.equal(glue,torch.as_tensor(expected,device=r.device)))
                    return {'valid':'views'},('tokens','logits'),'cells'
                r._try_p2_executor=execute
                with patch.object(torch.Tensor,'tolist',side_effect=AssertionError('Unexpected host fan-out')):
                    out=r._p2tree_rollout({},tok,c,a,r.split_k2_layout,[],[],a['temps'])
                self.assertEqual(out[0],{'valid':'views'})
                self.assertIsNone(r.split_k2_layout.fan_out_list)

    def test_fallback_restores_host_metadata_and_mask(self):
        with patch.dict(os.environ,ENV):
            r=runner();counts=[2,0,3,4];c=torch.tensor([counts],device=r.device);tok,a=args(r,c)
            expected=np.asarray([[int(j<=p) for j in range(4)] for p in [0,0,2,2,2,3]],dtype=np.uint8)
            class Arena:
                def to_pool(self,n):return 'pool'
            def rollout(*args,**kwargs):
                self.assertTrue(np.array_equal(kwargs['glue_rows_by_root'],expected))
                return Arena(),[],torch.zeros(27,32,device=r.device)
            with patch.object(p2_tree,'run_rollout_arena',side_effect=rollout):
                out=r._p2tree_rollout({},tok,c,a,r.split_k2_layout,[],[],a['temps'])
            self.assertEqual(r.split_k2_layout.fan_out_list,[counts])
            self.assertEqual(a['metadata_ints'],(1,3,2,9))
            self.assertEqual(out[0],'pool')

    def test_chain_batch_and_diagnostic_gates_keep_host_counts(self):
        cases=[dict(enabled=False),dict(env={'SSD_TREE_EXEC':'0'}),dict(env={'SSD_TREE_ARENA':'0'}),
               dict(env={'SSD_TREE_STAGE1':'1'}),dict(env={'SSD_TREE_LAB':'1'}),dict(batch=2)]
        for case in cases:
            with self.subTest(case=case),patch.dict(os.environ,{**ENV,**case.get('env',{})}):
                r=runner(enabled=case.get('enabled',True));B=case.get('batch',1)
                counts=torch.tensor([[3,2,4]]*B,device=r.device)
                r._update_phase2_layout_inplace(counts,2)
                self.assertEqual(r.split_k2_layout.fan_out_list,[[3,2,4]]*B)


if __name__=='__main__':unittest.main()
