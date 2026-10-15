"""PS metadata: full validity contract, strided inputs, GPU mask parity."""
import types
import unittest
import os
from unittest.mock import patch, Mock
import numpy as np
import torch
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.helpers.p2_prepare import copy_glue_rows, host_glue_rows, host_fanout


def runner(device='cuda'):
    r = object.__new__(DraftRunner)
    r.device = torch.device(device)
    r.block_size = 16
    r.config = types.SimpleNamespace(max_model_len=128)
    r.kv_cache = torch.empty((2, 1, 8, 16, 1, 1), device=device)
    return r


def inputs(device='cuda', F=3, W=15, R=10):
    return dict(rope0=torch.full((W,), 30, dtype=torch.int64, device=device),
                step_slot_maps=torch.arange(F*W, device=device).reshape(F,W)+32,
                step_context_lens=torch.full((F,1),32,dtype=torch.int64,device=device),
                dbt=torch.tensor([[0,1,-1,-1,-1,-1,-1,-1]],dtype=torch.int32,device=device),
                F=F,W=W,root_count=R)


def reference(a):
    # Independent scalar definition, including original first-reason priority.
    F,W=a['F'],a['W']; R=min(a.get('root_count',W),W)
    ctx=int(a['step_context_lens'][0].reshape(-1)[0])
    ropes=a['rope0'][:R].tolist()
    slots=a['step_slot_maps'][:F,:W].flatten().tolist()
    ctxs=[int(c.reshape(-1)[0]) for c in a['step_context_lens'][:F]]
    pages=a['dbt'][0].tolist(); need=(ctx+15)//16
    flags=[R<=0,ctx<=0,any(x<0 or x>=128 for x in ropes),
           any(x<0 or x>=128 for x in slots),any(x<=0 or x>128 for x in ctxs),
           any(x<0 or x>=8 for x in pages[:max(0,need)])]
    names=('no_roots','ctx','rope','slot','ctxlen','pages')
    return ctx,next((n for n,v in zip(names,flags) if v),None)


@unittest.skipUnless(torch.cuda.is_available(),'requires CUDA')
class TestP2Preparation(unittest.TestCase):
    def check(self, a, r=None):
        expected=reference(a)
        before={k:v.clone() for k,v in a.items() if torch.is_tensor(v)}
        r=r or runner()
        actual=r._p2_step_state(**a)
        self.assertEqual((actual.ctx_len,actual.reason),expected)
        self.assertTrue(hasattr(r,'_p2_state_snapshot'))
        for k,v in before.items():self.assertTrue(torch.equal(v,a[k]),k)
        return r,actual

    def test_all_valid_and_masked_sentinels(self):
        for F,W,R in ((1,1,1),(3,15,10),(8,26,19),(2,8,0),(3,15,15)):
            with self.subTest(F=F,W=W,R=R):
                a=inputs(F=F,W=W,R=R)
                a['step_slot_maps'].remainder_(128)
                a['rope0'][R:]=-1
                self.check(a)

    def test_each_guard_and_boundaries(self):
        for key,index,values in (
            ('rope0',(0,),[-1,0,127,128]),
            ('step_slot_maps',(2,14),[-1,0,127,128]),
            ('step_context_lens',(2,0),[-1,0,128,129]),
            ('step_context_lens',(0,0),[-300,-17,-1,0,1,16,17,32,33,128,129]),
            ('dbt',(0,1),[-1,0,7,8])):
            for v in values:
                with self.subTest(key=key,v=v):
                    a=inputs();a[key][index]=v;self.check(a)

    def test_flag_priority(self):
        a=inputs();a['rope0'][0]=-1;a['step_slot_maps'][1,0]=-1
        a['step_context_lens'][2,0]=129;a['dbt'][0,1]=-1
        self.assertEqual(self.check(a)[1].reason,'rope')
        a['root_count']=0
        self.assertEqual(self.check(a)[1].reason,'no_roots')
        a['root_count']=10;a['step_context_lens'][0,0]=0
        self.assertEqual(self.check(a)[1].reason,'ctx')

    def test_strides_and_integer_dtypes(self):
        for dtype in (torch.int32,torch.int64):
            a=inputs()
            for k in ('step_slot_maps','step_context_lens','dbt'):
                v=a[k].to(dtype);storage=torch.zeros(v.shape[0]*2,v.shape[1]*3,device='cuda',dtype=dtype)
                a[k]=storage[::2,::3];a[k].copy_(v)
            storage=torch.zeros(30,device='cuda',dtype=dtype)
            storage[::2].copy_(a['rope0']);a['rope0']=storage[::2]
            self.check(a)

    def test_buffer_reuse_and_changed_values(self):
        r,state=self.check(inputs()); ptr=r._p2_state_snapshot.data_ptr()
        a=inputs();a['rope0'][0]=128
        self.check(a,r);self.assertEqual(ptr,r._p2_state_snapshot.data_ptr())

    def test_page_tail_mask(self):
        a=inputs();a['dbt'][0,2:]=9999
        self.assertIsNone(self.check(a)[1].reason)
        a['step_context_lens'][0,0]=33
        self.assertEqual(self.check(a)[1].reason,'pages')

    def test_random_corrupt_metadata(self):
        g=torch.Generator().manual_seed(8721)
        for _ in range(100):
            a=inputs();a['root_count']=int(torch.randint(0,20,(),generator=g))
            key=('rope0','step_slot_maps','step_context_lens','dbt')[int(torch.randint(0,4,(),generator=g))]
            index=int(torch.randint(0,a[key].numel(),(),generator=g))
            a[key].view(-1)[index]=int(torch.randint(-5,150,(),generator=g))
            self.check(a)

    def test_device_glue_matches_host_and_keeps_tail(self):
        mask=torch.randint(0,2,(15,13),device='cuda',dtype=torch.uint8)
        for R,gw in ((10,7),(15,13),(0,1)):
            ex=types.SimpleNamespace(in_glue=torch.full((15,16),9,device='cuda',dtype=torch.uint8),
                in_glue_host=torch.empty((15,16),pin_memory=True,dtype=torch.uint8))
            copy_glue_rows(ex,mask,R,gw);device=ex.in_glue.clone()
            ex.in_glue.fill_(9)
            copy_glue_rows(ex,host_glue_rows(mask),R,gw)
            self.assertTrue(torch.equal(device,ex.in_glue))

    def test_layout_fan_map(self):
        for B in (1,3):
            r=runner();r.split_k2_layout=types.SimpleNamespace(MQ_LEN=15)
            counts=torch.tensor([[2,0,4,5,4]]*B,device='cuda')
            expected=torch.arange(5,device='cuda').repeat(B).repeat_interleave(counts.flatten())
            layout=r._update_phase2_layout_inplace(counts,4)
            self.assertTrue(torch.equal(expected,layout.fan_idx_hit))
            self.assertEqual(layout.fan_out_list,counts.tolist())

    def test_deferred_counts_reset_and_materialize(self):
        r=runner();r.split_k2_layout=types.SimpleNamespace(MQ_LEN=5)
        for counts in ([[2,0,3]], [[0,4,1]], [[1,1,1,2]]):
            gpu=torch.tensor(counts,device='cuda')
            layout=r._update_phase2_layout_inplace(gpu,len(counts[0])-1,
                                                  defer_host_fanout=True)
            self.assertIsNone(layout.fan_out_list)
            self.assertIsNone(layout.fan_out_list_miss)
            self.assertEqual(host_fanout(layout),counts)
            self.assertIs(host_fanout(layout),layout.fan_out_list_miss)
        gpu=torch.tensor([[3,2]],device='cuda')
        layout=r._update_phase2_layout_inplace(gpu,1)
        self.assertEqual(layout.fan_out_list,[[3,2]])

    def rollout_args(self, R=3):
        r=runner();r.config=types.SimpleNamespace(max_model_len=128,
            duet_phase2_k=2,duet_p2_active_root_count=5,duet_tree_c_tensor=2,
            duet_tree_nv=5,duet_tree_policy='dynamic',duet_tree_beta=1.,
            duet_tree_fanout_policy='fixed',sampler_x=1,async_fan_out=1,
            duet_tree_proxy_threshold=0.,duet_tree_conf_threshold=0.)
        r.hf_config=types.SimpleNamespace(vocab_size=32)
        r._lab_active=lambda:False
        r.split_k2_layout=types.SimpleNamespace(MQ_LEN=5)
        counts=torch.tensor([[2,0,1,2]],device='cuda')
        layout=r._update_phase2_layout_inplace(counts,3,defer_host_fanout=True)
        tree=dict(proxy_piv=torch.arange(1,R+1,device='cuda').view(1,R).float(),
            rope_positions=torch.arange(5,device='cuda')+30,
            glue_rows_override=torch.ones((5,4),device='cuda',dtype=torch.uint8),
            K_glue_override=3,_p2_ctx_len=32,
            block_tables=torch.tensor([[0,1]],device='cuda'),cache_hits_list=[1])
        args=({},torch.arange(5,device='cuda').view(1,5),counts,tree,layout,
              torch.arange(10,device='cuda').view(2,5),
              torch.full((2,1),32,device='cuda'),torch.ones(5,device='cuda'))
        return r,args

    def test_graph_uses_shape_without_host_counts(self):
        for R in (3,5):
            r,args=self.rollout_args(R);expected=object()
            r._try_p2_executor=Mock(return_value=expected)
            with patch.dict(os.environ,SSD_TREE_EXEC='1',SSD_TREE_ARENA='1',SSD_TREE_STAGE1='0'), \
                 patch('ssd.engine.helpers.p2_prepare.host_fanout',side_effect=AssertionError('unexpected readback')):
                self.assertIs(r._p2tree_rollout(*args),expected)
            call=r._try_p2_executor.call_args
            self.assertEqual(call.kwargs['root_count'],R)
            self.assertIsNone(call.args[-1])
            self.assertEqual(call.args[0].numel(),R)
            self.assertIsNone(args[4].fan_out_list)

    def test_declined_graph_materializes_before_arena(self):
        r,args=self.rollout_args();r._try_p2_executor=Mock(return_value=None)
        class ArenaReached(Exception):pass
        def arena(toks,piv,**kw):
            self.assertEqual(args[4].fan_out_list,[[2,0,1,2]])
            self.assertEqual(toks.tolist(),[0,1,2])
            self.assertIsInstance(kw['glue_rows_by_root'],np.ndarray)
            self.assertEqual(kw['K_glue'],3)
            raise ArenaReached
        with patch.dict(os.environ,SSD_TREE_EXEC='1',SSD_TREE_ARENA='1',SSD_TREE_STAGE1='0'), \
             patch('ssd.engine.helpers.tree_expansion_policy.TreeExpansionPolicy'), \
             patch('ssd.engine.helpers.p2_tree.run_rollout_arena',side_effect=arena):
            with self.assertRaises(ArenaReached):r._p2tree_rollout(*args)

    def test_diagnostic_gets_exact_root_positions(self):
        r,args=self.rollout_args();r._lab_active=lambda:True
        r._lab_dump=Mock();r._try_p2_executor=Mock(return_value=object())
        with patch.dict(os.environ,SSD_TREE_EXEC='1',SSD_TREE_ARENA='1',SSD_TREE_STAGE1='0'):
            r._p2tree_rollout(*args)
        self.assertEqual(r._lab_dump.call_args.args[-1],[(0,None),(0,None),(2,None)])
        self.assertEqual(args[4].fan_out_list,[[2,0,1,2]])

    def test_list_diagnostics_reference(self):
        a=inputs();expected=reference(a)
        a['step_slot_maps']=list(a['step_slot_maps']);a['step_context_lens']=list(a['step_context_lens'])
        r=runner();got=r._p2_step_state(**a)
        self.assertEqual((got.ctx_len,got.reason),expected)
        self.assertFalse(hasattr(r,'_p2_state_snapshot'))


class TestHostPreparation(unittest.TestCase):
    def test_cpu_reference_retained(self):
        a=inputs('cpu');got=runner('cpu')._p2_step_state(**a)
        self.assertEqual((got.ctx_len,got.reason),reference(a))

    def test_host_glue_noop(self):
        a=np.ones((3,4),dtype=np.uint8)
        self.assertIs(host_glue_rows(a),a)


if __name__=='__main__':unittest.main()
