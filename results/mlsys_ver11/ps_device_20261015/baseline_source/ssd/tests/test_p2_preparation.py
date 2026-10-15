"""PS metadata: full validity contract, strided inputs, GPU mask parity."""
import types
import unittest
import numpy as np
import torch
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.helpers.p2_prepare import copy_glue_rows, host_glue_rows


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
