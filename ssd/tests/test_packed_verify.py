import os
os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
import unittest
from types import SimpleNamespace
import torch
from ssd.engine.helpers.packed_verify import build_packed_layout, packed_shape


class TestPackedLayout(unittest.TestCase):
    def seq(self, start, table):
        return SimpleNamespace(num_tokens=start+5, num_cached_tokens=start,
            token_ids=list(range(start+5)), block_table=table)

    def test_short_row_keeps_frontier_and_own_physical_pages(self):
        seqs = [self.seq(3,[5,1,7]), self.seq(5,[3,0,2])]
        x = build_packed_layout(seqs, [4,2], 4, 4, 2, 8)
        self.assertEqual(x["cu_seqlens_q"], [0,5,8,8])
        self.assertEqual(x["positions"], [3,4,5,6,7,5,6,7])
        self.assertEqual(x["slot_mapping"], [23,4,5,6,7,1,2,3])
        self.assertEqual(x["context_lens"], [8,8,1])
        self.assertEqual(x["dense_rows"], [0,1,2,3,4,5,6,7,5,5])

    def test_batch_and_token_padding_are_separate_sequences(self):
        seqs = [self.seq(3,[5,1,7]) for _ in range(3)]
        shape = packed_shape(3,11,[1,2,4,8],4)
        self.assertEqual(shape,(4,12))
        x = build_packed_layout(seqs,[4,2,2],4,4,*shape)
        self.assertEqual(x["cu_seqlens_q"], [0,5,8,11,11,12])
        self.assertEqual(len(x["input_ids"]),11)

    def test_misaligned_frontier_fails_even_without_assert_statements(self):
        seq=self.seq(3,[0,1,2]); seq.num_cached_tokens=2
        with self.assertRaises(AssertionError):
            build_packed_layout([seq],[2],4,4,1,4)

    def test_zero_proposal_row_has_one_real_query(self):
        seqs=[self.seq(3,[5,1,7]),self.seq(6,[3,0,2])]
        x=build_packed_layout(seqs,[4,0],4,4,2,8)
        self.assertEqual(x["cu_seqlens_q"],[0,5,6,8])
        self.assertEqual(x["context_lens"],[8,7,1])
        self.assertEqual(x["dense_rows"][-5:],[5]*5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_captured_attention_dynamic_lengths_match_dense(self):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        from ssd.layers.attention import store_kvcache
        dev="cuda"; dtype=torch.float16
        torch.manual_seed(31)
        # Noncontiguous physical pages and distinct context lengths.
        bt=torch.tensor([[3,0],[1,2],[3,0]],device=dev,dtype=torch.int32)
        base_k=torch.randn(4,256,2,64,device=dev,dtype=dtype)
        base_v=torch.randn_like(base_k)
        cache_k=base_k.clone(); cache_v=base_v.clone()
        qs=torch.zeros(10,4,64,device=dev,dtype=dtype)
        ks=torch.zeros(10,2,64,device=dev,dtype=dtype);vs=torch.zeros_like(ks)
        slots=torch.full((10,),-1,device=dev,dtype=torch.int32)
        cu=torch.tensor([0,5,10,10],device=dev,dtype=torch.int32)
        lengths=torch.tensor([263,16,1],device=dev,dtype=torch.int32)
        def run():
            store_kvcache(ks,vs,cache_k,cache_v,slots)
            return flash_attn_with_kvcache(qs,cache_k,cache_v,page_table=bt,
                cache_seqlens=lengths,cu_seqlens_q=cu,max_seqlen_q=5,causal=True)
        with torch.inference_mode():
            for _ in range(2): run()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): output=run()
            for lens in ([5,3],[3,5],[3,3],[5,5]):
                cache_k.copy_(base_k);cache_v.copy_(base_v)
                n=sum(lens); prefix=[258,11]
                q=torch.randn(n,4,64,device=dev,dtype=dtype)
                k=torch.randn(n,2,64,device=dev,dtype=dtype);v=torch.randn_like(k)
                qs[:n].copy_(q);ks[:n].copy_(k);vs[:n].copy_(v)
                slots.fill_(-1)
                idx=[]
                for b,l in enumerate(lens):
                    idx.extend(int(bt[b,p//256])*256+p%256 for p in range(prefix[b],prefix[b]+l))
                slots[:n].copy_(torch.tensor(idx,device=dev,dtype=torch.int32))
                cu.copy_(torch.tensor([0,lens[0],n,10],device=dev,dtype=torch.int32))
                lengths.copy_(torch.tensor([prefix[b]+lens[b] for b in range(2)]+[1],device=dev,dtype=torch.int32))
                g.replay()
                offset=0
                for b,l in enumerate(lens):
                    expected=flash_attn_with_kvcache(q[offset:offset+l].unsqueeze(0),cache_k,cache_v,
                        page_table=bt[b:b+1],cache_seqlens=lengths[b:b+1],causal=True)
                    torch.testing.assert_close(output[offset:offset+l],expected.squeeze(0),atol=.003,rtol=.003)
                    offset+=l


if __name__ == "__main__": unittest.main()
