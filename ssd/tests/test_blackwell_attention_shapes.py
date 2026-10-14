"""70B TP2 *head geometry*, without claiming full-model/SM120 validation."""
from types import SimpleNamespace as NS
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class HeadGeometry(unittest.TestCase):
    @torch.inference_mode()
    def test_b1_b2_b8_alternating_graphs_and_noncontiguous_kv(self):
        from ssd.layers.attention import Attention
        from ssd.engine.helpers.batch_tree_forward import BatchedTreeForward
        from ssd.engine.helpers.tree_host_topology import context_topology
        dev=torch.device('cuda:0');page=256;hq,hkv,dim=32,4,128
        for dtype in (torch.float16,torch.bfloat16):
            class MiniAttention(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.emb=torch.nn.Embedding(97,hq*dim,device=dev,dtype=dtype)
                    self.pos=torch.nn.Embedding(2048,hq*dim,device=dev,dtype=dtype)
                    self.attn=Attention(hq,dim,dim**-.5,hkv,draft=True,
                                        speculate=True,draft_async=True)
                    self.attn.k_cache=torch.randn(80,page,hkv,dim,device=dev,dtype=dtype)*.1
                    self.attn.v_cache=torch.randn_like(self.attn.k_cache)*.1
                def project(self,ids,rope):
                    x=(self.emb(ids)+self.pos(rope))*.1
                    return x,x[:,:hkv*dim].contiguous(),x[:,-hkv*dim:].contiguous()
                def forward(self,ids,rope):
                    return self.attn(*self.project(ids,rope))
                def compute_logits(self,hidden,last_only):
                    return hidden
            torch.manual_seed(834)
            model=MiniAttention()
            r=NS(model=model,device=dev,block_size=page,num_tp_gpus=2,
                config=NS(max_blocks=8),hf_config=NS(torch_dtype=dtype,
                num_attention_heads=64,num_key_value_heads=8,head_dim=dim,hidden_size=hq*dim))
            forward=BatchedTreeForward(r,split=False)
            pointers={}
            for batch,width,prefix in [(1,5,250),(2,13,1734),(8,9,511),(2,13,1734),(1,5,250)]:
                rows=[]
                for i in range(batch):
                    size=width if i%2==0 else width-2
                    rows.append(dict(tokens=list(range(i+1,i+size+1)),
                        parents=[-1,-1]+list(range(size-3)),prefix=prefix+i,
                        blocks=[(i*9+j*7)%80 for j in range(8)]))
                # Choose disjoint physical pages per request with a permuted
                # global mapping, so no request shares KV accidentally.
                for i,row in enumerate(rows):row['blocks']=[(i*8+j)*7%79 for j in range(8)]
                expected=[]
                for row in rows:
                    depths,_,visible=context_topology(row['parents']);p=row['prefix']
                    ids=torch.tensor(row['tokens'],device=dev)
                    rope=torch.tensor([p]+[p+d+1 for d in depths],device=dev)
                    q,k,v=model.project(ids,rope);q=q.view(-1,hq,dim)
                    slots=torch.tensor([row['blocks'][j//page]*page+j%page for j in range(p)],device=dev)
                    k=torch.cat([model.attn.k_cache.view(-1,hkv,dim)[slots],k.view(-1,hkv,dim)])
                    v=torch.cat([model.attn.v_cache.view(-1,hkv,dim)[slots],v.view(-1,hkv,dim)])
                    k=k.repeat_interleave(hq//hkv,1).float();v=v.repeat_interleave(hq//hkv,1).float()
                    mask=torch.cat([torch.ones(len(ids),p,device=dev,dtype=torch.bool),
                                    torch.as_tensor(visible,device=dev).bool()],1)
                    logits=torch.einsum('qhd,khd->hqk',q.float(),k)*dim**-.5
                    prob=logits.masked_fill(~mask[None],float('-inf')).softmax(-1)
                    expected.append(torch.einsum('hqk,khd->qhd',prob,v).flatten(1))
                out=forward.run(rows,width).clone()
                for i,ref in enumerate(expected):
                    torch.testing.assert_close(out[i,:len(ref)].float(),ref,atol=.003,rtol=.015)
                for key,graph in forward.graphs.items():
                    address=graph['workspace'].data_ptr()
                    self.assertEqual(pointers.setdefault(key,address),address)
                # Prefix writes for one request must not alter another row.
                if batch>1:
                    model.attn.v_cache[rows[-1]['blocks'][0]].add_(.1)
                    changed=forward.run(rows,width)
                    torch.testing.assert_close(out[0],changed[0],atol=0,rtol=0)
            del forward,r,model
            torch.cuda.empty_cache()
