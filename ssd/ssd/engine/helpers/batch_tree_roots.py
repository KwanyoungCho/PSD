"""Batched P1 root preparation preserving the existing ranking policy."""
import torch
from ssd.engine.helpers.p2_tree import selected_q_probs_from_logits
from ssd.engine.helpers.tree_host_topology import context_topology


def root_candidates(logits,tokens,visibility,lengths,temps,roots_per_position,sampler_x,fan_out):
    B,P,V=logits.shape
    U=roots_per_position
    dev=logits.device
    idx=torch.arange(P,device=dev)
    candidates=logits.topk(U+1,-1).indices
    excluded=torch.cat([tokens[:,1:],tokens.new_full((B,1),-1)],1)
    excluded=torch.where(idx[None]<lengths[:,None]-1,excluded,-1)
    rank=torch.arange(U+1,device=dev).expand(B,P,U+1)
    keep=torch.where(candidates!=excluded[:,:,None],rank,U+1).topk(U,-1,largest=False).values
    chosen=candidates.gather(2,keep)
    parent=torch.where(visibility.bool()&(idx[None,:]<idx[:,None]),idx,0).amax(-1)
    offset=torch.arange(B,device=dev)[:,None]*P
    sources=(idx+offset)[:,:,None].expand(B,P,U+1).clone()
    sources[:,:,-1]=parent+offset
    ids=torch.cat([chosen,tokens[:,:,None]],2)
    probs=selected_q_probs_from_logits(logits.reshape(B*P,V),temps[:,None].expand(B,P).reshape(-1),
        ids.reshape(B*P,U+1),sampler_x,fan_out,source_rows=sources.reshape(B*P,U+1)).view(B,P,U+1)
    edge=probs[:,:,-1].clone();edge[:,0]=1
    reach=torch.exp(torch.bmm(visibility.float(),edge.clamp_min(1e-30).log()[:,:,None])).squeeze(-1)
    scores=probs[:,:,:U]*reach[:,:,None]*(idx[None,:,None]<lengths[:,None,None])
    return chosen.reshape(B,P*U),scores.reshape(B,P*U)


class BatchedP1Roots:
    @torch.inference_mode()
    def __init__(self,b,p,v,dtype,device,u,sampler_x,fan_out,pool=None):
        self.logits=torch.zeros(b,p,v,dtype=dtype,device=device)
        self.tokens=torch.zeros(b,p,dtype=torch.int64,device=device)
        self.visibility=torch.zeros(b,p,p,dtype=torch.uint8,device=device)
        self.lengths=torch.zeros(b,dtype=torch.int64,device=device)
        self.temps=torch.ones(b,dtype=torch.float32,device=device)
        self.context_ids=torch.arange(p,device=device).repeat_interleave(u)
        def run():return root_candidates(self.logits,self.tokens,self.visibility,self.lengths,self.temps,u,sampler_x,fan_out)
        for _ in range(2):run()
        torch.cuda.synchronize(device)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,pool=pool):self.out=run()

    @torch.inference_mode()
    def replay(self,rows,glue,temps):
        b=len(rows);p=self.tokens.shape[1]
        host_tokens=torch.zeros_like(self.tokens,device='cpu')
        host_vis=torch.zeros_like(self.visibility,device='cpu')
        lengths=[0]*self.tokens.shape[0]
        for i,row in enumerate(rows):
            n=len(row['tokens']);lengths[i]=n
            host_tokens[i,:n]=torch.tensor(row['tokens'])
            host_vis[i,:n,:n]=torch.as_tensor(context_topology(row['parents'])[2])
        self.logits.zero_();self.logits[:b,:glue.shape[1]].copy_(glue)
        self.tokens.copy_(host_tokens);self.visibility.copy_(host_vis)
        self.lengths.copy_(torch.tensor(lengths))
        self.temps[:b].copy_(torch.tensor(temps,dtype=torch.float32))
        self.graph.replay()
        return self.out
