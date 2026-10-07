"""Exact ordered-sibling residual ladders for independent batched trees."""
import torch
from ssd.engine.helpers.p2_tree import pack_piv, pack_tree_proxy_topology, q_probs_from_logits


def pack_topologies(parents, siblings, n, device,c_max=3):
    entries=[pack_tree_proxy_topology(p,s,n,c_max=c_max) for p,s in zip(parents,siblings)]
    return {k:torch.stack([e[k] for e in entries]).to(device) for k in entries[0]}


def ladder(tokens,p,q,topology,depth,exact=False,overlap_mix=0.0):
    """All contexts in parallel; siblings remain sequential within context.

    q[b,j] is the ORIGINAL parent proposal law for node j. Later siblings
    require conditioning that law on previous sibling tokens being removed.
    Returned residual is the target law after all children were rejected.
    """
    B,N=tokens.shape
    R=N+1
    dev=p.device
    bi=torch.arange(B,device=dev)[:,None]
    child=topology['child']
    cv=topology['child_valid']
    tok_ext=torch.cat([tokens,tokens.new_zeros(B,1)],1)
    q_ext=torch.cat([q,q.new_zeros(B,1,q.shape[-1])],1)
    residual=p.clone()
    draft=q_ext[bi,child[:,:,0]].clone()
    draft=torch.where(cv[:,:,:1],draft,torch.zeros_like(draft))
    all_reject=p.new_ones(B,R,1)
    alphas=[]
    pre=[]
    avg_alphas=[]; avg_pre=[]; avg_reject=all_reject.clone()
    for s in range(child.shape[-1]):
        tj=tok_ext.gather(1,child[:,:,s]).unsqueeze(-1)
        valid=cv[:,:,s:s+1]
        d=draft.gather(2,tj)
        if exact:
            # Finite-precision softmax may have fewer than C nonzero atoms.
            # WOR top-k then contains zero-mass padding after exhaustion;
            # those entries cannot be accepted as genuine proposals.
            valid=valid&(d>0)
        denom=torch.where(d>0,d,torch.ones_like(d)) if exact else d+1e-10
        alpha=(residual.gather(2,tj)/denom).clamp(max=1)*valid
        alphas.append(alpha.squeeze(-1))
        pre.append(all_reject.squeeze(-1))
        all_reject=all_reject*(1-alpha)
        if overlap_mix:
            avg=torch.minimum(residual,draft).sum(-1,keepdim=True).clamp(0,1)*valid
            avg_alphas.append(avg.squeeze(-1)); avg_pre.append(avg_reject.squeeze(-1))
            avg_reject=avg_reject*(1-avg)
        newer=(residual-draft).clamp_min(0)
        z=newer.sum(-1,keepdim=True)
        newer=torch.where(z>1e-12,newer/z.clamp_min(1e-30),torch.zeros_like(newer))
        residual=torch.where(valid,newer,residual)
        newer=draft.scatter(2,tj,0)
        z=newer.sum(-1,keepdim=True)
        newer=torch.where(z>1e-12,newer/z.clamp_min(1e-30),torch.zeros_like(newer))
        draft=torch.where(valid,newer,draft)
    alpha=torch.stack(alphas,-1)
    residual=torch.where(residual.sum(-1,keepdim=True)>1e-12,residual,p)
    if exact:
        # Verification consumes conditional accept probabilities and the
        # terminal residual; proxy reach scores would be discarded.
        return alpha,None,residual
    presib=torch.stack(pre,-1)
    par=(topology['par']+1).clamp(0,N)
    sib=topology['sib']
    base=alpha[bi,par,sib]*presib[bi,par,sib]*topology['node_valid']
    reach=base
    one=p.new_ones(B,1)
    for _ in range(max(0,depth-1)):
        reach=base*torch.cat([one,reach],1).gather(1,par)
    valid_ctx=torch.cat([torch.ones(B,1,dtype=torch.bool,device=dev),topology['node_valid']],1)
    term=torch.cat([one,reach],1)*all_reject.squeeze(-1)*valid_ctx
    if overlap_mix:
        avg_alpha=torch.stack(avg_alphas,-1)
        avg_presib=torch.stack(avg_pre,-1)
        avg_base=avg_alpha[bi,par,sib]*avg_presib[bi,par,sib]*topology['node_valid']
        avg_reach=avg_base
        for _ in range(max(0,depth-1)):
            avg_reach=avg_base*torch.cat([one,avg_reach],1).gather(1,par)
        avg_term=torch.cat([one,avg_reach],1)*avg_reject.squeeze(-1)*valid_ctx
        term=(1-overlap_mix)*term+overlap_mix*avg_term
    return alpha,term,residual


def candidates(exit_logits,q_logits,tokens,topology,wire_n,depth,top_k,
               target_temps=1.0,draft_temps=1.0,**policy):
    from ssd.engine.helpers.root_policy import tree_candidates
    return tree_candidates(exit_logits,q_logits,tokens,topology,wire_n,depth,top_k,
                           target_temps,draft_temps,**policy)[:2]


class BatchedTreeProxy:
    @torch.inference_mode()
    def __init__(self,b,n,v,dtype,device,wire_n,depth,top_k,pool=None,q_dtype=None,c_max=3,policy=None):
        self.exit=torch.zeros(b,n+1,v,dtype=dtype,device=device)
        self.q=torch.zeros(b,n,v,dtype=q_dtype or dtype,device=device)
        self.tokens=torch.zeros(b,n,dtype=torch.int64,device=device)
        self.c_max=c_max
        self.topology=pack_topologies([[]]*b,[[]]*b,n,device,c_max)
        self.tt=torch.ones(b,device=device); self.dt=torch.ones(b,device=device)
        def run(): return candidates(self.exit,self.q,self.tokens,self.topology,wire_n,depth,top_k,
                                     self.tt,self.dt,**(policy or {}))
        for _ in range(2): run()
        torch.cuda.synchronize(device)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,pool=pool): self.positions,self.packed=run()

    @torch.inference_mode()
    def prepare(self,parents,siblings,tokens,q,target_temps=1.0,draft_temps=1.0):
        B=len(parents)
        self.tt[:B].copy_(torch.as_tensor(target_temps,device=self.tt.device))
        self.dt[:B].copy_(torch.as_tensor(draft_temps,device=self.dt.device))
        N=self.tokens.shape[1]
        pad=self.tokens.shape[0]-B
        topo=pack_topologies(parents+[[]]*pad,siblings+[[]]*pad,N,self.tokens.device,self.c_max)
        for name in self.topology: self.topology[name].copy_(topo[name])
        self.tokens.zero_(); self.q.zero_()
        self.tokens[:B,:tokens.shape[1]].copy_(tokens)
        self.q[:B,:q.shape[1]].copy_(q)

    @torch.inference_mode()
    def replay(self,logits):
        self.exit.zero_()
        self.exit[:logits.shape[0]].copy_(logits)
        self.graph.replay()
        return self.positions,self.packed


@torch.inference_mode()
def verify_batch(logits_p,logits_q,tokens,topology,target_temps,draft_temps,
                 sampler_x=None,fan_out=1,greedy=False,uniforms=None):
    """Return path nodes and recovery, without a host sync at each sibling.

    Independent accept coins may be drawn before the walk. Unvisited nodes'
    coins are ignored. This is the same conditional residual ladder as the
    scalar verifier, with one final bounded result readback by the caller.
    """
    B,R,V=logits_p.shape
    N=R-1
    dev=logits_p.device
    bi=torch.arange(B,device=dev)[:,None]
    child=topology['child']
    ext=torch.cat([tokens,tokens.new_zeros(B,1)],1)
    child_tokens=ext.gather(1,child.reshape(B,-1)).reshape_as(child)
    if greedy:
        argmax=logits_p.argmax(-1)
        accepted=(child_tokens==argmax[:,:,None])&topology['child_valid']
        residual=None
    else:
        tt=torch.as_tensor(target_temps,device=dev,dtype=torch.float32)
        dt=torch.as_tensor(draft_temps,device=dev,dtype=torch.float32)
        p=(logits_p.float()/tt.clamp_min(1e-10)[:,None,None]).softmax(-1)
        # Mixed batches may include T=0 requests. Dividing by a tiny
        # temperature gives a uniform distribution over tied maxima;
        # greedy instead selects torch.argmax's first index deterministically.
        argmax=logits_p.argmax(-1,keepdim=True)
        greedy_p=torch.zeros_like(p).scatter(2,argmax,1)
        p=torch.where((tt==0)[:,None,None],greedy_p,p)
        q=q_probs_from_logits(logits_q.reshape(B*N,V),
                             dt[:,None].expand(B,N).reshape(-1),sampler_x,fan_out).view(B,N,V)
        alpha,_,residual=ladder(tokens,p,q,topology,N,exact=True)
        u=torch.rand(B,N,device=dev) if uniforms is None else uniforms
        ue=torch.cat([u,torch.ones(B,1,device=dev)],1)
        coins=ue.gather(1,child.reshape(B,-1)).reshape_as(child)
        accepted=(coins<alpha)&topology['child_valid']
    order=torch.arange(child.shape[-1],device=dev)
    first=torch.where(accepted,order,child.shape[-1]).min(-1).values
    selected=child.gather(2,first.clamp_max(child.shape[-1]-1)[:,:,None]).squeeze(-1)
    selected=torch.where(first<child.shape[-1],selected,-1)
    ctx=torch.zeros(B,1,dtype=torch.int64,device=dev)
    live=torch.ones(B,1,dtype=torch.bool,device=dev)
    path=[]
    for _ in range(N):
        node=selected.gather(1,ctx)
        live=live&(node>=0)
        path.append(torch.where(live,node,-1))
        ctx=torch.where(live,node+1,ctx)
    path=torch.cat(path,1)
    if greedy:
        recovery=argmax.gather(1,ctx)
    else:
        recovery=torch.multinomial(residual[bi[:,0],ctx[:,0]],1)
    return path,recovery,ctx


class BatchedTreeAccept:
    """Capture the entire exact walk, including independent CUDA RNG draws."""
    @torch.inference_mode()
    def __init__(self, proxy, sampler_x, fan_out, greedy, pool=None):
        self.proxy=proxy
        self.logits=torch.zeros_like(proxy.exit)
        b=proxy.tokens.shape[0]
        self.tt=torch.ones(b,device=proxy.tokens.device)
        self.dt=torch.ones_like(self.tt)
        def run():
            return verify_batch(self.logits,proxy.q,proxy.tokens,proxy.topology,
                                self.tt,self.dt,sampler_x,fan_out,greedy)
        for _ in range(2): run()
        torch.cuda.synchronize(proxy.tokens.device)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,pool=pool): self.out=run()

    @torch.inference_mode()
    def replay(self,logits,tt,dt):
        b=logits.shape[0]
        self.logits[:b].copy_(logits)
        self.tt[:b].copy_(torch.as_tensor(tt,device=self.tt.device))
        self.dt[:b].copy_(torch.as_tensor(dt,device=self.dt.device))
        self.graph.replay()
        return tuple(x[:b] for x in self.out)
