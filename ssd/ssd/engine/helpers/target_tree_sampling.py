"""Lossless target draws followed by cache-tree membership checks.

Conditional on ANY draft-selected tree and committed prefix, every visited
context draws from its target distribution. Missing children return that same
draw as recovery. Unvisited draws are discarded. This proof does not require
the pruned proposal law or ordered-WOR selection probabilities.
"""
import torch


def draw_targets(logits, temperatures, greedy=False, stochastic_only=False):
    if greedy:return logits.argmax(-1)
    t=torch.as_tensor(temperatures,dtype=torch.float32,device=logits.device)
    p=(logits.float()/t.clamp_min(1e-10).unsqueeze(-1)).softmax(-1)
    sampled=torch.multinomial(p.reshape(-1,p.shape[-1]),1).reshape(logits.shape[:-1])
    if not stochastic_only:
        sampled=torch.where(t==0,logits.argmax(-1),sampled)
    return sampled


def verify_dense(logits,tokens,topology,temperatures,*,greedy=False,
                 stochastic_only=False,draws=None):
    b,r,_=logits.shape;n=r-1;dev=logits.device
    if draws is None:
        t=torch.as_tensor(temperatures,dtype=torch.float32,device=dev)[:,None].expand(b,r)
        draws=draw_targets(logits,t,greedy,stochastic_only)
    child=topology['child'];valid=topology['child_valid']
    ext=torch.cat([tokens,tokens.new_zeros(b,1)],1)
    ct=ext.gather(1,child.reshape(b,-1)).reshape_as(child)
    match=(ct==draws[:,:,None])&valid
    order=torch.arange(child.shape[-1],device=dev)
    first=torch.where(match,order,child.shape[-1]).min(-1).values
    chosen=child.gather(2,first.clamp_max(child.shape[-1]-1)[:,:,None]).squeeze(-1)
    chosen=torch.where(first<child.shape[-1],chosen,-1)
    ctx=torch.zeros(b,1,dtype=torch.long,device=dev)
    live=torch.ones_like(ctx,dtype=torch.bool);path=[]
    for _ in range(n):
        node=chosen.gather(1,ctx);live=live&(node>=0)
        path.append(torch.where(live,node,-1));ctx=torch.where(live,node+1,ctx)
    paths=torch.cat(path,1) if path else tokens.new_empty(b,0)
    return paths,draws.gather(1,ctx),ctx


def verify_flat(logits,tokens,topo,n,temperatures,*,greedy=False,
                stochastic_only=False,draws=None):
    if draws is None:
        tt=torch.as_tensor(temperatures,dtype=torch.float32,device=logits.device)
        draws=draw_targets(logits,tt[topo['req']],greedy,stochastic_only)
    child=topo['child'];valid=topo['child_valid']
    match=(tokens[child]==draws[:,None])&valid
    order=torch.arange(child.shape[-1],device=logits.device)
    first=torch.where(match,order,child.shape[-1]).min(-1).values
    chosen=child.gather(1,first.clamp_max(child.shape[-1]-1)[:,None])[:,0]
    chosen=torch.where(first<child.shape[-1],chosen,-1)
    ctx=topo['roots'].clone();live=torch.ones_like(ctx,dtype=torch.bool);path=[]
    for _ in range(n):
        node=chosen[ctx];live=live&(node>=0)
        path.append(torch.where(live,topo['local'][node.clamp_min(0)]-1,-1))
        ctx=torch.where(live,node,ctx)
    paths=torch.stack(path,1) if path else tokens.new_empty(ctx.numel(),0)
    return paths,draws[ctx,None],topo['local'][ctx,None]
