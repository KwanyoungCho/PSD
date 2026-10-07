"""Frozen pre-optimization kernel from 81ea777; experimental control only."""
import torch

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
