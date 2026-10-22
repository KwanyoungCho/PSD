"""Exclude existing direct children without a vocabulary-sized copy.

The CUDA path filters the small sorted candidate table in one launch. Parent
indices are the same live inputs already used to compute P1 context reach.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=['P','K'])
def _filter_roots(candidates, returned, parents, output, validity,
                  P, K, U: tl.constexpr,
                  BK: tl.constexpr, BP: tl.constexpr, BU: tl.constexpr,
                  WRITE_VALID: tl.constexpr):
    row=tl.program_id(0)
    k=tl.arange(0,BK)
    j=tl.arange(0,BP)
    candidate=tl.load(candidates+row*K+k,k<K,other=-1)
    child=tl.load(returned+j,j<P,other=-2)
    parent=tl.load(parents+j,(j>0)&(j<P),other=-1)
    excluded=tl.sum(((candidate[:,None]==child[None,:]) &
                     (parent[None,:]==row) & (j[None,:]>0) &
                     (j[None,:]<P)).to(tl.int32),1)>0
    order=tl.sort(tl.where((k<K)&~excluded,k,BK),descending=False)
    u=tl.arange(0,BU)
    selected=tl.gather(order,u,0)
    token=tl.gather(candidate,tl.minimum(selected,BK-1),0)
    good=selected<K
    tl.store(output+row*U+u,tl.where(good,token,0),u<U)
    if WRITE_VALID:
        tl.store(validity+row*U+u,good,u<U)


def select_root_tokens(logits, returned, parent, count, *, chain=False):
    """Top ``count`` tokens outside each context's direct-child token set.

    At most P-1 child tokens can be excluded in a P-context tree, so the
    first U+P-1 vocabulary candidates suffice. A chain needs only U+1.
    Return validity only for tiny vocabularies that cannot guarantee U
    eligible tokens; the production vocabulary always satisfies the bound.
    Equal-logit ties retain the ordering produced by torch.topk.
    """
    p,v=logits.shape
    if count>v:
        raise ValueError(f'roots_per_position={count} exceeds vocab={v}')
    excluded_bound=min(p-1,1) if chain else p-1
    k=min(v,count+excluded_bound)
    candidates=logits.topk(k,dim=-1).indices
    guaranteed=count+excluded_bound<=v
    if logits.is_cuda:
        output=torch.empty((p,count),dtype=torch.int64,device=logits.device)
        valid=None if guaranteed else torch.empty_like(output,dtype=torch.bool)
        _filter_roots[(p,)](candidates,returned,parent,output,
            valid if valid is not None else output,P=p,K=k,U=count,
            # All native P<=24 root tables share one specialization. Do
            # not JIT a new kernel when a shorter live tree arrives later.
            BK=max(32,tr.next_power_of_2(k)),BP=max(32,tr.next_power_of_2(p)),
            BU=tr.next_power_of_2(count),WRITE_VALID=not guaranteed,
            num_warps=4)
        return output,valid
    # CPU correctness/reference path. No data-dependent host operation is
    # used by CUDA execution or CUDA graph capture.
    j=torch.arange(p,device=logits.device)
    children=(parent[None,:]==j[:,None]) & (j[None,:]>0)
    excluded=((candidates[:,:,None]==returned[None,None,:]) &
              children[:,None,:]).any(-1)
    order=torch.arange(k,device=logits.device).expand(p,k)
    selected=torch.where(excluded,k,order).topk(count,dim=-1,
        largest=False,sorted=True).values
    valid=selected<k
    output=candidates.gather(1,selected.clamp_max(k-1))
    output=torch.where(valid,output,0)
    return output,None if guaranteed else valid
