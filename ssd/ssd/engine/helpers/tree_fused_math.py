"""Exact fixed-shape fusion of mask construction and round-robin fanout."""
import torch
import triton
import triton.language as tl


@triton.jit
def _mask_kernel(Sel, Valid, Root, Anc, Glue, Prefix, GlueWidth, Out,
                 CANVAS:tl.constexpr, GW:tl.constexpr, WORDS:tl.constexpr,
                 PRIOR:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0)
    byte=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    bit=tl.arange(0,8)
    col=byte[:,None]*8+bit[None,:]
    live=tl.load(Valid+row)
    selected=tl.maximum(tl.load(Sel+row),0)
    root=tl.where(live,tl.load(Root+selected),0)
    prefix=tl.load(Prefix);gw=tl.load(GlueWidth)
    off=col-prefix
    glue=tl.load(Glue+root*GW+tl.minimum(tl.maximum(off,0),GW-1))
    value=tl.where((off>=0)&(off<gw),glue*live,col<prefix)
    spec=off-gw
    if PRIOR>0:
        safe=tl.minimum(tl.maximum(spec,0),PRIOR-1)
        word=tl.load(Anc+selected*WORDS+safe//63)
        ancestral=((word>>(safe%63))&1)*live
        value=tl.where((spec>=0)&(spec<PRIOR),ancestral,value)
    value=tl.where(col==prefix+gw+PRIOR+row,live,value)
    packed=tl.sum(value.to(tl.int32)*(1<<bit[None,:]),axis=1)
    tl.store(Out+row*(CANVAS//8)+byte,packed,byte<CANVAS//8)


def pack_mask(executor, wrapper, round_index):
    ex=executor;ar=ex.arena;sel,valid=ex._sel[round_index]
    canvas=wrapper._canvas_cols
    if canvas%8:raise ValueError('Mask canvas must be byte aligned')
    _mask_kernel[(sel.numel(),triton.cdiv(canvas//8,32))](
        sel,valid,ar.root,ar.anc_bits,ex.in_glue,ex.in_prefix_len,ex.in_glue_w,
        wrapper._custom_mask_buf,CANVAS=canvas,GW=ex.in_glue.shape[1],
        WORDS=ar.anc_words,PRIOR=ex.round_offsets[round_index],BLOCK=32)


@triton.jit
def _fan_kernel(Root, Pri, Sel, Valid, Remaining, Out,
                W:tl.constexpr, C:tl.constexpr, FUTURE:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.arange(0,BLOCK)
    valid=tl.load(Valid+i,i<W,False)
    selected=tl.maximum(tl.load(Sel+i,i<W,0),0)
    root=tl.load(Root+selected)
    pri=tl.load(Pri+selected).to(tl.float32)
    same=root[:,None]==root[None,:]
    earlier=(pri[None,:]>pri[:,None])|((pri[None,:]==pri[:,None])&(i[None,:]<i[:,None]))
    rank=tl.sum((same&earlier&valid[None,:]).to(tl.int32),1)
    count=tl.maximum(1,tl.sum((same&valid[None,:]).to(tl.int32),1))
    left=tl.load(Remaining+root)
    now=tl.where(left>0,tl.maximum(1,left-FUTURE),0)
    fan=tl.minimum(C,now//count+(rank<now%count))
    tl.store(Out+i,tl.where(valid,fan,0),i<W)


def fanout(arena, selected, valid, remaining, c_tensor, future_rounds):
    out=torch.empty_like(selected)
    _fan_kernel[(1,)](arena.root,arena.logpri,selected,valid,remaining,out,
        W=selected.numel(),C=int(c_tensor),FUTURE=max(0,int(future_rounds)),
        BLOCK=triton.next_power_of_2(selected.numel()))
    return out
