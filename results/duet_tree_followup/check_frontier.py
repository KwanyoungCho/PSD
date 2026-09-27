"""Actual selector/mask source checks on CPU, including mixed-depth cells."""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    torch.set_num_threads(1);rng=np.random.default_rng(92261)
    path=ROOT/'ssd/ssd/engine/helpers/p2_tree.py';tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_arena_select_global')
    for arg in node.args.args:arg.annotation=None
    source=ast.unparse(node);scope=dict(torch=torch,math=math)
    exec(compile(source,str(path),'exec'),scope);old=scope['_arena_select_global']
    if source.count('ar.depth == f')!=1:raise ValueError('Selector source changed')
    exec(compile(source.replace('ar.depth == f','ar.depth <= f'),str(path)+':frontier','exec'),scope)
    new=scope['_arena_select_global'];cases=0;relaxed=0
    for _ in range(500):
        r=int(rng.integers(1,5));cap=60;f=int(rng.integers(0,4));width=int(rng.integers(1,20));remaining=rng.integers(0,9,r)
        ar=SimpleNamespace(capacity=cap,device='cpu',n=torch.tensor(50),
            state=torch.tensor(rng.integers(0,2,cap)),valid=torch.tensor(rng.random(cap)>.1),
            depth=torch.tensor(rng.integers(0,f+2,cap)),raw_q=torch.tensor(rng.random(cap)),
            logpri=torch.tensor(np.log(rng.random(cap))),root=torch.tensor(rng.integers(0,r,cap)))
        ar.logpri[:r]=torch.tensor(np.log(rng.random(r)));piv=.01;conf=.03
        for fn,mode in [(old,'round'),(new,'frontier')]:
            sel,valid=fn(ar,width,f,4,torch.tensor(remaining),3-f,r,piv,conf)
            scores=ar.logpri.numpy();roots=ar.root.numpy();quota=np.where(remaining>0,np.maximum(1,remaining-(3-f)),0)
            eligible=[]
            for j in range(cap):
                d=int(ar.depth[j]);ok=j<50 and ar.state[j]==0 and ar.valid[j] and d<4 and (d==f if mode=='round' else d<=f)
                if f>0:ok=ok and scores[roots[j]]>=math.log(piv) and ar.raw_q[j]>=conf
                if ok:eligible.append(j)
            eligible.sort(key=lambda j:(-scores[j],j));used=np.zeros(r,int);expected=[]
            for j in eligible:
                if used[roots[j]]<quota[roots[j]]:expected.append(j);used[roots[j]]+=1
            got=sel[valid].tolist()
            if got!=expected[:width]:raise ValueError('Frontier selector differs from independent list implementation')
            if mode=='frontier':relaxed+=sum(int(ar.depth[j])<f for j in got)
            cases+=1
    # Execute production _pack_row_mask with mixed-depth ancestors at f=3.
    path=ROOT/'ssd/ssd/engine/helpers/p2_tree_executor.py';tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='P2TreeExecutor')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_pack_row_mask')
    scope=dict(torch=torch,PT=SimpleNamespace(_ANC_WORD_BITS=63))
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
    # Physical cells [0..8] precede current round; parents at depths 1 and 3.
    for plen in (0,7,129):
        dev='cpu';canvas=160;offset=9;gw=2;w=3
        anc=torch.tensor([[1,0],[1+(1<<4)+(1<<8),0],[0,0]],dtype=torch.long)
        ex=SimpleNamespace(round_widths=[3]*4,round_offsets=[0,3,6,9],dev=dev,
            arena=SimpleNamespace(root=torch.tensor([0,1,0]),anc_bits=anc),
            _sel={3:(torch.arange(3),torch.tensor([True,True,False]))},
            in_prefix_len=torch.tensor([plen]),in_glue=torch.tensor([[1,0],[1,1]],dtype=torch.uint8),
            in_glue_w=torch.tensor([gw]),lane_w=torch.arange(3))
        wr=SimpleNamespace(_canvas_cols=canvas,_custom_mask_buf=torch.zeros((w*canvas+7)//8,dtype=torch.uint8))
        scope['_pack_row_mask'](ex,wr,3)
        bits=np.unpackbits(wr._custom_mask_buf.numpy(),bitorder='little')[:w*canvas].reshape(w,canvas)
        expected=np.zeros_like(bits);expected[:,:plen]=1
        for lane,cells in [(0,[0]),(1,[0,4,8])]:
            expected[lane,plen:plen+gw]=ex.in_glue[lane].numpy()
            expected[lane,plen+gw+np.array(cells)]=1
            expected[lane,plen+gw+offset+lane]=1
        if not np.array_equal(bits,expected):raise ValueError('Mixed-depth ancestry mask mismatch')
    result=dict(passed=True,selector_cases=cases,older_depth_selections=relaxed,mixed_depth_mask_cases=3,
        scope='Actual production selector and mask tensor bodies on CPU; GPU replay still required')
    (HERE/'frontier_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
