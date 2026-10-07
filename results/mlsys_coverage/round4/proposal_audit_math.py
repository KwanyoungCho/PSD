"""Exact proposal-law audit helpers, copied from the prior research audit.

Only the functions required by Round4 are included; no sibling checkout needed.
Production helper bodies are loaded from this checkout, not copied snapshots.
"""
import ast
import math
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]

def production_functions():
    """Load actual pure helper bodies, avoiding engine/CUDA initialization."""
    import torch
    path = ROOT / 'ssd/ssd/engine/helpers/p2_tree.py'
    names = {'rerank_tree_indices', 'validate_tree_ints',
             'tree_policy_b_ladder', 'tree_verify_walk_tensor'}
    tree = ast.parse(path.read_text())
    module = ast.Module(body=[n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    scope = dict(torch=torch, math=math, np=np, defaultdict=defaultdict)
    exec(compile(module, str(path), 'exec'), scope)
    return {name: scope[name] for name in names}


def tensor_first_output(walk, par, sib, tok, keep, p, q, qnext):
    """Integrate actual production walk's coin branches, not Monte Carlo."""
    remap={j:i for i,j in enumerate(keep)}
    parents=[-1 if par[j]<0 else remap[par[j]] for j in keep]
    contexts=sorted(set(par[j] for j in keep))
    qrefs={ctx:i for i,ctx in enumerate(contexts)}
    ti=dict(valid=len(keep),u_valid=len(contexts),tok=[tok[j] for j in keep],
            parent_local=parents,sib_order=[sib[j] for j in keep],
            parent_q_ref=[qrefs[par[j]] for j in keep])
    probs=torch.tensor([[float(x) for x in (q if ctx<0 else qnext)] for ctx in contexts])
    plog=torch.tensor([[float(x) for x in p]]*(len(keep)+1)).log()
    out=np.zeros(len(p)); pending=[[]]
    class NeedCoin(Exception): pass
    while pending:
        choices=pending.pop(); state=dict(i=0,weight=1.,terminal=None)
        class Coin:
            def __lt__(self, alpha):
                if state['weight']==0: return True
                i=state['i']
                if i==len(choices): raise NeedCoin()
                chosen=choices[i];state['i']+=1
                state['weight']*=alpha if chosen else 1-alpha
                return chosen
        def mult(probs):
            state['terminal']=probs.double().numpy()
            return int(probs.argmax())
        try:
            path,_=walk(ti,plog,probs,1.,Coin,mult)
        except NeedCoin:
            pending.extend([choices+[False],choices+[True]])
            continue
        if path: out[ti['tok'][path[0]]]+=state['weight']
        else: out+=state['weight']*state['terminal']
    return out


def bias_audit(rerank, walk):
    # One root; two WOR siblings; expand the higher-q sibling once (W=1).
    # G=3, M=2. Branch sampling happens AFTER expansion choice, as in dynamic.
    p = [F(4,10),F(3,10),F(3,10)]
    q = [F(6,10),F(25,100),F(15,100)]
    qnext = [F(3,10),F(3,10),F(4,10)]
    totals = {name:[F(0)]*3 for name in ['full','production_rerank','fixed_prefix','target_match']}
    changes = F(0); states = []; tensor_totals={name:np.zeros(3) for name in ['full','production_rerank','fixed_prefix']}
    for first in range(3):
        for second in range(3):
            if second == first: continue
            parent = 0 if q[first] >= q[second] else 1
            for z in range(3):
                prob = q[first]*q[second]/(1-q[first])*qnext[z]
                par=[-1,-1,parent]; sib=[0,1,0]; tok=[first,second,z]
                raw=[q[first],q[second],qnext[z]]
                keep=rerank(par,sib,list(map(float,raw)),2)
                if keep != [0,1]: changes += prob
                states.append(dict(tokens=tok,parent=parent,probability=float(prob),keep=keep))
                for name,chosen in [('full',[0,1,2]),('production_rerank',keep),('fixed_prefix',[0,1])]:
                    tensor_totals[name]+=float(prob)*tensor_first_output(walk,par,sib,tok,chosen,p,q,qnext)
                    R=p[:]; D=q[:]; left=F(1); out=[F(0)]*3
                    for j in chosen:
                        if par[j] != -1: continue
                        t=tok[j]; a=min(F(1),R[t]/D[t]); out[t]+=left*a
                        left*=1-a
                        residual=[max(x-y,F(0)) for x,y in zip(R,D)]
                        total=sum(residual)
                        R=[x/total for x in residual] if total else [F(0)]*3
                        D[t]=F(0); total=sum(D)
                        D=[x/total for x in D] if total else [F(0)]*3
                    for t in range(3):
                        out[t]+=left*R[t]
                        totals[name][t]+=prob*out[t]
                    assert sum(out)==1
                # Fresh target draw is exact regardless of which tree survives.
                for t in range(3): totals['target_match'][t]+=prob*p[t]
    assert totals['full']==p and totals['fixed_prefix']==p and totals['target_match']==p
    assert totals['production_rerank']!=p
    for name,result in tensor_totals.items():
        assert np.max(np.abs(result-np.array(list(map(float,totals[name])))))<1e-6
    return dict(target=list(map(float,p)),draft=list(map(float,q)),
                next_draft=list(map(float,qnext)),states=len(states),
                first_output={k:list(map(float,v)) for k,v in totals.items()},
                first_output_exact={k:list(map(str,v)) for k,v in totals.items()},
                tv=float(sum(abs(x-y) for x,y in zip(totals['production_rerank'],p))/2),
                changed_tree_probability=float(changes),enumeration=states,
                production_tensor_walk_first_output={k:v.tolist() for k,v in tensor_totals.items()},
                scope='Actual production rerank AND tensor walk bodies; exhaustive states/coin branches, plus exact rational reference. Algorithmic counterexample, not measured LLM output bias.')
