"""Exact finite-state checks of score semantics, subset DP and pruning bias."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from core import (HERE, ROOT, production_functions, exact_ladder, path_product,
                  prefix_knapsack, closure_masks, valid_subset, compact)


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


def random_tree(rng,n):
    par=[]; sib=[]; counts=Counter()
    for j in range(n):
        p=int(rng.integers(-1,j));par.append(p);sib.append(counts[p]);counts[p]+=1
    return par,sib


def main():
    torch.set_num_threads(1)
    prod=production_functions();rng=np.random.default_rng(921)
    # Online lookup uses the same left-inclusive bins as NumPy searchsorted.
    # Verify priority arithmetic and, critically, that proposal q is untouched.
    from score_hook import tensor_priority
    boundaries=torch.tensor([.1,.3,.6],dtype=torch.double)
    table=torch.tensor([[.7,.8,.9,.95],[.3,.4,.5,.6],[.2,.3,.4,.5]],dtype=torch.double)
    raw_tensor=torch.tensor([[.1,.2,.7],[.01,.3,.8]],dtype=torch.double)
    saved_raw=raw_tensor.clone()
    parents=torch.tensor([0,0,0,1,1,1])
    log_parent=torch.tensor([.25,.5],dtype=torch.double).log()
    online=tensor_priority(log_parent,parents,raw_tensor,boundaries,table).exp().numpy().reshape(2,3)
    hand=np.array([[.2,.02,.015],[.35,.075,.0375]])
    online_error=float(np.max(np.abs(online-hand)))
    assert online_error<1e-12 and torch.equal(raw_tensor,saved_raw)
    # Production tensor ladder parity on small full distributions.
    maxerr=0.;dp_checks=0;subtree_checks=0;greedy_failures=0;example=None
    for trial in range(100):
        n=8;v=13;par,sib=random_tree(rng,n)
        p=rng.dirichlet(np.ones(v),n+1);qctx=rng.dirichlet(np.ones(v),n+1)
        q=np.array([qctx[pa+1] for pa in par]);tok=[];seen={}
        for j,pa in enumerate(par):
            d=q[j].copy();d[seen.get(pa,[])]=0;d/=d.sum()
            t=int(rng.choice(v,p=d));tok.append(t);seen.setdefault(pa,[]).append(t)
        calc=exact_ladder(par,sib,tok,p,q)
        got=prod['tree_policy_b_ladder'](par,sib,torch.tensor(tok),torch.tensor(p),torch.tensor(q))
        maxerr=max(maxerr,float(np.max(np.abs(got[0].numpy()-calc['alpha']))),
                   float(np.max(np.abs(got[1].numpy()-calc['terminal']))))
        raw=q[np.arange(n),tok];weights=calc['reach'];closure=closure_masks(par,sib)
        for budget in range(1,n+1):
            keep,value=prefix_knapsack(par,sib,weights,budget)
            values=[sum(weights[j] for j in range(n) if mask>>j&1)
                    for mask in range(1<<n) if mask.bit_count()<=budget and valid_subset(mask,closure)]
            assert abs(value-max(values))<1e-10
            dp_checks+=1
            reduced=compact(par,sib,tok,p,q,keep)
            assert abs(exact_ladder(*reduced)['al']-value)<1e-10
            subtree_checks+=1
            qweights=path_product(par,raw)
            qkeep,qvalue=prefix_knapsack(par,sib,qweights,budget)
            greedy=prod['rerank_tree_indices'](par,sib,raw,budget)
            gap=qvalue-sum(qweights[greedy])
            if gap>1e-10:
                greedy_failures+=1
                if example is None:example=dict(par=par,sib=sib,raw_q=raw.tolist(),budget=budget,
                    greedy=greedy,dp=qkeep,greedy_value=float(sum(qweights[greedy])),dp_value=qvalue)
    assert maxerr<1e-6
    # Perfect draft: ALL first-child edges accept, later siblings have zero reach.
    par=[-1,-1,0,0];sib=[0,1,0,1];tok=[0,1,1,0]
    p=np.tile([.4,.35,.25],(5,1));q=np.tile(p[0],(4,1))
    perfect=exact_ladder(par,sib,tok,p,q)
    assert np.allclose(perfect['reach'],[1,0,1,0])
    result=dict(seed=921,production_ladder_max_abs_error=maxerr,
        online_score_hand_calculation_max_error=online_error,online_score_preserves_raw_q=True,
        exact_dp_vs_enumeration_checks=dp_checks,subtree_reward_identity_checks=subtree_checks,
        greedy_q_objective_failures=greedy_failures,greedy_failure_example=example,
        perfect_draft=dict(reach=perfect['reach'].tolist(),q_product=path_product(par,q[np.arange(4),tok]).tolist()),
        rerank_selection_bias=bias_audit(prod['rerank_tree_indices'],prod['tree_verify_walk_tensor']),
        production_sha256=hashlib.sha256((ROOT/'ssd/ssd/engine/helpers/p2_tree.py').read_bytes()).hexdigest())
    (HERE/'math_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rerank_selection_bias'},indent=2))
    print('pruning bias:',result['rerank_selection_bias']['first_output'],'TV=',result['rerank_selection_bias']['tv'])


if __name__=='__main__':main()
