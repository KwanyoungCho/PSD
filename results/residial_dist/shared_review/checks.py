"""Independent numerical checks for the shared analysis and replay evaluator."""
import importlib.util
import ast
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from replay import candidates, entropy, hazard, normalize, score, source


def main():
    rng=np.random.default_rng(20260910)
    # Probability simplex draws with several concentration scales and q~p.
    max_bound_violation=-1.
    max_mixture_error=0.
    for scale in (.05,.2,1.,5.):
        p=rng.dirichlet(np.full(128,scale),3000)
        e=rng.dirichlet(np.full(128,scale),3000)
        q=rng.dirichlet(np.full(128,scale),3000)
        for a in (.01,.2,1.):
            qq=(1-a)*p+a*q
            r=np.maximum(p-qq,0);z=r.sum(1)
            re=np.maximum(e-qq,0);ze=re.sum(1)
            tv=.5*np.abs(r/z[:,None]-re/ze[:,None]).sum(1)
            bound=np.minimum(1,.5*np.abs(p-e).sum(1)/np.maximum(z,ze))
            max_bound_violation=max(max_bound_violation,float(np.max(tv-bound)))
            np.testing.assert_array_less(tv-bound,1e-12)
            np.testing.assert_allclose(np.minimum(p,qq)+z[:,None]*(r/z[:,None]),p,atol=1e-14)
        mixed=.37*e+.63*q
        expected=np.maximum(e-q,0);expected/=expected.sum(1,keepdims=True)
        actual=np.maximum(mixed-q,0);actual/=actual.sum(1,keepdims=True)
        max_mixture_error=max(max_mixture_error,float(np.max(np.abs(expected-actual))))
        np.testing.assert_allclose(expected,actual,atol=1e-13)
    p=np.array([.5,.25,.15,.1]);q=np.array([.45,.26,.17,.12]);e=.6*p+.1
    r=np.maximum(p-q,0);r/=r.sum()
    re=np.maximum(e-q,0);re/=re.sum()
    assert r[np.argmax(e)]==1 and r[np.argmax(re)]==0
    np.testing.assert_allclose(e-.6*q-.1,.6*(p-q),atol=1e-15)
    alpha=.5/.55;alpha_proxy=(.6*.5+.4/3)/.55
    lp=np.log(p);le=np.log(e)
    def kl_at_beta(beta):
        x=np.exp(beta*le);x/=x.sum()
        return np.sum(p*(lp-np.log(x)))
    derivative_numeric=(kl_at_beta(1+1e-5)-kl_at_beta(1-1e-5))/(2e-5)
    delta_entropy=-np.sum(e*le)+np.sum(p*lp)
    derivative_identity=kl_at_beta(1)-delta_entropy
    np.testing.assert_allclose(derivative_numeric,derivative_identity,atol=1e-10)
    # Unequal token counts must not change equal-prompt/seed weighting.
    from aggregate import estimate
    attrs=dict(dataset=np.array(['a']*6+['b']*2),
               seq_id=np.array([1,1,1,1,2,2,1,1]),
               seed=np.array([42,42,42,123,42,123,42,123]))
    values=np.array([1.,1.,1.,3.,10.,10.,100.,100.])
    est=estimate(attrs,values,np.ones(8,dtype=bool),boot=1000)
    np.testing.assert_allclose(est['mean'],53.)
    pair=estimate(attrs,np.full(8,5.),np.ones(8,dtype=bool),boot=1000)
    np.testing.assert_allclose([pair['mean'],*pair['ci95']],[5.,5.,5.])
    # Verify graph selection independently, using actual production function.
    root=Path(__file__).resolve().parents[3]
    # Load the production function verbatim without initializing the SSD
    # engine package (which requires server environment and CUDA imports).
    path=root/'ssd/ssd/engine/helpers/p2_tree.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='chain_proxy_candidates_fixed')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    chain_proxy_candidates_fixed=namespace['chain_proxy_candidates_fixed']
    torch.manual_seed(20260910)
    for _ in range(20):
        logits_e=torch.randn(5,128)*2
        logits_q=torch.randn(4,128)*2
        pe=logits_e.softmax(-1);pq=logits_q.softmax(-1)
        yy=pq.multinomial(1).squeeze(1)
        qq=torch.cat([pq,torch.zeros_like(pq[:1])])[None]
        yy_b=torch.cat([yy,yy.new_tensor([-1])])[None]
        hh=hazard(pe[None],qq,yy_b)
        for name,lam in [('residual',1.),('proxy',0.)]:
            pos,tok,value=chain_proxy_candidates_fixed(logits_e,logits_q,yy,17,15,False,name)
            s=source(pe[None],qq,yy_b,lam)
            scores,ids=candidates(s,hh,17)
            selected=scores.flatten().topk(15).indices
            pos2=selected//17;tok2=ids.flatten()[selected]
            assert set(zip(pos.tolist(),tok.tolist()))==set(zip(pos2.tolist(),tok2.tolist()))
            torch.testing.assert_close(value,scores.flatten()[selected],atol=1e-7,rtol=1e-6)
            gt=torch.rand_like(pe);gt/=gt.sum()
            torch.testing.assert_close(score(s,hh,gt[None])[0],gt[pos,tok].sum())
    # Lossless snapshot persistence, sampling schedule and no side effects.
    spec=importlib.util.spec_from_file_location('distribution_probe',root/'ssd/ssd/engine/helpers/distribution_probe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as d:
        probe=module.DistributionProbe([56,79],Path(d)/'probe.json',56,stride=2,flush_steps=2)
        pt=torch.tensor([[.5,.25,.15,.1],[.3,.4,.2,.1]])
        pe=torch.stack([.6*pt+.1,pt]);pq=pt[:1].clone();y=torch.tensor([0])
        before=[x.clone() for x in (pe,pt,pq,y)];state=torch.random.get_rng_state().clone()
        for sid in (1,1,1,2,2): probe.observe(pe,pq,pt,y,seq_id=sid)
        probe.dump();meta=json.loads(probe.out_path.read_text())
        assert (meta['n_steps'],meta['n_samples'],meta['n_rows'])==(5,3,6)
        assert torch.equal(state,torch.random.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(before,(pe,pt,pq,y)))
        with np.load(probe.out_path.parent/meta['chunks'][0]['file']) as z:
            np.testing.assert_array_equal(z['p_E'][:2],pe.transpose(0,1).numpy())
            np.testing.assert_array_equal(z['step'],[0,0,2,2])
    result=dict(simplex_cases=36000,max_bound_violation=max_bound_violation,
                max_draft_mixture_invariance_error=max_mixture_error,
                toy=dict(p=p.tolist(),q=q.tolist(),e=e.tolist(),r=r.tolist(),r_proxy=re.tolist(),
                         residual_hit=0,proxy_hit=1),
                hazard_example=dict(alpha=alpha,alpha_proxy=alpha_proxy,
                                    first_reject=1-alpha,first_reject_proxy=1-alpha_proxy,
                                    bonus=alpha**6,bonus_proxy=alpha_proxy**6),
                production_candidate_parity_cases=40,
                temperature_gradient=dict(numeric=float(derivative_numeric),identity=float(derivative_identity)),
                prompt_weighting_and_paired_bootstrap='PASS',
                snapshot_persistence_and_rng='PASS')
    (Path(__file__).parent/'checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
