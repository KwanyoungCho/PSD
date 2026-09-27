"""Meaningful numerical checks for the new decision rules and assumptions."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tf_replay',HERE/'replay.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def main():
    torch.set_num_threads(4);torch.manual_seed(913)
    n=256;v=256;radius=.25
    e=mod.normalize(torch.distributions.Dirichlet(torch.ones(v)*.7).sample((n,5)).double())
    noise=(torch.rand_like(e)*2-1)*radius/4
    p=mod.normalize(e*(1+noise))
    q=mod.normalize(torch.distributions.Dirichlet(torch.ones(v)*.4).sample((n,5)).double());q[:,-1]=0
    y=(q-p).argmax(-1);y[:,-1]=-1
    assert ((p>=(1-radius)*e-1e-12)&(p<=(1+radius)*e+1e-12)).all()
    pe,re=mod.source(e,q,y,0),mod.source(e,q,y,1)
    a=(p-q).clamp_min(0);base=pe.topk(17,-1).indices
    true_prefix=a.gather(-1,base).cumsum(-1)
    rankings=dict(mod.rerankings(e,q,y,pe,re))
    candidate=rankings['interval0.25']
    improvement=a.gather(-1,candidate).cumsum(-1)-true_prefix
    assert improvement.min()>-1e-12
    np.testing.assert_array_equal(rankings['interval1'].numpy(),base.numpy())
    for name,ids in rankings.items():
        assert not (ids.sort(-1).values[...,1:]==ids.sort(-1).values[...,:-1]).any(),name
    # Exact no-op on ties: interval1 must preserve the actual baseline topk.
    tied=torch.ones_like(e)/v;ty=torch.zeros(n,5,dtype=torch.long);ty[:,-1]=-1
    ts=mod.source(tied,q,ty,0)
    null=dict(mod.rerankings(tied,q,ty,ts,mod.source(tied,q,ty,1)))['interval1']
    np.testing.assert_array_equal(null.numpy(),ts.topk(17,-1).indices.numpy())
    # Mean proxy rejection hazard under a draw Y~q is exactly TV(e,q).
    qq=q[:,:-1];ee=e[:,:-1]
    rejection=(1-(ee/qq).clamp(max=1))
    torch.testing.assert_close((qq*rejection).sum(-1),.5*(ee-qq).abs().sum(-1),atol=1e-12,rtol=0)
    # Top-M renormalization can choose the wrong position even with true R/h.
    h=np.array([.6,.4]);top=np.array([.4,.9])
    assert np.argmax(h)==0 and np.argmax(h*top)==1
    # Log-relative sensitivity: smooth discount is 1; floor bounds it by 1/rho.
    ev=torch.rand(20000,dtype=torch.float64)*.8+.01
    ev2=ev*torch.exp(torch.randn_like(ev)*.4)
    qv=torch.rand_like(ev)*.99
    for beta in [.5,1.,2.]:
        s1=ev*(1-qv).pow(beta);s2=ev2*(1-qv).pow(beta)
        torch.testing.assert_close((s2/s1).log(),(ev2/ev).log(),atol=1e-12,rtol=0)
    for rho in [.25,.5,.75,.9]:
        s1=torch.maximum((ev-qv).clamp_min(0),rho*ev)
        s2=torch.maximum((ev2-qv).clamp_min(0),rho*ev2)
        assert ((s2/s1).log().abs()<=(ev2/ev).log().abs()/rho+1e-12).all()
    deterministic=torch.zeros_like(q);deterministic[:,:4,0]=1
    dy=torch.zeros(n,5,dtype=torch.long);dy[:,-1]=-1
    ps=mod.source(e,deterministic,dy,0);rs=mod.source(e,deterministic,dy,1)
    torch.testing.assert_close(ps,rs,atol=0,rtol=0)
    smooth=mod.mask_y(e*(1-deterministic),dy)
    torch.testing.assert_close(ps,smooth,atol=0,rtol=0)
    result={'passed':True,'random_contexts':n*5,'vocabulary':v,
            'interval_relative_radius':radius,'minimum_certified_prefix_improvement':float(improvement.min()),
            'interval1_exact_baseline_including_ties':True,'candidate_ids_unique':True,
            'mean_proxy_rejection_equals_proxy_draft_TV':True,
            'smooth_discount_relative_condition_one':True,
            'floor_relative_condition_bounded_by_inverse_floor':True,
            'deterministic_draft_y_exclusion_makes_sources_identical':True,
            'topM_renormalization_counterexample':{'legacy_selected_mass':.24,'optimal_selected_mass':.36},
            'scope':'Interval dominance requires bounds to hold and prefix-consistent selection; not an unconditional empirical guarantee.'}
    (HERE/'checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
