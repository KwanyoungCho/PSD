"""Freeze on development, then evaluate unchanged policies on new prompts."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'shared_review'))
from aggregate import load, estimate


def weights(d, mask=None):
    if mask is None: mask=np.ones(len(d['step']),bool)
    w=np.zeros(len(mask))
    ds_values=np.unique(d['dataset'][mask])
    for ds in ds_values:
        ds_mask=mask & (d['dataset']==ds)
        sids=np.unique(d['seq_id'][ds_mask])
        for sid in sids:
            sid_mask=ds_mask & (d['seq_id']==sid)
            seeds=np.unique(d['seed'][sid_mask])
            for seed in seeds:
                m=sid_mask & (d['seed']==seed)
                w[m]=1/(len(ds_values)*len(sids)*len(seeds)*m.sum())
    np.testing.assert_allclose(w.sum(),1)
    return w


def fit_stump(x,y,w,features):
    best=dict(kind='always_proxy',development_gain=0.)
    for j,name in enumerate(features):
        for t in np.unique(np.quantile(x[w>0,j],np.linspace(.05,.95,19))):
            for direction in ('above','below'):
                choose=x[:,j]>t if direction=='above' else x[:,j]<=t
                gain=float(w@(y*choose))
                if gain>best['development_gain']:
                    best=dict(kind='threshold',feature=name,index=j,threshold=float(t),direction=direction,development_gain=gain)
    return best


def predict_stump(x,model):
    if model['kind']=='always_proxy': return np.zeros(len(x),bool)
    v=x[:,model['index']]
    return v>model['threshold'] if model['direction']=='above' else v<=model['threshold']


def fit_ridge(x,y,w,alpha):
    mean=w@x
    std=np.sqrt(w@((x-mean)**2)).clip(1e-8)
    z=(x-mean)/std
    ym=float(w@y)
    coef=np.linalg.solve((z*w[:,None]).T@z+alpha*np.eye(x.shape[1]),z.T@(w*(y-ym)))
    return dict(alpha=alpha,mean=mean.tolist(),std=std.tolist(),coef=coef.tolist(),intercept=ym)


def predict_ridge(x,model):
    return ((x-model['mean'])/model['std'])@np.asarray(model['coef'])+model['intercept']>0


def freeze(d):
    w=weights(d);n=len(w)
    features=sorted(k for k in d if k.startswith('feature::'))
    x=np.column_stack([d[k] for k in features]).astype(float)
    y=(d['policy::residual']-d['policy::proxy']).astype(float)
    static={k:float(w@v) for k,v in d.items() if k.startswith('policy::')}
    # Group all seeds of each prompt together; stratified within dataset.
    folds=(d['seq_id']-1)%4
    cv_stump=np.zeros(n,bool)
    ridge_cv={alpha:np.zeros(n,bool) for alpha in (1e-4,.001,.01,.1,1.,10.)}
    for fold in range(4):
        train=folds!=fold;test=~train;wt=weights(d,train)
        cv_stump[test]=predict_stump(x[test],fit_stump(x,y,wt,features))
        for alpha in ridge_cv:
            model=fit_ridge(x,y,wt,alpha)
            ridge_cv[alpha][test]=predict_ridge(x[test],model)
    cv_gains={str(a):float(w@(y*c)) for a,c in ridge_cv.items()}
    alpha=max(ridge_cv,key=lambda a:cv_gains[str(a)])
    best_static=max(static,key=static.get)
    best_nontrivial=max((k for k in static if k!='policy::proxy'),key=static.get)
    model=dict(created_utc=datetime.now(timezone.utc).isoformat(),
               development_prompts=len(set(zip(d['dataset'],d['seq_id']))),
               features=features,best_static=best_static,best_nontrivial_static=best_nontrivial,
               static_development_means=static,
               stump=fit_stump(x,y,w,features),ridge=fit_ridge(x,y,w,alpha),
               cv_stump_gain=float(w@(y*cv_stump)),cv_ridge_gains=cv_gains,
               primary_endpoints=['best_static','stump','ridge'],
               note='Selected before any confirmation policy evaluation. All old prompts are development. Gates use only proxy/draft quantities.')
    model['development_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE/'development').glob('*.npz'))}
    model['analysis_sha256']=hashlib.sha256((HERE/'analyze.py').read_bytes()).hexdigest()
    path=HERE/'frozen_policies.json'
    if path.exists(): raise RuntimeError('Refusing to overwrite frozen selection')
    path.write_text(json.dumps(model,indent=2))
    print(json.dumps({k:v for k,v in model.items() if k not in ('development_hashes','ridge')},indent=2))
    return model


def summarize(d,records,model):
    n=len(d['step']);all_rows=np.ones(n,bool);w=weights(d)
    x=np.column_stack([d[k] for k in model['features']]).astype(float)
    base_p=d['policy::proxy'];base_r=d['policy::residual']
    choices=dict(stump=predict_stump(x,model['stump']),ridge=predict_ridge(x,model['ridge']))
    vals={k:v for k,v in d.items() if k.startswith(('policy::','global::','repair::'))}
    vals.update({k:np.where(c,base_r,base_p) for k,c in choices.items()})
    vals['best_static']=d[model['best_static']]
    vals['best_nontrivial_static']=d[model['best_nontrivial_static']]
    policies={k:dict(coverage=estimate(d,v,all_rows),gain_vs_proxy=estimate(d,v-base_p,all_rows),
                     gain_vs_residual=estimate(d,v-base_r,all_rows)) for k,v in vals.items()}
    h=d['row::h'][:,:4].astype(float)
    rowkeys=[k for k in d if k.startswith('row::')]
    def events(mask):
        mass=h*mask;total=mass.sum()
        return dict(rows=int(mask.sum()),event_fraction=float(total/h.sum()),
            means={k[5:]:float(np.sum(d[k][:,:4]*mass)/total) for k in rowkeys})
    groups={'all':events(np.ones_like(h,bool))}
    for name,edges in [('z',[0,.05,.1,.2,.4,.6,1.0001]),('delta',[0,.05,.1,.2,.3,.4,.6,1.0001])]:
        v=d['row::'+name][:,:4]
        for lo,hi in zip(edges[:-1],edges[1:]):
            m=(v>=lo)&(v<hi)
            if (h*m).sum()>0:groups[f'{name}_{lo:g}_{hi:g}']=events(m)
    z=d['row::z'][:,:4];delta=d['row::delta'][:,:4]
    for good in (True,False):
        for large in (True,False):
            m=((delta<.2)==good)&((z>=.4)==large)
            if (h*m).sum()>0:groups[f'error_{"low" if good else "high"}_z_{"large" if large else "small"}']=events(m)
    for name,mask in [('sharper',d['row::entropy_delta'][:,:4]<-.05),
                      ('flatter',d['row::entropy_delta'][:,:4]>.05),
                      ('same_entropy',np.abs(d['row::entropy_delta'][:,:4])<=.05),
                      ('rank_match',d['row::top1_match'][:,:4]>0),
                      ('rank_mismatch',d['row::top1_match'][:,:4]==0)]:
        groups[name]=events(mask)
    gain=base_r-base_p
    wins={}
    for name,m in [('residual_wins',gain>1e-6),('proxy_wins',gain< -1e-6),('tie',abs(gain)<=1e-6)]:
        wins[name]=dict(step_fraction=float(m.mean()),prompt_balanced_fraction=float(w@m),
                       conditional_gain=float((w*gain*m).sum()/(w@m)) if (w@m)>0 else None,
                       contribution=float(w@(gain*m)))
    # This is a paired prompt CI for the rejection-event probability integrated
    # decomposition, distinct from pooled conditional event averages above.
    decomp={k:estimate(d,(h*d['row::'+k][:,:4]).sum(1),all_rows)
            for k in ('available_gain','support_loss','rank_loss','residual','proxy','ideal_gain')}
    for policy in choices:
        wins[policy]=dict(residual_use_fraction=float(w@choices[policy]),
                         true_positive_contribution=float(w@(gain*choices[policy]*(gain>0))),
                         false_positive_contribution=float(w@(gain*choices[policy]*(gain<0))))
    # The original TV bound is for unmasked full-vocabulary distributions.
    zh=d['row::z_proxy'][:,:4]
    bound=np.minimum(1,delta/np.maximum(z,zh))
    assert np.nanmax(d['row::tv_residual'][:,:4]-bound)<3e-5
    tv_regret_sufficient=d['row::available_gain'][:,:4] > 2*d['row::tv_residual_excluded'][:,:4]+1e-6
    row_gain=d['row::residual'][:,:4]-d['row::proxy'][:,:4]
    assert not (tv_regret_sufficient & (row_gain< -2e-5)).any()
    return dict(records=records,totals=dict(runs=len(records),prompts=len(set(zip(d['dataset'],d['seq_id']))),
                sampled_steps=n,correction_contexts=n*4,verification_steps=sum(r['n_steps'] for r in records)),
                policies=policies,step_outcomes=wins,event_groups=groups,prompt_integrated_decomposition=decomp,
                diagnostics=dict(tv_bound_event_mean=float(np.sum(h*bound)/h.sum()),
                tv_bound_nontrivial_event_fraction=float(np.sum(h*(bound<1))/h.sum()),
                tv_regret_certificate_event_fraction=float(np.sum(h*tv_regret_sufficient)/h.sum()),
                uniform_fit_c_quantiles=np.quantile(d['row::uniform_c'][:,:4],[0,.25,.5,.75,1]).tolist()),
                weighting='Primary: equal datasets, prompts, seeds within prompt, sampled steps within generation. 4000 paired prompt bootstrap. Event groups: pooled true first-rejection h weights, descriptive.',
                frozen_policies_sha256=hashlib.sha256((HERE/'frozen_policies.json').read_bytes()).hexdigest())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','confirmation']);args=ap.parse_args()
    folder='development' if args.phase=='freeze' else 'confirmation'
    d,records=load(HERE/folder)
    assert len(records)==(8 if args.phase=='freeze' else 4)
    model=freeze(d) if args.phase=='freeze' else json.loads((HERE/'frozen_policies.json').read_text())
    assert model['analysis_sha256']==hashlib.sha256((HERE/'analyze.py').read_bytes()).hexdigest()
    result=summarize(d,records,model)
    (HERE/(folder+'_summary.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result['policies'][k] for k in ['policy::residual','policy::proxy','best_static','best_nontrivial_static','stump','ridge']},indent=2))


if __name__=='__main__':main()
