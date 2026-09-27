"""Prompt-cluster statistics and figures for the shared-analysis review."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from replay import TAUS, LAMBDAS


def load(folder):
    parts=[]
    records=[]
    for path in sorted(folder.glob('*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text())
        r=meta['record'];records.append(r)
        with np.load(path) as z: d={k:z[k] for k in z.files}
        d['dataset']=np.full(len(d['step']),r['dataset'])
        d['seed']=np.full(len(d['step']),r['seed'])
        parts.append(d)
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]},records


def clusters(d, values, mask):
    """Mean within generation, across seeds, then equally across prompts."""
    result={}
    for ds in np.unique(d['dataset']):
        ps=[]
        for sid in np.unique(d['seq_id'][mask & (d['dataset']==ds)]):
            seed_means=[]
            for seed in np.unique(d['seed']):
                m=mask & (d['dataset']==ds) & (d['seq_id']==sid) & (d['seed']==seed)
                if m.any(): seed_means.append(values[m].mean(axis=0))
            ps.append(np.mean(seed_means,axis=0))
        if ps:result[str(ds)]=np.asarray(ps)
    return result


def estimate(d, values, mask, boot=4000):
    c=clusters(d,values,mask)
    point=np.mean([a.mean(axis=0) for a in c.values()],axis=0)
    rng=np.random.default_rng(20260910)
    replicates=[]
    for a in c.values():
        index=rng.integers(len(a),size=(boot,len(a)))
        replicates.append(a[index].mean(axis=1))
    lo,hi=np.quantile(np.mean(replicates,axis=0),[.025,.975],axis=0)
    return dict(mean=np.asarray(point).tolist(),ci95=[np.asarray(lo).tolist(),np.asarray(hi).tolist()],
                dataset_means={k:a.mean(axis=0).tolist() for k,a in c.items()})


def pooled_stats(d,mask=None):
    # A secondary, explicitly event-weighted view; do not confuse with the
    # prompt-balanced candidate coverage tables.
    h=d['row::h_true'][:,:4]
    m=np.ones_like(h,dtype=bool) if mask is None else mask
    w=h*m
    avg=lambda k:float(np.sum(d['row::'+k][:,:4]*w)/w.sum())
    names=['delta','tvd','top1_match','residual_tvd','lost_mass','ds_recovered_lost',
           'local_residual','local_proxy','local_oracle','z','z_proxy',
           'thermal_tvd','thermal_lost','thermal_local_residual','thermal_local_proxy',
           'rankfixed_tvd','rankfixed_lost','rankfixed_local_residual','rankfixed_local_proxy']
    return dict(rows=int(m.sum()),row_fraction=float(m.mean()),
                reject_event_fraction=float(w.sum()/h.sum()),**{k:avg(k) for k in names})


def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--partial',action='store_true')
    args=ap.parse_args()
    out=Path(__file__).parent
    d,records=load(out/'metrics')
    if not args.partial and len(records)!=8:raise ValueError('expected 8 completed replay runs')
    exact_meta=None
    if (out/'exact_calibration.npz').exists():
        exact_meta=json.loads((out/'exact_calibration.json').read_text())
        with np.load(out/'exact_calibration.npz') as z:
            lookup={(str(ds),int(seed),int(step)):i for i,(ds,seed,step) in enumerate(zip(z['dataset'],z['seed'],z['step']))}
            ids=[lookup[(str(ds),int(seed),int(step))] for ds,seed,step in zip(d['dataset'],d['seed'],d['step'])]
            for k in z.files:
                if k not in ('dataset','seed','step','seq_id'):d['exact::'+k]=z[k][ids]
    train=d['seq_id']<=16;test=~train;all_rows=np.ones(len(train),dtype=bool)
    # Selection is restricted to training prompts for EVERY tuning objective.
    fit_train={str(t):estimate(d,d[f'fit::tau{t:g}__kl'].mean(1),train)['mean'] for t in TAUS}
    tau_kl=min(TAUS,key=lambda t:fit_train[str(t)])
    train_grid={k:estimate(d,v,train)['mean'] for k,v in d.items()
                if k.startswith('grid::') and k.count('__')==1}
    best_joint=max(train_grid,key=train_grid.get)
    best_residual=max((k for k in train_grid if k.endswith('__lambda1')),key=train_grid.get)
    best_lambda=max((k for k in train_grid if k.startswith('grid::tau1__')),key=train_grid.get)
    base_r=d['grid::tau1__lambda1'];base_p=d['grid::tau1__lambda0']
    selections=dict(tau_kl=tau_kl,joint=best_joint,residual_tau=best_residual,lambda_only=best_lambda)
    temperature_gradient={}
    for name,mask in [('train',train),('test',test),('all',all_rows)]:
        temperature_gradient[name]=dict(
            kl=estimate(d,d['fit::tau1__kl'].mean(1),mask)['mean'],
            delta_entropy=estimate(d,d['fit::tau1__entropy_delta'].mean(1),mask)['mean'],
            beta_gradient=estimate(d,(d['fit::tau1__kl']-d['fit::tau1__entropy_delta']).mean(1),mask))
    keys={
        'residual':'grid::tau1__lambda1',
        'proxy':'grid::tau1__lambda0',
        'KL_calibrated_residual':f'grid::tau{tau_kl:g}__lambda1',
        'KL_calibrated_proxy':f'grid::tau{tau_kl:g}__lambda0',
        'coverage_calibrated_residual':best_residual,
        'lambda_only':best_lambda,
        'joint_tau_lambda':best_joint,
        'oracle_residual_proxy_h':'step::oracle_residual__proxy_h__m17',
        'residual_true_h':'step::residual__true_h__m17',
        'proxy_true_h':'step::proxy__true_h__m17',
        'oracle_residual_true_h':'step::oracle_residual__true_h__m17',
        'oracle_target_true_h':'step::oracle_target__true_h__m17',
        'ceiling':'step::ceiling',
    }
    if exact_meta:
        selections['tau_exact_KL']=exact_meta['tau']
        keys.update(exact_KL_residual='exact::residual',exact_KL_proxy='exact::proxy')
    policies={}
    for name,key in keys.items():
        policies[name]=dict(key=key,train=estimate(d,d[key],train),
            test=estimate(d,d[key],test),all=estimate(d,d[key],all_rows),
            test_gain_vs_residual=estimate(d,d[key]-base_r,test),
            test_gain_vs_proxy=estimate(d,d[key]-base_p,test))
    write_csv(out/'policies.csv',[dict(policy=k,key=v['key'],train=v['train']['mean'],
         test=v['test']['mean'],test_lo=v['test']['ci95'][0],test_hi=v['test']['ci95'][1],
         test_gain_vs_residual=v['test_gain_vs_residual']['mean'],
         gain_lo=v['test_gain_vs_residual']['ci95'][0],gain_hi=v['test_gain_vs_residual']['ci95'][1],
         test_gain_vs_proxy=v['test_gain_vs_proxy']['mean']) for k,v in policies.items()])
    # Preserve all tau curves, including held-out values, while keeping the
    # selected policy fixed by the training objective above.
    curves=[]
    for tau in TAUS:
        fits={metric:estimate(d,d[f'fit::tau{tau:g}__{metric}'].mean(1),test)['mean']
              for metric in ('kl','tvd','entropy_delta')}
        for lam in LAMBDAS:
            k=f'grid::tau{tau:g}__lambda{lam:g}'
            s=estimate(d,d[k],test)
            row=dict(tau=tau,lam=lam,train=train_grid[k],test=s['mean'],test_lo=s['ci95'][0],test_hi=s['ci95'][1],**fits)
            if lam in (0.,1.):
                row.update({hn:estimate(d,d[k+'__'+hn],test)['mean'] for hn in ('fixed_h','true_h')})
            else:row.update(fixed_h=None,true_h=None)
            if lam==1.:row['h_only']=estimate(d,d[k+'__h_only'],test)['mean']
            else:row['h_only']=None
            curves.append(row)
    write_csv(out/'calibration_grid.csv',curves)
    categories={'sharper':d['row::delta'][:,:4]<-.05,
                'similar':np.abs(d['row::delta'][:,:4])<=.05,
                'flatter':d['row::delta'][:,:4]>.05}
    conditional={k:pooled_stats(d,m) for k,m in categories.items()}
    conditional.update({
        'top1_match':pooled_stats(d,d['row::top1_match'][:,:4]>0),
        'top1_mismatch':pooled_stats(d,d['row::top1_match'][:,:4]==0),
        'high_overlap':pooled_stats(d,d['row::z'][:,:4]<.2),
        'low_overlap':pooled_stats(d,d['row::z'][:,:4]>=.2),
    })
    write_csv(out/'conditional.csv',[dict(group=k,**v) for k,v in conditional.items()])
    residual_gap=d['row::local_residual'][:,:4]-d['row::local_proxy'][:,:4]
    def corr(x,y,w):
        a=np.sum(x*w)/w.sum();b=np.sum(y*w)/w.sum()
        return float(np.sum(w*(x-a)*(y-b))/np.sqrt(np.sum(w*(x-a)**2)*np.sum(w*(y-b)**2)))
    w=d['row::h_true'][:,:4]
    secondary={k:estimate(d,v,all_rows) for k,v in d.items() if k.startswith('step::')}
    secondary.update(pooled=pooled_stats(d),conditional=conditional,
        delta_local_gap_correlation=corr(d['row::delta'][:,:4],residual_gap,w),
        delta_lost_mass_correlation=corr(d['row::delta'][:,:4],d['row::lost_mass'][:,:4],w),
        entropy_mean_rows=float(d['row::delta'][:,:4].mean()),
        top1_match_rows=float(d['row::top1_match'][:,:4].mean()),
        thermal_entropy_max_error=float(np.abs(d['row::thermal_entropy_error']).max()),
        thermal_entropy_p99_error=float(np.quantile(np.abs(d['row::thermal_entropy_error']),.99)),
        rankfixed_entropy_max_error=float(np.abs(d['row::rankfixed_entropy_error']).max()),
        nontrivial_bound_fraction=float((d['row::stronger_bound'][:,:4]<1).mean()))
    # Exact target ties impose a minimum temperature-family entropy log(tie
    # count). Exclude infeasible matches from *both* sides of the control.
    feasible=np.abs(d['row::thermal_entropy_error'])<1e-3
    secondary['thermal_feasible_rows']=pooled_stats(d,feasible[:,:4])
    feasible_steps=feasible.all(1)
    secondary['thermal_feasible_step_fraction']=float(feasible_steps.mean())
    secondary['thermal_feasible_steps']={k:estimate(d,d['step::'+k],feasible_steps)
        for k in ('residual__true_h__m17','proxy__true_h__m17',
                  'thermal_residual_true_h','thermal_proxy_true_h')}
    # Compare DS/PS policies using the exact same DS, and compare total unique
    # root budgets 15 or 27 to the corresponding no-DS reference.
    ds_pairs={}
    for n in (3,15):
        res=d[f'step::ds12_add{n}_residual'];pro=d[f'step::ds12_add{n}_proxy']
        ds_pairs[f'ds12_add{n}_proxy_minus_residual']=estimate(d,pro-res,all_rows)
        for name,a in [('residual',res),('proxy',pro)]:
            ds_pairs[f'ds12_add{n}_{name}_vs_equal_total_ps_only']=estimate(d,a-d[f'step::ps_only{12+n}_{name}'],all_rows)
    checks=json.loads((out/'checks.json').read_text())
    old=[]
    for p in Path('ssd/experiments/proxy_source_ablation/probe_ds_seed_20260909/out').glob('*.json'):
        a=json.loads(p.read_text());i=a['layers'].index(56);j=a['budgets'].index(15)
        old.append(np.asarray(a['hit'])[i,:,:,j])
    prior=np.mean(old,axis=0)
    result=dict(records=records,totals=dict(runs=len(records),generations=sum(r['numseqs'] for r in records),
        verification_steps=sum(r['n_steps'] for r in records),sampled_steps=len(train),
        contexts=len(train)*5,reject_positions=len(train)*4,
        unique_prompts=len(set(zip(d['dataset'],d['seq_id'])))),
        bootstrap='4000 paired resamples of prompt clusters within each dataset; seeds averaged within prompt',
        selections=selections,training_KL=fit_train,temperature_gradient=temperature_gradient,
        policies=policies,secondary=secondary,
        ds_pairs=ds_pairs,checks=checks,prior_12run_matrix=prior.tolist())
    if exact_meta:
        result['exact_calibration']=dict(metadata=exact_meta,
            train={k:estimate(d,v,train) for k,v in d.items() if k.startswith('exact::')},
            test={k:estimate(d,v,test) for k,v in d.items() if k.startswith('exact::')})
    (out/'results.json').write_text(json.dumps(result,indent=2))
    # Figures are English to remain portable across environments/fonts.
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    for lam,label,color in [(1.,'Residual','#3268ad'),(0.,'Proxy only','#d05d47')]:
        c=[x for x in curves if x['lam']==lam]
        axs[0].plot(TAUS,[x['test'] for x in c],'-o',label=label,color=color)
        axs[0].plot(TAUS,[x['fixed_h'] for x in c],'--',alpha=.55,color=color)
    axs[0].axvline(tau_kl,color='.5',ls=':',label=f'Train KL optimum ({tau_kl:g})')
    if exact_meta:
        axs[0].scatter([exact_meta['tau']]*2,[policies[k]['test']['mean'] for k in ('exact_KL_residual','exact_KL_proxy')],
                       marker='*',s=100,c=['#3268ad','#d05d47'],zorder=5,label='Continuous train KL optimum')
    axs[0].set(xlabel='Proxy calibration temperature',ylabel='Held-out recovery coverage',title='Solid: recompute h; dashed: fixed h')
    axs[0].legend(fontsize=8)
    c=[x for x in curves if x['lam']==1.]
    axs[1].plot(TAUS,[x['kl'] for x in c],'-o',label='KL(target || proxy)')
    axs[1].plot(TAUS,[x['tvd'] for x in c],'-s',label='TV(target, proxy)')
    axs[1].set(xlabel='Proxy calibration temperature',title='Held-out distribution error')
    axs[1].legend(fontsize=9)
    names=['Residual','Proxy only','Tuned residual','Tuned tau/lambda','Oracle']
    selected=[policies[k]['test'] for k in ['residual','proxy','coverage_calibrated_residual','joint_tau_lambda','ceiling']]
    vals=np.array([a['mean'] for a in selected]);bounds=np.array([a['ci95'] for a in selected]).T
    axs[2].barh(names,vals,color=['#3268ad','#d05d47','#65a488','#8176b5','#888888'])
    axs[2].errorbar(vals,np.arange(5),xerr=np.abs(bounds-vals),fmt='none',color='black',capsize=3)
    axs[2].set(xlim=(0,1),xlabel='Held-out coverage (95% prompt CI)',title='Selection uses training prompts only')
    axs[2].invert_yaxis()
    for ext in ('png','pdf'):fig.savefig(out/f'01_calibration.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    cats=list(categories);x=np.arange(3);width=.35
    axs[0].bar(x-width/2,[conditional[k]['local_residual'] for k in cats],width,label='Residual',color='#3268ad')
    axs[0].bar(x+width/2,[conditional[k]['local_proxy'] for k in cats],width,label='Proxy only',color='#d05d47')
    axs[0].set(xticks=x,xticklabels=cats,ylabel='True rejection-weighted top-3 coverage',title='Real proxy: condition on entropy difference')
    axs[0].legend()
    pool=secondary['thermal_feasible_rows']
    axs[1].bar([0,1,2],[pool['lost_mass'],pool['rankfixed_lost'],pool['thermal_lost']],color=['#3268ad','#9475b4','#8caf83'])
    axs[1].set(xticks=[0,1,2],xticklabels=['Real proxy','Rank repaired','Temperature\nsurrogate'],ylabel='Erased true residual mass',title='Same entropy: two oracle controls')
    vals=[secondary['step::ds12']['mean'],secondary['step::ds12_add3_residual']['mean'],secondary['step::ds12_add3_proxy']['mean'],secondary['step::ps_only15_residual']['mean'],secondary['step::ps_only15_proxy']['mean']]
    axs[2].barh(['DS12','DS12 + residual3','DS12 + proxy3','Residual15','Proxy15'],vals,color=['#999999','#3268ad','#d05d47','#8aa8cc','#dea294'])
    axs[2].set(xlim=(0,1),xlabel='Recovery coverage',title='Fixed q-head DS; equal total budget 15')
    axs[2].invert_yaxis()
    for ext in ('png','pdf'):fig.savefig(out/f'02_mechanisms.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(totals=result['totals'],selections=selections,
        test={k:dict(value=v['test']['mean'],gain_vs_proxy=v['test_gain_vs_proxy']) for k,v in policies.items()},
        pooled=secondary['pooled'],conditional=conditional,ds_pairs=ds_pairs),indent=2))


if __name__=='__main__':main()
