"""Prespecified-policy ablations and multiplicity-aware confirmation summaries."""
import csv
import json
from pathlib import Path
import numpy as np
import aggregate as agg

HERE=Path(__file__).resolve().parent

def bootstrap(clusters,indices,reference,reps=20000):
    rng=np.random.default_rng(917);draw=[];point=[]
    for a in clusters.values():
        x=a[:,indices]-a[:,reference,None]
        ix=rng.integers(len(x),size=(reps,len(x)))
        draw.append(x[ix].mean(1));point.append(x.mean(0))
    v=np.mean(draw,0);mu=np.mean(point,0)
    # Six primary contrasts: selected vs baseline/optimized-proxy, 3 T.
    bounds=np.quantile(v,[.05/12,1-.05/12],axis=0)
    return [{'mean':float(m),'ci_bonferroni_six':[float(lo),float(hi)]}
            for m,lo,hi in zip(mu,*bounds)]

def main():
    keys,data,records,local,steps=agg.read(HERE/'confirmation')
    assert len(records)==3
    # Put reconstructed legacy-temperature controls on the same prompt weights.
    legacy=['control::legacy_temperature_proxy','control::legacy_temperature_residual']
    keys=keys+legacy
    for path in (HERE/'confirmation').glob('*.npz'):
        with np.load(path) as z:
            t=float(z['temperature'][0]);ds=z['dataset'];pid=z['prompt_id']
            vals=np.column_stack([z[k] for k in legacy])
            for name,a in data[t].items():
                extra=np.stack([vals[(ds==name)&(pid==p)].mean(0) for p in np.unique(pid[ds==name])])
                data[t][name]=np.column_stack([a,extra])
    frozen=json.loads((HERE/'frozen.json').read_text());out={};rows=[]
    for t,cl in data.items():
        choice=frozen['selection'][str(t)];key=choice['overall']
        src,norm,hn=key.removeprefix('policy::').split('__')
        base=keys.index(choice['original_proxy']);bestproxy=keys.index(choice['proxy_allocation'])
        same=keys.index(f'policy::proxy__{norm}__{hn}')
        names={
          'original_proxy':choice['original_proxy'],
          'original_residual':choice['original_residual'],
          'full_only':'policy::proxy__full__original',
          'hazard_only':f'policy::proxy__topm__{hn}',
          'allocation_only':f'policy::proxy__{norm}__{hn}',
          'source_only':f'policy::{src}__topm__original',
          'source_plus_full':f'policy::{src}__full__original',
          'source_plus_hazard':f'policy::{src}__topm__{hn}',
          'selected':key,'optimized_proxy':choice['proxy_allocation'],
          'optimized_residual':choice['residual_allocation'],
          'legacy_unscaled_proxy':legacy[0],'legacy_unscaled_residual':legacy[1]}
        ix=[keys.index(k) for k in names.values()]
        values=agg.ci(cl,ix);vsbase=agg.ci(cl,ix,base);vssame=agg.ci(cl,ix,same)
        ablation={n:{'key':names[n],'coverage':val,'vs_base':dif,'vs_identical_proxy_allocation':fair}
                  for n,val,dif,fair in zip(names,values,vsbase,vssame)}
        for n,r in ablation.items():
            rows.append({'T':t,'ablation':n,'coverage_pct':100*r['coverage']['mean'],
                         'gain_pp':100*r['vs_base']['mean'],
                         'lo_pp':100*r['vs_base']['ci95'][0],'hi_pp':100*r['vs_base']['ci95'][1]})
        # Diagnostic quantities use rejection-event weights, unlike main tables.
        d=local[t];h=d['row::h'][:,:4].astype(float);z=d['row::z'][:,:4]
        use=z>1e-7;w=h*use;w/=w.sum()
        get=lambda k:d[k][:,:4].astype(float)
        g=get('exchange::gamma');ein=get('exchange::error_in');eout=get('exchange::error_out')
        def weighted_ratio(x):return float((w*np.divide(x,z,out=np.zeros_like(x),where=use)).sum())
        sign={'gain_proxy_raw_over_Z':weighted_ratio(g),'incoming_error_over_Z':weighted_ratio(ein),
              'outgoing_error_over_Z':weighted_ratio(eout),'total_error_over_Z':weighted_ratio(ein-eout),
              'true_gain_over_Z':weighted_ratio(get('exchange::true_gain_raw')),
              'deleted_out_over_Z':weighted_ratio(get('exchange::deleted_out'))}
        gap=get('local::residual')-get('local::proxy');win=gap>1e-6;loss=gap< -1e-6
        sign['win_event_fraction']=float((w*win).sum());sign['loss_event_fraction']=float((w*loss).sum())
        sign['mean_gain_on_winning_events']=float((w*win*gap).sum()/(w*win).sum())
        sign['mean_gain_on_losing_events']=float((w*loss*gap).sum()/(w*loss).sum())
        assert abs(sign['true_gain_over_Z']-(sign['gain_proxy_raw_over_Z']-sign['total_error_over_Z']))<1e-6
        out[str(t)]={'ablation':ablation,
             'primary_selected_vs_original':bootstrap(cl,[keys.index(key)],base)[0],
             'primary_selected_vs_optimized_proxy':bootstrap(cl,[keys.index(key)],bestproxy)[0],
             'signed_exchange_top3_rejection_weighted':sign,
             'method':'20,000 paired prompt bootstrap; six primary contrasts use 99.1667% Bonferroni intervals; pointwise 95% ablation intervals are secondary.'}
    (HERE/'diagnostics.json').write_text(json.dumps(out,indent=2))
    with (HERE/'ablations.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print(json.dumps({t:{k:v for k,v in r.items() if k.startswith('primary')} for t,r in out.items()},indent=2))

if __name__=='__main__':main()
