"""Development selection and independent confirmation, no model fitting."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def read(folder):
    records=[];clusters={};locals_={};steps={};keys=None
    for path in sorted(folder.glob('*.npz')):
        with np.load(path) as f:d={k:f[k] for k in f.files}
        pk=sorted(k for k in d if k.startswith('policy::'))
        if keys is None:keys=pk
        assert keys==pk
        m=np.column_stack([d[k] for k in pk]).astype(np.float64)
        t=float(d['temperature'][0]);records.append(json.loads(path.with_suffix('.json').read_text()))
        for ds in np.unique(d['dataset']):
            for pid in np.unique(d['prompt_id'][d['dataset']==ds]):
                mask=(d['dataset']==ds)&(d['prompt_id']==pid)
                clusters.setdefault((t,str(ds),int(pid)),[]).append(m[mask].mean(0))
        if t not in locals_:locals_[t]={};steps[t]=[]
        for k,v in d.items():
            if k.startswith(('local::','row::','exchange::','oracle::','control::legacy_temperature')):
                locals_[t].setdefault(k,[]).append(v)
        steps[t].append(len(m))
    data={}
    for t in sorted(set(k[0] for k in clusters)):
        data[t]={ds:np.stack([np.mean(v,axis=0) for (tt,dd,pid),v in sorted(clusters.items()) if tt==t and dd==ds])
                 for ds in sorted(set(k[1] for k in clusters if k[0]==t))}
    locals_={t:{k:np.concatenate(v) for k,v in dic.items()} for t,dic in locals_.items()}
    return keys,data,records,locals_,steps


def mean(clusters):return np.mean([a.mean(0) for a in clusters.values()],axis=0)


def ci(clusters,indices,reference=None):
    # Matrix bootstrap only for prespecified endpoints, all tested means in CSV.
    rng=np.random.default_rng(20260913);rep=[];original=[]
    for a in clusters.values():
        x=a[:,indices]
        if reference is not None:x=x-a[:,reference,None]
        original.append(x.mean(0))
        idx=rng.integers(len(x),size=(4000,len(x)))
        rep.append(x[idx].mean(1))
    point=np.mean(original,axis=0);lo,hi=np.quantile(np.mean(rep,axis=0),[.025,.975],axis=0)
    return [{'mean':float(p),'ci95':[float(l),float(h)]} for p,l,h in zip(point,lo,hi)]


def family(key):
    source=key.split('::')[1].split('__')[0]
    if source=='proxy':return 'proxy_allocation'
    if source=='residual':return 'residual_allocation'
    if source.startswith(('keep','interval')):return 'rank_only'
    if source.startswith(('argmax','z_gate','qpeak','epeak')):return 'local_gate'
    if source.startswith('source_tau'):return 'source_temperature'
    return 'bounded_subtraction'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['development','confirmation']);args=ap.parse_args()
    keys,data,records,local,steps=read(HERE/args.stage)
    assert set(data)=={.5,.7,1.},set(data)
    assert len(records)==(14 if args.stage=='development' else 3),len(records)
    base=keys.index('policy::proxy__topm__original');res=keys.index('policy::residual__topm__original')
    frozen=HERE/'frozen.json'
    if args.stage=='development':
        assert not frozen.exists(),'Refusing to overwrite frozen selection'
        selection={}
        for t,clusters in data.items():
            avg=mean(clusters)
            chosen={name:keys[max((i for i,k in enumerate(keys) if family(k)==name),key=lambda i:avg[i])]
                    for name in sorted(set(family(k) for k in keys))}
            chosen['overall']=keys[int(avg.argmax())]
            chosen['source_only']=keys[max((i for i,k in enumerate(keys) if k.endswith('__topm__original')),key=lambda i:avg[i])]
            chosen['original_proxy']=keys[base];chosen['original_residual']=keys[res]
            selection[str(t)]=chosen
        config={'created_utc':datetime.now(timezone.utc).isoformat(),'selection':selection,
                'primary':['overall','proxy_allocation','source_only'],'policy_count':len(keys),
                'replay_sha256':hashlib.sha256((HERE/'replay.py').read_bytes()).hexdigest(),
                'development_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'development').glob('*.npz')},
                'note':'No model fitting. Scalar/rule choices selected solely on development. Confirmation labels unseen at freeze.'}
        frozen.write_text(json.dumps(config,indent=2))
    else:config=json.loads(frozen.read_text())
    assert config['replay_sha256']==hashlib.sha256((HERE/'replay.py').read_bytes()).hexdigest()
    output={};rows=[]
    for t,clusters in data.items():
        avg=mean(clusters);sel=config['selection'][str(t)]
        names=list(sel);indices=[keys.index(sel[n]) for n in names]
        optim=keys.index(sel['proxy_allocation'])
        absolute=ci(clusters,indices);diff=ci(clusters,indices,base);fair=ci(clusters,indices,optim)
        selected={n:{'policy':sel[n],'coverage':a,'vs_original_proxy':b,'vs_optimized_proxy':c,
                     'dataset_means':{ds:float(v[:,i].mean()) for ds,v in clusters.items()}}
                  for n,i,a,b,c in zip(names,indices,absolute,diff,fair)}
        for i,k in enumerate(keys):rows.append({'T':t,'policy':k,'family':family(k),'coverage':avg[i],'gain_vs_proxy':avg[i]-avg[base]})
        ld=local[t];h=ld['row::h'][:,:4].astype(float)
        pooled={k:float(np.sum(v[:,:4]*h)/h.sum()) for k,v in ld.items() if v.ndim==2}
        pooled.update({k:float(v.mean()) for k,v in ld.items() if v.ndim==1})
        gamma=ld['exchange::gamma'][:,:4];err=ld['exchange::error_in'][:,:4]-ld['exchange::error_out'][:,:4]
        z=ld['row::z'][:,:4];gap=ld['local::residual'][:,:4]-ld['local::proxy'][:,:4]
        nz=z>1e-7
        assert np.max(abs((gamma-err)[nz]/z[nz]-gap[nz]))<2e-4
        output[str(t)]={'selected':selected,'development_or_confirmation':args.stage,
                        'unique_prompts':sum(len(v) for v in clusters.values()),'sampled_steps':sum(steps[t]),
                        'pooled_rejection_event_diagnostics':pooled,
                        'original_residual_local_win_event_fraction':float(np.sum(h*(gap>1e-6))/h.sum()),
                        'original_residual_local_loss_event_fraction':float(np.sum(h*(gap< -1e-6))/h.sum()),
                        'top_exploratory_means':[{'policy':keys[i],'coverage':float(avg[i]),'gain':float(avg[i]-avg[base])} for i in np.argsort(avg)[-15:][::-1]]}
    (HERE/(args.stage+'_summary.json')).write_text(json.dumps({'temperatures':output,'records':records,
        'frozen_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
        'weighting':'equal datasets, prompts, then seeds per prompt; 4000 paired prompt bootstrap; diagnostic rows separately h-weighted'},indent=2))
    with (HERE/(args.stage+'_policies.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for t,r in output.items():
        print('T',t,'prompts',r['unique_prompts'])
        for name,s in r['selected'].items():print(name,s['policy'],s['coverage'],s['vs_original_proxy'],s['vs_optimized_proxy'])


if __name__=='__main__':main()
