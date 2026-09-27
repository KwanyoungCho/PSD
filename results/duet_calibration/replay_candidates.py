"""Training-free scalar selection from full-vocabulary snapshots (CPU only)."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from campaign import HERE


def hz(alpha):
    reach=torch.cat([torch.ones_like(alpha[:,:1]),alpha.cumprod(1)],1)
    return torch.cat([reach[:,:-1]*(1-alpha),reach[:,-1:]],1)


def evaluate(manifest,n_prompts):
    started=time.perf_counter();meta=json.loads(manifest.read_text());acc={};counts=np.zeros(n_prompts,dtype=int);samples=0
    torch.set_num_threads(1)
    M=32;budgets=[8,15,24];omegas=[0,.25,.5,.75,1]
    for chunk in meta['chunks']:
        with np.load(manifest.parent/chunk['file']) as z:
            arrays={k:z[k] for k in ['p_T','p_D','p_E','y','step','seq_id']}
        grouped={}
        for sid in np.unique(arrays['step']):
            mask=arrays['step']==sid;ids=arrays['seq_id'][mask]
            if int(ids[0])==1:continue  # Excluded warmup generate().
            if len(set(ids))!=1 or not 2<=int(ids[0])<=n_prompts+1:raise ValueError('Unexpected prompt IDs')
            grouped.setdefault(int(mask.sum()),[]).append((int(ids[0])-2,{k:arrays[k][mask] for k in ['p_T','p_D','p_E','y']}))
        for rows,group in grouped.items():
            prompt=np.array([idx for idx,z in group]);np.add.at(counts,prompt,1);samples+=len(group)
            tensors={key:torch.from_numpy(np.stack([z[key] for idx,z in group])) for key in ['p_T','p_D','p_E','y']}
            p,q,es,y=(tensors[k] for k in ['p_T','p_D','p_E','y']);K=rows-1;V=p.shape[-1]
            for arr in [p,es]:
                if not torch.isfinite(arr).all() or (arr<0).any() or (arr.sum(-1)-1).abs().max()>1e-5:raise ValueError('Invalid distribution')
            qy=q[:,:K].gather(-1,y[:,:K,None]).squeeze(-1)
            if (qy<=0).any():raise ValueError('Sampled draft token has zero probability')
            true_h=hz((p[:,:K].gather(-1,y[:,:K,None]).squeeze(-1)/qy).clamp(max=1))
            a=(p-q).clamp_min(0);a[:,-1]=p[:,-1]
            r=a/a.sum(-1,keepdim=True).clamp_min(1e-30)
            gt=(true_h[...,None]*r).double().reshape(len(group),-1)
            for li,layer in enumerate(meta['layers']):
                e=es[:,:,li,:]
                hhat=hz((e[:,:K].gather(-1,y[:,:K,None]).squeeze(-1)/qy).clamp(max=1))
                hbar=hz(torch.minimum(e[:,:K],q[:,:K]).sum(-1).clamp(0,1))
                sources={'proxy':e.clone(),'residual':(e-q).clamp_min(0),
                         'smooth0.5':e*(1-q).clamp_min(0).sqrt(),'smooth1':e*(1-q),
                         'floor0.75':torch.maximum((e-q).clamp_min(0),.75*e)}
                for source,s in sources.items():
                    s[:,:K].scatter_(-1,y[:,:K,None],0);s[:,-1]=e[:,-1]
                    values,tokens=s.topk(M,dim=-1)
                    for norm,denom in [('topm',values.sum(-1,keepdim=True)),('full',s.sum(-1,keepdim=True))]:
                        probs=values/denom.clamp_min(1e-30)
                        for omega in omegas:
                            h=(1-omega)*hhat+omega*hbar
                            order=(h[...,None]*probs).reshape(len(group),-1).topk(max(budgets),dim=-1).indices
                            token=tokens.reshape(len(group),-1).gather(-1,order)
                            gains=gt.gather(-1,(order//M)*V+token).cumsum(-1)
                            for B in budgets:
                                vals=gains[:,B-1].numpy()
                                if (vals<0).any() or (vals>1+1e-5).any():raise ValueError('Invalid coverage')
                                key=f'L{layer}|B{B}|{source}|{norm}|w{omega:g}'
                                if key not in acc:acc[key]=np.zeros(n_prompts)
                                np.add.at(acc[key],prompt,vals)
    if not (counts>0).all():raise ValueError('Missing prompt snapshots')
    values={key:(v/counts).tolist() for key,v in acc.items()}
    return dict(meta=meta,counts=counts.tolist(),n_samples=samples,n_policies=len(acc),seconds=time.perf_counter()-started,
                prompt_coverage=values,mean_coverage={key:float(np.mean(v)) for key,v in values.items()},
                objective='Pre-P1-dedup theoretical root coverage; M32 and top-B, not actual DUET wire/cache hit or TPS.')


def evaluate_campaign():
    plan=json.loads((HERE/'plan.json').read_text());root=HERE/'runs'
    cal=evaluate(root/'probe_calibration/chain_e56_k4_2_w15_s4913/distributions.json',len(plan['datasets']['calibration']['mapping']))
    (HERE/'candidate_calibration.json').write_text(json.dumps(cal,indent=2))
    selected={};selected_proxy={}
    for layer in cal['meta']['layers']:
        keys=[k for k in cal['mean_coverage'] if k.startswith(f'L{layer}|B15|')]
        selected[str(layer)]=max(keys,key=cal['mean_coverage'].get)
        selected_proxy[str(layer)]=max([k for k in keys if '|proxy|' in k],key=cal['mean_coverage'].get)
    frozen=HERE/'candidate_frozen.json'
    if frozen.exists():raise ValueError('Refuse candidate reselection')
    frozen.write_text(json.dumps(dict(selection=selected,selection_proxy_only=selected_proxy,
        selected_before_validation_scoring=True,calibration_seconds=cal['seconds']),indent=2))
    val=evaluate(root/'probe_validation/chain_e56_k4_2_w15_s5913/distributions.json',len(plan['datasets']['validation']['mapping']))
    (HERE/'candidate_validation.json').write_text(json.dumps(val,indent=2))
    results=[]
    datasets=[m['dataset'] for m in plan['datasets']['validation']['mapping']]
    groups=[np.flatnonzero(np.array(datasets)==name) for name in dict.fromkeys(datasets)]
    rng=np.random.default_rng(913)
    boot=np.array([np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(4000)])
    for layer,key in selected.items():
        baseline=f'L{layer}|B15|proxy|topm|w0'
        proxy=selected_proxy[layer]
        difference=100*(np.array(val['prompt_coverage'][key])-np.array(val['prompt_coverage'][baseline]))
        source_difference=100*(np.array(val['prompt_coverage'][key])-np.array(val['prompt_coverage'][proxy]))
        results.append(dict(layer=int(layer),selected=key,calibration=cal['mean_coverage'][key],
                            validation=val['mean_coverage'][key],proxy_baseline=val['mean_coverage'][baseline],
                            calibrated_proxy_policy=proxy,calibrated_proxy=val['mean_coverage'][proxy],
                            gain_vs_baseline_pp=float(difference.mean()),gain_vs_baseline_ci95_pp=np.quantile(difference[boot].mean(1),[.025,.975]).tolist(),
                            gain_vs_calibrated_proxy_pp=float(source_difference.mean()),
                            gain_vs_calibrated_proxy_ci95_pp=np.quantile(source_difference[boot].mean(1),[.025,.975]).tolist()))
    (HERE/'candidate_summary.json').write_text(json.dumps(dict(results=results,calibration_seconds=cal['seconds'],validation_seconds=val['seconds'],
                                                            configs_per_step=cal['n_policies'],scope=cal['objective']),indent=2))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':evaluate_campaign()
