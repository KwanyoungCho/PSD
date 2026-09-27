"""Secondary budget sensitivity of the already frozen rules; no new selection."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
RAW=ROOT/'ssd/experiments/proxy_source_ablation/probe_training_free_20260912/out/confirmation'
spec=importlib.util.spec_from_file_location('tf',HERE/'replay.py')
tf=importlib.util.module_from_spec(spec);spec.loader.exec_module(tf)

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    frozen=json.loads((HERE/'frozen.json').read_text());result={}
    for done in sorted(RAW.glob('*.complete.json')):
        record=json.loads(done.read_text());t=record['temperature'];meta=json.loads((RAW/record['manifest']).read_text())
        choices={k:frozen['selection'][str(t)][k] for k in ['original_proxy','original_residual','proxy_allocation','overall']}
        _,norm,weight=choices['overall'].removeprefix('policy::').split('__')
        choices['identical_allocation_proxy']=f'policy::proxy__{norm}__{weight}'
        parts={};attrs=[]
        for chunk in meta['chunks']:
            with np.load(RAW/chunk['file']) as d:
                n=len(d['step'])//5
                tensor=lambda k:torch.from_numpy(d[k]).cuda()
                p,q,e,y=tensor('p_T').reshape(n,5,-1),tensor('p_D').reshape(n,5,-1),tensor('p_E')[:,0].reshape(n,5,-1),tensor('y').reshape(n,5)
                attrs.append(d['seq_id'][::5]);h=tf.hazard(p,q,y);gt=h[...,None]*tf.normalize((p-q).clamp_min(0))
                weights=tf.pos_weights(e,q,y)
                needed={k.split('::')[1].split('__')[0] for k in choices.values()};source={}
                for name,s in tf.sources(e,q,y):
                    if name in needed:source[name]=s
                for name,key in choices.items():
                    sn,norm,hn=key.removeprefix('policy::').split('__');s=source[sn]
                    for budget in [5,10,15,25,40]:
                        m=max(17,budget);val,ids=s.topk(m,-1)
                        den=val.sum(-1,keepdim=True) if norm=='topm' else s.sum(-1,keepdim=True)
                        pred=weights[hn][...,None]*val/den.clamp_min(1e-10)
                        chosen=pred.flatten(1).topk(budget,-1).indices
                        value=gt.gather(-1,ids).flatten(1).gather(-1,chosen).sum(-1)
                        parts.setdefault(f'{name}:B{budget}',[]).append(value.cpu().numpy())
                # Probability mass of actual rejection events assigned zero estimated h.
                missed=(h[:,:4]*(weights['original'][:,:4]<1e-9)).sum(-1)
                parts.setdefault('zero_proxy_h_rejection_mass',[]).append(missed.cpu().numpy())
        ids=np.concatenate(attrs);values={k:np.concatenate(v) for k,v in parts.items()}
        mapping={x['seq_id']:x for x in record['mapping']}
        tables={}
        for key,a in values.items():
            by_ds={}
            for sid in sorted(set(ids)):
                by_ds.setdefault(mapping[int(sid)]['dataset'],[]).append(float(a[ids==sid].mean()))
            tables[key]={'mean':float(np.mean([np.mean(v) for v in by_ds.values()])),
                         'datasets':{ds:float(np.mean(v)) for ds,v in by_ds.items()}}
        result[str(t)]=tables
        print(t,{k:v['mean'] for k,v in tables.items()},flush=True)
    if len(result)!=3:raise RuntimeError('Need all three confirmation runs')
    (HERE/'budget_checks.json').write_text(json.dumps({'secondary':True,'temperatures':result,
       'note':'Frozen sources and allocation rules, same saved trajectories; B=5/10/15/25/40, M=max(17,B). No policy reselection; not throughput measurements.'},indent=2))

if __name__=='__main__':main()
