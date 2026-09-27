"""Check actual 17-slot wire / 15-root prefix against replay on recorded rows."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from runtime_policy import make_policy
import aggregate as agg

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
RAW=ROOT/'ssd/experiments/proxy_source_ablation/probe_training_free_20260912/out/confirmation'
spec=importlib.util.spec_from_file_location('tf',HERE/'replay.py');tf=importlib.util.module_from_spec(spec);spec.loader.exec_module(tf)

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    frozen=json.loads((HERE/'frozen.json').read_text());out=[];summaries={}
    for done in sorted(RAW.glob('*.complete.json')):
        rec=json.loads(done.read_text());t=rec['temperature'];meta=json.loads((RAW/rec['manifest']).read_text())
        chosen=frozen['selection'][str(t)];_,norm,weight=chosen['overall'].removeprefix('policy::').split('__')
        choices={k:chosen[k] for k in ['original_proxy','proxy_allocation','overall']}
        choices['identical_allocation_proxy']=f'policy::proxy__{norm}__{weight}'
        pol=sorted(set(choices.values()));parts={k:[] for k in pol};attributes=[];errors={k:[] for k in pol};maxscore={k:0. for k in pol}
        for chunk in meta['chunks']:
          with np.load(RAW/chunk['file']) as z:
            n=len(z['step'])//5;get=lambda k:torch.from_numpy(z[k].copy()).cuda()
            p,q,e,y=get('p_T').reshape(n,5,-1),get('p_D').reshape(n,5,-1),get('p_E')[:,0].reshape(n,5,-1),get('y').reshape(n,5)
            h=tf.hazard(p,q,y);gt=h[...,None]*tf.normalize((p-q).clamp_min(0))
            z_e=e.clamp_min(1e-38).log()*t;z_q=q[:,:4].clamp_min(1e-38).log()*t
            eq=(z_e/t).softmax(-1);qq=torch.cat([(z_q/t).softmax(-1),torch.zeros_like(q[:,:1])],1)
            hs=tf.pos_weights(eq,qq,y)
            attributes.append(z['seq_id'][::5].copy())
            for key in pol:
                sn,norm,hn=key.removeprefix('policy::').split('__')
                source=dict(tf.sources(eq,qq,y))[sn]
                val,ids=source.topk(17,-1);den=val.sum(-1,keepdim=True) if norm=='topm' else source.sum(-1,keepdim=True)
                cov,_=tf.evaluate(ids,val,hs[hn],gt,den)
                ideal=(hs[hn][...,None]*val/den.clamp_min(1e-10)).flatten(1).topk(15,-1).values.sum(-1)
                fn=make_policy(key,t,None);observed=[]
                for i in range(n):
                    pos,tok,score=fn(z_e[i],z_q[i],y[i,:4],17,17,False)
                    observed.append(gt[i,pos[:15],tok[:15]].sum())
                    err=float(abs(score[:15].sum()-ideal[i]))
                    maxscore[key]=max(maxscore[key],err)
                observed=torch.stack(observed);delta=(observed-cov).abs()
                parts[key].append(observed.cpu().numpy());errors[key].append((observed-cov).cpu().numpy())
        seq=np.concatenate(attributes);values={k:np.concatenate(v) for k,v in parts.items()}
        for key in pol:
            delta=np.concatenate(errors[key]);assert maxscore[key]<2e-6
            out.append({'temperature':t,'policy':key,'contexts':len(seq),
                        'predicted_objective_max_error':maxscore[key],
                        'coverage_max_abs_difference':float(abs(delta).max()),
                        'coverage_mean_signed_difference':float(delta.mean()),
                        'contexts_above_3e_5':int((abs(delta)>3e-5).sum())})
        mapping={x['seq_id']:x for x in rec['mapping']};cl={}
        for sid in sorted(set(seq)):
            cl.setdefault(mapping[int(sid)]['dataset'],[]).append([float(values[k][seq==sid].mean()) for k in pol])
        cl={ds:np.asarray(rows) for ds,rows in cl.items()}
        base=pol.index(choices['original_proxy']);same=pol.index(choices['identical_allocation_proxy'])
        ix=[pol.index(k) for k in choices.values()]
        summaries[str(t)]={name:{'policy':choices[name],'coverage':c,'vs_original_proxy':b,'vs_identical_allocation_proxy':s}
             for name,c,b,s in zip(choices,agg.ci(cl,ix),agg.ci(cl,ix,base),agg.ci(cl,ix,same))}
        print(t,summaries[str(t)]['overall'],flush=True)
    assert len(summaries)==3
    (HERE/'runtime_snapshot_checks.json').write_text(json.dumps({'passed':True,'records':out,'summaries':summaries,
      'scope':'All 3,112 confirmation steps, reconstructed float32 logits. Every runtime 17-slot wire prefix has the optimal predicted 15-root score within 2e-6. Ties can change true coverage; paired results reported rather than claiming token-ID equality. Not bitwise reconstruction of original model logits.'},indent=2))

if __name__=='__main__':main()
