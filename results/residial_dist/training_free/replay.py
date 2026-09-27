"""Training-free policies on fixed, full-precision probability snapshots.

No policy receives p or true h. Those are used exclusively by evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import importlib.util
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
_spec=importlib.util.spec_from_file_location('shared_policy_replay',HERE.parent/'shared_review/replay.py')
_shared=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_shared)
hazard,normalize,source,entropy=_shared.hazard,_shared.normalize,_shared.source,_shared.entropy


def pos_weights(e,q,y):
    h=hazard(e,q,y);out={'original':h}
    k=q.shape[1]-1
    def hz(alpha):
        reach=torch.cat([torch.ones_like(alpha[:,:1]),alpha.cumprod(1)],1)
        v=reach.clone();v[:,:-1]*=1-alpha;return v
    for power in (.5,.75,1.25,1.5):out[f'power{power:g}']=normalize(h.pow(power))
    for mix in (.02,.05,.1,.2):out[f'uniform{mix:g}']=(1-mix)*h+mix/h.shape[1]
    for boost in (1.25,1.5,2.,3.):
        v=h.clone();v[:,-1]*=boost;out[f'bonus{boost:g}']=normalize(v)
    overlap=torch.minimum(e[:,:-1],q[:,:-1]).sum(-1).clamp(0,1)
    expected=hz(overlap);out['expected_rejection']=expected
    for mix in (.25,.5,.75):out[f'expected_mix{mix:g}']=(1-mix)*h+mix*expected
    alpha=(e[:,:-1].gather(-1,y[:,:-1,None]).squeeze(-1)/(q[:,:-1].gather(-1,y[:,:-1,None]).squeeze(-1)+1e-10)).clamp(max=1)
    for floor in (.02,.05,.1):out[f'accept_clip{floor:g}']=hz(alpha.clamp(floor,1-floor))
    for tau in (.8,.9,1.1,1.25):out[f'h_tau{tau:g}']=hazard((e.clamp_min(1e-38).log()/tau).softmax(-1),q,y)
    return out


def mask_y(s,y):
    s=s.clone();s[:,:-1].scatter_(-1,y[:,:-1,None],0);return s


def sources(e,q,y):
    pe,re=source(e,q,y,0.),source(e,q,y,1.)
    yield 'proxy',pe
    yield 'residual',re
    for lam in (.1,.25,.5,.75):yield f'lambda{lam:g}',source(e,q,y,lam)
    for floor in (.25,.5,.75,.9):yield f'floor{floor:g}',torch.maximum(re,floor*pe)
    negative=mask_y((q-e).clamp_min(0),y)
    for beta in (.1,.25,.5,1.):
        yield f'twosided{beta:g}',re+beta*negative
        yield f'capped_negative{beta:g}',re+beta*torch.minimum(negative,pe)
    for beta in (.5,1.,2.,4.):yield f'complement_power{beta:g}',mask_y(e*(1-q).clamp_min(0).pow(beta),y)
    uncertainty=mask_y(torch.sqrt(e*(1-e).clamp_min(0)),y)
    uncertainty[:,-1]=0
    for bonus in (.01,.03,.1):yield f'uncertainty{bonus:g}',re+bonus*uncertainty
    ey=e[:,:-1].gather(-1,y[:,:-1,None]);qy=q[:,:-1].gather(-1,y[:,:-1,None])
    for name,lam in [('anchor_y',(ey/(qy+1e-10)).clamp(max=1)),
                     ('anchor_peak',(e[:,:-1].amax(-1,keepdim=True)/(q[:,:-1].amax(-1,keepdim=True)+1e-10)).clamp(max=1)),
                     ('complement',((1-qy)/(1-ey+1e-10)).clamp(0,1))]:
        lam=torch.cat([lam,torch.zeros_like(lam[:,:1])],1)
        yield name,mask_y((e-lam*q).clamp_min(0),y)
    diff=e.argmax(-1)!=q.argmax(-1)
    for name,gate in [('argmax_diff',diff),('argmax_same',~diff)]:
        yield name,torch.where(gate[...,None],re,pe)
    z=.5*(e-q).abs().sum(-1)
    for threshold in (.2,.4,.6):
        yield f'z_gate{threshold:g}',torch.where((z>=threshold)[...,None],re,pe)
    for threshold in (.5,.7,.9):
        for name,gate in [('qpeak',q.amax(-1)>=threshold),('epeak',e.amax(-1)>=threshold)]:
            yield f'{name}{threshold:g}',torch.where(gate[...,None],re,pe)
    for tau in (.8,.9,1.1):
        cal=(e.clamp_min(1e-38).log()/tau).softmax(-1)
        yield f'source_tau{tau:g}',source(cal,q,y,1.)


def rerankings(e,q,y,pe,re):
    """Preserve proxy score multiset, alter token identities only.

    Interval policy swaps two ranks, rather than moving a token and shifting
    intermediate ranks. Under valid bounds, each certified swap improves
    every local prefix sum. Global ties may select non-prefix slots, so the
    theorem is stated for prefix-consistent selection separately from replay.
    """
    pv,pi=pe.topk(17,-1)
    ri=re.topk(64,-1).indices
    for keep in (1,2,3,5,8,12):
        head=pi[...,:keep]
        overlap=(ri[...,None]==head[...,None,:]).any(-1)
        order=overlap.int().argsort(dim=-1,stable=True)
        tail=ri.gather(-1,order)[...,:17-keep]
        yield f'keep{keep}',torch.cat([head,tail],-1)
    ei=pe.topk(64,-1).indices
    # Put the exact baseline top17 first. Stable sorting then preserves its
    # tie order; a no-op rule must not win merely by a different tie choice.
    pool=torch.cat([pi,ei,ri],-1)
    width=pool.shape[-1]
    same=pool[...,None,:]==pool[...,:,None]
    duplicate=(same & torch.ones(width,width,device=e.device,dtype=torch.bool).tril(-1)).any(-1)
    ev=pe.gather(-1,pool).masked_fill(duplicate,-torch.inf)
    order=ev.argsort(dim=-1,descending=True,stable=True)
    pool=pool.gather(-1,order)
    valid=(~duplicate).gather(-1,order)
    ee=e.gather(-1,pool);qq=q.gather(-1,pool)
    for radius in (0.,.05,.1,.25,.5,.75,1.):
        lo=((1-radius)*ee-qq).clamp_min(0).masked_fill(~valid,-torch.inf)
        hi=((1+radius)*ee-qq).clamp_min(0)
        perm=torch.arange(width,device=e.device).expand_as(pool).clone()
        for j in range(17):
            remaining=perm[...,j:]
            low,where=lo.gather(-1,remaining).max(-1)
            current=perm[...,j]
            certified=low>hi.gather(-1,current[...,None]).squeeze(-1)+1e-9
            pos=torch.where(certified,where+j,torch.full_like(where,j))
            selected=perm.gather(-1,pos[...,None]).squeeze(-1)
            perm.scatter_(-1,pos[...,None],current[...,None])
            perm[...,j]=selected
        yield f'interval{radius:g}',pool.gather(-1,perm[...,:17])


def evaluate(ids,values,h,gt,denom):
    scores=h[...,None]*values/denom.clamp_min(1e-10)
    chosen=scores.flatten(1).topk(15,-1).indices
    coverage=gt.gather(-1,ids).flatten(1).gather(-1,chosen).sum(-1)
    return coverage,chosen


@torch.inference_mode()
def measure(p,q,e,y,temperature=1.):
    h=hazard(p,q,y);hs=pos_weights(e,q,y)
    a=(p-q).clamp_min(0);b=(e-q).clamp_min(0)
    r=normalize(a);gt=h[...,None]*r
    assert (h.sum(-1)-1).abs().max()<2e-6
    out={};cached={}
    for name,s in sources(e,q,y):
        values,ids=s.topk(17,-1)
        cached[name]=(values,ids,s.sum(-1,keepdim=True))
    pe,re=source(e,q,y,0.),source(e,q,y,1.)
    for name,ids in rerankings(e,q,y,pe,re):cached[name]=(cached['proxy'][0],ids,cached['proxy'][2])
    # Full factorial for source vs allocation: sorting is cached, so crossing
    # the small 5x17 tables is cheap and doesn't hide allocation improvements.
    for name,(values,ids,fullmass) in cached.items():
        for norm,denom in [('topm',values.sum(-1,keepdim=True)),('full',fullmass)]:
            for hn,hh in hs.items():
                out[f'policy::{name}__{norm}__{hn}']=evaluate(ids,values,hh,gt,denom)[0]
        out[f'control::{name}__true_h']=evaluate(ids,values,h,gt,values.sum(-1,keepdim=True))[0]
        out[f'local::{name}']=r.gather(-1,ids[...,:3]).sum(-1)
    out['oracle::global']=gt.flatten(1).topk(15,-1).values.sum(-1)
    for name in ['proxy','residual']:
        eligible=torch.zeros_like(gt,dtype=torch.bool)
        eligible.scatter_(-1,cached[name][1],True)
        out[f'oracle::pool_{name}']=gt.masked_fill(~eligible,0).flatten(1).topk(15,-1).values.sum(-1)
    # Current production chain uses unscaled logits in candidate selection,
    # even when its actual target/draft sampling uses T!=1. Reconstruct that
    # legacy source from stored T-probabilities; float32 underflow is audited.
    if temperature==1.:
        legacy_e,legacy_q=e,q
    else:
        legacy_e,legacy_q=normalize(e.pow(temperature)),normalize(q.pow(temperature))
    for name,lam in [('proxy',0.),('residual',1.)]:
        val,ids=source(legacy_e,legacy_q,y,lam).topk(17,-1)
        out[f'control::legacy_temperature_{name}']=evaluate(ids,val,hazard(legacy_e,legacy_q,y),gt,val.sum(-1,keepdim=True))[0]
    out['oracle::switch']=torch.maximum(out['policy::proxy__topm__original'],out['policy::residual__topm__original'])
    out['row::h']=h
    out['row::z']=a.sum(-1)
    out['row::delta']=.5*(p-e).abs().sum(-1)
    out['row::entropy_difference']=entropy(e)-entropy(p)
    out['row::top1_correct']=(p.argmax(-1)==e.argmax(-1)).float()
    # Exact exchange decomposition at k=3. I=Sr\Se, O=Se\Sr.
    ir=cached['residual'][1][...,:3];ie=cached['proxy'][1][...,:3]
    incoming=~(ir[...,None]==ie[...,None,:]).any(-1)
    outgoing=~(ie[...,None]==ir[...,None,:]).any(-1)
    def mass(x,ids,mask):return (x.gather(-1,ids)*mask).sum(-1)
    b=source(e,q,y,1.)
    error=b-a
    gamma=mass(b,ir,incoming)-mass(b,ie,outgoing)
    assert gamma.min()>-2e-6
    er_in=mass(error,ir,incoming);er_out=mass(error,ie,outgoing)
    direct=mass(a,ir,incoming)-mass(a,ie,outgoing)
    torch.testing.assert_close(direct,gamma-(er_in-er_out),atol=2e-6,rtol=0)
    for name,v in dict(gamma=gamma,error_in=er_in,error_out=er_out,true_gain_raw=direct,
                       deleted_out=mass(a,ie,outgoing & (b.gather(-1,ie)==0)),
                       positive_error_in=mass(error.clamp_min(0),ir,incoming),
                       negative_error_out=mass((-error).clamp_min(0),ie,outgoing)).items():out['exchange::'+name]=v
    return {k:v.cpu().numpy() for k,v in out.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',nargs='+',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args();torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for folder in args.input:
        for done in sorted(folder.glob('*.complete.json')):
            rec=json.loads(done.read_text());dest=args.output/(done.name.replace('.complete.json','.npz'))
            if dest.exists():
                assert json.loads(dest.with_suffix('.json').read_text())['analysis_sha256']==sha;continue
            manifest=folder/rec['manifest'];meta=json.loads(manifest.read_text())
            assert meta['layers'][0]==56 and meta['exit_layer']==56
            parts=[];attrs=[]
            for j,ch in enumerate(meta['chunks']):
                with np.load(folder/ch['file']) as d:
                    n=len(d['step'])//5
                    np.testing.assert_array_equal(d['position'],np.tile(np.arange(5),n))
                    get=lambda k:torch.from_numpy(d[k].copy()).to(args.device)
                    p,q,e,y=get('p_T'),get('p_D'),get('p_E'),get('y')
                    parts.append(measure(p.reshape(n,5,-1),q.reshape(n,5,-1),e[:,0].reshape(n,5,-1),y.reshape(n,5),meta['temperature']))
                    attr={k:d[k][::5].copy() for k in ['step','seq_id','local_step']}
                    if 'mapping' in rec:
                        mapping={m['seq_id']:m for m in rec['mapping']}
                        attr['dataset']=np.array([mapping[int(s)]['dataset'] for s in attr['seq_id']])
                        attr['prompt_id']=np.array([mapping[int(s)]['source_row'] for s in attr['seq_id']])
                    else:
                        attr['dataset']=np.full(n,rec['dataset'])
                        offset=32 if folder.name=='out' and 'probe_direct_' in str(folder) else 0
                        attr['prompt_id']=attr['seq_id']+offset
                    attr['seed']=np.full(n,rec['seed']);attr['temperature']=np.full(n,meta['temperature'])
                    attrs.append(attr)
                if j%4==0:print(f'{dest.stem} {j+1}/{len(meta["chunks"])}',flush=True)
            data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
            data.update({k:np.concatenate([p[k] for p in attrs]) for k in attrs[0]})
            previous=HERE.parent/'direct_comparison'/('confirmation' if rec['seed']==7 else 'development')/dest.name
            if previous.exists():
                with np.load(previous) as z:
                    np.testing.assert_array_equal(data['step'],z['step'])
                    for name in ['proxy','residual']:
                        np.testing.assert_allclose(data[f'policy::{name}__topm__original'],z['policy::'+name],atol=3e-5,rtol=0)
            np.savez_compressed(dest,**data)
            dest.with_suffix('.json').write_text(json.dumps({'record':rec,'analysis_sha256':sha,'manifest':str(manifest),
                'policies':sum(k.startswith('policy::') for k in data),'topm':17,'budget':15},indent=2))
            print(f'finished {dest.stem}: {len(data["step"])} steps, {len(data)} metrics',flush=True)


if __name__=='__main__':main()
