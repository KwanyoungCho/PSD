"""Fit a continuous global temperature by convex KL optimization on TRAIN only.

The fixed grid is kept as the original ablation. This refinement avoids
rounding the KL optimum to a coarse grid before evaluating held-out coverage.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch

from replay import entropy, hazard, normalize, score, source


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    device='cuda:0'
    base=Path('ssd/experiments/proxy_source_ablation/probe_replay_20260910/out')
    out=Path(__file__).parent
    done=sorted(base.glob('*.complete.json'))
    if len(done)!=8:raise ValueError('requires eight completed runs')
    # At most ~1 GB of training log-probabilities retained on the free GPU.
    entries=[];total_target_loge=0.;total_target_logp=0.;total_weight=0.
    started=time.time()
    for path in done:
        record=json.loads(path.read_text());m=json.loads((base/record['manifest']).read_text())
        counts={int(k):(int(v)+m['stride']-1)//m['stride'] for k,v in m['seq_steps'].items()}
        for chunk in m['chunks']:
            with np.load(base/chunk['file']) as z:
                mask=z['seq_id']<=16
                if not mask.any():continue
                w=torch.tensor([1/(4*16*2*5*counts[int(sid)]) for sid in z['seq_id'][mask]],device=device,dtype=torch.float64)
                le=torch.from_numpy(z['p_E'][mask,0].copy()).to(device).clamp_min(1e-38).log()
                p=torch.from_numpy(z['p_T'][mask].copy()).to(device)
                total_target_loge+=float(((p*le).sum(1).double()*w).sum())
                total_target_logp+=float(((p*p.clamp_min(1e-38).log()).sum(1).double()*w).sum())
                total_weight+=float(w.sum())
                entries.append((le,w))
    assert abs(total_weight-1)<1e-10
    def objective(beta):
        value=total_target_logp-beta*total_target_loge
        grad=-total_target_loge;curvature=0.
        for le,w in entries:
            logits=beta*le
            v=logits.softmax(1)
            mean=(v*le).sum(1)
            var=(v*(le-mean[:,None]).square()).sum(1)
            value+=float((torch.logsumexp(logits,1).double()*w).sum())
            grad+=float((mean.double()*w).sum())
            curvature+=float((var.double()*w).sum())
        return value,grad,curvature
    beta=1.;history=[]
    for iteration in range(20):
        value,g,h=objective(beta)
        history.append(dict(iteration=iteration,beta=beta,tau=1/beta,kl=value,gradient=g,curvature=h))
        if abs(g)<2e-6:break
        proposal=max(.05,beta-g/h)
        # Convex line search, using only training data.
        for _ in range(20):
            if objective(proposal)[0]<=value+1e-9:break
            proposal=(proposal+beta)/2
        beta=proposal
    if abs(history[-1]['gradient'])>=2e-5:raise RuntimeError('KL optimization did not converge')
    tau=1/beta
    print(f'Continuous TRAIN optimum tau={tau:.8f}; KL={history[-1]["kl"]:.6f}; gradient={history[-1]["gradient"]:.3g}',flush=True)
    del entries,le,p,w
    torch.cuda.empty_cache()
    parts=[]
    # Evaluate frozen tau on both splits. Held-out information was not used
    # by the optimization above.
    for path in done:
        record=json.loads(path.read_text());m=json.loads((base/record['manifest']).read_text())
        for chunk in m['chunks']:
            with np.load(base/chunk['file']) as z:
                n=len(z['step'])//5
                p=torch.from_numpy(z['p_T'].copy()).to(device).reshape(n,5,-1)
                e=torch.from_numpy(z['p_E'][:,0].copy()).to(device).reshape(n,5,-1)
                q=torch.from_numpy(z['p_D'].copy()).to(device).reshape(n,5,-1)
                y=torch.from_numpy(z['y'].copy()).to(device).reshape(n,5)
                c=(e.clamp_min(1e-38).log()*beta).softmax(-1)
                h=hazard(p,q,y);hh=hazard(e,q,y);hc=hazard(c,q,y)
                gt=h[...,None]*normalize((p-q).clamp_min(0))
                r,pr=source(c,q,y,1.),source(c,q,y,0.)
                data=dict(residual=score(r,hc,gt),proxy=score(pr,hc,gt),
                    residual_fixed_h=score(r,hh,gt),residual_h_only=score(source(e,q,y,1.),hc,gt),
                    kl=(p*(p.clamp_min(1e-38).log()-c.clamp_min(1e-38).log())).sum(-1).mean(1),
                    tvd=(.5*(p-c).abs().sum(-1)).mean(1),
                    entropy_delta=(entropy(c)-entropy(p)).mean(1))
                a={k:v.cpu().numpy() for k,v in data.items()}
                a.update(seq_id=z['seq_id'][::5],step=z['step'][::5],
                    dataset=np.full(n,record['dataset']),seed=np.full(n,record['seed']))
                parts.append(a)
    arrays={k:np.concatenate([a[k] for a in parts]) for k in parts[0]}
    np.savez_compressed(out/'exact_calibration.npz',**arrays)
    (out/'exact_calibration.json').write_text(json.dumps(dict(tau=tau,beta=beta,
        history=history,weight_sum=total_weight,seconds=time.time()-started,
        objective='equal dataset / training prompt / seed; mean over sampled step and K+1 context',
        split='train prompt 1..16 only; fixed before held-out evaluation'),indent=2))


if __name__=='__main__':main()
