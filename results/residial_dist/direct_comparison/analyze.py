"""Direct, paired candidate-policy comparisons on immutable probability snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent/'shared_review'))
from replay import hazard, normalize, source, score, entropy

GAMMAS = (0., .1, .25, .5, .75, 1.)


def policy_sources(e, q, y):
    pe, re = source(e, q, y, 0.), source(e, q, y, 1.)
    yield 'proxy', pe, e
    yield 'residual', re, e
    for lam in (.1, .25, .5, .75):
        yield f'lambda_{lam:g}', source(e, q, y, lam), e
    en, rn = normalize(pe), normalize(re)
    for weight in (.1, .25, .5, .75):
        yield f'mix_{weight:g}', (1-weight)*en+weight*rn, e
    ratio = e/(e+q).clamp_min(1e-38)
    for beta in (.25, .5, 1., 2.):
        s = e*ratio.pow(beta)
        s[:, :-1].scatter_(-1, y[:, :-1, None], 0)
        yield f'relative_{beta:g}', s, e
    for tau in (.8, .9, 1.1, 1.25):
        calibrated = (e.clamp_min(1e-38).log()/tau).softmax(-1)
        for name, lam in [('residual', 1.), ('proxy', 0.)]:
            yield f'tau_{tau:g}_{name}', source(calibrated, q, y, lam), calibrated
    calibrated = (e.clamp_min(1e-38).log()/.9).softmax(-1)
    en, rn = normalize(source(calibrated,q,y,0.)), normalize(source(calibrated,q,y,1.))
    for weight in (.25, .5, .75):
        yield f'tau_0.9_mix_{weight:g}', (1-weight)*en+weight*rn, calibrated


def coverage(s, r, k=3):
    ids = s.topk(k, -1).indices
    return r.gather(-1, ids).sum(-1), ids


def fit_uniform_tv(p, e):
    """Exact scalar TV minimizer over c in [0,1], via a weighted median.

    sum |e-u-c(p-u)| = sum |p-u| |(e-u)/(p-u)-c|.
    c depends on the TRUE target: diagnostic only.
    """
    u = 1/p.shape[-1]
    d = p-u
    ratio = torch.where(d != 0, (e-u)/torch.where(d != 0,d,torch.ones_like(d)), torch.zeros_like(d))
    values, order = ratio.sort(-1)
    weights = d.abs().gather(-1, order)
    cdf = weights.cumsum(-1)
    idx = (cdf >= .5*weights.sum(-1, keepdim=True)).int().argmax(-1)
    c = values.gather(-1, idx[...,None]).squeeze(-1).clamp(0,1)
    model = c[...,None]*p+(1-c[...,None])*u
    tv = .5*(e-model).abs().sum(-1)
    best_endpoint = torch.minimum(.5*(e-p).abs().sum(-1), .5*(e-u).abs().sum(-1))
    assert (tv-best_endpoint).max() < 3e-6
    return c, model, tv


@torch.inference_mode()
def measure(p, q, e, y):
    h, hh = hazard(p,q,y), hazard(e,q,y)
    a = (p-q).clamp_min(0)
    b = (e-q).clamp_min(0)
    z, zh = a.sum(-1), b.sum(-1)
    r, rh = normalize(a), normalize(b)
    gt = h[...,None]*r
    pe, re = source(e,q,y,0.), source(e,q,y,1.)
    en, rn = normalize(pe), normalize(re)
    data = {}
    cov = {}
    for name, s, cal_e in policy_sources(e,q,y):
        hc = hh if cal_e is e else hazard(cal_e,q,y)
        cov[name] = score(s,hc,gt)
    for name, s in [('residual',re),('proxy',pe),('oracle_residual',source(p,q,y,1.)),('oracle_target',source(p,q,y,0.))]:
        data[f'global::{name}_true_h'] = score(s,h,gt)
        if name.startswith('oracle'):
            data[f'global::{name}_proxy_h'] = score(s,hh,gt)
    data.update({f'policy::{k}':v for k,v in cov.items()})
    data['global::ceiling'] = gt.flatten(1).topk(15,-1).values.sum(-1)
    data['global::oracle_switch'] = torch.maximum(cov['residual'],cov['proxy'])
    ce, ie = coverage(pe,r)
    cr, ir = coverage(re,r)
    cs, istar = coverage(r,r)
    cp, _ = coverage(source(p,q,y,0.),r)
    # The actual top-k implementation can fill with zero-score tokens.
    # Include those actual selected IDs in the support-constrained oracle so
    # support + ranking regret is exact and nonnegative, even in that case.
    reachable = re > 0
    reachable.scatter_(-1,ir,True)
    support_oracle = r.masked_fill(~reachable,-torch.inf).topk(3,-1).values.sum(-1)
    support_loss, rank_loss = cs-support_oracle, support_oracle-cr
    torch.testing.assert_close(cr-ce, (cs-ce)-support_loss-rank_loss, atol=2e-6, rtol=0)
    assert support_loss.min() > -2e-6 and rank_loss.min() > -2e-6
    delta = .5*(p-e).abs().sum(-1)
    eta = (p-e).abs().amax(-1)
    # Same selected sets, compared under b and a. Certificate follows from
    # |sum_(Sr\Se)(a-b)-sum_(Se\Sr)(a-b)| <= min(2delta,2k*eta).
    gamma_raw = b.gather(-1,ir).sum(-1)-b.gather(-1,ie).sum(-1)
    allowed = (ir[:,:-1] != y[:,:-1,None]).all(-1)
    certified = gamma_raw[:,:-1] > torch.minimum(2*delta[:,:-1],6*eta[:,:-1])+1e-6
    assert not (certified & allowed & (cr[:,:-1] < ce[:,:-1]-2e-5)).any()
    row = dict(h=h, h_proxy=hh, z=z, z_proxy=zh, delta=delta, eta=eta,
               entropy_delta=entropy(e)-entropy(p),
               tv_residual=.5*(r-rh).abs().sum(-1), tv_proxy=.5*(r-e).abs().sum(-1),
               tv_residual_excluded=.5*(r-rn).abs().sum(-1), tv_proxy_excluded=.5*(r-en).abs().sum(-1),
               residual=cr, proxy=ce, oracle=cs, target_only=cp,
               available_gain=cs-ce, ideal_gain=cs-cp,
               support_loss=support_loss, rank_loss=rank_loss,
               support_mass=(r*(e<=q)).sum(-1),
               zeros_selected=(re.gather(-1,ir)==0).any(-1).float(),
               gamma_raw=gamma_raw,
               residual_topk_margin=a.topk(4,-1).values[...,2]-a.topk(4,-1).values[...,3],
               top1_match=(p.argmax(-1)==e.argmax(-1)).float())
    row['certificate'] = torch.cat([certified & allowed, torch.zeros_like(certified[:,:1])],1).float()
    for gamma in GAMMAS:
        repaired = (1-gamma)*e+gamma*p
        hc = hazard(repaired,q,y)
        for name, lam in [('residual',1.),('proxy',0.)]:
            s = source(repaired,q,y,lam)
            data[f'repair::{gamma:g}_{name}'] = score(s,hc,gt)
            row[f'repair_{gamma:g}_{name}'] = coverage(s,r)[0]
    c, uniform_model, fit_tv = fit_uniform_tv(p,e)
    row.update(uniform_c=c, uniform_fit_tv=fit_tv,
               uniform_fit_residual=coverage(source(uniform_model,q,y,1.),r)[0],
               uniform_fit_proxy=coverage(source(uniform_model,q,y,0.),r)[0])
    # Observable step features ONLY. No p, true h, target entropy, or R.
    he, hq = entropy(e[:,:-1]), entropy(q[:,:-1])
    feats = {
        'mean_proxy_draft_tv': (.5*(e[:,:-1]-q[:,:-1]).abs().sum(-1)).mean(1),
        'mean_proxy_entropy':he.mean(1), 'mean_draft_entropy':hq.mean(1),
        'mean_entropy_difference':(he-hq).mean(1),
        'mean_proxy_peak':e[:,:-1].amax(-1).mean(1),
        'mean_draft_peak':q[:,:-1].amax(-1).mean(1),
        'argmax_agreement':(e[:,:-1].argmax(-1)==q[:,:-1].argmax(-1)).float().mean(1),
        'proxy_bonus_probability':hh[:,-1], 'position_entropy':entropy(hh),
        'mean_residual_peak':rn[:,:-1].amax(-1).mean(1),
        'mean_source_tv':(.5*(rn[:,:-1]-en[:,:-1]).abs().sum(-1)).mean(1),
        'mean_proxy_top17_mass':pe[:,:-1].topk(17,-1).values.sum(-1).mean(1),
        'mean_residual_top17_mass':rn[:,:-1].topk(17,-1).values.sum(-1).mean(1),
    }
    data.update({f'feature::{k}':v for k,v in feats.items()})
    data.update({f'row::{k}':v for k,v in row.items()})
    return {k:v.cpu().numpy() for k,v in data.items()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith('cuda'):
        # Share only OUR inference GPU, reserving <3 GiB of its 24 GiB.
        torch.cuda.set_per_process_memory_fraction(.12, torch.device(args.device))
    args.output.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for done in sorted(args.input.glob('*.complete.json')):
        record=json.loads(done.read_text())
        dest=args.output/done.name.replace('.complete.json','.npz')
        if dest.exists():
            assert json.loads(dest.with_suffix('.json').read_text())['analysis_sha256']==sha
            continue
        manifest=done.parent/record['manifest']; meta=json.loads(manifest.read_text())
        assert meta['layers']==[56,79] and meta['temperature']==1.
        parts=[];attrs=[]
        for j,chunk in enumerate(meta['chunks']):
            with np.load(manifest.parent/chunk['file']) as f:
                n=len(f['step'])//5
                np.testing.assert_array_equal(f['position'],np.tile(np.arange(5),n))
                get=lambda name:torch.from_numpy(f[name].copy()).to(args.device)
                p,q,all_e,y=get('p_T'),get('p_D'),get('p_E'),get('y')
                part=measure(p.reshape(n,5,-1),q.reshape(n,5,-1),all_e[:,0].reshape(n,5,-1),y.reshape(n,5))
                parts.append(part)
                attrs.append({k:f[k][::5] for k in ['step','seq_id','local_step']})
            if j%4==0:print(f'{dest.stem}: chunk {j+1}/{len(meta["chunks"])}',flush=True)
        data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        data.update({k:np.concatenate([p[k] for p in attrs]) for k in attrs[0]})
        old=HERE.parent/'shared_review/metrics'/dest.name
        if old.exists():
            with np.load(old) as previous:
                np.testing.assert_array_equal(data['step'],previous['step'])
                for name,lam in [('proxy',0),('residual',1)]:
                    np.testing.assert_allclose(data['policy::'+name],previous[f'grid::tau1__lambda{lam}'],atol=3e-5,rtol=0)
                    np.testing.assert_allclose(data['row::'+name],previous['row::local_'+name],atol=3e-5,rtol=0)
        np.savez_compressed(dest,**data)
        dest.with_suffix('.json').write_text(json.dumps(dict(record=record,analysis_sha256=sha,manifest=str(manifest),
            main_topm=17,budget=15,local_k=3,uniform_fit='exact TV minimizing c in [0,1], oracle'),indent=2))
        print(f'finished {dest.stem}: {len(data["step"])} steps',flush=True)


if __name__=='__main__':main()
