"""Counterfactual candidate replay on lossless full-vocabulary snapshots.

Run on a free GPU, independently of the decoder. Probabilities are fixed:
no policy evaluated here changes the sampled generation trajectories.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

TAUS = (.65, .8, .9, 1., 1.1, 1.25, 1.5)
LAMBDAS = (0., .25, .5, .75, 1.)
EPS = 1e-10
SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def entropy(p):
    return -(p * p.clamp_min(1e-38).log()).sum(-1)


def normalize(x):
    return x / x.sum(-1, keepdim=True).clamp_min(1e-38)


def hazard(p, q, y):
    k = q.shape[1] - 1
    ids = y[:, :k, None]
    alpha = (p[:, :k].gather(-1, ids).squeeze(-1)
             / (q[:, :k].gather(-1, ids).squeeze(-1) + EPS)).clamp(max=1)
    reach = torch.cat([torch.ones_like(alpha[:, :1]), alpha.cumprod(1)], 1)
    h = reach.clone()
    h[:, :k] *= 1-alpha
    return h


def source(e, q, y, lam):
    s = (e - lam*q).clamp_min(0)
    # Bonus q is zero; never exclude its placeholder token id.
    s[:, :-1].scatter_(-1, y[:, :-1, None], 0)
    return s


def candidates(s, h, m=17, full_norm=False):
    values, ids = s.topk(m, -1)
    denom = s.sum(-1, keepdim=True) if full_norm else values.sum(-1, keepdim=True)
    scores = h[..., None] * values / denom.clamp_min(EPS)
    return scores, ids


def score(s, h, gt, m=17, budget=15, full_norm=False, exclude=None):
    scores, ids = candidates(s, h, m, full_norm)
    if exclude is not None:
        scores = scores.masked_fill(exclude.gather(-1, ids), -torch.inf)
    selected = scores.flatten(1).topk(budget, -1).indices
    values = gt.gather(-1, ids).flatten(1).gather(1, selected)
    return values.sum(-1)


def local_coverage(s, r, n=3):
    ids = s.topk(n, -1).indices
    return r.gather(-1, ids).sum(-1)


def entropy_matched(p, desired_entropy):
    """Oracle rank-preserving temperature surrogate, NOT a deployable policy."""
    logits = p.clamp_min(1e-38).log()
    lo = torch.full_like(desired_entropy, .05)
    hi = torch.full_like(desired_entropy, 20.)
    for _ in range(22):
        tau = (lo+hi)/2
        v = torch.softmax(logits/tau[..., None], -1)
        low_entropy = entropy(v) < desired_entropy
        lo = torch.where(low_entropy, tau, lo)
        hi = torch.where(low_entropy, hi, tau)
    tau = (lo+hi)/2
    return torch.softmax(logits/tau[..., None], -1), tau


@torch.inference_mode()
def measure(p, q, e, final_replica, y):
    b, rows, v = p.shape
    ht, he = entropy(p), entropy(e)
    raw_r = (p-q).clamp_min(0)
    r = normalize(raw_r)
    rt = source(p, q, y, 1.)
    se, re = source(e, q, y, 0.), source(e, q, y, 1.)
    h, hh = hazard(p, q, y), hazard(e, q, y)
    gt = h[..., None]*r
    torch.testing.assert_close(h.sum(1), torch.ones(b, device=p.device), atol=2e-6, rtol=0)
    torch.testing.assert_close(gt.sum((1, 2)), torch.ones(b, device=p.device), atol=2e-6, rtol=0)
    steps, diag, grid, fits = {}, {}, {}, {}
    steps['ceiling'] = gt.flatten(1).topk(15, -1).values.sum(-1)
    for name, s in [('residual', re), ('proxy', se), ('oracle_residual', rt),
                    ('oracle_target', source(p, q, y, 0.))]:
        for hn, hw in [('proxy_h', hh), ('true_h', h)]:
            for m in (17, 64):
                steps[f'{name}__{hn}__m{m}'] = score(s, hw, gt, m)
            steps[f'{name}__{hn}__full'] = score(s, hw, gt, 64, full_norm=True)
        steps[f'{name}__equal3'] = local_coverage(s, gt).sum(1)
    # Full-vocabulary normalization + exact h/r must achieve the oracle.
    torch.testing.assert_close(steps['oracle_residual__true_h__full'], steps['ceiling'], atol=3e-6, rtol=0)
    # DS reference: three unprepared q-head alternatives at each rejection
    # position, zero at bonus (no q was collected there). This is a fixed
    # 12-root set, NOT the asynchronous P1 scheduler/tree.
    ds = torch.zeros_like(q, dtype=torch.bool)
    q_alt = q.clone()
    q_alt[:, :-1].scatter_(-1, y[:, :-1, None], 0)
    ds[:, :-1].scatter_(-1, q_alt[:, :-1].topk(3, -1).indices, True)
    steps['ds12'] = (gt*ds).sum((1, 2))
    for name, s in [('residual', re), ('proxy', se)]:
        for budget in (3, 15):
            steps[f'ds12_add{budget}_{name}'] = steps['ds12']+score(s, hh, gt, 64, budget, exclude=ds)
            steps[f'ps_only{12+budget}_{name}'] = score(s, hh, gt, 64, 12+budget)
    rh = normalize((e-q).clamp_min(0))
    tvd = .5*(p-e).abs().sum(-1)
    z, zh = raw_r.sum(-1), (e-q).clamp_min(0).sum(-1)
    lost = (r*(e <= q)).sum(-1)
    diag.update(entropy_target=ht, entropy_proxy=he, delta=he-ht, tvd=tvd,
                top1_match=(p.argmax(-1)==e.argmax(-1)).float(),
                z=z, z_proxy=zh, residual_tvd=.5*(r-rh).abs().sum(-1),
                lost_mass=lost, ds_recovered_lost=(r*(e <= q)*ds).sum(-1),
                h_true=h, h_proxy=hh,
                local_residual=local_coverage(re, r), local_proxy=local_coverage(se, r),
                local_oracle=local_coverage(rt, r),
                final_head_tvd=.5*(p-final_replica).abs().sum(-1),
                stronger_bound=(tvd/torch.maximum(z,zh).clamp_min(1e-38)).clamp(max=1))
    # Bound concerns unmasked residuals; bonus row is not a residual.
    valid = (z[:, :-1] > 1e-7) & (zh[:, :-1] > 1e-7)
    violations = (diag['residual_tvd'][:, :-1]-diag['stronger_bound'][:, :-1])[valid]
    if violations.numel() and violations.max().item() > 2e-4:
        raise AssertionError('residual stability bound failed')
    steps['h_tvd'] = .5*(h-hh).abs().sum(1)
    steps['bonus_true'] = h[:, -1]
    steps['bonus_proxy'] = hh[:, -1]
    steps['h_zero_true_mass'] = (h*(hh == 0)).sum(1)
    steps['lost_joint_mass'] = (h[:, :-1]*lost[:, :-1]).sum(1)
    steps['ds_recovered_lost_joint'] = (h[:, :-1]*diag['ds_recovered_lost'][:, :-1]).sum(1)
    lp, le = p.clamp_min(1e-38).log(), e.clamp_min(1e-38).log()
    for tau in TAUS:
        calibrated = e if tau == 1. else torch.softmax(le/tau, -1)
        hc = hazard(calibrated, q, y)
        name = f'tau{tau:g}'
        fits[name+'__kl'] = (p*(lp-calibrated.clamp_min(1e-38).log())).sum(-1)
        fits[name+'__tvd'] = .5*(p-calibrated).abs().sum(-1)
        fits[name+'__entropy_delta'] = entropy(calibrated)-ht
        fits[name+'__lost'] = (r*(calibrated <= q)).sum(-1)
        for lam in LAMBDAS:
            s = source(calibrated, q, y, lam)
            key = f'{name}__lambda{lam:g}'
            grid[key] = score(s, hc, gt)
            if lam in (0., 1.):
                grid[key+'__fixed_h'] = score(s, hh, gt)
                grid[key+'__true_h'] = score(s, h, gt)
                if lam == 1.:
                    grid[key+'__h_only'] = score(re, hc, gt)
        fits[name+'__local_residual'] = local_coverage(source(calibrated, q, y, 1.), r)
    # Match each context's *observed* entropy, retaining the exact target
    # order. This isolates a constructed rank-preserving error pattern.
    thermal, tau_oracle = entropy_matched(p, he)
    hs = hazard(thermal, q, y)
    sr, sp = source(thermal, q, y, 1.), source(thermal, q, y, 0.)
    diag.update(thermal_entropy_error=entropy(thermal)-he,
                thermal_tau=tau_oracle,
                thermal_tvd=.5*(thermal-p).abs().sum(-1),
                thermal_lost=(r*(thermal <= q)).sum(-1),
                thermal_local_residual=local_coverage(sr,r),
                thermal_local_proxy=local_coverage(sp,r))
    steps['thermal_residual'] = score(sr, hs, gt)
    steps['thermal_proxy'] = score(sp, hs, gt)
    steps['thermal_residual_true_h'] = score(sr, h, gt)
    steps['thermal_proxy_true_h'] = score(sp, h, gt)
    # Additional oracle control: preserve the ENTIRE observed proxy
    # probability multiset, changing only its assignment to token identities.
    # This is stronger than merely matching entropy with a temperature.
    rank_fixed = torch.empty_like(e)
    rank_fixed.scatter_(-1,p.argsort(-1,descending=True),e.sort(-1,descending=True).values)
    hr = hazard(rank_fixed,q,y)
    rr, rp = source(rank_fixed,q,y,1.),source(rank_fixed,q,y,0.)
    diag.update(rankfixed_entropy_error=entropy(rank_fixed)-he,
                rankfixed_tvd=.5*(rank_fixed-p).abs().sum(-1),
                rankfixed_lost=(r*(rank_fixed<=q)).sum(-1),
                rankfixed_local_residual=local_coverage(rr,r),
                rankfixed_local_proxy=local_coverage(rp,r))
    steps['rankfixed_residual'] = score(rr,hr,gt)
    steps['rankfixed_proxy'] = score(rp,hr,gt)
    steps['rankfixed_residual_true_h'] = score(rr,h,gt)
    steps['rankfixed_proxy_true_h'] = score(rp,h,gt)
    # Same-head oracle control uses the final replica as reference for both.
    r2 = normalize((final_replica-q).clamp_min(0))
    h2 = hazard(final_replica, q, y)
    steps['samehead_residual'] = score(re, hh, h2[...,None]*r2)
    steps['samehead_proxy'] = score(se, hh, h2[...,None]*r2)
    # Check all fixed-cost 15-candidate policies against the exact ceiling.
    for k, a in {**grid, **{k:a for k,a in steps.items() if '__' in k}}.items():
        if (a-steps['ceiling']).max().item() > 3e-6:
            raise AssertionError(f'policy beats oracle: {k}')
    return {name:{k:a.cpu().numpy() for k,a in d.items()}
            for name,d in [('step',steps),('row',diag),('grid',grid),('fit',fits)]}


def process(done, out, device):
    record = json.loads(done.read_text())
    manifest = done.parent/record['manifest']
    meta = json.loads(manifest.read_text())
    if meta['layers'] != [56, 79] or meta['temperature'] != 1.:
        raise ValueError('unexpected probe configuration')
    groups = {k:[] for k in ('step','row','grid','fit')}
    attrs = []
    for c in meta['chunks']:
        with np.load(manifest.parent/c['file']) as z:
            # Campaign contract is K=4 throughout; fail rather than silently
            # reshape a variable-length chain.
            n = len(z['step'])//5
            np.testing.assert_array_equal(z['position'],np.tile(np.arange(5),n))
            t = {k:torch.from_numpy(z[k].copy()).to(device) for k in ('p_E','p_T','p_D','y')}
            results = measure(t['p_T'].reshape(n,5,-1),t['p_D'].reshape(n,5,-1),
                              t['p_E'][:,0].reshape(n,5,-1),t['p_E'][:,1].reshape(n,5,-1),
                              t['y'].reshape(n,5))
            for key in groups: groups[key].append(results[key])
            attrs.append({k:z[k][::5] for k in ('seq_id','step','local_step','prefix_len')})
    arrays = {k:np.concatenate([a[k] for a in attrs]) for k in attrs[0]}
    for group, chunks in groups.items():
        for key in chunks[0]: arrays[group+'::'+key] = np.concatenate([c[key] for c in chunks])
    dest = out/(done.name.replace('.complete.json','.npz'))
    np.savez_compressed(dest, **arrays)
    (dest.with_suffix('.json')).write_text(json.dumps(dict(record=record,probe=meta,
        replay_sha256=SCRIPT_SHA256,
        policies=dict(taus=TAUS,lambdas=LAMBDAS,main_topm=17,budget=15),
        train='seq_id 1..16 in every dataset; same prompt split across seeds',
        test='seq_id 17..32 in every dataset; never used to select hyperparameters'),indent=2))
    print(f'[replayed] {dest.stem}: {len(arrays["step"])} sampled steps',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=Path('ssd/experiments/proxy_source_ablation/probe_replay_20260910/out'))
    ap.add_argument('--out',type=Path,default=Path(__file__).parent/'metrics')
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    for done in sorted(args.input.glob('*.complete.json')):
        dest=args.out/done.name.replace('.complete.json','.npz')
        if dest.exists() and dest.with_suffix('.json').exists():
            if json.loads(dest.with_suffix('.json').read_text()).get('replay_sha256')==SCRIPT_SHA256:
                continue
        process(done,args.out,args.device)


if __name__=='__main__':
    main()
