"""Frozen scalar calibration and counterfactual diagnostics on served trees.

Post-sampling subset choices below are NOT deployable lossless pruning rules.
They isolate score quality on one realized candidate pool; no TPS claim.
"""
from collections import defaultdict
import hashlib
import json
import numpy as np
from core import HERE, exact_ladder, groups, path_product, prefix_knapsack, production_functions


def reach_from_alpha(par, sib, alpha):
    out = np.zeros(len(par))
    for ctx, children in sorted(groups(par, sib).items()):
        left = 1.0 if ctx < 0 else out[ctx]
        for j in children:
            out[j] = left * alpha[j]
            left *= 1.0 - alpha[j]
    return out


def load_trees(manifest, folder=None):
    trees = []
    for meta in manifest['snapshots']:
        path = (HERE/'probe' if folder is None else folder)/meta['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('Snapshot checksum mismatch')
        with np.load(path) as z:
            par = z['par'].astype(int); sib = z['sib'].astype(int); tok = z['tok'].astype(int)
            p = z['p'].astype(float); q = z['q'].astype(float)
            p /= p.sum(1, keepdims=True); q /= q.sum(1, keepdims=True)
            calc = exact_ladder(par, sib, tok, p, q)
            raw = q[np.arange(len(par)), tok]
            trees.append(dict(meta, split=manifest['plan']['mapping'][meta['prompt']]['split'],
                              par=par, sib=sib, raw=raw, calc=calc, observed_al=len(z['path'])))
    return trees


def fit(trees, bins, shrink):
    stats = defaultdict(lambda: [0., 0.])
    for t in trees:
        for j, (a, w) in enumerate(zip(t['calc']['alpha'], t['calc']['attempt'])):
            phase = str(t['phase']); sibling = str(t['sib'][j]); b = str(np.searchsorted(bins, t['raw'][j], side='right') - 1)
            for key in ['all', phase, phase+':'+sibling, phase+':'+sibling+':'+b]:
                stats[key][0] += w*a; stats[key][1] += w
    means = {}
    for key in sorted(stats, key=lambda k: (k != 'all', k.count(':'), k)):
        num, den = stats[key]
        parent = ':'.join(key.split(':')[:-1]) if ':' in key else 'all'
        prior = means.get(parent, .5)
        means[key] = num/den if key == 'all' and den > 0 else (num + shrink*prior)/(den+shrink)
    return dict(bins=bins, shrinkage=shrink, means=means, sufficient_statistics=dict(stats),
                label='Expected conditional acceptance; weighted by exact attempt probability.',
                training_prompts=sorted({t['prompt'] for t in trees}),
                validation_used_for_fitting=False)


def predict(t, model, mode):
    if mode == 'raw_q': return t['raw'].copy()
    answer = []
    for sibling, q in zip(t['sib'], t['raw']):
        phase = str(t['phase']); key = phase+':'+str(sibling)
        mean = model['means'].get(key, model['means'].get(phase, model['means']['all']))
        if mode == 'phase_sibling_q_bin':
            key += ':'+str(np.searchsorted(model['bins'], q, side='right') - 1)
            mean = model['means'].get(key, mean)
        answer.append(mean)
    return np.asarray(answer)


def prompt_mean(rows, field):
    by = defaultdict(list)
    for row in rows: by[row['prompt']].append(row[field])
    return {p: float(np.mean(x)) for p,x in by.items()}


def paired(rows, first, second):
    a = prompt_mean(rows, first); b = prompt_mean(rows, second)
    differences = np.array([a[p]-b[p] for p in sorted(a)])
    rng = np.random.default_rng(922)
    samples = differences[rng.integers(0, len(differences), (2000, len(differences)))].mean(1)
    return dict(mean_delta=float(differences.mean()), ci95=np.quantile(samples, [.025,.975]).tolist(),
                positive_prompts=int((differences>1e-10).sum()), prompts=len(differences))


def main():
    import torch
    torch.set_num_threads(1)
    manifest = json.loads((HERE/'probe/manifest.json').read_text())
    if len(manifest['records']) != len(manifest['plan']['mapping']):
        raise ValueError('Refuse to analyze an incomplete campaign')
    trees = load_trees(manifest)
    spec = manifest['plan']['analysis_prespecified']; modes = spec['estimators']
    model = fit([t for t in trees if t['split']=='calibration'], spec['q_bins'], spec['shrinkage_attempt_mass'])
    frozen = HERE/'calibration_frozen.json'
    serialized = json.dumps(model, indent=2)
    if frozen.exists() and frozen.read_text() != serialized:
        raise ValueError('Refuse to replace different frozen calibration')
    frozen.write_text(serialized)  # Persist calibration BEFORE looking at validation scores.
    production = production_functions()
    rows=[]; node_rows=[]; diagnostics=[]
    for t in trees:
        if t['split'] != 'validation': continue
        par=t['par']; sib=t['sib']; calc=t['calc']; n=len(par)
        qpath=path_product(par,t['raw'])
        score=dict(q_path=qpath)
        row=dict(prompt=t['prompt'], phase=t['phase'], file=t['file'], n=n,
                 true_al=calc['al'], observed_al=t['observed_al'], q_path_al=float(qpath.sum()))
        for mode in modes:
            alpha=predict(t,model,mode); rho=reach_from_alpha(par,sib,alpha); score[mode]=rho
            row[mode+'_al']=float(rho.sum())
            row[mode+'_alpha_mse']=float(np.sum(calc['attempt']*(alpha-calc['alpha'])**2)/max(calc['attempt'].sum(),1e-12))
            row[mode+'_reach_mse']=float(np.mean((rho-calc['reach'])**2))
            row[mode+'_al_abs_error']=abs(float(rho.sum())-calc['al'])
        row['q_path_reach_mse']=float(np.mean((qpath-calc['reach'])**2))
        row['q_path_al_abs_error']=abs(float(qpath.sum())-calc['al'])
        rows.append(row)
        for j in range(n):
            node_rows.append(dict(prompt=t['prompt'],phase=t['phase'],sibling=int(sib[j]),q=float(t['raw'][j]),
                                  alpha=float(calc['alpha'][j]),reach=float(calc['reach'][j]),attempt=float(calc['attempt'][j])))
        # Fixed pool diagnostic; evaluate exact conditional AL after pruning.
        for budget in [2,4,6]:
            if budget >= n: continue
            greedy=production['rerank_tree_indices'](par.tolist(),sib.tolist(),t['raw'].tolist(),budget)
            d=dict(prompt=t['prompt'],phase=t['phase'],file=t['file'],budget=budget,full_al=calc['al'],
                   q_greedy=float(calc['reach'][greedy].sum()))
            for name, weights in dict(score, oracle=calc['reach']).items():
                keep,_=prefix_knapsack(par,sib,weights,budget)
                d[name+'_dp']=float(calc['reach'][keep].sum())
            diagnostics.append(d)
    summary={}
    for phase in ['all']+sorted({r['phase'] for r in rows}):
        selected=[r for r in rows if phase=='all' or r['phase']==phase]
        fields=[k for k in rows[0] if k not in ['prompt','phase','file','n']]
        summary[str(phase)]=dict(trees=len(selected),prompts=len({r['prompt'] for r in selected}),
            prompt_balanced={k:float(np.mean(list(prompt_mean(selected,k).values()))) for k in fields},
            paired_reach_mse={mode:paired(selected,mode+'_reach_mse','q_path_reach_mse') for mode in modes})
    budget_summary={}
    for budget in [2,4,6]:
        selected=[r for r in diagnostics if r['budget']==budget]
        if not selected: continue
        fields=['q_greedy','q_path_dp']+[m+'_dp' for m in modes]+['oracle_dp','full_al']
        budget_summary[str(budget)]=dict(trees=len(selected),prompts=len({r['prompt'] for r in selected}),
            prompt_balanced={k:float(np.mean(list(prompt_mean(selected,k).values()))) for k in fields},
            paired_vs_greedy={k:paired(selected,k,'q_greedy') for k in fields if k!='q_greedy'})
    sibling_summary=[]
    for phase in sorted({n['phase'] for n in node_rows}):
        for sibling in sorted({n['sibling'] for n in node_rows if n['phase']==phase}):
            ns=[n for n in node_rows if n['phase']==phase and n['sibling']==sibling]
            den=sum(n['attempt'] for n in ns)
            sibling_summary.append(dict(phase=phase,sibling=sibling,nodes=len(ns),attempt_mass=den,
                attempt_weighted_q=sum(n['attempt']*n['q'] for n in ns)/max(den,1e-12),
                conditional_acceptance=sum(n['reach'] for n in ns)/max(den,1e-12),
                total_reach=sum(n['reach'] for n in ns)))
    result=dict(scope='Served-tree held-out diagnostics; no new topology rollout, no TPS, no lossless pruning claim.',
        calibration_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
        calibration_trees=sum(t['split']=='calibration' for t in trees),validation_trees=len(rows),
        summary=summary,budget_diagnostics=budget_summary,sibling_diagnostics=sibling_summary)
    (HERE/'analysis.json').write_text(json.dumps(result,indent=2))
    (HERE/'validation_rows.json').write_text(json.dumps(dict(trees=rows,nodes=node_rows,subsets=diagnostics),indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
