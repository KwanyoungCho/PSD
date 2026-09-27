"""Grouped out-of-fold diagnostics. Fixed-pool DP is never a deployed policy."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from features import groups, extract
from calibrate import MODES, fit, predict, frozen_predict, from_edges
from core import prefix_knapsack, production_functions

HERE = Path(__file__).resolve().parent
POLICIES = ['q_path', 'phase_sibling_q_bin']
SCORES = ['q_path', 'frozen', 'same8_depth', 'same8_rich'] + MODES


def macro(rows, field):
    by = defaultdict(list)
    for r in rows:
        if field in r: by[(r['group'], r['question_id'])].append(r[field])
    tasks = defaultdict(list)
    for (g, q), vals in by.items(): tasks[g].append(float(np.mean(vals)))
    return float(np.mean([np.mean(v) for v in tasks.values()])) if tasks else None


def interval(rows, first, second=None):
    by = defaultdict(list)
    for r in rows:
        if first in r and (second is None or second in r):
            by[(r['group'], r['question_id'])].append(r[first] - (r[second] if second else 0))
    tasks = defaultdict(list)
    for (g,q), vals in by.items(): tasks[g].append(float(np.mean(vals)))
    rng = np.random.default_rng(25322); samples = np.zeros(2000)
    for vals in tasks.values():
        a = np.asarray(vals); samples += a[rng.integers(len(a), size=(2000,len(a)))].mean(axis=1) / len(tasks)
    return dict(mean=macro(rows, first)-(macro(rows, second) if second else 0),
        ci95=np.quantile(samples, [.025,.975]).tolist(), questions=len(by)) if by else None


def decorate(t):
    par = t['par']; reach = np.asarray(t['reach']); alpha = np.asarray(t['alpha'])
    parent = np.array([1. if p < 0 else reach[p] for p in par])
    # Local transition remains defined even if the incoming reach is zero.
    # Dividing reach/parent and filling zero would give the oracle extra help.
    _, beta = from_edges(t, alpha)
    t['_parent'] = parent; t['_beta'] = beta


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--only-policy',choices=POLICIES);args=ap.parse_args()
    trees=[]; coverage=[]
    for policy in (POLICIES[:1] if args.smoke else [args.only_policy] if args.only_policy else POLICIES):
        folder=HERE/'runs'/(('smoke_' if args.smoke else '')+policy)
        completion=json.loads((folder/'completion.json').read_text())
        validated=json.loads((folder/'validated.json').read_text())
        if completion['exit_code'] or (not args.smoke and validated['turns']!=560): raise ValueError('Incomplete diagnostic run')
        records=[json.loads(s) for s in (folder/'records.jsonl').read_text().splitlines()]
        expected_hits=sum(e['cache_hit'] for r in records for e in r['metrics']['phase_events'])
        ts=[json.loads(s) for s in (folder/'trees.jsonl').read_text().splitlines()]
        if len(ts)!=expected_hits: raise ValueError('Missing served trees')
        for t in ts: decorate(t)
        trees.extend(ts)
        coverage.append(dict(policy=policy,turns=len(records),questions=validated['questions'],trees=len(ts),
            nonfinal_trees=sum(not t['is_final_event'] for t in ts),nodes=sum(t['n'] for t in ts),
            questions_with_hits=len({t['question_id'] for t in ts}),raw_snapshots=sum('raw_file' in t for t in ts)))
    questions=json.loads((HERE.parent/'duet_tree_al_full/questions.json').read_text())
    counts=defaultdict(int); folds={}
    for q in questions:
        folds[q['question_id']]=counts[q['group']]%5;counts[q['group']]+=1
    frozen=json.loads((HERE.parent/'duet_tree_analysis/calibration_frozen.json').read_text())
    oldfolder=HERE.parent/'duet_tree_analysis/probe'
    oldmanifest=json.loads((oldfolder/'manifest.json').read_text());oldcal=[]
    for meta in oldmanifest['snapshots']:
        if oldmanifest['plan']['mapping'][meta['prompt']]['split']!='calibration':continue
        file=oldfolder/meta['file']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Old calibration snapshot changed')
        with np.load(file) as z:sample=extract(**dict(z))
        sample.update(phase=meta['phase']);oldcal.append(sample)
    same8={name:fit(oldcal,mode) for name,mode in [('same8_depth','depth_q'),('same8_rich','draft_rich')]}
    # Overlap is known retrospectively for EXPANDED nodes. Its predictor uses
    # only features already present when the node was created, before expanding it.
    gain_training=[]
    for t in trees:
        kids=groups(t['par'],t['sib']); ids=[p for p in kids if p>=0]
        if not ids:continue
        gt={k:t[k] for k in ['question_id','phase','policy']}
        for key in ['raw','q_cond','depth','sib','q_entropy']:
            gt[key]=[t[key][j] for j in ids]
        gt.update(n=len(ids),alpha=[t['overlap'][kids[j][0]] for j in ids],
            attempt=[t['reach'][j] for j in ids])
        gain_training.append(gt)
    models={}; crossmodels={}; gain_models={}
    for fold in range(5):
        training=[t for t in trees if folds[t['question_id']]!=fold]
        for mode in MODES:
            models[(fold,mode)]=fit(training,mode)
            models[(fold,mode)]['training_question_ids']=sorted({t['question_id'] for t in training})
            if any(folds[q]==fold for q in models[(fold,mode)]['training_question_ids']):raise ValueError('Fold leakage')
        gain_models[fold]=fit([t for t in gain_training if folds[t['question_id']]!=fold],'draft_rich')
        # Rich-table transfer is evaluated on held-out questions from the OTHER policy.
        for policy in POLICIES:
            train=[t for t in training if t['policy']==policy]
            if train: crossmodels[(fold,policy)]=fit(train,'draft_rich')
    prefix='smoke_' if args.smoke else args.only_policy+'_' if args.only_policy else ''
    (HERE/(prefix+'crossfit_models.json')).write_text(json.dumps(
        dict(folds=folds,models={f'{f}:{m}':v for (f,m),v in models.items()},gain_models=gain_models,same8=same8,
             scope='Exploratory scalar calibration. Held-out questions and both turns/policies excluded from fitting.'),indent=2))
    production=production_functions(); rows=[]; subsets=[]; strata=[]; expansion=[]; prospective=[]; examples=[]
    for ti,t in enumerate(trees):
        true=np.asarray(t['reach']); a=np.asarray(t['alpha']); attempts=np.asarray(t['attempt']); par=t['par']
        fa=frozen_predict(t,frozen); fr,fb=from_edges(t,fa)
        scores=dict(q_path=np.asarray(t['q_path']),frozen=fr); locals_=dict(frozen=fb); alphas=dict(frozen=fa)
        for name,model in same8.items():
            aa=predict(t,model);scores[name],locals_[name]=from_edges(t,aa);alphas[name]=aa
        fold=folds[t['question_id']]
        predicted_g=predict(t,gain_models[fold])
        for mode in MODES:
            edge=predict(t,models[(fold,mode)]); rho,beta=from_edges(t,edge,beta=mode=='direct_beta')
            scores[mode]=rho;locals_[mode]=beta
            if mode!='direct_beta':alphas[mode]=edge
        for policy in POLICIES:
            if (fold,policy) in crossmodels:
                edge=predict(t,crossmodels[(fold,policy)])
                scores['rich_from_'+policy]=from_edges(t,edge)[0]
        meta={k:t[k] for k in ['question_id','group','uid','phase','policy','serial','is_final_event']}
        r=dict(meta,n=t['n'],true_al=t['true_al'],observed_al=t['observed_al'],coin_variance=t['coin_variance'],
            oracle_observed_mse=(t['observed_al']-t['true_al'])**2,
            observed_minus_exact=t['observed_al']-t['true_al'])
        for name,rho in scores.items():
            r[name+'_reach_mse']=float(np.mean((rho-true)**2))
            r[name+'_pred_al']=float(rho.sum());r[name+'_al_bias']=float(rho.sum()-t['true_al'])
            r[name+'_al_mse']=float((rho.sum()-t['true_al'])**2)
            r[name+'_observed_mse']=float((rho.sum()-t['observed_al'])**2)
        for name,aa in alphas.items():
            r[name+'_alpha_mse']=float(np.sum(attempts*(aa-a)**2)/max(attempts.sum(),1e-14))
        # Oracle substitutions diagnose factor errors; not causal/additive attribution.
        r['oracle_parent_frozen_local_mse']=float(np.mean((t['_parent']*fb-true)**2))
        fparent=np.array([1. if p<0 else fr[p] for p in par])
        r['frozen_parent_oracle_local_mse']=float(np.mean((fparent*t['_beta']-true)**2))
        kids=groups(t['par'],t['sib']); cap={1:4,2:2}[t['phase']]
        depth=np.r_[0,np.asarray(t['depth'])]; terminal=np.asarray(t['terminal'])
        leaf=np.array([ctx not in kids for ctx in [-1]+list(range(t['n']))])
        r['early_leaf_terminal']=float(terminal[leaf&(depth<cap)].sum())
        r['depth_cap_terminal']=float(terminal[depth==cap].sum())
        r['internal_rejection_terminal']=float(terminal[~leaf].sum())
        r['early_leaf_extra_upper']=float(np.dot(terminal[leaf],(cap-depth)[leaf]))
        r['internal_rejection_gap']=float(np.dot(terminal[~leaf],(cap-depth)[~leaf]))
        q_threshold=.03 if t['phase']==2 else 0.
        below=np.r_[False,np.asarray(t['raw'])<q_threshold]
        r['early_leaf_below_q_threshold_mass']=float(terminal[leaf&(depth<cap)&below].sum())
        r['unused_node_budget']={1:8,2:6}[t['phase']]-t['n']
        r['underfilled_tree']=float(r['unused_node_budget']>0)
        if abs(r['early_leaf_extra_upper']+r['internal_rejection_gap']-(cap-t['true_al']))>1e-8:
            raise ValueError('Structural gap identity')
        r['zero_reach_node_fraction']=float(np.mean(true<1e-12))
        r['low_reach_node_fraction']=float(np.mean(true<.01))
        rows.append(r)
        for j in range(t['n']):
            strata.append(dict(meta,depth=t['depth'][j],sibling=t['sib'][j],q=t['raw'][j],
                alpha=a[j],attempt=attempts[j],reach=true[j],q_path=scores['q_path'][j],frozen=fr[j],
                draft_rich=scores['draft_rich'][j],refit_q=scores['refit_q'][j],
                overlap=t['overlap'][j],entropy=t['q_entropy'][j],q_cond=t['q_cond'][j],
                frozen_alpha=fa[j],p_rank=t['p_rank'][j],p_token=t['p_token'][j]))
        for budget in [2,4,6]:
            if budget>=t['n']:continue
            diag=dict(meta,budget=budget,full_al=t['true_al'])
            ids=production['rerank_tree_indices'](par,t['sib'],t['raw'],budget)
            diag['q_greedy']=float(true[ids].sum())
            selected_sets={}
            for name,weights in dict(scores,oracle=true).items():
                ids,_=prefix_knapsack(par,t['sib'],weights,budget)
                diag[name]=float(true[ids].sum())
                selected_sets[name]=set(ids)
            for name in ['q_path','refit_q','draft_rich','direct_beta','oracle']:
                diag[name+'_changed_vs_frozen']=float(selected_sets[name]!=selected_sets['frozen'])
                diag[name+'_win_vs_frozen']=float(diag[name]>diag['frozen']+1e-10)
                diag[name+'_loss_vs_frozen']=float(diag[name]<diag['frozen']-1e-10)
            subsets.append(diag)
            if budget==4 and diag['oracle']-diag['frozen']>.6:
                examples.append(dict(diag,par=par,sib=t['sib'],tok=t['tok'],raw=t['raw'],
                    alpha=t['alpha'],reach=t['reach'],frozen_reach=fr.tolist(),
                    q_path=t['q_path'],raw_file=t.get('raw_file')))
        # Only compare observed, expanded parents with equal depth AND equal fanout.
        buckets=defaultdict(list)
        for parent,children in kids.items():
            if parent<0:continue
            buckets[(t['depth'][parent],len(children))].append(parent)
        for (d,c),ps in buckets.items():
            if len(ps)<2:continue
            gains=np.array([true[kids[p]].sum() for p in ps])
            oracle=int(gains.argmax());best=float(gains[oracle])
            v=dict(meta,depth=d,fanout=c,choices=len(ps),oracle_gain=best,
                rho_choice_gain=float(gains[np.argmax(true[ps])]))
            for name in ['q_path','frozen','draft_rich']:
                v[name+'_choice_gain']=float(gains[np.argmax(scores[name][ps])])
            v['rho_misrank']=float(best-v['rho_choice_gain']>1e-10)
            v['rho_regret']=best-v['rho_choice_gain'];expansion.append(v)
        # Integrate the first new candidate over q: g1 = sum min(p,q).
        # These are only nodes that were actually expanded, hence known next q.
        bydepth=defaultdict(list)
        for parent in kids:
            if parent>=0:bydepth[t['depth'][parent]].append(parent)
        for d,ps in bydepth.items():
            if len(ps)<2:continue
            overlaps=np.array([t['overlap'][kids[p][0]] for p in ps])
            gains=true[ps]*overlaps; best=float(gains.max())
            v=dict(meta,depth=d,choices=len(ps),oracle_gain=best,
                rho_choice_gain=float(gains[np.argmax(true[ps])]))
            priorities={name:scores[name][ps] for name in ['q_path','frozen','draft_rich']}
            priorities['frozen_times_pred_g']=fr[ps]*predicted_g[ps]
            priorities['rich_times_pred_g']=scores['draft_rich'][ps]*predicted_g[ps]
            priorities['frozen_times_oracle_g']=fr[ps]*overlaps
            for name,values in priorities.items():v[name+'_choice_gain']=float(gains[np.argmax(values)])
            fi=int(np.argmax(fr[ps]));ri=int(np.argmax(scores['draft_rich'][ps]));gi=int(np.argmax(fr[ps]*predicted_g[ps]))
            v['rich_choice_changed']=float(ri!=fi);v['g_choice_changed']=float(gi!=fi)
            v['frozen_misrank']=float(best-v['frozen_choice_gain']>1e-10)
            if d==1 and any(t['sib'][p]==0 for p in ps):
                v['frozen_selects_first_sibling']=float(t['sib'][ps[fi]]==0)
                v['oracle_selects_later_sibling']=float(t['sib'][ps[int(gains.argmax())]]>0 and best>1e-12)
            v['rho_misrank']=float(best-v['rho_choice_gain']>1e-10)
            v['rho_regret']=best-v['rho_choice_gain']
            v['g_prediction_mse']=float(np.sum(true[ps]*(predicted_g[ps]-overlaps)**2)/max(true[ps].sum(),1e-14))
            prospective.append(v)
        if ti%5000==0:print('analyzed',ti,'/',len(trees),flush=True)
    result=dict(coverage=coverage,weighting='Equal task, then equal original question, then mean over eligible trees. Both policies pooled unless specified.',
        folds=5,bootstrap=2000,scope='Diagnostic observer trajectories, not online policy AL gain.')
    result['fold_coverage']={str(f):dict(allocated_training_questions=sum(v!=f for v in folds.values()),
        observed_training_questions=len(models[(f,'refit_q')]['training_question_ids']),
        allocated_test_questions=sum(v==f for v in folds.values()),
        observed_test_questions=len({t['question_id'] for t in trees if folds[t['question_id']]==f})) for f in range(5)}
    summaries={}
    for scope in ['all_events','nonfinal']:
        selected=[r for r in rows if scope=='all_events' or not r['is_final_event']]
        for phase in ['all',1,2]:
            rs=[r for r in selected if phase=='all' or r['phase']==phase]
            fields=[k for k in rows[0] if k not in ['question_id','group','uid','phase','policy','serial','is_final_event']]
            summaries[f'{scope}:{phase}']=dict(trees=len(rs),means={k:macro(rs,k) for k in fields},
                reach_vs_q={s:interval(rs,s+'_reach_mse','q_path_reach_mse') for s in SCORES[1:]},
                reach_vs_frozen={s:interval(rs,s+'_reach_mse','frozen_reach_mse') for s in MODES},
                reach_vs_refit={s:interval(rs,s+'_reach_mse','refit_q_reach_mse') for s in MODES if s!='refit_q'},
                observed_minus_exact=interval(rs,'observed_minus_exact'))
    result['scores']=summaries
    result['per_policy']={}
    for policy in POLICIES:
        rs=[r for r in rows if r['policy']==policy and not r['is_final_event']]
        if rs:result['per_policy'][policy]={k:macro(rs,k) for k in rows[0] if k.endswith(('_reach_mse','_al_bias'))}
    result['budgets']={}
    for phase in ['all',1,2]:
        for budget in [2,4,6]:
            rs=[r for r in subsets if r['budget']==budget and not r['is_final_event'] and (phase=='all' or r['phase']==phase)]
            if not rs:continue
            names=['q_greedy']+SCORES+['oracle']
            result['budgets'][f'{phase}:{budget}']=dict(trees=len(rs),means={k:macro(rs,k) for k in names},
                differences_vs_frozen={k:interval(rs,k,'frozen') for k in names if k!='frozen'},
                choice_rates={k:macro(rs,k) for k in rs[0] if k.endswith(('_changed_vs_frozen','_win_vs_frozen','_loss_vs_frozen'))})
    result['expansion']={}
    for phase in ['all',1,2]:
        rs=[r for r in expansion if not r['is_final_event'] and (phase=='all' or r['phase']==phase)]
        if rs: result['expansion'][str(phase)]=dict(comparisons=len(rs),
            means={k:macro(rs,k) for k in ['oracle_gain','rho_choice_gain','q_path_choice_gain','frozen_choice_gain','draft_rich_choice_gain','rho_misrank','rho_regret']},
            rho_regret_ci=interval(rs,'rho_regret'))
    result['prospective_onechild']={}
    for phase in ['all',1,2]:
        rs=[r for r in prospective if not r['is_final_event'] and (phase=='all' or r['phase']==phase)]
        if rs:result['prospective_onechild'][str(phase)]=dict(comparisons=len(rs),
            means={k:macro(rs,k) for k in rs[0] if k.endswith(('_gain','_regret','_misrank','_mse','_changed'))},
            root_branch_comparisons=sum('frozen_selects_first_sibling' in r for r in rs),
            root_branch={k:macro([r for r in rs if r['depth']==1],k) for k in ['frozen_selects_first_sibling','oracle_selects_later_sibling']},
            differences_vs_frozen={k:interval(rs,k,'frozen_choice_gain') for k in rs[0] if k.endswith('_choice_gain') and k!='frozen_choice_gain'})
    result['strata']=[]
    for axis in ['depth','sibling','q_bin','overlap_bin']:
        for phase in [1,2]:
            bins=defaultdict(list)
            for r in strata:
                if r['phase']!=phase or r['is_final_event']:continue
                b=r[axis] if axis in r else int(np.searchsorted([.1,.3,.6,.9] if axis=='q_bin' else [.25,.5,.75,.9],r['q' if axis=='q_bin' else 'overlap']))
                bins[b].append(r)
            for b,rs in sorted(bins.items()):
                attempt=sum(r['attempt'] for r in rs)
                result['strata'].append(dict(axis=axis,phase=phase,bin=int(b),nodes=len(rs),
                    attempt_mass=attempt,alpha=sum(r['attempt']*r['alpha'] for r in rs)/max(attempt,1e-14),
                    predicted_alpha=sum(r['attempt']*r['frozen_alpha'] for r in rs)/max(attempt,1e-14),
                    q_path_mse=float(np.mean([(r['q_path']-r['reach'])**2 for r in rs])),
                    frozen_mse=float(np.mean([(r['frozen']-r['reach'])**2 for r in rs])),
                    rich_mse=float(np.mean([(r['draft_rich']-r['reach'])**2 for r in rs]))))
    result['high_confidence_errors']={}
    for phase in [1,2]:
        rs=[r for r in strata if r['phase']==phase and r['q']>=.9 and r['sibling']==0 and not r['is_final_event']]
        w=sum(r['attempt'] for r in rs)
        result['high_confidence_errors'][str(phase)]=dict(nodes=len(rs),attempt_mass=w,
            alpha_below_half=sum(r['attempt'] for r in rs if r['alpha']<.5)/max(w,1e-14),
            alpha_below_tenth=sum(r['attempt'] for r in rs if r['alpha']<.1)/max(w,1e-14))
    out=prefix+'analysis.json'
    (HERE/out).write_text(json.dumps(result,indent=2,allow_nan=False))
    if not args.smoke and not args.only_policy:
        (HERE/'tree_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (HERE/'subset_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in subsets))
        (HERE/'expansion_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in expansion))
        (HERE/'prospective_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prospective))
        examples.sort(key=lambda r:r['oracle']-r['frozen'],reverse=True)
        (HERE/'failure_examples.json').write_text(json.dumps(examples[:30],indent=2))
    print(json.dumps(dict(coverage=coverage,primary=result['scores']['nonfinal:all']),indent=2))


if __name__=='__main__':main()
