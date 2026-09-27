"""Full-corpus AL inference; refuses a final comparison with missing jobs."""
import importlib.util
import itertools
import json
from pathlib import Path
import hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_old_online_analysis',HERE.parent/'duet_tree_al_full/analyze.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def main():
    plan=json.loads((HERE/'plan.json').read_text());questions=json.loads((HERE/'questions.json').read_text())
    expected=[f'{q["question_id"]}_t{t}' for q in questions for t in range(len(q['turns']))]
    raw={};coverage=[]
    for job in plan['jobs']:
        folder=HERE/'runs'/f's{job["seed"]}_{job["policy"]}'
        complete=folder/'completion.json';valid=folder/'validated.json';path=folder/'records.jsonl'
        done=complete.exists() and valid.exists() and json.loads(complete.read_text())['exit_code']==0
        rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        coverage.append(dict(**job,complete=done,turns=len(rows)))
        if done:
            if [r['uid'] for r in rows]!=expected:raise ValueError('Incomplete corpus')
            if any(r['policy']!=job['policy'] or r['seed']!=job['seed'] for r in rows):raise ValueError('Wrong policy/seed')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=json.loads(valid.read_text())['records_sha256']:raise ValueError('Changed records')
            raw[job['policy'],job['seed']]=old.question_counts(rows,questions)
    (HERE/'online_coverage.json').write_text(json.dumps(coverage,indent=2))
    if not all(x['complete'] for x in coverage):raise ValueError('Online campaign incomplete; no final winner inference')
    policies=plan['policies'];seeds=plan['seeds'];groups=plan['groups'];summaries={};by_seed={}
    metrics=['al_hit','al_p1','al_p2','al_equal_phase','al_all_step_sensitivity','al_emitted_all_steps','hit_rate']
    for policy in policies:
        pooled=[r for s in seeds for r in raw[policy,s].values()]
        gg={g:old.aggregate([r for r in pooled if r['group']==g]) for g in groups}
        summaries[policy]=dict(groups=gg,event_weighted=old.aggregate(pooled),
            equal_task_mean={k:float(np.mean([gg[g][k] for g in groups])) for k in metrics})
    for seed in seeds:
        by_seed[str(seed)]={p:{g:old.aggregate([r for r in raw[p,seed].values() if r['group']==g]) for g in groups} for p in policies}
    boot={p:{k:[] for k in metrics} for p in policies};rng=np.random.default_rng(9222026)
    ix={k:i for i,k in enumerate(old.FIELDS)}
    for group in groups:
        ids=[q['question_id'] for q in questions if q['group']==group]
        indices=rng.integers(0,len(ids),(5000,len(ids)))
        for policy in policies:
            a=np.array([[sum(raw[policy,s][i][k] for s in seeds) for k in old.FIELDS] for i in ids])
            a=a[indices].sum(1)
            div=lambda n,d:a[:,ix[n]]/a[:,ix[d]]
            values=dict(al_hit=div('hit_sum','hit_n'),al_p1=div('p1_sum','p1_n'),al_p2=div('p2_sum','p2_n'),
                al_all_step_sensitivity=div('raw_hit_sum','raw_hit_n'),al_emitted_all_steps=div('emitted_hit_sum','raw_hit_n'),
                hit_rate=div('raw_hit_n','raw_step_n'))
            values['al_equal_phase']=(values['al_p1']+values['al_p2'])/2
            for k,v in values.items():
                if not np.isfinite(v).all():raise ValueError('Undefined bootstrap ratio')
                boot[policy][k].append(v)
    boot={p:{k:np.mean(v,axis=0) for k,v in dd.items()} for p,dd in boot.items()}
    comparisons={}
    for baseline,candidate in itertools.combinations(policies,2):
        comparisons[candidate+' - '+baseline]={}
        for k in metrics:
            diff=boot[candidate][k]-boot[baseline][k]
            base=summaries[baseline]['equal_task_mean'][k];point=summaries[candidate]['equal_task_mean'][k]-base
            comparisons[candidate+' - '+baseline][k]=dict(delta=point,relative_pct=100*point/base,
                pointwise_ci95=np.quantile(diff,[.025,.975]).tolist())
    primary=[];centered=[]
    for candidate,baseline in plan['primary_contrasts']:
        point=summaries[candidate]['equal_task_mean']['al_hit']-summaries[baseline]['equal_task_mean']['al_hit']
        diff=boot[candidate]['al_hit']-boot[baseline]['al_hit'];se=float(diff.std(ddof=1))
        centered.append((diff-point)/max(se,1e-15))
        primary.append(dict(candidate=candidate,baseline=baseline,delta=point,bootstrap_se=se))
    critical=float(np.quantile(np.max(np.abs(centered),axis=0),.95))
    for row in primary:
        radius=critical*row['bootstrap_se'];row['simultaneous_ci95']=[row['delta']-radius,row['delta']+radius]
    result=dict(complete=True,primary='equal_task_mean.al_hit',coverage=coverage,summaries=summaries,by_seed=by_seed,
        comparisons=comparisons,primary_contrasts=primary,simultaneous_critical_value=critical,
        inference='5000 paired task-stratified question bootstrap replicates; max-standardized-error simultaneous intervals for six prespecified contrasts.',
        limitations=plan['limits'])
    (HERE/'online_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(primary,indent=2))


if __name__=='__main__':main()
