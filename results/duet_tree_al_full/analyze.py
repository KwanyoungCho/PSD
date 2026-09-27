"""AL-primary, full-coverage, question-clustered analysis; no subset winner picking."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
FIELDS=['hit_n','hit_sum','p1_n','p1_sum','p2_n','p2_sum','step_n','raw_hit_n','raw_hit_sum','raw_step_n','output_tokens','wall_s',
        'hit_nodes','p1_nodes','p2_nodes','all_spec_sum','terminal_events','emitted_hit_sum']


def question_counts(rows,questions):
    by={q['question_id']:dict(question_id=q['question_id'],group=q['group'],**{k:0. for k in FIELDS},hist=[0]*5,
                            p1_hist=[0]*5,p2_hist=[0]*5,
                            turns=0,stop_counts=Counter(),input_max=0) for q in questions}
    for r in rows:
        a=by[r['question_id']];a['turns']+=1;a['output_tokens']+=r['output_tokens'];a['wall_s']+=r['wall_s']
        a['stop_counts'][r['stop']]+=1;a['input_max']=max(a['input_max'],r['input_tokens'])
        events=r['metrics']['phase_events']
        a['raw_step_n']+=len(events)
        a['terminal_events']+=bool(events)
        for e in events:
            if e['cache_hit']:a['raw_hit_n']+=1;a['raw_hit_sum']+=e['accepted_spec_len']
        # Async suffix = current correction root + accepted descendants;
        # the next recovery token is kept pending for the following step.
        # Only the last suffix can be clipped by EOS/output cap.
        remaining=r['output_tokens']
        for i,e in enumerate(events):
            emitted=min(remaining,e['accepted_len'])
            if emitted<=0 or (i<len(events)-1 and emitted!=e['accepted_len']):
                raise ValueError('Output clipping occurred before the last event')
            if e['cache_hit']:a['emitted_hit_sum']+=max(0,emitted-1)
            remaining-=emitted
        if events and remaining:raise ValueError('Output count exceeds logged suffixes')
        # Final step may contain accepted tokens that were clipped by EOS/cap.
        for e in events[:-1]:
            a['step_n']+=1
            a['all_spec_sum']+=e['accepted_spec_len']
            if not e['cache_hit']:continue
            n=e['accepted_spec_len'];phase=e['source']
            if not isinstance(n,int) or not 0<=n<=({1:4,2:2}[phase]):raise ValueError('Invalid AL')
            a['hit_n']+=1;a['hit_sum']+=n;a[f'p{phase}_n']+=1;a[f'p{phase}_sum']+=n;a['hist'][n]+=1
            a[f'p{phase}_hist'][n]+=1
            nodes=e['valid_k']
            if not isinstance(nodes,int) or not n<=nodes<=({1:8,2:6}[phase]):raise ValueError('Invalid hit node count')
            a['hit_nodes']+=nodes;a[f'p{phase}_nodes']+=nodes
    return by


def aggregate(rows):
    sums={k:float(sum(r[k] for r in rows)) for k in FIELDS}
    ratio=lambda num,den:sums[num]/sums[den] if sums[den]>0 else None
    hist=np.array([r['hist'] for r in rows]).sum(0)
    result=dict(sums,al_hit=ratio('hit_sum','hit_n'),al_p1=ratio('p1_sum','p1_n'),al_p2=ratio('p2_sum','p2_n'),
                al_all_step_sensitivity=ratio('raw_hit_sum','raw_hit_n'),hit_rate=ratio('raw_hit_n','raw_step_n'),
                al_emitted_all_steps=ratio('emitted_hit_sum','raw_hit_n'),
                tps_descriptive=ratio('output_tokens','wall_s'),
                question_seed_zero_hits=sum(r['hit_n']==0 for r in rows),histogram=hist.tolist(),
                tail={str(k):float(hist[k:].sum()/max(1,hist.sum())) for k in range(1,5)})
    result['al_equal_phase']=(result['al_p1']+result['al_p2'])/2 if result['al_p1'] is not None and result['al_p2'] is not None else None
    result['al_all_decode_steps']=ratio('all_spec_sum','step_n')
    result['mean_hit_verify_nodes']=ratio('hit_nodes','hit_n')
    result['phase1_share_of_hits']=ratio('p1_n','hit_n')
    result['phase_histograms']={}
    result['phase_tails']={}
    for phase in (1,2):
        h=np.array([r[f'p{phase}_hist'] for r in rows]).sum(0)
        result['phase_histograms'][str(phase)]=h.tolist()
        result['phase_tails'][str(phase)]={str(k):float(h[k:].sum()/max(1,h.sum())) for k in range(1,5)}
        result[f'mean_p{phase}_verify_nodes']=ratio(f'p{phase}_nodes',f'p{phase}_n')
    result['stop_counts']=dict(sum((Counter(r['stop_counts']) for r in rows),Counter()))
    result['input_max']=max(r['input_max'] for r in rows)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    plan=json.loads((HERE/'plan.json').read_text());questions=json.loads((HERE/'questions.json').read_text())
    expected=[f'{q["question_id"]}_t{t}' for q in questions for t in range(len(q['turns']))]
    raw={};coverage=[]
    for job in plan['jobs']:
        folder=HERE/'runs'/f"s{job['seed']}_{job['policy']}"
        path=folder/'records.jsonl';complete=folder/'completion.json';validated=folder/'validated.json'
        rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        done=complete.exists() and json.loads(complete.read_text())['exit_code']==0 and validated.exists()
        coverage.append(dict(job,turns=len(rows),complete=done))
        if done:
            if [r['uid'] for r in rows]!=expected:raise ValueError('Not the full corpus')
            if any(r['seed']!=job['seed'] or r['policy']!=job['policy'] for r in rows):raise ValueError('Policy/seed mismatch')
            v=json.loads(validated.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest()!=v['records_sha256']:raise ValueError('Raw checksum mismatch')
            raw[(job['policy'],job['seed'])]=question_counts(rows,questions)
    (HERE/'coverage.json').write_text(json.dumps(coverage,indent=2))
    if not all(c['complete'] for c in coverage):
        print(json.dumps(coverage,indent=2))
        if args.partial:return
        raise ValueError('Refuse final inference on an incomplete campaign')
    policies=plan['policies'];seeds=plan['seeds'];groups=plan['groups']
    summaries={};by_seed={}
    for policy in policies:
        pooled=[r for seed in seeds for r in raw[policy,seed].values()]
        summaries[policy]=dict(overall_event_weighted=aggregate(pooled),
            groups={g:aggregate([r for r in pooled if r['group']==g]) for g in groups})
        keys=['al_hit','al_p1','al_p2','al_equal_phase','al_all_step_sensitivity','al_emitted_all_steps','hit_rate']
        summaries[policy]['equal_task_mean']={k:float(np.mean([summaries[policy]['groups'][g][k] for g in groups])) for k in keys}
    for seed in seeds:
        by_seed[str(seed)]={p:{g:aggregate([r for r in raw[p,seed].values() if r['group']==g]) for g in groups} for p in policies}
    # Paired, task-stratified question resampling; all turns/seeds stay in cluster.
    rng=np.random.default_rng(20260921);boot={p:defaultdict(list) for p in policies};per_group_ci={}
    for group in groups:
        ids=[q['question_id'] for q in questions if q['group']==group]
        indices=rng.integers(0,len(ids),(5000,len(ids)))
        values={}
        for policy in policies:
            a=np.array([[sum(raw[policy,s][i][k] for s in seeds) for k in FIELDS] for i in ids])
            sampled=a[indices].sum(1);ix={k:FIELDS.index(k) for k in FIELDS}
            div=lambda n,d:sampled[:,ix[n]]/sampled[:,ix[d]]
            metrics=dict(al_hit=div('hit_sum','hit_n'),al_p1=div('p1_sum','p1_n'),al_p2=div('p2_sum','p2_n'),
                         al_all_step_sensitivity=div('raw_hit_sum','raw_hit_n'),
                         al_emitted_all_steps=div('emitted_hit_sum','raw_hit_n'),hit_rate=div('raw_hit_n','raw_step_n'))
            metrics['al_equal_phase']=(metrics['al_p1']+metrics['al_p2'])/2
            values[policy]=metrics
            for k,v in metrics.items():
                if not np.all(np.isfinite(v)):raise ValueError('Undefined bootstrap ratio')
                boot[policy][k].append(v)
        per_group_ci[group]={k:dict(delta=summaries[policies[1]]['groups'][group][k]-summaries[policies[0]]['groups'][group][k],
            ci95=np.quantile(values[policies[1]][k]-values[policies[0]][k],[.025,.975]).tolist()) for k in values[policies[0]]}
    inference={}
    for k in boot[policies[0]]:
        base_boot=np.mean(boot[policies[0]][k],axis=0)
        candidate_boot=np.mean(boot[policies[1]][k],axis=0)
        delta=candidate_boot-base_boot
        point=summaries[policies[1]]['equal_task_mean'][k]-summaries[policies[0]]['equal_task_mean'][k]
        inference[k]=dict(delta=point,ci95=np.quantile(delta,[.025,.975]).tolist(),
            relative_change_pct=100*point/summaries[policies[0]]['equal_task_mean'][k],
            relative_ci95_pct=np.quantile(100*(candidate_boot/base_boot-1),[.025,.975]).tolist())
    # Explicit macro-question sensitivity on common estimable questions only.
    common=[];macro=[]
    for q in questions:
        qid=q['question_id'];rates=[]
        for policy in policies:
            nums=sum(raw[policy,s][qid]['hit_sum'] for s in seeds);den=sum(raw[policy,s][qid]['hit_n'] for s in seeds)
            rates.append(nums/den if den else None)
        if all(v is not None for v in rates):common.append(qid);macro.append(rates)
    result=dict(plan_sha256=hashlib.sha256((HERE/'plan.json').read_bytes()).hexdigest(),complete=True,coverage=coverage,
        primary='equal_task_mean.al_hit',summaries=summaries,paired_inference=inference,by_group_inference=per_group_ci,
        by_seed=by_seed,macro_question_sensitivity=dict(questions=len(common),excluded_undefined=480-len(common),
            baseline=float(np.mean(np.array(macro)[:,0])),candidate=float(np.mean(np.array(macro)[:,1]))),
        limitations=plan['limits'])
    (HERE/'analysis.json').write_text(json.dumps(result,indent=2))
    (HERE/'question_counts.json').write_text(json.dumps({f'{p}_s{s}':list(v.values()) for (p,s),v in raw.items()},indent=2))
    print(json.dumps(dict(primary={p:summaries[p]['equal_task_mean'] for p in policies},paired=inference,coverage=coverage),indent=2))


if __name__=='__main__':main()
