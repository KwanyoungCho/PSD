"""Native-chain phase timings and source-conditional P(AL >= x)."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
RAW=Path('/data/chokwans99/duet-chain40-k10-20261016')
sys.path.insert(0,str(ROOT/'ssd/tools/duet_timeline'))
from plot_consecutive_steps import read_profiles
from plot_model_wait_timelines import command_index
from export_profile_timelines import normalize_outcomes,preparation_intervals,preparation_affected
from phase_averages import sample,metrics,summarize


def survival(rows, draws=4000):
    """Inclusive survival. Bootstrap questions, keeping both passes together."""
    result=[];uids=sorted({r['uid'] for r in rows});thresholds=np.arange(1,12)
    weights=np.random.default_rng(9101).multinomial(len(uids),np.ones(len(uids))/len(uids),size=draws)
    for state in (1,2,0):
        group=[r for r in rows if r['state']==state]
        vals=np.asarray([r['accepted_len'] for r in group])
        assert len(vals)>0 and np.all((vals>=1)&(vals<=11))
        counts=np.asarray([sum(r['uid']==u for r in group) for u in uids])
        successes=np.asarray([[sum(r['uid']==u and r['accepted_len']>=x for r in group)
                               for x in thresholds] for u in uids])
        den=weights@counts;boot=(weights[den>0]@successes)/den[den>0,None]
        lo,hi=np.quantile(boot,[.025,.975],axis=0)
        points=[dict(x=int(x),count=int((vals>=x).sum()),probability=float((vals>=x).mean()),
                     lower95=float(l),upper95=float(h)) for x,l,h in zip(thresholds,lo,hi)]
        result.append(dict(state=state,steps=len(vals),questions_with_source=int((counts>0).sum()),
            mean_al=float(vals.mean()),histogram=dict(Counter(map(int,vals))),points=points,
            bootstrap_valid_draws=len(boot)))
    return result


def analyze(name):
    folder=RAW/name;run=json.loads((folder/'run.json').read_text());assert run['status']=='complete'
    outcomes=normalize_outcomes(run);groups,commands,hashes=read_profiles(folder)
    index=command_index(commands);preps=preparation_intervals(folder)
    expected={(q['epoch'],s['step_id']) for q in outcomes['requests'] for s in q['steps']}
    assert expected==set(groups['draft'])==set(groups['target'])
    shapes={}
    for path in folder.glob('chain_shapes_request_*.json'):
        data=json.loads(path.read_text())
        for item in data['records']:
            key=data['epoch'],item['step_id'];assert key not in shapes;shapes[key]=item
            assert item['ds_roots']==item['ps_roots']==item['ds_width']==item['ps_width']==40
            assert item['k1']==item['k2']==10
    assert set(shapes)==expected
    for key,events in groups['draft'].items():
        counts=Counter(e['label'] for e in events)
        assert counts['phase1_replay']==counts['phase2_replay']==10,(key,counts)
    rows=[];all_rows=[];skipped=Counter()
    for q in outcomes['requests']:
        ep=q['epoch'];steps=q['steps']
        for i,s in enumerate(steps):
            assert 1<=s['accepted_len']<=11 and s['verify_width']==10
            reason='warmup' if q['warmup'] else 'terminal' if i==len(steps)-1 else 'cap_clip' if s['cap'] or s['clipped'] else None
            if reason:skipped[reason]+=1;continue
            row=dict(epoch=ep,step_id=s['step_id'],uid=q['uid'],state=s['source'],next_state=steps[i+1]['source'])
            row.update({k:s[k] for k in ('accepted_len','emitted_len','context_tokens','verify_width')})
            all_rows.append(row.copy())
            if i<10:skipped['initial10']+=1;continue
            if preparation_affected(groups,ep,s['step_id'],preps):skipped['preparation']+=1;continue
            record=sample(groups,index,row,10,10,name);record.update(metrics(record))
            record['rounds']={phase:[(e['wall_end_ns']-e['wall_start_ns'])/1e6 for e in sorted(
                (e for e in groups['draft'][(ep,s['step_id'])] if e['label']==label),key=lambda e:e['wall_start_ns'])]
                for phase,label in [('ds','phase1_replay'),('ps','phase2_replay')]}
            assert all(len(a)==10 for a in record['rounds'].values())
            assert record['accepted_len']==record['emitted_len']
            rows.append(record)
    summaries=summarize(rows)
    for r in summaries:
        sub=[s for s in rows if r['state']=='all' or s['state']==r['state']]
        for phase in ('ds','ps'):
            r[phase+'_negative_margin_steps']=sum(s[phase+'_margin_ms']<0 for s in sub)
            r[phase+'_margin_p05_ms']=float(np.quantile([s[phase+'_margin_ms'] for s in sub],.05))
            r[phase+'_rounds_mean_ms']=np.mean([s['rounds'][phase] for s in sub],axis=0).tolist()
    for path in [folder/'run.json',folder/'COMMAND.json',folder/'RESOLVED_CONFIG.json',
                 *folder.glob('preparation_*.jsonl'),*folder.glob('chain_shapes_request_*.json')]:
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(model=name,args=run['args'],resolved=json.loads((folder/'RESOLVED_CONFIG.json').read_text()),
        eligibility=dict(included=len(rows),skipped=dict(skipped)),summary=summaries,
        survival=survival(rows),all_nonterminal_survival=survival(all_rows),rows=rows,
        source_hashes=hashes,measured_requests=len(run['groups']))


def main():
    runs=[analyze(n) for n in ('7b','13b') if (RAW/n/'run.json').exists() and json.loads((RAW/n/'run.json').read_text())['status']=='complete']
    assert runs
    (HERE/'SAMPLES.json.gz').write_bytes(gzip.compress(json.dumps(runs).encode(),mtime=0))
    (HERE/'SUMMARY.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in runs],indent=2)+'\n')
    for r in runs:print(r['model'],r['eligibility'],r['summary'][-1])


if __name__=='__main__':main()
