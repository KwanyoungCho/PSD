"""Full-question aggregation; paired question bootstrap, preserving seed cluster."""
from pathlib import Path
from collections import defaultdict
import json,csv
import numpy as np
HERE=Path(__file__).resolve().parent

def collect(report):
    n=len(report['question_indexes']); per=np.zeros((n,12)); samples=[]; full_batch=[]; miss_lengths=defaultdict(int)
    physical_queries=[]; node_by_source=[[],[],[]]
    for cell in report['cells']:
        events=cell['metrics']['phase_events']
        ids=sorted({e['seq_id'] for e in events})
        if len(ids)!=n:raise ValueError('Question/event identity mismatch')
        lookup={sid:j for j,sid in enumerate(ids)}
        for e in events:
            x=per[lookup[e['seq_id']]]
            x[0]+=e['accepted_len'];x[1]+=1;x[2]+=e['emitted_len']
            if not e.get('output_cap_reached') and not e.get('clipped'):
                x[3]+=e['accepted_len'];x[4]+=1
            source=e['source'];x[5+source]+=1;x[8+source]+=e['accepted_len']
            x[11]+=e['valid_k']
            node_by_source[source].append(e['valid_k'])
        samples.append(cell['summary'])
        steps=defaultdict(list)
        for e in events:
            steps[e['step_id']].append(e)
            if e['source']==0:miss_lengths[e['valid_k']]+=1
        full_batch.extend([list(v) for v in steps.values() if len(v)==cell['batch']])
        physical_queries.extend(v[0]['physical_batch_queries'] for v in steps.values()
                                if len(v)==cell['batch'] and 'physical_batch_queries' in v[0])
    x=per.sum(0)
    return dict(questions=n,question_indexes=report['question_indexes'],cells=len(samples),verify_events=int(x[1]),
        observed_miss_valid_k=dict(miss_lengths),
        full_batch_steps=len(full_batch),
        full_batch_all_hit_fraction=np.mean([all(e['cache_hit'] for e in step) for step in full_batch]) if full_batch else None,
        al=x[0]/x[1],emitted_al=x[2]/x[1],boundary_excluded_al=x[3]/x[4],
        old_counter_overcount_pct=100*(x[0]/x[2]-1),boundary_excluded_events=int(x[1]-x[4]),
        mean_valid_nodes=x[11]/x[1],miss_rate=x[5]/x[1],p1_hit=x[6]/x[1],p2_hit=x[7]/x[1],
        mean_valid_nodes_by_source=[float(np.mean(v)) if v else None for v in node_by_source],
        mean_full_batch_physical_queries=float(np.mean(physical_queries)) if physical_queries else None,
        al_by_source=[x[8+i]/x[5+i] if x[5+i] else None for i in range(3)],
        tps=[s['decode_tps'] for s in samples],
        tps_mean=np.mean([s['decode_tps'] for s in samples]),
        later_pass_tps_mean=np.mean([s['decode_tps'] for s in samples[1:]]) if len(samples)>1 else None,
        boundary_excluded_step_tps=[s['boundary_excluded_step_tps'] for s in samples],
        capped_requests=[sum(len(o['token_ids'])>=cap for o,cap in zip(c['outputs'],report['output_caps'])) for c in report['cells']],
        per_question=per.tolist())

groups=defaultdict(dict)
for path in sorted(HERE.glob('*/*.json')):
    folder=path.parent.name
    if not any(folder.endswith(s) for s in ['_full','_b1','_follow','_b1_short','_budget','_replica','_budget_control']):continue
    report=json.loads(path.read_text())
    if not isinstance(report,dict) or report.get('status')!='complete' or 'cells' not in report:continue
    model='llama2' if folder.startswith('llama2') else 'llama3'
    name=path.stem.removeprefix(model+'_')
    if folder.endswith(('_budget','_budget_control')):
        group=model+'_optimized'
    elif folder.endswith('_follow'):
        group=model+('_optimized' if name in ['baseline','fused','fused_bulk','fused_bulk_parallel'] else '_full')
    elif folder.endswith('_b1_short'):group=model+'_b1'
    else:group=folder
    row=collect(report)|dict(group=group,name=name,path=str(path.relative_to(HERE)),
        batch=report['cells'][0]['batch'],temperature=report['cells'][0]['temperature'],
        source_sha256=report['source_sha256'],git_diff_sha256=report['git_diff_sha256'])
    groups[group][name]=row
rows=[]
for group,reports in groups.items():
    baseline=reports.get('q_path') or reports.get('baseline') or reports.get('legacy_q_path')
    if baseline:
        n=len(baseline['per_question'])
        bootstrap=np.random.default_rng(20261007).integers(0,n,(2000,n))
        for row in reports.values():
            reference=reports.get('legacy_q_path',baseline) if row['name'].startswith('legacy_') else baseline
            if row['cells']!=reference['cells']:continue
            base=np.array(reference['per_question']);base_draw=base[bootstrap].sum(1)
            value=np.array(row['per_question']);draw=value[bootstrap].sum(1)
            delta=draw[:,0]/draw[:,1]-base_draw[:,0]/base_draw[:,1]
            clean=draw[:,3]/draw[:,4]-base_draw[:,3]/base_draw[:,4]
            row.update(reference=reference['name'],delta_al=row['al']-reference['al'],
                delta_al_ci95=np.quantile(delta,[.025,.975]).tolist(),
                delta_clean_al=row['boundary_excluded_al']-reference['boundary_excluded_al'],
                delta_clean_al_ci95=np.quantile(clean,[.025,.975]).tolist(),
                clean_al_relative_pct=100*(row['boundary_excluded_al']/reference['boundary_excluded_al']-1),
                al_relative_pct=100*(row['al']/reference['al']-1))
            h=np.array([row['miss_rate'],row['p1_hit'],row['p2_hit']])
            hb=np.array([reference['miss_rate'],reference['p1_hit'],reference['p2_hit']])
            av=np.array(row['al_by_source']); ab=np.array(reference['al_by_source'])
            row['al_decomposition']=dict(
                source_order=['miss','p1','p2'],
                mixture_term=((h-hb)*(av+ab)/2).tolist(),
                conditional_al_term=((av-ab)*(h+hb)/2).tolist(),
                note='Exact arithmetic decomposition of untrimmed AL, not a same-prefix causal estimate')
            # Prespecified calibration questions are removed from BOTH arms.
            eligible=[j for j,q in enumerate(row['question_indexes']) if q not in [0,60,120,180,240,300,360,420]]
            if eligible:
                ix=np.random.default_rng(20261008).choice(eligible,(2000,len(eligible)))
                v=value[eligible].sum(0);b=base[eligible].sum(0)
                vv=value[ix].sum(1);bb=base[ix].sum(1)
                row['heldout472']=dict(questions=len(eligible),al=v[0]/v[1],reference_al=b[0]/b[1],
                    relative_pct=100*((v[0]/v[1])/(b[0]/b[1])-1),
                    delta_ci95=np.quantile(vv[:,0]/vv[:,1]-bb[:,0]/bb[:,1],[.025,.975]).tolist(),
                    boundary_excluded_al=v[3]/v[4],reference_boundary_excluded_al=b[3]/b[4],
                    clean_relative_pct=100*((v[3]/v[4])/(b[3]/b[4])-1),
                    clean_delta_ci95=np.quantile(vv[:,3]/vv[:,4]-bb[:,3]/bb[:,4],[.025,.975]).tolist())
    rows.extend(reports.values())
(HERE/'RESULTS.json').write_text(json.dumps(rows,indent=2))
fields=['group','name','questions','cells','verify_events','al','emitted_al','boundary_excluded_al','mean_valid_nodes','miss_rate','p1_hit','p2_hit','tps_mean','al_relative_pct','delta_al_ci95']
with (HERE/'RESULTS.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
for r in rows:print(r['group'],r['name'],r['cells'],round(r['al'],4),round(r['tps_mean'],1),r.get('delta_al_ci95'))
