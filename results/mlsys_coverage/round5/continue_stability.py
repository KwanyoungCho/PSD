"""B8-only protocol amendment after a tuning-step tail audit.

Use six passes in one process, three warm then three measured. Choose the
median of whole-pass unfiltered TPS*, never delete slow individual steps.
Lane0 does selection; lane1 obtains exact-point diagnostic profiles. Both
wait for their older work to finish, then validate a new winner on full480.
"""
import argparse,copy,json,subprocess,sys,time
import numpy as np
from make_plans import HERE,save
from make_warm_plan import from_row
from make_tree_plan import kwargs_from
from analyze_screen import analyze
from result_metrics import read,collect
from continue_postopt import wait_campaign


def signature(row):
    return json.dumps(dict(params=kwargs_from(row),trim=row['env'].get('SSD_TREE_LADDER_TRIM','0'),
        stream=row['env'].get('SSD_BATCH_TREE_PROXY_STREAM','0')),sort_keys=True)


def candidates(model):
    rows=[]
    # Original fast trim/stream controls and the two previously profiled
    # post-optimization finalists. Read plans, not final480 scores.
    for job in read(HERE/f'{model}_postopt_l0_warm_plan.json'):
        if job['batch']==8:
            row=analyze(HERE/job['origin'])
            row['stability_baseline']=job['trim_control']
            rows.append(row)
    # The original L3 short-P2 finalists also had large timing tails. Test
    # both implementations so K2 and stream effects are not conflated.
    if model=='llama3':
        for name in ('warm_duet_tree_1','warm_duet_tree_2'):
            origin=analyze(HERE/f'{model}_b8_warm/{model}_b8_{name}.json')
            for stream in ('0','1'):
                row=copy.deepcopy(origin)
                row['env'].update(SSD_TREE_LADDER_TRIM='1',SSD_BATCH_TREE_PROXY_STREAM=stream)
                row['stability_baseline']=False;rows.append(row)
    unique={}
    for row in rows:unique.setdefault(signature(row),row)
    return list(unique.values())


def tail(cell):
    steps=[s for s in cell['metrics']['decode_steps'] if not(s['output_cap_reached'] or s['clipped'])]
    seconds=np.array([s['seconds'] for s in steps]);mask=seconds>5*np.median(seconds)
    return dict(clean_steps=len(steps),median_ms=float(np.median(seconds)*1000),
        max_ms=float(seconds.max()*1000),over_5median_steps=int(mask.sum()),
        over_5median_time_fraction=float(seconds[mask].sum()/seconds.sum()),
        tps=cell['summary']['boundary_excluded_step_tps'])


def run(a,stage,jobs):
    name=f'{a.model}_stability_l{a.lane}_{stage}';plan=HERE/f'{name}_plan.json'
    save(plan.name,jobs)
    subprocess.run([sys.executable,'ssd/bench/mlsys_campaign.py','--plan',str(plan),
        '--directory',str(HERE/name),'--gpus',a.gpus,'--port',str(a.port)],check=True)
    wait_campaign(HERE/name)
    return HERE/name


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--lane',type=int,choices=(0,1),required=True)
    p.add_argument('--gpus',required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
    while not(HERE/f'{a.model}_postopt_l{a.lane}_DONE.json').exists():time.sleep(5)
    rows=candidates(a.model);jobs=[]
    frozen=HERE/f'{a.model}_STABILITY_FROZEN.json'
    if a.lane==0:
        if frozen.exists():raise FileExistsError('Stability choices are immutable')
        for i,row in enumerate(rows):
            item=from_row(row,f'stability_warm_{i}',extra=['--max-new-tokens','128','--seeds',*map(str,range(7000,7006))],
                env={'SSD_PROFILE_DUET':'0','SSD_TREE_SHAPE_METRICS':'0','SSD_SEED':'29'})
            item.update(stage='stability_warm',batch=8,baseline=row['stability_baseline'],
                diagnostic_origin=f'{a.model}_stability_l1_profile/{a.model}_b8_stability_profile_{i}.json')
            jobs.append(item)
        # SSD has no large-tail problem in its original warm finalists. This
        # repeated-pass control verifies that on the same new protocol.
        ssd=read(HERE/f'{a.model}_b8_FROZEN.json')['presets']['ssd_fast']
        jobs.append(from_row(analyze(HERE/ssd['origin']), 'stability_ssd_control',
            extra=['--max-new-tokens','128','--seeds',*map(str,range(7000,7006))],
            env={'SSD_PROFILE_DUET':'0','SSD_TREE_SHAPE_METRICS':'0','SSD_SEED':'29'}))
        directory=run(a,'warm',jobs);scores=[]
        for job in jobs:
            raw=read(directory/(job['name']+'.json'))
            if len(raw['cells'])!=6:raise ValueError('Six passes required')
            for i in range(6):collect(raw,i)
            stats=[tail(c) for c in raw['cells']];measured=stats[3:]
            scores.append(dict(name=job['name'],origin=str((directory/(job['name']+'.json')).relative_to(HERE)),
                args=raw['args'],env=raw['env'],baseline=job.get('baseline',False),
                diagnostic_origin=job.get('diagnostic_origin'),passes=stats,
                median_tps=float(np.median([c['tps'] for c in measured])),
                measured_tps_range=[min(c['tps'] for c in measured),max(c['tps'] for c in measured)],
                median_al=float(np.median([c['summary']['boundary_excluded_al'] for c in raw['cells'][3:]]))))
        group=[r for r in scores if r['args']['mode']=='duet-tree'];base=next(r for r in group if r['baseline'])
        winner=max(group,key=lambda r:r['median_tps']);ratio=winner['median_tps']/base['median_tps']
        if ratio<1.02:winner=base
        selected=dict(winner,tuning_tps=winner['median_tps'],tuning_al=winner['median_al'],
            observed_best_ratio=ratio,rule='Three warm + three measured tuning48 passes; median whole-pass TPS*, >=2% over original trim control to change; no step-tail deletion.',
            selected_existing=None)
        # Reuse full validation iff algorithm parameters AND implementation
        # switches match, irrespective of the scores of those full runs.
        for kind,source in [('trim',f'{a.model}_stream_l0_profile/{a.model}_b8_stream_l0_profile_s0.json'),
                            ('stream',f'{a.model}_stream_l0_profile/{a.model}_b8_stream_l0_profile_s1.json')]:
            if signature(winner)==signature(analyze(HERE/source)):selected['selected_existing']=kind
        old=read(HERE/f'{a.model}_POSTOPT_FROZEN.json')['presets']['8']
        if selected['selected_existing'] is None and signature(winner)==signature(old):selected['selected_existing']='postopt'
        save(frozen.name,dict(model=a.model,batch=8,selection='Tuning48-only stability amendment; original decisions retained',
            candidates=scores,preset=selected))
    else:
        for i,row in enumerate(rows):
            item=from_row(row,f'stability_profile_{i}',profile=True,extra=['--max-new-tokens','128','--seeds','7100'],
                env={'SSD_PROFILE_DUET_DETAIL':'1','SSD_SEED':'0'})
            item.update(stage='stability_profile',batch=8);jobs.append(item)
        run(a,'profile',jobs)
        while not frozen.exists():time.sleep(5)
    point=read(frozen)['preset'];rep=1 if a.lane==0 else 0
    if point['selected_existing'] is None:
        row=analyze(HERE/point['origin']);seed=6100+rep*100
        item=from_row(row,f'stability_final_r{rep}',extra=['--prompts',str(HERE.parent/'questions.json'),
            '--limit','0','--max-new-tokens','128','--seeds',str(seed),str(seed+1)],
            env={'SSD_PROFILE_DUET':'0','SSD_TREE_SHAPE_METRICS':'0','SSD_SEED':str(41+rep)})
        item.update(stage='stability_full480',batch=8,replicate=rep)
        run(a,'full',[item])
    save(f'{a.model}_stability_l{a.lane}_DONE.json',dict(status='complete',new_full_jobs=int(point['selected_existing'] is None)))


if __name__=='__main__':main()
