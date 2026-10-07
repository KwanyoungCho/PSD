"""Close a small tuning-only neighborhood after overlap changes deadlines.

Lane0 profiles four causal neighbors and warm-checks finalists. Lane1 waits
for that immutable decision. A retained implementation/parameter point
reuses its existing paired full480 controls; a new point gets both repeats.
"""
import argparse,json,subprocess,sys,time
from make_plans import HERE,save
from make_warm_plan import from_row
from analyze_screen import analyze
from make_tree_plan import kwargs_from
from result_metrics import read


def wait_campaign(directory):
    while True:
        mp=directory/'campaign.json';pp=directory/'plan.json'
        if mp.exists() and pp.exists():
            try:rows=read(mp);jobs=read(pp)
            except json.JSONDecodeError:time.sleep(2);continue
            if any(r['status']!='complete' for r in rows):raise RuntimeError(f'Prerequisite failed: {directory}')
            if len(rows)==len(jobs):return mp,len(jobs)
        time.sleep(5)


def run(a,stage,jobs,after,count):
    name=f'{a.model}_postopt_l{a.lane}_{stage}'
    plan=HERE/f'{name}_plan.json';directory=HERE/name
    save(plan.name,jobs)
    subprocess.run([sys.executable,str(HERE.parent/'round4/after_campaign.py'),
        '--after',str(after),'--count',str(count),'--plan',str(plan),'--directory',str(directory),
        '--gpus',a.gpus,'--port',str(a.port)],check=True)
    return directory


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--lane',type=int,choices=[0,1],required=True)
    p.add_argument('--gpus',required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
    after,count=wait_campaign(HERE/f'{a.model}_stream_l{a.lane}_full')
    # GPU4 belongs to the final stream-enabled regression before reuse here.
    if a.model=='llama3' and a.lane==0:
        while not (HERE/'FINAL_STREAM_REGRESSIONS.json').exists():time.sleep(5)
    frozen_path=HERE/f'{a.model}_POSTOPT_FROZEN.json'
    if a.lane==0:
        if frozen_path.exists():raise FileExistsError('Post-optimization choices are immutable')
        base={};profiles=[]
        for b in (1,8):
            path=HERE/f'{a.model}_stream_l0_profile/{a.model}_b{b}_stream_l0_profile_s1.json'
            base[b]=analyze(path)
            for tag,change in [('k1_down',{'k1':max(1,base[b]['args']['k1']-1)}),
                               ('k2_down',{'k2':max(1,base[b]['args']['k2']-1)}),
                               ('exit_earlier',{'exit_layer':max(0,base[b]['args']['exit_layer']-2)}),
                               ('exit_later',{'exit_layer':min(31,base[b]['args']['exit_layer']+2)})]:
                if all(base[b]['args'][k]==v for k,v in change.items()):continue
                row=dict(base[b],args=dict(base[b]['args'],**change))
                item=from_row(row,'postopt_profile_'+tag,profile=True,
                    extra=['--max-new-tokens','128','--seeds','6800'],
                    env={'SSD_PROFILE_DUET_DETAIL':'1','SSD_SEED':'0'})
                item.update(stage='postopt_profile',reason='Overlap advances target completion; check shorter phase or neighboring exit',batch=b)
                profiles.append(item)
        directory=run(a,'profile',profiles,after,count);after,count=wait_campaign(directory)
        warm=[]
        for b in (1,8):
            rows=[analyze(directory/(j['name']+'.json')) for j in profiles if j['batch']==b]
            picked=sorted(rows,key=lambda r:r.get('profile_steady_tps') or 0,reverse=True)[:2]
            picked+=[base[b]]
            control=analyze(HERE/f'{a.model}_stream_l0_profile/{a.model}_b{b}_stream_l0_profile_s0.json')
            picked+=[control]
            for i,row in enumerate(picked):
                item=from_row(row,f'postopt_warm_{i}',extra=['--max-new-tokens','128','--seeds','6900','6901'],
                    env={'SSD_PROFILE_DUET':'0','SSD_TREE_SHAPE_METRICS':'0','SSD_SEED':'19'})
                item.update(stage='postopt_warm',batch=b,baseline=i==2,trim_control=i==3)
                warm.append(item)
        directory=run(a,'warm',warm,after,count);after,count=wait_campaign(directory)
        decisions={}
        for b in (1,8):
            candidates=[(j,analyze(directory/(j['name']+'.json'))) for j in warm if j['batch']==b]
            basejob,baserow=next((j,r) for j,r in candidates if j['baseline'])
            winnerjob,winner=max(candidates,key=lambda jr:jr[1]['summary']['boundary_excluded_step_tps'])
            ratio=winner['summary']['boundary_excluded_step_tps']/baserow['summary']['boundary_excluded_step_tps']
            # Prespecified small-gain guard; not a significance test.
            if ratio<1.02:winnerjob,winner=basejob,baserow
            same=kwargs_from(winner)==kwargs_from(base[b])
            stream=winner['env'].get('SSD_BATCH_TREE_PROXY_STREAM','0')=='1'
            decisions[str(b)]=dict(origin=winner['path'],args=winner['args'],env=winner['env'],
                tuning_tps=winner['summary']['boundary_excluded_step_tps'],tuning_al=winner['summary']['boundary_excluded_al'],
                observed_best_ratio=ratio,selected_existing=('stream' if stream else 'trim') if same else None,
                rule='Top two causal-neighbor profiles plus stream/trim baseline warm48; require >=2% over stream baseline to switch. This is a noise guard, not a confidence test.')
        save(frozen_path.name,dict(model=a.model,selection='tuning48 only after implementation changes',presets=decisions))
    else:
        while not frozen_path.exists():time.sleep(5)
    decisions=read(frozen_path)['presets'];jobs=[]
    for b in (1,8):
        point=decisions[str(b)]
        if point['selected_existing']:continue
        row=analyze(HERE/point['origin']);rep=0 if ((b==1)==(a.lane==0)) else 1;seed=6100+rep*100
        item=from_row(row,f'postopt_final_r{rep}',extra=['--prompts',str(HERE.parent/'questions.json'),
            '--limit','0','--max-new-tokens','128','--seeds',str(seed),str(seed+1)],
            env={'SSD_SEED':str(41+rep),'SSD_PROFILE_DUET':'0','SSD_TREE_SHAPE_METRICS':'0'})
        item.update(stage='postopt_full480',batch=b,replicate=rep);jobs.append(item)
    if jobs:
        directory=run(a,'full',jobs,after,count);wait_campaign(directory)
    save(f'{a.model}_postopt_l{a.lane}_DONE.json',dict(status='complete',new_full_jobs=len(jobs)))

if __name__=='__main__':main()
