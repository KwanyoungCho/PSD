"""Separate side-stream intervention after the trim-only matched controls."""
import argparse,json,subprocess,sys,time
from make_plans import HERE,save
from make_warm_plan import from_row
from analyze_screen import analyze
from result_metrics import read
from continue_opt import parity


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--lane',type=int,choices=[0,1],required=True);p.add_argument('--gpus',required=True)
    p.add_argument('--port',type=int,required=True);a=p.parse_args()
    prefix=f'{a.model}_stream_l{a.lane}';prior=HERE/f'{a.model}_opt_l{a.lane}_full'
    manifest=prior/'campaign.json';plan=prior/'plan.json'
    while True:
        if manifest.exists() and plan.exists():
            try:rows=read(manifest);expected=len(read(plan))
            except json.JSONDecodeError:time.sleep(2);continue
            if any(r['status']!='complete' for r in rows):raise RuntimeError('Trim prerequisite failed')
            ready=not(a.model=='llama2' and a.lane==1) or (HERE/'FINAL_REGRESSIONS.json').exists()
            if len(rows)==expected and ready:break
        time.sleep(5)
    profiles=[];full=[]
    for b in (1,8):
        frozen=read(HERE/f'{a.model}_b{b}_FROZEN.json')['presets']['duet_fast']
        row=analyze(HERE/frozen['origin'])
        for stream in (0,1):
            extra=['--max-new-tokens','128','--seeds','6800']
            if b==1 and stream==0:extra+=['--preflight-test','tests.test_root_policy,tests.test_phase_budget']
            item=from_row(row,f'stream_l{a.lane}_profile_s{stream}',profile=True,extra=extra,
                env={'SSD_TREE_LADDER_TRIM':'1','SSD_BATCH_TREE_PROXY_STREAM':str(stream),
                     'SSD_PROFILE_DUET_DETAIL':'1','SSD_SEED':'0'})
            item.update(stage='stream_profile',batch=b,stream=stream);profiles.append(item)
        rep=0 if ((b==1)==(a.lane==0)) else 1;seed=6100+rep*100
        item=from_row(row,f'stream_fast_r{rep}',extra=[
            '--prompts',str(HERE.parent/'questions.json'),'--limit','0',
            '--max-new-tokens','128','--seeds',str(seed),str(seed+1)],
            env={'SSD_TREE_LADDER_TRIM':'1','SSD_BATCH_TREE_PROXY_STREAM':'1','SSD_SEED':str(41+rep),
                 'SSD_PROFILE_DUET':'0','SSD_TREE_SHAPE_METRICS':'0'})
        item.update(stage='stream_full480',role='duet_fast',batch=b,replicate=rep);full.append(item)
    smoke=from_row(row,f'stream_l{a.lane}_greedy_smoke',extra=['--limit','8','--max-new-tokens','24','--temperatures','0'],
        env={'SSD_TREE_LADDER_TRIM':'1','SSD_BATCH_TREE_PROXY_STREAM':'1','SSD_SEED':'0'})
    smoke.update(stage='stream_greedy_smoke',batch=8);profiles.append(smoke)
    for stage,jobs in [('profile',profiles),('full',full)]:
        future=HERE/f'{prefix}_{stage}_plan.json';save(future.name,jobs);dest=HERE/f'{prefix}_{stage}'
        subprocess.run([sys.executable,str(HERE.parent/'round4/after_campaign.py'),
            '--after',str(manifest),'--count',str(expected),'--plan',str(future),
            '--directory',str(dest),'--gpus',a.gpus,'--port',str(a.port)],check=True)
        if stage=='profile':
            checks=[]
            for b in (1,8):
                pair=[j for j in jobs if j.get('batch')==b and j['stage']=='stream_profile']
                checks.append(dict(batch=b,**parity(read(dest/(pair[0]['name']+'.json')),read(dest/(pair[1]['name']+'.json')))))
            save(f'{prefix}_PARITY.json',checks)
            manifest=dest/'campaign.json';expected=len(jobs)
        else:
            checks=[]
            for job in jobs:
                baseline=prior/f"{a.model}_b{job['batch']}_opt_fast_r{job['replicate']}.json"
                checks.append(dict(batch=job['batch'],**parity(read(dest/(job['name']+'.json')),read(baseline))))
            save(f'{prefix}_FULL_PARITY.json',checks)
    print('STREAM_COMPLETE',prefix,flush=True)

if __name__=='__main__':main()
