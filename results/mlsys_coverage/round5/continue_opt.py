"""Wait for exclusive GPU lane; require exact profile parity before full runs."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from result_metrics import read,collect
from make_plans import HERE,save


def parity(a,b):
    if a['question_indexes']!=b['question_indexes']:raise ValueError('Different questions')
    if len(a['cells'])!=len(b['cells']):raise ValueError('Different passes')
    keys=('seq_id','step_id','accepted_len','emitted_len','valid_k','source','cache_hit',
          'output_cap_reached','clipped')
    for x,y in zip(a['cells'],b['cells']):
        if [o['token_ids'] for o in x['outputs']]!=[o['token_ids'] for o in y['outputs']]:
            raise ValueError('Token output mismatch')
        ex=[tuple(e[k] for k in keys) for e in x['metrics']['phase_events']]
        ey=[tuple(e[k] for k in keys) for e in y['metrics']['phase_events']]
        if ex!=ey:raise ValueError('Proposal/acceptance event mismatch')
    collect(a);collect(b)
    return dict(token_outputs_equal=True,proposal_acceptance_events_equal=True,passes=len(a['cells']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--lane',type=int,choices=[0,1],required=True)
    p.add_argument('--gpus',required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
    # Final r1 on each lane is the last main campaign that owns these GPUs.
    b=8 if a.lane==0 else 1
    directory=HERE/f'final_{a.model}_b{b}_r1';manifest=directory/'campaign.json'
    plan=HERE/f'{a.model}_b{b}_final_r1_plan.json'
    while True:
        if manifest.exists() and plan.exists():
            try:rows=read(manifest);expected=len(read(plan))
            except json.JSONDecodeError:time.sleep(2);continue
            if any(r['status']!='complete' for r in rows):raise RuntimeError('Final prerequisite failed')
            if len(rows)==expected:break
        time.sleep(5)
    subprocess.run([sys.executable,str(HERE/'make_opt_plan.py'),'--model',a.model,'--lane',str(a.lane)],check=True)
    prefix=f'{a.model}_opt_l{a.lane}'
    for stage in ('profile','full'):
        future=HERE/f'{prefix}_{stage}_plan.json';dest=HERE/f'{prefix}_{stage}'
        subprocess.run([sys.executable,str(HERE.parent/'round4/after_campaign.py'),
            '--after',str(manifest),'--count',str(expected),'--plan',str(future),
            '--directory',str(dest),'--gpus',a.gpus,'--port',str(a.port)],check=True)
        if stage=='profile':
            jobs=read(future)
            check=parity(read(dest/(jobs[0]['name']+'.json')),read(dest/(jobs[1]['name']+'.json')))
            save(f'{prefix}_PARITY.json',check)
            manifest=dest/'campaign.json';expected=len(jobs)
    print('OPT_COMPLETE',prefix,flush=True)

if __name__=='__main__':main()
