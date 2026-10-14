"""Sequential, fail-closed 70B native functional gate; AR control runs last."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',required=True)
    p.add_argument('--draft',required=True)
    p.add_argument('--base-target',help='Optional original 70B checkpoint: greedy tree/AR native-loader gate')
    p.add_argument('--gpus',default='0,1,2')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    devices=a.gpus.split(',')
    if len(devices)!=3 or len(set(devices))!=3:raise ValueError('Exactly three different GPUs required')
    report=dict(status='running',runs=[],started=time.time())
    def save():
        tmp=out/'status.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(out/'status.json')
    jobs=[('tree','last',0,'full',1),('tree','frontier',0,'topm',1),
          ('tree','last',.7,'full',1),('tree','frontier',.7,'topm',1),
          ('tree','last',.7,'full',2),('ar','last',0,'full',1)]
    env=dict(os.environ,PYTHONPATH=str(ROOT/'ssd'),SSD_HF_CACHE='/tmp',SSD_DATASET_DIR='/tmp',
             SSD_CUDA_ARCH='12.0',SSD_DEBUG_TOKENS='0',SSD_TREE_EXEC_WARMUP='all',
             SSD_DIST_PORT='29895',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    for name in ('SSD_TREE_EXPANSION_POLICY','SSD_TREE_LEGACY_CALIBRATION','LD_PRELOAD'):
        env.pop(name,None)
    jobs=[(a.target,'layerskip',*job) for job in jobs]
    if a.base_target:
        jobs += [(a.base_target,'base70b','tree','last',0,'full',1),
                 (a.base_target,'base70b','ar','last',0,'full',1)]
    for target,model,mode,expansion,temp,norm,batch in jobs:
        name=f'{model}_{mode}_{expansion}_t{temp}_{norm}_b{batch}'
        gpus=','.join(devices if mode=='tree' else devices[:2])
        cmd=[sys.executable,'-B',str(HERE/'probe.py'),'--target',target,'--draft',a.draft,
             '--mode',mode,'--expansion',expansion,'--temperature',str(temp),'--normalization',norm,
             '--batch',str(batch),'--output',str(out/f'{name}.json')]
        row=dict(name=name,command=cmd,gpus=gpus,started=time.time());report['runs'].append(row);save()
        with (out/f'{name}.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=gpus),
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row['pid']=proc.pid;save()
            try:code=proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
        row.update(returncode=code,ended=time.time());save()
        if code:
            report.update(status='failed',ended=time.time());save()
            raise RuntimeError(f'{name} failed ({code}); log retained')
    def outputs(name):
        r=json.loads((out/f'{name}.json').read_text())
        if r['status']!='complete' or not r['source_unchanged']:raise ValueError(f'Invalid result {name}')
        return r['inputs'],[x for g in r['groups'] for x in g['outputs']]
    parity=[]
    for model in (['layerskip','base70b'] if a.base_target else ['layerskip']):
        inputs,ar=outputs(model+'_ar_last_t0_full_b1')
        names=['tree_last_t0_full_b1']
        if model=='layerskip':names+=['tree_frontier_t0_topm_b1']
        for name in names:
            x,tokens=outputs(model+'_'+name)
            parity.append(dict(name=model+'_'+name,identical_inputs=x==inputs,prompts=len(tokens),
                               matches=sum(a==b for a,b in zip(ar,tokens)),same_count=len(tokens)==len(ar)))
    report.update(greedy_parity=parity,ended=time.time(),status='complete' if all(
        r['identical_inputs'] and r['same_count'] and r['matches']==r['prompts'] for r in parity) else 'failed')
    save()
    if report['status']!='complete':raise RuntimeError('Greedy AR parity failed')
    print(json.dumps(dict(status='complete',greedy_parity=parity)))


if __name__=='__main__':main()
