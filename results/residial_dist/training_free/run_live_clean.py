"""Repeat the main allocation contrast with no concurrent GPU2 analysis."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent;SSD=HERE.parents[2]/'ssd';OUT=HERE/'live_clean'

def main():
    OUT.mkdir(exist_ok=True)
    frozen=json.loads((HERE/'frozen.json').read_text())
    policies={'matched_proxy':'policy::proxy__topm__original',
              'optimized_proxy':frozen['selection']['0.7']['proxy_allocation']}
    orders=[['matched_proxy','optimized_proxy'],['optimized_proxy','matched_proxy']]
    plan={'temperature':.7,'seeds':[903,904],'policies':policies,'orders':orders,'num_prompts':16,
          'reason':'Main allocation contrast, fresh sampling seeds, analysis GPU idle. Earlier runtime screen overlapped GPU2 analysis and is treated as exploratory.',
          'data':'Same 16 prompt texts as the pilot; new seeds, not a new unseen-prompt claim.'}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    while not (HERE/'live/original_residual_seed802.complete.json').exists():time.sleep(5)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='3,4,5,6,7',
      SSD_DATASET_DIR=str(HERE/'live_dataset'),SSD_PROFILE='0',SSD_PROFILE_DUET='0',SSD_PROFILE_DUET_DETAIL='0',
      SSD_TREE_EXEC='0',SSD_TREE_ARENA='0',SSD_TREE_PROXY_GRAPH='0',SSD_TREE_EXEC_WARMUP='0',
      SSD_DUET_EXIT_REPLICA='1',SSD_DUET_PROXY_ON_DRAFT='0',SSD_DUET_PROBE_LAYERS='',SSD_DUET_PROBE_OUT='',SSD_TF_T='.7')
    for j,order in enumerate(orders):
        seed=903+j
        for arm in order:
            tag=f'{arm}_seed{seed}';done=OUT/(tag+'.complete.json');data=OUT/(tag+'.json');log=OUT/(tag+'.log')
            if done.exists():continue
            cmd=json.loads((HERE/f'live/{arm}_seed801.complete.json').read_text())['command']
            cmd[cmd.index('--seed')+1]=str(seed)
            env.update(SSD_TF_POLICY=policies[arm],SSD_TF_LIVE_OUT=str(data),SSD_DIST_PORT=str(20200+2*j+order.index(arm)))
            idle=subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
            used,util=map(int,idle.split(','))
            if used>100 or util>0:raise RuntimeError('Analysis GPU is not idle: '+idle)
            before=hashlib.sha256((HERE/'runtime_policy.py').read_bytes()).hexdigest()
            print('[run]',tag,flush=True);start=time.time()
            with log.open('w') as f:r=subprocess.run(cmd,cwd=SSD,env=env,stdout=f,stderr=subprocess.STDOUT)
            text=log.read_text(errors='replace')
            if r.returncode or not data.exists() or 'Traceback (most recent call last):' in text or 'Final Decode Throughput:' not in text:
                raise RuntimeError(str(log))
            assert before==hashlib.sha256((HERE/'runtime_policy.py').read_bytes()).hexdigest()
            done.write_text(json.dumps({'arm':arm,'seed':seed,'elapsed_s':time.time()-start,'command':cmd,
                        'runtime_policy_sha256':before,'gpu2_before':idle,'no_concurrent_analysis':True},indent=2))
            print('[done]',tag,flush=True)

if __name__=='__main__':main()
