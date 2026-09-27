"""Secondary live control: the existing unscaled residual rule, same hardware."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];SSD=ROOT/'ssd';OUT=HERE/'live'

def main():
    plan={'arm':'original_residual','seeds':[801,802],'temperature':.7,'candidate_temperature':1.,
          'source':'residual','policy':'legacy','note':'Canonical original DUET PS rule, added as a secondary runtime control after the first four-arm repetition. No parameter tuning or winner selection.'}
    (OUT/'original_residual_plan.json').write_text(json.dumps(plan,indent=2))
    while not (OUT/'legacy_seed802.complete.json').exists():time.sleep(5)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='3,4,5,6,7',
      SSD_DATASET_DIR=str(HERE/'live_dataset'),SSD_PROFILE='0',SSD_PROFILE_DUET='0',SSD_PROFILE_DUET_DETAIL='0',
      SSD_TREE_EXEC='0',SSD_TREE_ARENA='0',SSD_TREE_PROXY_GRAPH='0',SSD_TREE_EXEC_WARMUP='0',
      SSD_DUET_EXIT_REPLICA='1',SSD_DUET_PROXY_ON_DRAFT='0',SSD_DUET_PROBE_LAYERS='',SSD_DUET_PROBE_OUT='',
      SSD_TF_T='.7',SSD_TF_POLICY='legacy')
    for seed in [801,802]:
        tag=f'original_residual_seed{seed}';done=OUT/(tag+'.complete.json');log=OUT/(tag+'.log');data=OUT/(tag+'.json')
        if done.exists():continue
        original=json.loads((OUT/f'legacy_seed{seed}.complete.json').read_text());cmd=original['command']
        cmd[cmd.index('--duet_proxy_source')+1]='residual'
        env.update(SSD_TF_LIVE_OUT=str(data),SSD_DIST_PORT=str(20100+seed-801))
        before=hashlib.sha256((HERE/'runtime_policy.py').read_bytes()).hexdigest()
        print('[run]',tag,flush=True);start=time.time()
        with log.open('w') as f:r=subprocess.run(cmd,cwd=SSD,env=env,stdout=f,stderr=subprocess.STDOUT)
        text=log.read_text(errors='replace')
        if r.returncode or not data.exists() or 'Traceback (most recent call last):' in text or 'Final Decode Throughput:' not in text:
            raise RuntimeError(str(log))
        if 'captured target chain proxy graphs (K=[4, 8], source=residual)' not in text:raise RuntimeError('Wrong source')
        assert before==hashlib.sha256((HERE/'runtime_policy.py').read_bytes()).hexdigest()
        done.write_text(json.dumps({'arm':'original_residual','seed':seed,'elapsed_s':time.time()-start,
                                  'command':cmd,'runtime_policy_sha256':before,'secondary_control':True},indent=2))
        print('[done]',tag,flush=True)

if __name__=='__main__':main()
