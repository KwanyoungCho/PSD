"""Prespecified no-probe timing screen on the frozen T=0.7 rules."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2];SSD=ROOT/'ssd'
CAMPAIGN=SSD/'experiments/proxy_source_ablation/probe_training_free_20260912'

def main():
    info=json.loads((CAMPAIGN/'datasets.json').read_text())['stages']['confirmation']
    dataset=HERE/'live_dataset/alpaca/alpaca_data_10000.jsonl'
    dataset.parent.mkdir(parents=True,exist_ok=True)
    dataset.write_text('\n'.join(Path(info['file']).read_text().splitlines()[:16])+'\n')
    frozen=json.loads((HERE/'frozen.json').read_text())
    policies={'legacy':'legacy','matched_proxy':'policy::proxy__topm__original',
              'optimized_proxy':frozen['selection']['0.7']['proxy_allocation'],
              'selected':frozen['selection']['0.7']['overall']}
    orders=[['legacy','matched_proxy','optimized_proxy','selected'],
            ['selected','optimized_proxy','matched_proxy','legacy']]
    out=HERE/'live';out.mkdir(exist_ok=True)
    plan={'temperature':.7,'seeds':[801,802],'policies':policies,'orders':orders,
          'num_prompts':16,'mapping':info['mapping'][:16],'output_len':256,
          'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
          'frozen_sha256':hashlib.sha256((HERE/'frozen.json').read_bytes()).hexdigest(),
          'note':'Small prespecified throughput screen; two seeds, reverse arm order. No distribution probes. All arms use identical prompts and output limits. Not a full Mirror-SD comparison.'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    while not (CAMPAIGN/'out/confirmation/mixed_t0.5_seed271.complete.json').exists():time.sleep(10)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='3,4,5,6,7',SSD_DATASET_DIR=str(dataset.parents[1]),
               SSD_PROFILE='0',SSD_PROFILE_DUET='0',SSD_PROFILE_DUET_DETAIL='0',
               SSD_TREE_EXEC='0',SSD_TREE_ARENA='0',SSD_TREE_PROXY_GRAPH='0',
               SSD_TREE_EXEC_WARMUP='0',SSD_DUET_EXIT_REPLICA='1',SSD_DUET_PROXY_ON_DRAFT='0',
               SSD_DUET_PROBE_LAYERS='',SSD_DUET_PROBE_OUT='',SSD_TF_T='0.7')
    for repeat,order in enumerate(orders):
        for arm in order:
            seed=801+repeat;tag=f'{arm}_seed{seed}'
            data=out/(tag+'.json');log=out/(tag+'.log');done=out/(tag+'.complete.json')
            if done.exists():continue
            if data.exists():raise RuntimeError('Incomplete live result: '+tag)
            env.update(SSD_TF_POLICY=policies[arm],SSD_TF_LIVE_OUT=str(data),SSD_DIST_PORT=str(20000+repeat*10+order.index(arm)))
            cmd=[str(SSD/'.venv/bin/python'),'-O',str(HERE/'runtime_policy.py'),
                 '--llama','--size','70','--gpus','5',
                 '--model_path','/home/chokwans99/awq_calibrated/layerskip_llama2_70b',
                 '--draft_path','/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0',
                 '--quant_awq','--quant_awq_artifact','/home/chokwans99/awq_artifacts/layerskip70b_awq_tp4',
                 '--alpaca','--numseqs','16','--output_len','256','--b','1','--temp','.7','--seed',str(seed),
                 '--async','--spec','--duet','--duet_exit_layer','56',
                 '--duet_phase1_k','8','--duet_phase2_k','4','--duet_draft_fan_out','3','--duet_p2_budget','15',
                 '--duet_p1_tree_policy','off','--duet_p2_tree_policy','off','--duet_only_proxy','--duet_proxy_source','proxy']
            print('[run] '+tag,flush=True);start=time.time()
            with log.open('w') as f:r=subprocess.run(cmd,cwd=SSD,env=env,stdout=f,stderr=subprocess.STDOUT)
            text=log.read_text(errors='replace')
            if r.returncode or not data.exists() or 'Final Decode Throughput:' not in text or 'Traceback (most recent call last):' in text:
                raise RuntimeError(str(log))
            result=json.loads(data.read_text())
            if result['prompts']!=16 or result['output_tokens']<=0:raise RuntimeError('Invalid live results')
            record={'arm':arm,'seed':seed,'elapsed_s':time.time()-start,'command':cmd,
                    'runtime_policy_sha256':hashlib.sha256((HERE/'runtime_policy.py').read_bytes()).hexdigest()}
            done.write_text(json.dumps(record,indent=2));print('[done] '+tag,flush=True)

if __name__=='__main__':main()
