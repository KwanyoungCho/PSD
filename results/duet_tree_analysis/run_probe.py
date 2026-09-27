"""Launch one bounded unpruned tree diagnostic; preserve all prior artifacts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    source=json.loads((ROOT/'results/duet_calibration/plan.json').read_text())
    mapping=[];items=[]
    for split in ['calibration','validation']:
        info=source['datasets'][split]
        texts=Path(info['file']).read_text().splitlines()
        for text,meta in zip(texts,info['mapping']):
            items.append(text);mapping.append(dict(meta,split=split))
    data=HERE/'datasets/alpaca/alpaca_data_10000.jsonl';data.parent.mkdir(parents=True,exist_ok=True)
    data.write_text('\n'.join(items)+'\n')
    plan=dict(mapping=mapping,stride=4,seed=921,temperature=.7,output_len=128,
              config=dict(exit=56,k1=4,k2=2,c=3,roots=15,p1_nodes=8,p2_nodes=6,verify_equals_generated=True),
              dataset_scope='Disjoint within this analysis; reused finite prompt bank, NOT globally unseen research data.',
              analysis_prespecified=dict(q_bins=[0,.01,.03,.1,.3,.6,.9,1.00001],
                  estimators=['raw_q','phase_sibling_mean','phase_sibling_q_bin'],
                  shrinkage_attempt_mass=5,validation_reselection=False),
              purpose='Conditional acceptance and reach scores on existing served trees; not a TPS or new policy rollout trial.')
    plan_path=HERE/'probe_plan.json'
    if plan_path.exists():raise RuntimeError('Refuse to overwrite existing probe plan')
    plan_path.write_text(json.dumps(plan,indent=2))
    dest=HERE/'probe';dest.mkdir(exist_ok=True)
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('SSD_DUET_PROBE','SSD_E0_','SSD_CONF_','SSD_TREE_TOPO_TRACE','SSD_TREE_CALIB_TRACE','SSD_TREE_NODE_AUDIT')):env.pop(k)
    env.update(CUDA_VISIBLE_DEVICES='3,4,5,6,7',SSD_DATASET_DIR=str(data.parents[1]),
        SSD_HF_CACHE='/data/chokwans99/models',SSD_CUDA_ARCH='8.9',TORCH_CUDA_ARCH_LIST='8.9',
        SSD_ATTN_BACKEND='auto',SSD_PROFILE='0',SSD_PROFILE_DUET='0',SSD_TREE_EXEC='1',
        SSD_TREE_ARENA='1',SSD_TREE_PROXY_GRAPH='1',SSD_TREE_EXEC_WARMUP='1,2,3,4',SSD_DUET_EXIT_REPLICA='1',
        SSD_ASYNC_PROXY_SEND='1',SSD_PROXY_STREAM='0',SSD_DUET_PROXY_ON_DRAFT='0',SSD_CHAIN_PROXY_GRAPH='1',
        SSD_FORCE_SPLIT_K1K2='1',SSD_DUET_JIT_SHORT='1',SSD_TREE_VERIFY_WORKSPACE_MB='256',
        SSD_TREE_EXEC_WORKSPACE_MB='128',SSD_P1_TREE_EXEC_WORKSPACE_MB='128',
        DUET_TREE_AUDIT_OUT=str(dest),SSD_DIST_PORT='24921')
    cmd=[str(ROOT/'ssd/.venv/bin/python'),'-O',str(HERE/'collect.py'),
        '--llama','--size','70','--gpus','5','--model_path','/home/chokwans99/awq_calibrated/layerskip_llama2_70b',
        '--draft_path','/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0','--quant_awq',
        '--quant_awq_artifact','/home/chokwans99/awq_artifacts/layerskip70b_awq_tp4',
        '--alpaca','--numseqs',str(len(items)),'--output_len','128','--b','1','--temp','.7','--seed','921',
        '--max_model_len','2048','--async','--spec','--duet','--duet_exit_layer','56','--duet_k1','4',
        '--duet_k2','2','--duet_p1_fanout','3','--duet_p2_budget','15','--duet_p1_tree_policy','on',
        '--duet_p2_tree_policy','on','--duet_tree_root_count','15','--duet_tree_c_tensor','3',
        '--duet_p1_roots_per_position','3','--duet_p1_tree_max_nodes','8','--duet_p1_tree_verify_nodes','8',
        '--duet_p2_tree_max_nodes','6','--duet_p2_tree_verify_nodes','6']
    (dest/'command.json').write_text(json.dumps(dict(command=cmd,env={k:v for k,v in env.items() if k.startswith(('SSD_','DUET_','CUDA_VISIBLE'))}),indent=2))
    start=time.time()
    with (dest/'run.log').open('w') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT/'ssd',env=env,stdout=f,stderr=subprocess.STDOUT)
        (dest/'process.json').write_text(json.dumps(dict(pid=proc.pid,started=start)))
        code=proc.wait()
    (dest/'completion.json').write_text(json.dumps(dict(exit_code=code,wall_s=time.time()-start),indent=2))
    if code:raise RuntimeError(f'Probe failed, inspect {dest / "run.log"}')
    print('probe complete')


if __name__=='__main__':main()
