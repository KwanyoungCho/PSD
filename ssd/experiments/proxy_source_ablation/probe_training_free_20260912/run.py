"""Actual-temperature generation; mixed dataset with exact prompt provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

HERE=Path(__file__).resolve().parent
SSD=HERE.parents[2]
ROOT=SSD.parent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['development','confirmation'],required=True)
    ap.add_argument('--temperatures',nargs='+',type=float,required=True)
    ap.add_argument('--gpus',default='3,4,5,6,7')
    ap.add_argument('--port',type=int,default=19840)
    args=ap.parse_args()
    info=json.loads((HERE/'datasets.json').read_text())['stages'][args.stage]
    n=len(info['mapping']);out=HERE/'out'/args.stage;out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus,SSD_DATASET_DIR=str(HERE/'datasets'/args.stage),
               SSD_PROFILE='0',SSD_PROFILE_DUET='0',SSD_PROFILE_DUET_DETAIL='0',
               SSD_TREE_EXEC='0',SSD_TREE_ARENA='0',SSD_TREE_PROXY_GRAPH='0',
               SSD_TREE_EXEC_WARMUP='0',SSD_DUET_EXIT_REPLICA='1',
               SSD_DUET_PROBE_LAYERS='56',SSD_DUET_PROBE_KIND='distribution',SSD_DUET_PROBE_STRIDE='8')
    seed=101 if args.stage=='development' else 271
    for j,t in enumerate(args.temperatures):
        tag=f'mixed_t{t:g}_seed{seed}'
        manifest=out/(tag+'.json');done=out/(tag+'.complete.json');log=out/(tag+'.log')
        if done.exists():
            print(f'[skip] {tag}',flush=True);continue
        if manifest.exists():raise RuntimeError(f'Incomplete artifact exists: {manifest}')
        cmd=[str(SSD/'.venv/bin/python'),'-O',str(SSD/'bench/bench.py'),
             '--llama','--size','70','--gpus','5',
             '--model_path','/home/chokwans99/awq_calibrated/layerskip_llama2_70b',
             '--draft_path','/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0',
             '--quant_awq','--quant_awq_artifact','/home/chokwans99/awq_artifacts/layerskip70b_awq_tp4',
             '--alpaca','--numseqs',str(n),'--output_len','256','--b','1','--temp',str(t),'--seed',str(seed),
             '--async','--spec','--duet','--duet_exit_layer','56',
             '--duet_phase1_k','8','--duet_phase2_k','4','--duet_draft_fan_out','3','--duet_p2_budget','15',
             '--duet_p1_tree_policy','off','--duet_p2_tree_policy','off','--duet_only_proxy']
        env.update(SSD_DUET_PROBE_OUT=str(manifest),SSD_DIST_PORT=str(args.port+j))
        print(f'[run] {args.stage} T={t:g} prompts={n}',flush=True);start=time.time()
        with log.open('w') as f:r=subprocess.run(cmd,cwd=SSD,env=env,stdout=f,stderr=subprocess.STDOUT)
        text=log.read_text(errors='replace')
        assert r.returncode==0 and manifest.exists(),str(log)
        assert 'Final Decode Throughput:' in text and 'max|probe-engine|=0.000e+00 OK' in text,str(log)
        assert not any(s in text for s in ['Traceback (most recent call last):','PROBE_FLUSH_FAILED','falling back to random']),str(log)
        meta=json.loads(manifest.read_text());assert meta['temperature']==t and meta['layers']==[56]
        attrs=[];maxerr=0.;hashes={}
        for ch in meta['chunks']:
            path=out/ch['file'];hashes[ch['file']]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as z:
                for key in ['p_T','p_D','p_E']:
                    a=z[key];assert a.dtype==np.float32 and np.isfinite(a).all() and (a>=0).all()
                    if key=='p_D':
                        assert (a[z['is_bonus']]==0).all();a=a[~z['is_bonus']]
                    err=float(abs(a.sum(-1,dtype=np.float64)-1).max());assert err<3e-6
                    maxerr=max(maxerr,err)
                attrs.append({k:z[k][~z['is_bonus']] for k in ['seq_id','local_step','step','position']})
        attrs={k:np.concatenate([a[k] for a in attrs]) for k in attrs[0]}
        assert set(attrs['seq_id'])==set(range(1,n+1))
        assert len(set(attrs['step']))==meta['n_samples']
        assert len(attrs['step'])==4*meta['n_samples']
        for sid,count in meta['seq_steps'].items():
            ids=(attrs['seq_id']==int(sid))&(attrs['position']==0)
            np.testing.assert_array_equal(attrs['local_step'][ids],np.arange(0,count,meta['stride']))
        record=dict(dataset='mixed',stage=args.stage,temperature=t,seed=seed,numseqs=n,output_len=256,
                    n_steps=meta['n_steps'],n_samples=meta['n_samples'],n_rows=meta['n_rows'],
                    elapsed_s=time.time()-start,manifest=manifest.name,command=cmd,mapping=info['mapping'],
                    dataset_sha256=info['sha256'],max_probability_sum_error=maxerr,snapshot_sha256=hashes,
                    validation='exit, log, exact tap, finite normalized float32, all prompt/step IDs and stride')
        done.write_text(json.dumps(record,indent=2))
        print(f'[done] {tag}: {meta["n_steps"]} steps / {meta["n_samples"]} sampled; {record["elapsed_s"]:.1f}s',flush=True)


if __name__=='__main__':main()
