"""Reproducible short calibration and disjoint evaluation; no full-grid fitting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SSD = ROOT / 'ssd'
PY = SSD / '.venv/bin/python'


def config(exit=56, k1=10, k2=4, budget=15, mode='chain', n1=12, v1=8, n2=8, v2=8,
           candidate='legacy', fanout=3):
    tag = f'{mode}_e{exit}_k{k1}_{k2}_w{budget}'
    if mode != 'chain': tag += f'_n{n1}v{v1}_n{n2}v{v2}'
    if candidate != 'legacy': tag += '_' + candidate.removeprefix('policy::')
    if fanout != 3: tag += f'_f{fanout}'
    return dict(tag=tag, exit=exit, k1=k1, k2=k2, budget=budget, mode=mode,
                n1=n1, v1=v1, n2=n2, v2=v2, candidate=candidate, fanout=fanout)


def initialize():
    if (HERE / 'plan.json').exists(): return
    source = json.loads((ROOT / 'ssd/experiments/proxy_source_ablation/probe_training_free_20260912/datasets.json').read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/chokwans99/awq_calibrated/layerskip_llama2_70b')
    used, datasets = set(), {}
    for stage, start, per_dataset in [('calibration', 89, 2), ('validation', 25, 4), ('confirmation', 73, 4)]:
        rows = {}
        for name, info in source['source_files'].items():
            lines = Path(info['path']).read_text().splitlines()
            selected = []
            for row in list(range(start - 1, len(lines))) + list(range(0, start - 1)):
                item = json.loads(lines[row]); tokens = tokenizer.encode(item['text'], add_special_tokens=False)
                digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
                if digest in used or not 16 <= len(tokens) <= 768: continue
                used.add(digest)
                selected.append((item, dict(dataset=name, source_row=row+1, input_tokens=len(tokens), token_sha256=digest)))
                if len(selected) == per_dataset: break
            if len(selected) != per_dataset: raise RuntimeError('Insufficient disjoint prompts: ' + name)
            rows[name] = selected
        ordered = [rows[name][j] for j in range(per_dataset) for name in rows]
        dst = HERE / 'datasets' / stage / 'alpaca/alpaca_data_10000.jsonl'
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text('\n'.join(json.dumps(item) for item, meta in ordered) + '\n')
        datasets[stage] = dict(file=str(dst), mapping=[m for _, m in ordered], sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
    anchors = [config(), config(k1=6), config(k2=2), config(exit=40), config(exit=72)]
    validation = [config(exit=e, k1=k1, k2=k2) for e in [40,56,72] for k1 in [4,6,10] for k2 in [2,4]]
    validation.append(config(k1=8))
    random.Random(913).shuffle(validation)
    extras = [config(budget=8), config(budget=24), config(mode='p2tree', v2=4),
              config(mode='p2tree'), config(mode='fulltree'),
              config(candidate='policy::proxy__topm__original'),
              config(candidate='policy::proxy__full__expected_mix0.25')]
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), datasets=datasets,
                calibration=anchors, validation=validation, extensions=extras,
                temperature=.7, output_len=128, seeds=dict(calibration=913, validation=1913, confirmation=2913),
                primary='No-profile output tokens / summed generate wall time, includes prefill, excludes load and one warmup.',
                dataset_scope='Disjoint within this campaign. Existing finite prompt bank; some prompts were used in earlier, different experiments.',
                freeze='Calibration-only model and top-3 shortlist must be written before validation execution.',
                scope='B1, full P1+P2 DUET, 70B AWQ target + BF16 TinyLlama draft. No assumption of monotonic quality or constant all-shape latency.')
    (HERE / 'plan.json').write_text(json.dumps(plan, indent=2))


def run_one(cfg, stage, profile=False, seed=None, output_len=None, dataset_info=None, probe_layers=None):
    plan = json.loads((HERE / 'plan.json').read_text())
    data_stage = 'calibration' if stage == 'extensions' else stage
    info = dataset_info or plan['datasets'][data_stage]
    seed = seed or plan['seeds'][data_stage]
    dest = HERE / 'runs' / stage / f'{cfg["tag"]}_s{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    done = dest / 'complete.json'
    if done.exists(): return
    if (dest / 'result.json').exists(): raise RuntimeError('Incomplete result: ' + str(dest))
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('SSD_DUET_PROBE', 'SSD_E0_', 'SSD_CONF_')): env.pop(key)
    tree = cfg['mode'] != 'chain'
    env.update(CUDA_VISIBLE_DEVICES='3,4,5,6,7', SSD_DATASET_DIR=str(Path(info['file']).parents[1]),
               SSD_HF_CACHE='/data/chokwans99/models', SSD_CUDA_ARCH='8.9', TORCH_CUDA_ARCH_LIST='8.9',
               SSD_ATTN_BACKEND='auto',
               SSD_PROFILE='0', SSD_PROFILE_DUET=str(int(profile)), SSD_PROFILE_DUET_DETAIL='1',
               SSD_PROFILE_DUET_MAX_EVENTS='250000', SSD_PROFILE_DIR=str(dest),
               SSD_TREE_EXEC=str(int(tree)), SSD_TREE_ARENA=str(int(tree)), SSD_TREE_PROXY_GRAPH=str(int(tree)),
               SSD_TREE_EXEC_WARMUP='all' if tree else '0', SSD_DUET_EXIT_REPLICA='1',
               SSD_ASYNC_PROXY_SEND='1', SSD_PROXY_STREAM='0', SSD_DUET_PROXY_ON_DRAFT='0',
               SSD_CHAIN_PROXY_GRAPH='1', SSD_FORCE_SPLIT_K1K2='1', SSD_DUET_JIT_SHORT='1',
               SSD_TREE_VERIFY_WORKSPACE_MB='256', SSD_TREE_EXEC_WORKSPACE_MB='128', SSD_P1_TREE_EXEC_WORKSPACE_MB='128',
               DUET_CAL_OUT=str(dest/'result.json'), DUET_CAL_CANDIDATE=cfg['candidate'], DUET_CAL_TEMP='.7',
               SSD_DIST_PORT=str(21000 + int(hashlib.md5(str(dest).encode()).hexdigest()[:5],16)%9000))
    if probe_layers:
        env.update(SSD_DUET_PROBE_LAYERS=','.join(map(str,probe_layers)),SSD_DUET_PROBE_KIND='distribution',
                   SSD_DUET_PROBE_STRIDE='8',SSD_DUET_PROBE_OUT=str(dest/'distributions.json'))
    runner=HERE/('run_probe.py' if probe_layers else 'run_bench.py')
    cmd = [str(PY), '-O', str(runner), '--llama', '--size', '70', '--gpus', '5',
           '--model_path', '/home/chokwans99/awq_calibrated/layerskip_llama2_70b',
           '--draft_path', '/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0',
           '--quant_awq', '--quant_awq_artifact', '/home/chokwans99/awq_artifacts/layerskip70b_awq_tp4',
           '--alpaca', '--numseqs', str(len(info['mapping'])), '--output_len', str(output_len or plan['output_len']),
           '--b','1','--temp','.7','--seed',str(seed),'--max_model_len','2048',
           '--async','--spec','--duet','--duet_exit_layer',str(cfg['exit']),
           '--duet_k1',str(cfg['k1']),'--duet_k2',str(cfg['k2']),'--duet_p1_fanout',str(cfg.get('fanout',3)),
           '--duet_p2_budget',str(cfg['budget']), '--duet_p1_tree_policy','on' if cfg['mode']=='fulltree' else 'off',
           '--duet_p2_tree_policy','on' if tree else 'off']
    if tree:
        cmd += ['--duet_tree_root_count',str(cfg['budget']), '--duet_tree_c_tensor','3',
                '--duet_p1_roots_per_position','3', '--duet_p1_tree_max_nodes',str(cfg['n1']),
                '--duet_p1_tree_verify_nodes',str(cfg['v1']), '--duet_p2_tree_max_nodes',str(cfg['n2']),
                '--duet_p2_tree_verify_nodes',str(cfg['v2'])]
    start = time.time()
    print('[run]',stage,cfg['tag'],'profile',profile,flush=True)
    with (dest/'run.log').open('w') as f:
        proc = subprocess.Popen(cmd,cwd=SSD,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try: rc = proc.wait(timeout=900)
        finally:
            try: os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    log = (dest/'run.log').read_text(errors='replace')
    if rc or not (dest/'result.json').exists() or any(x in log for x in ['Traceback (most recent call last):','falling back to random']):
        raise RuntimeError(f'Run failed: {dest}')
    result = json.loads((dest/'result.json').read_text())
    if len(result['records']) != len(info['mapping']): raise RuntimeError('Missing prompts')
    for rec, mapping in zip(result['records'],info['mapping']):
        if rec['prompt_token_sha256'] != mapping['token_sha256']: raise RuntimeError('Prompt mismatch')
    record = dict(config=cfg,stage=stage,seed=seed,profile=profile,elapsed_s=time.time()-start,command=cmd,
                  dataset_sha256=info['sha256'], completed_utc=datetime.now(timezone.utc).isoformat(),
                  result_sha256=hashlib.sha256((dest/'result.json').read_bytes()).hexdigest())
    done.write_text(json.dumps(record,indent=2))
    print('[done]',stage,cfg['tag'],round(record['elapsed_s'],1),'s',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage', choices=['init','calibration','validation','extensions']); args=ap.parse_args()
    initialize()
    if args.stage=='init': return
    if args.stage=='validation' and not (HERE/'frozen.json').exists(): raise RuntimeError('Freeze calibration model first')
    plan=json.loads((HERE/'plan.json').read_text())
    for cfg in plan[args.stage]: run_one(cfg,args.stage,profile=args.stage in ('calibration','extensions'))


if __name__=='__main__': main()
