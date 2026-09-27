"""Explicit secondary experiment: diversify exits, then independently confirm.

Registered after three primary validation cells were inspected. Therefore
validation comparisons of this refinement are exploratory; confirmation uses
untouched prompts and two new seeds. The primary frozen recommendation stays.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from campaign import HERE, ROOT, config, run_one
from calibrator import load_run, metrics


def register():
    path=HERE/'refinement_plan.json'
    if path.exists():return json.loads(path.read_text())
    plan=json.loads((HERE/'plan.json').read_text());frozen=json.loads((HERE/'frozen.json').read_text())
    shortlist=[]
    for e in [40,56,72]:
        options=[p for p in frozen['predictions'] if p['config']['exit']==e]
        shortlist.append(max(options,key=lambda p:p['tps'])['config'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/home/chokwans99/awq_calibrated/layerskip_llama2_70b')
    used={m['token_sha256'] for st in plan['datasets'].values() for m in st['mapping']}
    sources=json.loads((ROOT/'ssd/experiments/proxy_source_ablation/probe_training_free_20260912/datasets.json').read_text())['source_files']
    selected={}
    for name,info in sources.items():
        values=[]
        for row,line in enumerate(Path(info['path']).read_text().splitlines(),1):
            item=json.loads(line);ids=tokenizer.encode(item['text'],add_special_tokens=False)
            h=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            if h in used or not 16<=len(ids)<=768:continue
            used.add(h);values.append((item,dict(dataset=name,source_row=row,input_tokens=len(ids),token_sha256=h)))
            if len(values)==4:break
        if len(values)!=4:raise ValueError('Not enough independent selection prompts')
        selected[name]=values
    ordered=[selected[name][i] for i in range(4) for name in selected]
    dst=HERE/'datasets/selection/alpaca/alpaca_data_10000.jsonl';dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text('\n'.join(json.dumps(x) for x,m in ordered)+'\n')
    out=dict(registered_utc=datetime.now(timezone.utc).isoformat(),primary_validation_seen=3,
             method='One model-best configuration per exit, measured on 16 new selection prompts, no profiler. Select observed throughput winner.',
             shortlist=shortlist,dataset=dict(file=str(dst),mapping=[m for x,m in ordered],sha256=hashlib.sha256(dst.read_bytes()).hexdigest()),
             selection_seed=2413,confirmation_seeds=[2913,3913],confirmation_output_len=256,
             note='Secondary adaptation, not retroactively pre-registered primary. Final confirmation remains independent.')
    path.write_text(json.dumps(out,indent=2));return out


def main():
    ref=register();plan=json.loads((HERE/'plan.json').read_text())
    boundary=HERE/'boundary_plan.json'
    if boundary.exists():
        additions=[x['config'] for x in json.loads(boundary.read_text())['predictions']]
        ref['shortlist']=list({c['tag']:c for c in ref['shortlist']+additions}.values())
    # The preceding supplementary runner may preserve a failure rather than completion.
    extension=json.loads((HERE/'extension_plan.json').read_text())['configs']
    def ext_finished(c):
        p=HERE/'runs/extensions'/f'{c["tag"]}_s913'
        return (p/'complete.json').exists() or (p/'failure.txt').exists()
    while not all(ext_finished(c) for c in extension):time.sleep(10)
    for cfg in [config(fanout=1),config(fanout=2)]:run_one(cfg,'extensions',profile=True)
    # Supplementary dataset metadata is deliberately separate from immutable plan.json.
    for cfg in ref['shortlist']:
        run_one(cfg,'selection',profile=False,seed=ref['selection_seed'],dataset_info=ref['dataset'])
    measured=[load_run(HERE/'runs/selection'/f'{c["tag"]}_s2413') for c in ref['shortlist']]
    chosen=max(measured,key=lambda r:metrics(r)['tps'])['config']
    validation=json.loads((HERE/'validation_summary.json').read_text())
    primary=json.loads((HERE/'frozen.json').read_text())
    bytag={c['tag']:c for c in plan['validation']+ref['shortlist']}
    # Includes validation oracle, the primary and secondary recommendations,
    # the default reference, and the latency-only control. Deduplicate exact arms.
    names=[chosen['tag'],primary['selection']['top1'],validation['oracle'],
           config(k1=8)['tag'],primary['selection']['gap_only']]
    configs=[bytag[tag] for tag in dict.fromkeys(names)]
    freeze=dict(selected=chosen,confirmation=configs,seeds=ref['confirmation_seeds'],
                frozen_utc=datetime.now(timezone.utc).isoformat(),selection=[dict(config=r['config'],metrics=metrics(r)) for r in measured],
                output_len=ref['confirmation_output_len'])
    dst=HERE/'refinement_frozen.json'
    if not dst.exists():dst.write_text(json.dumps(freeze,indent=2))
    else:freeze=json.loads(dst.read_text());configs=freeze['confirmation']
    for j,seed in enumerate(ref['confirmation_seeds']):
        order=configs if j==0 else list(reversed(configs))
        for cfg in order:run_one(cfg,'confirmation',profile=False,seed=seed,output_len=ref['confirmation_output_len'])
    print('Independent confirmation finished.',flush=True)


if __name__=='__main__':main()
