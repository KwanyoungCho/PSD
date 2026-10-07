"""Frozen screening inputs and explicit topology-aware campaign plans."""
import collections
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
MODELS={
    'llama2':('/home/chokwans99/models_mlsys/layerskip-llama2-7b',
              '/home/chokwans99/models_mlsys/AMD-Llama-135m'),
    'llama3':('/data/chokwans99/models/layerskip-llama3-8B',
              '/home/chokwans99/models_mlsys/Qwama-0.5B-Instruct')}


def save(name, obj):
    path=HERE/name
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(obj,indent=2)+'\n')
    temporary.replace(path)


def job(model,b,name,mode='duet-tree',extra=(),env=None,profile=True):
    target,draft=MODELS[model]
    name=f'{model}_b{b}_{name}'
    args=['--target',target,'--draft',draft,'--mode',mode,
          '--prompts',str(HERE/'tuning48.json'),'--limit','0',
          '--max-new-tokens','96','--temperatures','.7','--seeds','5100',
          '--batches',str(b)]
    options={}
    if mode=='duet-tree':
        args+=['--p1-tree','--root-source','complement','--root-normalization','full',
               '--root-overlap-mix','.25']
        options.update(SSD_BATCHED_TREE='1',SSD_TREE_EXPANSION_POLICY='reach_gain_frontier',
            SSD_TREE_REACH_CALIBRATION=str(HERE.parent/'round4/calibration/reach.json'),
            SSD_TREE_GAIN_CALIBRATION=str(HERE.parent/'round4/calibration/gain.json'),
            SSD_DUET_MISS_K='4',SSD_TREE_FUSED_MATH='1',SSD_BATCH_TREE_BULK_EXPORT='1',
            SSD_TREE_PARALLEL_INSERT='0')
    if profile:
        options.update(SSD_PROFILE_DUET='1',SSD_PROFILE_DUET_MAX_EVENTS='100000',SSD_TREE_SHAPE_METRICS='1',
                       SSD_PROFILE_DIR=str(HERE/'profiles'/name))
    options.update(env or {})
    return dict(name=name,args=args+list(extra),env=options)


def main():
    source=json.loads((HERE.parent/'questions.json').read_text())
    groups=collections.defaultdict(list)
    for i,q in enumerate(source):groups[q['group']].append(i)
    selected=sorted(i for ids in groups.values() for i in ids[::10])
    save('SPLIT.json',dict(tuning=selected,confirmation=[i for i in range(len(source)) if i not in selected],
                          rule='each group original offsets0,10,...,70; frozen before new results'))
    save('tuning48.json',[source[i] for i in selected])
    for model in MODELS:
        for b in (1,8):
            save(f'{model}_b{b}_ssd_screen_plan.json',[
                job(model,b,f'ssd_k{k}_f{f}',mode='ssd',extra=['--k1',str(k),
                    '--draft-fan-out',str(f),'--proxy-fan-out','0'])
                for k in (2,4,6,8) for f in (1,3,5)])
        save(f'{model}_smoke_plan.json',[
            job(model,2,'root_smoke',extra=['--limit','6','--max-new-tokens','16',
                '--temperatures','0','.7','1'],profile=False)])
        for b in (1,8):
            save(f'{model}_b{b}_duet_anchor_plan.json',[
                job(model,b,f'duet_e{e}_k4_2',extra=['--exit-layer',str(e)])
                for e in (16,21,26,30)])


if __name__=='__main__':main()
