"""Frozen fixed-parameter comparisons; no result-driven tuning."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
base=root.parent
for model,draft,exit_layer in [('llama2','AMD-Llama-135m',21),('llama3','Qwama-0.5B-Instruct',16)]:
    target=('/home/chokwans99/models_mlsys/layerskip-llama2-7b' if model=='llama2'
            else '/data/chokwans99/models/layerskip-llama3-8B')
    args=['--target',target,
          '--draft',f'/home/chokwans99/models_mlsys/{draft}',
          '--mode','duet-tree','--p1-tree','--k1','4','--k2','2',
          '--exit-layer',str(exit_layer),'--p1-nodes','8','--p2-nodes','4',
          '--tree-width','3','--p1-roots','2','--draft-fan-out','2',
          '--proxy-fan-out','1','--root-source','residual',
          '--root-normalization','topm','--root-overlap-mix','0',
          '--temperatures','.7','--input-cap','512','--limit','0']
    env=dict(SSD_TREE_EXEC='1',SSD_TREE_EXPANSION_POLICY='q_path',
             SSD_TREE_FUSED_MATH='0',SSD_BATCH_TREE_BULK_EXPORT='0',
             SSD_TREE_PARALLEL_INSERT='0',SSD_TREE_LADDER_TRIM='0',
             SSD_BATCH_TREE_PROXY_STREAM='0',SSD_PACKED_TREE_VERIFY='0',
             SSD_PROFILE_DUET_DETAIL='1',SSD_SEED='71')
    jobs=[]
    for b in (1,8,2,4):
        arms=['legacy','graph0','graph1','trim1'] if b==1 else ['graph0','graph1','trim1']
        # Alternate order across B to reduce a monotonic warm-machine bias.
        if b in (2,8):arms.reverse()
        for arm in arms:
            jobs.append(dict(name=f'{model}_b{b}_{arm}_profile',
                args=args+['--batches',str(b),'--prompts',str(base/'round5/tuning48.json'),
                           '--max-new-tokens','96','--seeds','8100','8101'],
                env=env|dict(SSD_BATCHED_TREE='0' if arm=='legacy' else '1',
                             SSD_BATCH_TREE_ACCEPT_GRAPH='0' if arm=='graph0' else '1',
                             SSD_TREE_LADDER_TRIM='1' if arm=='trim1' else '0',
                             SSD_PROFILE_DUET='1'),
                kind='diagnostic',batch=b,arm=arm))
    (root/f'{model}_profile_plan.json').write_text(json.dumps(jobs,indent=2))
    jobs=[]
    for b in (1,8):
        for rep in (0,1):
            arms=['graph0','graph1','trim1']
            if rep:arms.reverse()
            for arm in arms:
                jobs.append(dict(name=f'{model}_b{b}_{arm}_full_r{rep}',
                    args=args+['--batches',str(b),'--prompts',str(base/'questions.json'),
                               '--max-new-tokens','128','--seeds',str(8200+rep*10),str(8201+rep*10)],
                    env=env|dict(SSD_BATCHED_TREE='1',SSD_PROFILE_DUET='0',
                                 SSD_PROFILE_DUET_DETAIL='0',SSD_SEED=str(72+rep),
                                 SSD_BATCH_TREE_ACCEPT_GRAPH='0' if arm=='graph0' else '1',
                                 SSD_TREE_LADDER_TRIM='1' if arm=='trim1' else '0'),
                    kind='full480',batch=b,arm=arm,replicate=rep))
    (root/f'{model}_full_plan.json').write_text(json.dumps(jobs,indent=2))
