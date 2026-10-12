"""Current full-model chain controls: general, fast, fast+packed."""
import json
from pathlib import Path
here=Path(__file__).resolve().parent
for model in ('llama2','llama3'):
    plan=json.loads((here/f'{model}_b8_full_plan.json').read_text());base=plan[0];args=list(base['args'])
    args[args.index('--mode')+1]='duet-chain';args.remove('--p1-tree')
    args[args.index('--prompts')+1]=str(here.parent/'round5/tuning48.json')
    args[args.index('--max-new-tokens')+1]='96';idx=args.index('--seeds');args=args[:idx]+['--seeds','8300','8301']
    jobs=[]
    for b in (1,8,2,4):
        local=list(args);local[local.index('--batches')+1]=str(b)
        arms=['general','fast','packed']
        if b in (2,8):arms.reverse()
        for arm in arms:
            jobs.append(dict(name=f'{model}_chain_b{b}_{arm}_profile',args=local,
                env=base['env']|dict(SSD_BATCHED_TREE='0',SSD_FAST_VERIFY='0' if arm=='general' else '1',
                    SSD_PACKED_VERIFY='1' if arm=='packed' else '0',SSD_PROFILE_DUET='1',
                    SSD_PROFILE_DUET_DETAIL='1',SSD_SEED='73',
                    SSD_PROFILE_DIR=str(here/'profiles'/f'{model}_chain_b{b}_{arm}_profile')),
                batch=b,arm=arm,kind='chain_diagnostic'))
    (here/f'{model}_chain_profile_plan.json').write_text(json.dumps(jobs,indent=2))
