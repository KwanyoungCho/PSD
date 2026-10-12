import json
from pathlib import Path
here=Path(__file__).resolve().parent
for model in ('llama2','llama3'):
    base=json.loads((here/f'{model}_b8_full_plan.json').read_text())[0];jobs=[]
    for kind in ('profile','full'):
      for b in (1,8):
       for host in (0,1):
        args=list(base['args']);args[args.index('--batches')+1]=str(b)
        args[args.index('--temperatures')+1]='0'
        args[args.index('--max-new-tokens')+1]='64'
        args[args.index('--prompts')+1]=str(here.parent/('round5/tuning48.json' if kind=='profile' else 'questions.json'))
        index=args.index('--seeds');args=args[:index]+['--seeds','8400','8401','--preflight-test','mlsys_review_greedy_hook']
        name=f'{model}_greedy_b{b}_host{host}_{kind}'
        jobs.append(dict(name=name,args=args,kind='greedy_'+kind,batch=b,host=host,
          env=base['env']|dict(SSD_REVIEW_GREEDY_HOST=str(host),SSD_BATCH_TREE_ACCEPT_GRAPH='1',
              SSD_TREE_LADDER_TRIM='0',SSD_PROFILE_DUET='1' if kind=='profile' else '0',
              SSD_PROFILE_DUET_DETAIL='1' if kind=='profile' else '0',SSD_SEED='74',
              SSD_PROFILE_DIR=str(here/'profiles'/name))))
    (here/f'{model}_greedy_review_plan.json').write_text(json.dumps(jobs,indent=2))
