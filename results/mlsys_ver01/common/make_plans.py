import json
from pathlib import Path
R=Path(__file__).resolve().parent;root=R.parents[2]
checkouts={'paper':'/home/chokwans99/PSD-paper-review','before':'/home/chokwans99/PSD-ver01-common-base','after':str(root)}
def job(arm,name=None,mode='tree',temp=1,batch=1,limit=0,seed=10801,cpus='8-15,40-47',**extra):
    args=['--mode',mode,'--prompts',str(R.parent/'questions.json'),'--output-tokens','96','--warm-prompts','16','--seed',str(seed),'--temperature',str(temp),'--batch',str(batch)]
    if limit:args+=['--limit',str(limit)]
    for k,v in extra.items():args+=['--'+k.replace('_','-'),str(v)]
    return dict(name=name or arm,arm='base',bench='mlsys_ver01_common.py',draft_seed=231,args=args,env=dict(PAPER_CPUSET=cpus,PAPER_CHECKOUT=checkouts[arm],PAPER_BASELINE=str(int(arm=='paper'))))
def save(name,items):(R/(name+'_plan.json')).write_text(json.dumps(items,indent=2))
save('smoke',[job('after','smoke',limit=8)])
save('tree_r0',[job(a,cpus='8-15,40-47') for a in ['paper','before','after']])
save('tree_r1',[job(a,seed=10802,cpus='16-23,48-55') for a in ['after','before','paper']])
save('chain',[job(a,mode='chain',cpus='0-7,32-39') for a in ['before','after']])
save('features',[job('after','tree_t07',temp=.7,limit=24,cpus='24-31,56-63'),
 job('after','tree_g_gt_m',p1_verify=4,p2_verify=2,limit=24,cpus='24-31,56-63'),
 job('after','tree_t0',temp=0,limit=24,cpus='24-31,56-63'),
 *[job('after',mode+'_b4',mode=mode,batch=4,limit=24,cpus='24-31,56-63') for mode in ['chain','ssd','sd','ar']]])
for arm in ('before','after','paper'):
    j=job(arm,'profile_'+arm,limit=24)
    dest=R/'profiles'/j['name'];dest.mkdir(parents=True,exist_ok=True)
    j['env'].update(SSD_PROFILE_DUET='1',SSD_PROFILE_DUET_DETAIL='1',SSD_PROFILE_DIR=str(dest))
    save('profile_'+arm,[j])
