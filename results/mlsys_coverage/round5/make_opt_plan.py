"""Implementation-only follow-up: frozen parameters, paired seeds/hardware."""
import argparse,json
from make_plans import HERE,save
from make_warm_plan import from_row
from analyze_screen import analyze


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--lane',type=int,choices=[0,1],required=True);a=p.parse_args()
    # lane0 is the original B1 pair, lane1 the original B8 pair.
    frozen={b:json.loads((HERE/f'{a.model}_b{b}_FROZEN.json').read_text()) for b in (1,8)}
    def preset(b,role):
        data=frozen[b]['presets'];key=data[role].get('alias',role)
        return analyze(HERE/data[key]['origin'])
    profiles=[]
    for trim in (0,1):
        extra=['--max-new-tokens','128','--seeds','6700']
        if trim==0:extra+=['--preflight-test','tests.test_root_policy,tests.test_phase_budget']
        item=from_row(preset(8,'duet_fast'),f'opt_l{a.lane}_profile_trim{trim}',profile=True,
            extra=extra,env={'SSD_TREE_LADDER_TRIM':str(trim),'SSD_PROFILE_DUET_DETAIL':'1','SSD_SEED':'0'})
        item.update(stage='optimization_profile',trim=trim)
        profiles.append(item)
    jobs=[]
    for b in (1,8):
        rep=0 if ((b==1)==(a.lane==0)) else 1
        seed=6100+rep*100
        item=from_row(preset(b,'duet_fast'),f'opt_fast_r{rep}',extra=[
            '--prompts',str(HERE.parent/'questions.json'),'--limit','0',
            '--max-new-tokens','128','--seeds',str(seed),str(seed+1)],
            env={'SSD_TREE_LADDER_TRIM':'1','SSD_SEED':str(41+rep),
                 'SSD_PROFILE_DUET':'0','SSD_TREE_SHAPE_METRICS':'0'})
        item.update(stage='optimization_full480',role='duet_fast',replicate=rep,batch=b)
        jobs.append(item)
    if a.lane==1:
        item=from_row(preset(8,'duet_al'),'opt_al_r0',extra=[
            '--prompts',str(HERE.parent/'questions.json'),'--limit','0',
            '--max-new-tokens','128','--seeds','6100','6101'],
            env={'SSD_TREE_LADDER_TRIM':'1','SSD_SEED':'41',
                 'SSD_PROFILE_DUET':'0','SSD_TREE_SHAPE_METRICS':'0'})
        item.update(stage='optimization_full480',role='duet_al',replicate=0,batch=8)
        jobs.append(item)
    save(f'{a.model}_opt_l{a.lane}_profile_plan.json',profiles)
    save(f'{a.model}_opt_l{a.lane}_full_plan.json',jobs)

if __name__=='__main__':main()
