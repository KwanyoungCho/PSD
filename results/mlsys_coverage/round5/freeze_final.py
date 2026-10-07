"""Freeze one warm tuning cell before reading any full480 confirmation result."""
import argparse,hashlib,json
from make_plans import HERE,save
from make_warm_plan import from_row
from analyze_screen import analyze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--batch',type=int,choices=[1,8],required=True);args=parser.parse_args()
    model,b=args.model,args.batch;output=HERE/f'{model}_b{b}_FROZEN.json'
    if output.exists():raise FileExistsError('Frozen choices cannot be overwritten')
    manifest=json.loads((HERE/f'{model}_b{b}_warm/campaign.json').read_text())
    expected=len(json.loads((HERE/f'{model}_b{b}_warm_plan.json').read_text()))
    if len(manifest)!=expected or any(r['status']!='complete' for r in manifest):raise RuntimeError('Warm tuning incomplete')
    rows=[analyze(HERE/f'{model}_b{b}_warm/{m["name"]}.json') for m in manifest]
    rows=[r for r in rows if r['env'].get('SSD_PROFILE_DUET','0')=='0' and len(r['all_cells'])>=2]
    picked={};points={};jobs={0:[],1:[]};fast_rows={}
    for mode in ('ssd','duet-tree'):
        prefix='ssd' if mode=='ssd' else 'duet'
        group=[r for r in rows if r['args']['mode']==mode]
        fast=max(group,key=lambda r:r['summary']['boundary_excluded_step_tps'])
        fast_rows[mode]=fast
        quality=max(group,key=lambda r:r['summary']['boundary_excluded_al'])
        for tag,row in [('fast',fast),('al',quality)]:
            role=f'{prefix}_{tag}'
            if row['path'] in picked:
                points[role]=dict(alias=picked[row['path']],origin=row['path']);continue
            picked[row['path']]=role
            points[role]=dict(origin=row['path'],args=row['args'],env=row['env'],
                tuning_al=row['summary']['boundary_excluded_al'],tuning_tps=row['summary']['boundary_excluded_step_tps'])
            for rep in (0,1):
                seeds=[6100+rep*100,6101+rep*100]
                item=from_row(row,f'final_{role}_r{rep}',extra=[
                    '--prompts',str(HERE.parent/'questions.json'),'--limit','0',
                    '--max-new-tokens','128','--seeds',*map(str,seeds)],
                    env={'SSD_SEED':str(41+rep),'SSD_PROFILE_DUET':'0','SSD_TREE_SHAPE_METRICS':'0'})
                item.update(role=role,replicate=rep,stage='full480',selection='warm48 only')
                jobs[rep].append(item)
    for rep,items in jobs.items():
        items.sort(key=lambda j:j['role'],reverse=rep==1)
    if b==8:
        # Use the pair that finishes B8 tuning first for full-corpus transfer,
        # long-output checks and a same-algorithm implementation control while
        # the other pair finishes B1 tuning. These never change frozen choices.
        for new_b in (2,4):
            for mode,row in fast_rows.items():
                label='ssd' if mode=='ssd' else 'duet'
                seed=6400 if new_b==2 else 6500
                item=from_row(row,f'scale_{label}',batch=new_b,extra=[
                    '--prompts',str(HERE.parent/'questions.json'),'--limit','0','--max-new-tokens','128',
                    '--seeds',str(seed),str(seed+1)],env={'SSD_SEED':str(42+new_b),'SSD_PROFILE_DUET':'0'})
                item.update(stage='batch_transfer',parameters_from_batch=8,role=label)
                jobs[0].append(item)
        for mode,row in fast_rows.items():
            label='ssd' if mode=='ssd' else 'duet'
            item=from_row(row,f'long_{label}',extra=[
                '--prompts',str(HERE.parent/'questions.json'),'--limit','0','--max-new-tokens','256',
                '--input-cap','1024','--seeds','6600','6601'],env={'SSD_SEED':'46','SSD_PROFILE_DUET':'0'})
            item.update(stage='long_transfer',parameters_from_batch=8,role=label)
            jobs[0].append(item)
        row=fast_rows['duet-tree']
        item=from_row(row,'implementation_off',extra=[
            '--prompts',str(HERE.parent/'questions.json'),'--limit','0','--max-new-tokens','128',
            '--seeds','6100','6101'],env={'SSD_SEED':'41','SSD_PROFILE_DUET':'0',
                'SSD_TREE_FUSED_MATH':'0','SSD_BATCH_TREE_BULK_EXPORT':'0','SSD_TREE_PARALLEL_INSERT':'0'})
        item.update(stage='implementation_control',role='duet_fast')
        jobs[0].append(item)
        item=from_row(row,'final_breakdown',profile=True,extra=['--max-new-tokens','128','--seeds','6700'],
            env={'SSD_PROFILE_DUET_DETAIL':'1'})
        item.update(stage='final_breakdown',role='duet_fast')
        jobs[0].append(item)
    for rep,items in jobs.items():
        save(f'{model}_b{b}_final_r{rep}_plan.json',items)
    common=dict(model=model,tuning_source_sha256=hashlib.sha256((HERE/'tuning48.json').read_bytes()).hexdigest(),
        selection_rule='Warm last-pass boundary-excluded TPS and AL separately; no full/432 feedback',
        primary_temperature=.7,output_cap=128,
        topology_note='Each SSD/DUET repeat uses identical paired GPUs. Repeat1 moves to the other same-model GPU pair; absolute TPS is kept separate.')
    save(output.name,dict(common,batch=b,presets=points))
    paths=[HERE/f'{model}_b{x}_FROZEN.json' for x in (1,8)]
    if all(p.exists() for p in paths):
        master=HERE/f'{model}_FROZEN.json'
        combined=dict(common,presets={str(x):json.loads(p.read_text())['presets'] for x,p in zip((1,8),paths)})
        if master.exists() and json.loads(master.read_text())!=combined:raise ValueError('Conflicting frozen master')
        if not master.exists():save(master.name,combined)
    print(model,b,'frozen',[(k,len(v)) for k,v in jobs.items()])

if __name__=='__main__':main()
