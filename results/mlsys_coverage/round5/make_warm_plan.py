"""Confirm profile-screen finalists with warm, uninstrumented tuning runs.

This is still parameter selection on the frozen48, never final-test feedback.
"""
import argparse,json,copy
from make_plans import HERE,job,save
from make_tree_plan import kwargs_from
from analyze_screen import analyze


def rows_for(model,batch):
    rows=[]
    for directory in HERE.glob(f'{model}_b{batch}_*'):
        if not directory.is_dir() or any(s in directory.name for s in ('warm','final','opt','scale','long')):continue
        for path in directory.glob(f'{model}*.json'):
            try:r=analyze(path)
            except (KeyError,TypeError):continue
            if (r and not r['timing_excluded'] and r.get('profile_steady_tps')
                    and r['args']['limit']==0 and r['args']['max_new_tokens']==96):rows.append(r)
    return rows


def from_row(row,name,batch=None,profile=False,extra=(),env=None):
    args=row['args'];model=row['name'].split('_')[0]
    b=batch or args['batches'][0]
    params=(kwargs_from(row) if args['mode']=='duet-tree' else
            {k:args[k] for k in ('k1','draft_fan_out','proxy_fan_out')})
    cli=[]
    for key,value in params.items():
        if value is not None:cli+=['--'+key.replace('_','-'),str(value)]
    runtime={k:v for k,v in row['env'].items() if k.startswith('SSD_') and
        k not in ('SSD_DIST_PORT','SSD_PROFILE_DIR','SSD_PROFILE_DUET','SSD_PROFILE_DUET_MAX_EVENTS',
                  'SSD_TREE_SHAPE_METRICS','SSD_SEED')}
    runtime.update(env or {})
    result=job(model,b,name,args['mode'],cli+list(extra),runtime,profile)
    result['origin']=row['path']
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--batch',type=int,required=True);a=p.parse_args()
    rows=rows_for(a.model,a.batch);jobs=[];choices=[]
    for mode in ('ssd','duet-tree'):
        group=[r for r in rows if r['args']['mode']==mode]
        picked=sorted(group,key=lambda r:r['profile_steady_tps'],reverse=True)[:3]
        picked+=sorted(group,key=lambda r:r['summary']['boundary_excluded_al'],reverse=True)[:2]
        seen=set()
        for row in picked:
            params=kwargs_from(row) if mode=='duet-tree' else {k:row['args'][k] for k in ('k1','draft_fan_out')}
            key=json.dumps(params,sort_keys=True)
            if key in seen:continue
            seen.add(key)
            name=f'warm_{mode.replace("-","_")}_{len(seen)}'
            jobs.append(from_row(row,name,extra=['--seeds','5200','5201'],env={'SSD_SEED':'17'}))
            choices.append(dict(name=jobs[-1]['name'],screen=row['name'],params=params,
                profile_tps=row['profile_steady_tps'],screen_al=row['summary']['boundary_excluded_al']))
    # One final measured neighborhood after interaction/phase-transfer jobs.
    # A C=1 tree has N=K; increasing K while freezing N would be a no-op.
    # Add the required node slot as well, profile it, then warm-confirm it.
    duet=[r for r in rows if r['args']['mode']=='duet-tree']
    base=max(duet,key=lambda r:r['profile_steady_tps']);known={json.dumps(kwargs_from(r),sort_keys=True) for r in duet}
    neighbors=[]
    for phase,node in [('k1','p1_nodes'),('k2','p2_nodes')]:
        params=kwargs_from(base);params[phase]+=1
        if params[phase]>16:continue
        params[node]=max(params[node],params[phase]);neighbors.append((f'grow_{phase}',params))
    for exit in (max(1,base['args']['exit_layer']-2),min(31,base['args']['exit_layer']+2)):
        neighbors.append((f'exit_{exit}',dict(kwargs_from(base),exit_layer=exit)))
    for name,params in neighbors:
        sig=json.dumps(params,sort_keys=True)
        if sig in known:continue
        known.add(sig)
        candidate=copy.deepcopy(base);candidate['args'].update(params)
        prof=from_row(candidate,'warm_diagnostic_'+name,profile=True)
        prof['reason']='Final depth/exit neighbor after joint parameter changes; profiling remains diagnostic'
        jobs.append(prof)
        candidate['path']=f'{a.model}_b{a.batch}_warm/{prof["name"]}.json'
        jobs.append(from_row(candidate,'warm_confirm_'+name,extra=['--seeds','5200','5201'],
            env={'SSD_SEED':'17','SSD_PROFILE_DUET':'0'}))
        choices.append(dict(name=jobs[-1]['name'],screen=prof['name'],params=params,
            reason='New local neighbor; paired diagnostic and warm confirmation are both required'))
    save(f'{a.model}_b{a.batch}_warm_plan.json',jobs)
    save(f'{a.model}_b{a.batch}_warm_choices.json',choices)
    print(a.model,a.batch,len(jobs))

if __name__=='__main__':main()
