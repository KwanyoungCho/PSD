"""Close local phase neighborhoods after measured tree-budget interventions."""
import argparse,json
from make_plans import HERE,save
from make_tree_plan import kwargs_from,make_job
from make_warm_plan import rows_for


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--batch',type=int,required=True);a=p.parse_args()
    rows=[r for r in rows_for(a.model,a.batch) if r['args']['mode']=='duet-tree']
    fast=max(rows,key=lambda r:r['profile_steady_tps']);quality=max(rows,key=lambda r:r['summary']['boundary_excluded_al'])
    anchor=json.loads((HERE/f'{a.model}_b{a.batch}_tree_decision.json').read_text())
    origin=anchor['parameters'];jobs=[];decisions=[];seen=set()
    smoke=make_job(a.model,a.batch,'transfer_smoke',dict(origin,k1=1,k2=3,exit_layer=12,
        draft_fan_out=1,p1_nodes=2,p2_nodes=6,p1_roots=2,p2_budget=3,tree_root_count=None),
        'Gate reversed phase lengths on real models before measuring them; excluded from tuning selection')
    smoke['args']+=['--limit','8','--max-new-tokens','24','--temperatures','0','.7']
    smoke['env']['SSD_PROFILE_DUET']='0'
    jobs.append(smoke)
    def signature(params,env):return json.dumps([params,env.get('SSD_DUET_MISS_K','4')],sort_keys=True)
    for row in rows:seen.add(signature(kwargs_from(row),row['env']))
    def add(name,params,reason,env=None):
        env=env or {};width=max(params['k1']+params['k2'],params['p1_nodes'],params['p2_nodes'])
        if width<int(env.get('SSD_DUET_MISS_K','4')):return
        key=signature(params,env)
        if key in seen:return
        seen.add(key)
        row=make_job(a.model,a.batch,'combo_'+name,params,reason);row['env'].update(env)
        jobs.append(row)
    # Merge improvements from distinct knobs, then measure their interaction.
    combined=dict(origin)
    families=[('coverage',['p1_roots','draft_fan_out']),('shape',['tree_width','p1_nodes','p2_nodes']),
              ('p2',['p2_budget','tree_root_count']),('allocation',['tree_beta']),
              ('floors',['tree_proxy_threshold','tree_conf_threshold'])]
    tree=[r for r in rows if '_tree/' in r['path'] and '_al_' not in r['name']]
    for family,fields in families:
        candidates=[r for r in tree if any(kwargs_from(r)[k]!=origin[k] for k in fields)]
        # The common root-only screen changes one family per job.
        candidates=[r for r in candidates if all(kwargs_from(r)[k]==origin[k] for k in origin if k not in fields)]
        if not candidates:continue
        winner=max(candidates,key=lambda r:r['profile_steady_tps'])
        base=next(r for r in rows if r['name']==anchor['profile_efficient_anchor'])
        if (winner['profile_steady_tps']>base['profile_steady_tps']*1.01 and
            winner['summary']['boundary_excluded_al']>=base['summary']['boundary_excluded_al']*.97):
            changes={k:kwargs_from(winner)[k] for k in fields};combined.update(changes)
            decisions.append(dict(family=family,source=winner['name'],changes=changes))
    add('combined',combined,'Combine individually measured gains; never assume their percentages add')
    base=kwargs_from(fast)
    for exit in sorted({max(1,base['exit_layer']-2),min(31,base['exit_layer']+2)}):
        add(f'e{exit}',dict(base,exit_layer=exit),'Exit neighbors after narrowing tree/phase work')
    for phase in ('k1','k2'):
        cap=base['p1_nodes' if phase=='k1' else 'p2_nodes']
        for k in sorted({max(1,base[phase]-1),min(cap,base[phase]+1)}):
            add(f'{phase}_{k}',dict(base,**{phase:k}),'Serial-round neighbor at fixed node/root capacity')
    for miss in (2,6):
        add(f'miss{miss}',base,'Independent miss depth: test exposed JIT cost versus miss AL after root/tree tuning',{'SSD_DUET_MISS_K':str(miss)})
    # A short P2 hit can have lower AL than a deeper miss chain. Its late
    # arrival at the old exit is not evidence that longer P2 is impossible:
    # move work from P1 to P2 and deliver a proxy earlier at fixed small W2.
    transfers=([(2,4,12),(2,4,16),(3,3,16),(3,3,20)] if a.model=='llama2'
        else [(1,3,10),(1,3,14),(2,3,12),(2,3,16),(2,4,12),(2,4,16)])
    for k1,k2,exit in transfers:
        params=dict(base,k1=k1,k2=k2,exit_layer=exit,draft_fan_out=1,
            p1_roots=2,p1_nodes=2*k1,p2_nodes=2*k2,p2_budget=3,tree_root_count=None)
        add(f'transfer_{k1}_{k2}_e{exit}',params,
            'Move serial work from P1 to P2: earlier proxy plus narrow P2 budget tests whether shallow P2 hits limit AL')
    qb=kwargs_from(quality)
    for phase in ('k1','k2'):
        cap=qb['p1_nodes' if phase=='k1' else 'p2_nodes'];k=qb[phase]+1
        if k<=cap:add(f'al_{phase}_{k}',dict(qb,**{phase:k}),'Check remaining AL depth headroom at fixed node budget')
    # If the interaction combination moved the operating point, check its
    # observed P1-alignment exit estimate as well, then warm-confirm finalists.
    if combined!=origin:
        for exit in sorted({max(1,base['exit_layer']-2),base['exit_layer']}):
            add(f'combined_e{exit}',dict(combined,exit_layer=exit),'Recheck combined tree changes with efficient exit neighbor')
    save(f'{a.model}_b{a.batch}_combo_plan.json',jobs)
    save(f'{a.model}_b{a.batch}_combo_decision.json',dict(efficient=fast['name'],quality=quality['name'],
        independent_changes=decisions,combined=combined,jobs=len(jobs),
        limitation='Finite coordinate neighborhood; global optimum over all combinations is not established'))
    print(a.model,a.batch,len(jobs))

if __name__=='__main__':main()
