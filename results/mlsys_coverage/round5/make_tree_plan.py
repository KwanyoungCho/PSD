"""Diagnose root coverage, continuation width, node capacity and exit neighbors."""
import argparse,json,math
from make_plans import HERE,job,save
from analyze_screen import analyze

FIELDS=['k1','k2','exit_layer','draft_fan_out','proxy_fan_out','p1_nodes','p2_nodes',
        'p1_roots','p2_budget','tree_root_count','tree_width','tree_beta',
        'tree_proxy_threshold','tree_conf_threshold']


def kwargs_from(row):
    args=row['args']
    out={k:args.get(k) for k in FIELDS}
    out['p1_nodes']=out['p1_nodes'] or out['k1']*2
    out['p2_nodes']=out['p2_nodes'] or out['k2']*2
    return out


def make_job(model,batch,name,parameters,reason):
    args=[]
    for k,v in parameters.items():
        if v is not None:args+=['--'+k.replace('_','-'),str(v)]
    row=job(model,batch,name,extra=args)
    row['reason']=reason
    return row


def rows_for(model,batch):
    rows=[]
    for pattern in (f'{model}_b{batch}_duet_anchor',f'{model}_b{batch}_refine*'):
        for d in HERE.glob(pattern):
            if not d.is_dir():continue
            for p in d.glob(f'{model}*.json'):
                r=analyze(p)
                if r and not r.get('timing_excluded'):rows.append(r)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--batch',type=int,required=True)
    a=p.parse_args();rows=rows_for(a.model,a.batch)
    fast=max(rows,key=lambda r:r.get('profile_steady_tps') or 0)
    quality=max(rows,key=lambda r:r['summary']['boundary_excluded_al'])
    base=kwargs_from(fast);jobs=[];seen=set()
    def add(name,changes,reason,origin=base):
        params=dict(origin,**changes)
        sig=json.dumps(params,sort_keys=True)
        if sig in seen or params==origin:return
        seen.add(sig)
        jobs.append(make_job(a.model,a.batch,'tree_'+name,params,reason))
    add('roots1_compute1',dict(p1_roots=1,draft_fan_out=1),
        'Reduce both initial P1 roots and later continuation lanes; root-only intervention already measured')
    add('roots2_compute1',dict(p1_roots=2,draft_fan_out=1),
        'Keep P1 root coverage, reduce later forwards to expose the continuation-width cost')
    add('roots3',dict(p1_roots=3),'Broaden P1 root coverage at fixed continuation budget')
    add('compute3',dict(draft_fan_out=3),'Increase continuation lanes at fixed root coverage to test AL lost to unexpanded roots')
    add('width1',dict(tree_width=1,p1_nodes=base['k1'],p2_nodes=base['k2']),
        'Single-child continuation; less target node work but no sibling recovery')
    add('width2',dict(tree_width=2),'Reduce sibling width using the measured gain-curve prefix')
    add('nodes_low',dict(p1_nodes=base['k1'],p2_nodes=base['k2']),
        'Reduce target query capacity and generated nodes at fixed serial phase rounds')
    add('nodes_high',dict(p1_nodes=min(16,3*base['k1']),p2_nodes=min(16,3*base['k2'])),
        'Increase root-local node allowance; check accepted-depth ceiling and target-query overhead')
    for budget in (3,8,12):
        add(f'p2budget{budget}',dict(p2_budget=budget,tree_root_count=None),
            'P2 root/forward budget changes coverage and draft width independently of K')
    for beta in (0,1):
        add(f'beta{beta}',dict(tree_beta=beta),'Equal versus probability-proportional root node allocation')
    add('no_floors',dict(tree_proxy_threshold=0,tree_conf_threshold=0),
        'Full-vocabulary normalization reduces absolute root scores; test whether old pruning floors suppress useful depth')
    layer_ms=fast['profile']['batch_target_pre']['p50']/base['exit_layer']
    predicted=max(1,min(31,round(base['exit_layer']-fast['profile']['p1_slack_ms']['p50']/layer_ms)))
    for exit in sorted({predicted,max(1,predicted-2),min(31,predicted+2)}):
        add(f'align_exit{exit}',dict(exit_layer=exit),
            'Move proxy arrival toward measured P1 completion; save post-exit time without excessive proxy-quality loss')
    if quality['name']!=fast['name']:
        qb=kwargs_from(quality)
        add('al_width1',dict(tree_width=1,p1_nodes=qb['k1'],p2_nodes=qb['k2']),
            'AL-best phase anchor: compare deep single-child continuation',qb)
        add('al_width2',dict(tree_width=2),'AL-best anchor: test a narrower sibling tree',qb)
        add('al_compute3',dict(draft_fan_out=3),'AL-best anchor: test more continuation lanes',qb)
    save(f'{a.model}_b{a.batch}_tree_plan.json',jobs)
    save(f'{a.model}_b{a.batch}_tree_decision.json',dict(
        profile_efficient_anchor=fast['name'],al_anchor=quality['name'],parameters=base,
        predicted_exit=predicted,jobs=len(jobs),warning='Profile-based screen only; warm uninstrumented confirmation must follow'))
    print(a.model,a.batch,fast['name'],quality['name'],len(jobs))


if __name__=='__main__':main()
