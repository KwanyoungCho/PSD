"""Whole-step paired comparisons and draft/target event profiles."""
from pathlib import Path
from collections import defaultdict
import json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE/'raw/runs' if (HERE/'raw/runs').exists() else Path('/data/chokwans99/duet-local-followup-20261017/runs')
import analysis_common as common
common.RAW=ROOT
sys.path.insert(0,str(HERE.parents[1]/'ssd/tools/duet_timeline'))
from plot_consecutive_steps import read_profiles


def profiles(tag):
    folder=ROOT/tag
    if not (folder/'status.json').exists():return None
    d=json.loads((folder/'run.json').read_text());groups,_,_=read_profiles(folder)
    result={}
    for source in (1,2,0,'all'):
        rows=[]
        for i,g in enumerate(d['groups']):
            ep=len(d['warm_groups'])+i
            for s in g['steps']:
                e=s['events'][0];sid=e['step_id'];src=e['source'] if e['cache_hit'] else 0
                if sid<=10 or s['cap'] or s['clipped'] or (source!='all' and src!=source):continue
                row={'step_ms':s['seconds']*1000}
                for role in ('target','draft'):
                    by=defaultdict(list)
                    for ev in groups[role][ep,sid]:by[ev['label']].append(ev)
                    for label,evs in by.items():row[role+':'+label]=sum(ev['wall_end_ns']-ev['wall_start_ns'] for ev in evs)/1e6
                rows.append(row)
        keys=sorted({k for r in rows for k in r})
        result[str(source)]={'n':len(rows),**{k:float(np.mean([r.get(k,0) for r in rows])) for k in keys}}
    return result


def main():
    all_pairs=[('base_fused_p0_a','fused_p0_b'),('base_fused_p0_d','fused_p0_c'),('base_fused_p1_a','fused_p1_b')]
    complete=[p for p in all_pairs if all(common.load(x) for x in p)]
    result={}
    if complete:result['fused']=common.compare(complete)
    if common.load('heldout_fused') and common.load('heldout_base_fused'):
        result['heldout']=common.compare([('heldout_base_fused','heldout_fused')])
    if common.load('heldout_fused_bulk') and common.load('heldout_base_fused'):
        result['heldout_final']=common.compare([('heldout_base_fused','heldout_fused_bulk')])
    for name,pairs in {'bulk':[('fused_bulk_reference','fused_bulk_candidate'),('fused_bulk_reference2','fused_bulk_candidate2')]}.items():
        complete=[p for p in pairs if all(common.load(x) for x in p)]
        if complete:result[name]=common.compare(complete)
    if common.load('fused_bulk_candidate'):
        result['final_12']=common.compare([('base_fused_p0_d','fused_bulk_candidate')])
    if common.load('fused_bulk_candidate') and common.load('heldout_fused_bulk'):
        result['selected_joint']=common.compare([('base_fused_p0_d','fused_bulk_candidate'),('heldout_base_fused','heldout_fused_bulk')])
    (HERE/'COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
    pp={tag:profiles(tag) for tag in ('baseline_fused_profile','fused_profile','fused_bulk_profile')}
    (HERE/'PROFILES.json').write_text(json.dumps(pp,indent=2)+'\n')
    for name,r in result.items():
        print(name,r['status'],r.get('reduction_pct'),r.get('ci95_pct'))
        if r['status']=='complete':print(json.dumps(r['sources'],indent=2))


if __name__=='__main__':main()
