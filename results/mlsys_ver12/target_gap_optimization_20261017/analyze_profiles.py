"""Explain whole target gaps from identical native-event trace populations."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'ssd/tools/duet_timeline'))
from plot_consecutive_steps import read_profiles


def one(rows,label):
    out=[e for e in rows if e['label']==label]
    if len(out)!=1:raise ValueError((label,len(out)))
    return out[0]


def summarize(folder):
    j=json.loads((folder/'run.json').read_text())
    groups,_,hashes=read_profiles(folder)
    rows=[]
    for group_index,g in enumerate(j['groups']):
        epoch=len(j['warm_groups'])+group_index
        for s in g['steps']:
            if s['cap'] or s['clipped'] or s['batch']!=1:continue
            e=s['events'][0];sid=e['step_id']
            if sid<=10:continue
            target=groups['target'][(epoch,sid)];previous=groups['target'][(epoch,sid-1)]
            points=[one(previous,'final_logits')['wall_end_ns'],
                one(previous,'verify_sample_accept')['wall_end_ns'],
                one(previous,'target_postprocess')['wall_end_ns'],
                one(target,'target_send_request')['wall_start_ns'],
                one(target,'target_send_request')['wall_end_ns'],
                one(target,'target_recv_response_wait')['wall_end_ns'],
                one(target,'graph_pre')['wall_start_ns']]
            assert all(b>=a for a,b in zip(points,points[1:]))
            names=['accept_ms','bookkeeping_ms','schedule_request_prepare_ms','request_send_ms','response_ms','response_to_forward_ms']
            row={name:(b-a)/1e6 for name,a,b in zip(names,points,points[1:])}
            row['whole_gap_ms']=(points[-1]-points[0])/1e6
            assert abs(sum(row[k] for k in names)-row['whole_gap_ms'])<1e-8
            row.update(state=e['source'] if e['cache_hit'] else 0,epoch=epoch,step_id=sid,
                       step_ms=s['seconds']*1000)
            for name in ['graph_pre','graph_post','final_logits','verify_setup']:
                ev=one(target,name);row[name+'_ms']=(ev['wall_end_ns']-ev['wall_start_ns'])/1e6
            spec=one(target,'target_spec_wait');setup=one(target,'verify_setup')
            row['speculator_after_recv_ms']=(spec['wall_end_ns']-points[-2])/1e6
            row['verifier_runner_input_ms']=(setup['wall_start_ns']-spec['wall_end_ns'])/1e6
            rows.append(row)
    summaries=[]
    for state in (1,2,0,'all'):
        selected=[r for r in rows if state=='all' or r['state']==state]
        if not selected:continue
        summaries.append(dict(state=state,n=len(selected),**{k:mean(r[k] for r in selected) for k in rows[0] if k.endswith('_ms')}))
    return dict(tag=folder.name,summary=summaries,source_sha256=hashes,
                profile_on=True,first_steps_excluded=10,rows=len(rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=[]
    for model in ('7b','13b'):
        for tag in (f'before_{model}_profile_v3',f'snapshot_{model}_profile',f'inputs_{model}_profile',
                    f'before_{model}_profile_warm',f'restore_{model}_profile_warm'):
            folder=args.raw/'runs'/tag
            if not (folder/'status.json').exists():continue
            if json.loads((folder/'status.json').read_text())['status']!='complete':continue
            row=summarize(folder);result.append(row)
            all_rows=next(s for s in row['summary'] if s['state']=='all')
            print(tag,{k:round(v,4) if isinstance(v,float) else v for k,v in all_rows.items()},flush=True)
    args.output.write_text(json.dumps(dict(status='complete' if len(result)==10 else 'partial',runs=result,
        note='Profile-ON diagnostic only; not primary whole-step performance evidence. Same initial10/cap exclusions for all arms. Nested component labels are not additive. Early 7B profile_v3 included cold parent-q gather JIT and is not used for final improvement claims; profile_warm is the warm-cache attribution pair.'),indent=2)+'\n')


if __name__=='__main__':main()
