"""Explain phase deadlines using paired step events, including mixed misses."""
import collections,json
from pathlib import Path
import numpy as np
from make_plans import HERE,save
from analyze_screen import analyze,grouped,stats


def detail(row,sample_step=None):
    raw=json.loads((HERE/row['path']).read_text());cell=raw['cells'][-1]
    paths=[HERE/p for p in row['profile_paths']]
    tg=grouped(paths[0],'target_send_request')
    dg=grouped(paths[1],'batch_miss_draft' if raw['args']['mode']=='duet-tree' else 'draft_recv_request')
    events=collections.defaultdict(list)
    for e in cell['metrics']['phase_events']:events[e['step_id']].append(e)
    captures={sid for sid,es in list(tg.items())+list(dg.items()) if any('capture' in k for k in es)}
    excluded=captures|{sid+1 for sid in captures};steps=[];boundary=[]
    for sid,evs in events.items():
        if sid<20 or sid in excluded or len(evs)!=cell['batch']:continue
        if any(e.get('output_cap_reached') or e.get('clipped') for e in evs):
            boundary.append(sid)
            continue
        te=tg.get(sid,{});de=dg.get(sid,{});both=dict(te,**de)
        required=['batch_p1_total','batch_p2_total','batch_proxy_receive','batch_target_pre',
                  'batch_target_post','batch_target_final_logits','target_postprocess']
        if any(k not in both for k in required):continue
        end=lambda k:both[k][-1]['wall_end_ns']/1e6
        origin=both['batch_target_pre'][0]['wall_start_ns']/1e6
        metrics={k:sum(v['cuda_ms'] for v in vs) for k,vs in both.items() if all('cuda_ms' in v for v in vs)}
        metrics.update(p1_slack_ms=end('batch_proxy_receive')-end('batch_p1_total'),
            p2_model_slack_ms=end('batch_target_final_logits')-end('batch_p2_total'),
            p2_ready_slack_ms=end('target_postprocess')-end('batch_p2_total'),
            p2_dispatch_gap_ms=both['batch_p2_total'][0]['wall_start_ns']/1e6-
                max(end('batch_p1_total'),end('batch_proxy_receive')))
        for key,label in [('proxy_arrival_from_pre_ms','batch_proxy_receive'),
                ('p1_finish_from_pre_ms','batch_p1_total'),('p2_finish_from_pre_ms','batch_p2_total'),
                ('target_final_from_pre_ms','batch_target_final_logits'),
                ('target_ready_from_pre_ms','target_postprocess')]:
            metrics[key]=end(label)-origin
        next_events=events.get(sid+1,[]);next_target=tg.get(sid+1,{})
        same_requests=sorted(e['seq_id'] for e in evs)==sorted(e['seq_id'] for e in next_events)
        if (same_requests and sid+1 not in excluded and
                not any(e.get('output_cap_reached') or e.get('clipped') for e in next_events) and
                'target_send_request' in next_target and 'target_spec_wait' in next_target):
            next_start=next_target['target_send_request'][0]['wall_start_ns']/1e6
            metrics['p2_next_request_slack_ms']=next_start-end('batch_p2_total')
            metrics['target_next_request_gap_ms']=next_start-end('target_postprocess')
            metrics['next_target_wait_ms']=sum(e['cuda_ms'] for e in next_target['target_spec_wait'])
        timeline={k:dict(start_ms=vs[0]['wall_start_ns']/1e6-origin,
                        end_ms=vs[-1]['wall_end_ns']/1e6-origin)
                  for k,vs in both.items() if k in required+['batch_glue','batch_miss_draft',
                      'target_spec_wait','batch_tree_accept','batch_tree_commit','batch_proxy_side','batch_target_proxy']}
        steps.append(dict(step=sid,all_hit=all(e['cache_hit'] for e in evs),
            hit_rate=sum(e['cache_hit'] for e in evs)/len(evs),metrics=metrics,timeline=timeline))
    groups={}
    for name,ss in [('full_batch',steps),('all_hit',[s for s in steps if s['all_hit']]),
                    ('some_miss',[s for s in steps if not s['all_hit']])]:
        keys=set(k for s in ss for k in s['metrics'])
        groups[name]=dict(n=len(ss),metrics={k:stats([s['metrics'][k] for s in ss if k in s['metrics']]) for k in sorted(keys)},
            hidden={k:float(np.mean([s['metrics'][k]>=0 for s in ss if k in s['metrics']]))
                    if any(k in s['metrics'] for s in ss) else None
                    for k in ('p1_slack_ms','p2_model_slack_ms','p2_ready_slack_ms','p2_next_request_slack_ms')})
    # A real step near the median total duration, not concatenated medians.
    sample=None
    if steps:
        duration=lambda s:s['timeline']['target_postprocess']['end_ms']-s['timeline']['batch_target_pre']['start_ms']
        center=float(np.median([duration(s) for s in steps]));sample=min(steps,key=lambda s:abs(duration(s)-center))
        if sample_step is not None:
            sample=next(s for s in steps if s['step']==sample_step)
    return dict(name=row['name'],path=row['path'],args=row['args'],groups=groups,
                eligible_step_ids=[s['step'] for s in steps],
                boundary_excluded_step_ids=boundary,
                _step_endpoints={s['step']:{k:v for k,v in s['metrics'].items()
                    if k.endswith(('_from_pre_ms','_slack_ms','_gap_ms')) or k=='next_target_wait_ms'} for s in steps},
                source_order=row['source_order'],conditional_al=row['conditional_al'],
                all_hit_rate=row['all_hit_rate'],representative_step=sample)


def signature(row):
    from make_tree_plan import kwargs_from
    return json.dumps(dict(params=kwargs_from(row),miss=row['env'].get('SSD_DUET_MISS_K','4')),sort_keys=True)


def main():
    screen=json.loads((HERE/'SCREEN_RESULTS.json').read_text());chosen={};roles={}
    for row in screen:
        if row['name'].startswith(('llama2_b1_duet_e','llama2_b8_duet_e','llama3_b1_duet_e','llama3_b8_duet_e')):
            chosen[row['name']]=row;roles[row['name']]=['anchor']
    for model in ('llama2','llama3'):
        for b in (1,8):
            frozen=json.loads((HERE/f'{model}_b{b}_FROZEN.json').read_text())['presets']
            for role in ('duet_fast','duet_al'):
                point=frozen[frozen[role].get('alias',role)]
                matches=[r for r in screen if r['name'].startswith(f'{model}_b{b}_') and
                    r.get('profile_paths') and r['args']['max_new_tokens']==96 and
                    r['args']['limit']==0 and signature(r)==signature(point)]
                if not matches:raise ValueError(f'No diagnostic at frozen {model} B{b} {role}')
                row=matches[-1];chosen[row['name']]=row;roles.setdefault(row['name'],[]).append(f'frozen_{role}')
    for directory in HERE.glob('*_profile'):
        if not directory.is_dir():continue
        for path in directory.glob('llama*.json'):
            row=analyze(path)
            if row and row.get('profile_paths'):
                chosen[row['name']]=row
                role='postopt_neighbor' if '_postopt_' in directory.name else 'implementation_profile'
                roles.setdefault(row['name'],[]).append(role)
    output=[]
    for name in sorted(chosen):
        row=chosen[name];data=detail(row)
        if '_profile_s1' in name:
            control=next(x for x in output if x['name']==name.replace('_profile_s1','_profile_s0'))
            common=set(control['eligible_step_ids']) & set(data['eligible_step_ids'])
            if not common:raise ValueError(f'No common eligible stream-control step: {name}')
            paired=min(common,key=lambda sid:abs(sid-control['representative_step']['step']))
            if paired!=control['representative_step']['step']:
                control['representative_step']=detail(chosen[control['name']],paired)['representative_step']
            data['representative_step']=detail(row,paired)['representative_step']
            keys=data['_step_endpoints'][paired]
            data['paired_stream_endpoint_deltas']=dict(control=control['name'],steps=len(common),
                definition='Same-step enabled minus disabled; each endpoint is relative to its target-pre origin.',
                metrics={key:stats([data['_step_endpoints'][sid][key]-control['_step_endpoints'][sid][key]
                          for sid in sorted(common) if key in data['_step_endpoints'][sid]
                          and key in control['_step_endpoints'][sid]]) for key in keys})
        data['roles']=roles[name];output.append(data)
    for row in output:row.pop('_step_endpoints')
    save('BREAKDOWN.json',output)
    lines=['# Paired phase breakdown','',
        'Instrumented diagnostic medians (ms), excluding labeled lazy graph-capture steps and the following step, the first20 steps, incomplete batches and cap/clipping boundary batches. '
        'Historical SCREEN_RESULTS component medians included complete boundary batches; their profiled TPS selection already excluded them. '
        'Older screening profiles lack the later target/glue forward-capture labels; their total timings may still contain unlabeled captures. '
        'P2-next-request slack, when available, uses the next request of the same unchanged full batch, excluding boundaries/captures. '
        'This separates a strict current-target deadline from actual next-request exposure; next-step waits are not assigned to the current step. '
        'Negative slack means a missed boundary. All-hit and some-miss groups, real representative timelines and sample counts are retained in BREAKDOWN.json. '
        'Do not add separate medians to reconstruct a measured timeline or interpret profiled TPS as final speed.','',
        '| Setting | Role | n | target pre | proxy | post | P1 | P2 | P2 transition gap | P1 slack | P2 final-logits slack | P2 ready slack |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in output:
        g=row['groups']['full_batch'];m=g['metrics']
        keys=['batch_target_pre','batch_target_proxy','batch_target_post','batch_p1_total','batch_p2_total',
              'p2_dispatch_gap_ms','p1_slack_ms','p2_model_slack_ms','p2_ready_slack_ms']
        values=[f"{m[k]['p50']:.2f}" if k in m else '-' for k in keys]
        lines.append('| '+ ' | '.join([row['name'],','.join(row['roles']),str(g['n'])]+values)+' |')
    (HERE/'BREAKDOWN_TABLES.md').write_text('\n'.join(lines)+'\n')
    print('Diagnostic settings',len(output))

if __name__=='__main__':main()
