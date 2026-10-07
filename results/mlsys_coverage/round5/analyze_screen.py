"""Per-job AL/cache/timing diagnostics; profile TPS is explicitly diagnostic."""
from pathlib import Path
import argparse, collections, json
import numpy as np

HERE=Path(__file__).resolve().parent


def stats(values):
    values=[float(v) for v in values]
    return None if not values else dict(n=len(values),mean=float(np.mean(values)),
        p50=float(np.median(values)),p90=float(np.quantile(values,.9)))


def grouped(path,marker):
    rows=json.loads(path.read_text())
    starts=[i for i,e in enumerate(rows) if e['label']==marker and e.get('step_id')==1]
    if starts:rows=rows[starts[-1]:]
    out=collections.defaultdict(lambda:collections.defaultdict(list))
    for e in rows:
        if e.get('step_id') is not None:out[e['step_id']][e['label']].append(e)
    return out


def analyze(report):
    data=json.loads(report.read_text())
    if data.get('status')!='complete':return None
    cell=data['cells'][-1];b=cell['batch'];events=cell['metrics']['phase_events']
    evs=collections.defaultdict(list)
    for e in events:evs[e['step_id']].append(e)
    row=dict(name=report.stem,path=str(report.relative_to(HERE)),args=data['args'],
        env=data['env'],summary=cell['summary'],all_cells=[c['summary'] for c in data['cells']])
    source=lambda e:int(e['cache_hit']) if data['args']['mode']=='ssd' else e.get('source')
    row['source_order']=['miss','cache_hit','unused'] if data['args']['mode']=='ssd' else ['miss','p1','p2']
    row['conditional_al']={str(s):stats([e['accepted_len'] for e in events
        if source(e)==s and not e.get('output_cap_reached') and not e.get('clipped')]) for s in (0,1,2)}
    excluded=json.loads((HERE/'EXCLUSIONS.json').read_text()) if (HERE/'EXCLUSIONS.json').exists() else []
    row['timing_excluded']=any(x['job']==report.stem for x in excluded)
    row['valid_nodes']=stats([e['valid_k'] for e in events])
    row['tree_shape']={k:stats([e[k] for e in events if k in e]) for k in
        ('tree_max_depth','tree_sibling_width','reached_depth_ceiling','node_utilization')}
    row['all_hit_rate']=np.mean([all(e['cache_hit'] for e in a) for a in evs.values() if len(a)==b])
    recorded_prof=Path(data['env'].get('SSD_PROFILE_DIR','/nonexistent'))
    # Raw provenance keeps the original absolute checkout. Offline replay on
    # another server must prefer the restored profiles in THIS checkout.
    local_prof=HERE/'profiles'/recorded_prof.name
    prof=local_prof if local_prof.is_dir() else recorded_prof
    # Retries retain failed initialization traces. Filesystem iteration order
    # is not chronological: use each process's newest completed trace.
    target=sorted(prof.glob('*target_rank0*json'),key=lambda p:p.stat().st_mtime_ns)
    draft=sorted(prof.glob('*draft*json'),key=lambda p:p.stat().st_mtime_ns)
    if target and draft:
        row['profile_paths']=[str(target[-1].relative_to(HERE)),str(draft[-1].relative_to(HERE))]
        tg=grouped(target[-1],'target_send_request')
        dg=grouped(draft[-1],'batch_miss_draft' if data['args']['mode']=='duet-tree' else 'draft_recv_request')
        metrics=collections.defaultdict(list);steps=[]
        capture={sid for sid,es in list(tg.items())+list(dg.items()) if any('capture' in k for k in es)}
        blocked=capture|{sid+1 for sid in capture}
        steady=[]
        for step in cell['metrics'].get('decode_steps',[]):
            es=events[step['event_start']:step['event_end']]
            if not es:continue
            sid=es[0]['step_id']
            if sid>=20 and sid not in blocked and sid in tg and sid in dg and len(es)==b and not step['output_cap_reached'] and not step['clipped']:
                steady.append(step)
        row['profile_steady_tps']=(sum(s['emitted_tokens'] for s in steady)/sum(s['seconds'] for s in steady)) if steady else None
        row['profile_steady_steps']=len(steady)
        row['profile_steady_step_ms']=1000*sum(s['seconds'] for s in steady)/len(steady) if steady else None
        row['profile_steady_al']=sum(s['emitted_tokens'] for s in steady)/(len(steady)*b) if steady else None
        row['profile_capture_steps']=sorted(capture)
        for sid,es in evs.items():
            if sid<20 or len(es)!=b or sid in blocked:continue
            te=tg.get(sid,{});de=dg.get(sid,{})
            if any('capture' in k for k in de):continue
            both=dict(te,**de)
            for k,vs in both.items():
                if all('cuda_ms' in v for v in vs):metrics[k].append(sum(v['cuda_ms'] for v in vs))
            end=lambda label:both[label][-1]['wall_end_ns']/1e6
            if all(k in both for k in ('batch_p1_total','batch_proxy_receive','batch_p2_total','target_postprocess')):
                p1=end('batch_proxy_receive')-end('batch_p1_total')
                p2=end('target_postprocess')-end('batch_p2_total')
                metrics['p1_slack_ms'].append(p1);metrics['p2_slack_ms'].append(p2)
                steps.append(dict(step=sid,p1_slack=p1,p2_slack=p2))
                if 'batch_target_final_logits' in both:
                    model=end('batch_target_final_logits')-end('batch_p2_total')
                    metrics['p2_model_slack_ms'].append(model)
                    steps[-1]['p2_model_slack']=model
            if all(k in both for k in ('ssd_populate_cache','target_postprocess')):
                metrics['ssd_build_slack_ms'].append(end('target_postprocess')-end('ssd_populate_cache'))
        row['profile']={k:stats(v) for k,v in metrics.items()}
        row['phase_hidden']={name:float(np.mean([s[name]>=0 for s in steps if name in s]))
            for name in ('p1_slack','p2_slack','p2_model_slack')
            if any(name in s for s in steps)}
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--glob',default='*/*json');a=parser.parse_args()
    rows=[]
    for p in sorted(HERE.glob(a.glob)):
        if p.name in ('plan.json','campaign.json'):continue
        try:r=analyze(p)
        except (KeyError,TypeError):continue
        if r:rows.append(r)
    (HERE/'SCREEN_RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:
        s=r['summary'];p=r.get('profile',{})
        timing={k:round(p[k]['mean'],2) for k in ('target_spec_wait','verify_replay','batch_target_pre',
            'batch_target_post','batch_p1_total','batch_p2_total','p1_slack_ms','p2_slack_ms',
            'ssd_build_tree','ssd_decode_tree','ssd_populate_cache','ssd_build_slack_ms') if p.get(k)}
        print(r['name'],f"AL*={s['boundary_excluded_al']:.3f} TPSdiag={s['decode_tps']:.1f} steady={r.get('profile_steady_tps') or 0:.1f} hit={s['cache_hit']:.3f}",timing)


if __name__=='__main__':main()
