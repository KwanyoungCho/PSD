"""Question-cluster statistics for measured (last) passes, including cap audit."""
import collections,gzip,hashlib,json
from pathlib import Path
import numpy as np

# Per-question sufficient statistics, so bootstrap resamples complete questions.
COLS=['accepted','events','clean_accepted','clean_events','emitted',
      'miss_count','p1_count','p2_count','miss_clean_count','p1_clean_count','p2_clean_count',
      'miss_clean_accepted','p1_clean_accepted','p2_clean_accepted','valid_nodes']


def read(path):
    path=Path(path)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)


def collect(report,cell_index=-1):
    c=report['cells'][cell_index];events=c['metrics']['phase_events']
    ids=sorted({e['seq_id'] for e in events});n=len(report['question_indexes'])
    if len(ids)!=n:raise ValueError('Sequence/question identity mismatch')
    mapping={sid:i for i,sid in enumerate(ids)}
    out=np.zeros((n,len(COLS)),dtype=np.float64)
    batches=collections.defaultdict(list)
    for e in events:
        x=out[mapping[e['seq_id']]]
        # SSD phase_source is always zero, even on hits. Its source1 column
        # below means ordinary cache hit, not DUET P1.
        src=int(e['cache_hit']) if report['args']['mode']=='ssd' else e['source']
        clean=not(e['output_cap_reached'] or e['clipped'])
        x[0]+=e['accepted_len'];x[1]+=1;x[4]+=e['emitted_len'];x[5+src]+=1;x[14]+=e['valid_k']
        if clean:x[2]+=e['accepted_len'];x[3]+=1;x[8+src]+=1;x[11+src]+=e['accepted_len']
        batches[e['step_id']].append(e)
    total=sum(len(o['token_ids']) for o in c['outputs'])
    steps=c['metrics']['decode_steps'];clean_steps=[s for s in steps if not(s['output_cap_reached'] or s['clipped'])]
    checks=dict(
        returned_matches_events=total==out[:,4].sum(),
        returned_matches_steps=total==sum(s['emitted_tokens'] for s in steps),
        returned_matches_decode_counter=total==c['metrics']['decode_total_tokens'],
        step_event_partition=sum(s['event_end']-s['event_start'] for s in steps)==len(events),
        caps_respected=all(0<len(o['token_ids'])<=cap for o,cap in zip(c['outputs'],report['output_caps'])))
    checks={key:bool(value) for key,value in checks.items()}
    if not all(checks.values()):raise ValueError(checks)
    row=aggregate(out)
    query_steps=[es for es in batches.values() if 'physical_batch_queries' in es[0]]
    query_utilization=(sum(sum(e['valid_k']+1 for e in es) for es in query_steps)/
                       sum(es[0]['physical_batch_queries'] for es in query_steps)) if query_steps else None
    query_shapes=collections.Counter((len(es),es[0]['physical_batch_queries'],
                         max(e['valid_k']+1 for e in es)) for es in query_steps)
    if abs(row['cache_hit']-c['summary']['cache_hit'])>1e-12:
        raise ValueError('Cache source labels disagree with the saved hit counter')
    if abs(row['boundary_excluded_al']-c['summary']['boundary_excluded_al'])>1e-12:
        raise ValueError('AL event aggregation disagrees with the saved counter')
    row.update(per_question=out.tolist(),question_indexes=report['question_indexes'],checks=checks,
        source_order=['miss','cache_hit','unused'] if report['args']['mode']=='ssd' else ['miss','p1','p2'],
        target_seed=c['seed'],draft_seed=report['env'].get('SSD_SEED'),batch=c['batch'],temperature=c['temperature'],
        decode_tps=total/c['metrics']['decode_total_time'],
        boundary_excluded_step_tps=sum(s['emitted_tokens'] for s in clean_steps)/sum(s['seconds'] for s in clean_steps),
        timed_step_mean_tokens=sum(s['emitted_tokens'] for s in clean_steps)/len(clean_steps),
        timed_step_mean_ms=1000*sum(s['seconds'] for s in clean_steps)/len(clean_steps),
        timed_step_count=len(clean_steps),
        end_to_end_tps=c['summary']['end_to_end_tps'],
        output_cap_requests=sum(len(o['token_ids'])==cap for o,cap in zip(c['outputs'],report['output_caps'])),
        complete_batch_all_hit_fraction=float(np.mean([all(e['cache_hit'] for e in b) for b in batches.values() if len(b)==c['batch']])),
        boundary_excluded_steps=len(steps)-len(clean_steps),output_tokens=total,
        physical_query_utilization=query_utilization,
        query_shape_counts=[dict(active_batch=key[0],physical_queries=key[1],
            max_logical_queries=key[2],steps=value) for key,value in sorted(query_shapes.items())],
        truncated_prompts=report['truncated_prompts'],
        prompt_tokens_p50=float(np.median([len(p) for p in report['prompt_ids']])),
        prompt_tokens_p90=float(np.quantile([len(p) for p in report['prompt_ids']],.9)),
        source_sha256=report['source_sha256'],prompt_sha256=report['prompt_sha256'],
        output_caps_sha256=hashlib.sha256(json.dumps(report['output_caps']).encode()).hexdigest(),
        cold_decode_tps=report['cells'][0]['summary']['decode_tps'],pass_count=len(report['cells']))
    for metric in ('decode_tps','boundary_excluded_step_tps'):
        if not np.isclose(row[metric],c['summary'][metric],rtol=1e-12,atol=1e-12):
            raise ValueError(f'{metric} disagrees with the saved summary')
    return row


def assert_paired_runs(left,right):
    for key in ('batch','temperature','target_seed','draft_seed','question_indexes',
                'prompt_sha256','output_caps_sha256'):
        if left[key]!=right[key]:raise ValueError(f'Unpaired benchmark field: {key}')
    if 'gpus' in left and 'gpus' in right and left['gpus']!=right['gpus']:
        raise ValueError('Unpaired GPU IDs')


def aggregate(per):
    x=np.asarray(per).sum(0)
    return dict(questions=len(per),events=int(x[1]),clean_events=int(x[3]),
        al=x[0]/x[1],boundary_excluded_al=x[2]/x[3],emitted_al=x[4]/x[1],
        boundary_excluded_events=int(x[1]-x[3]),mean_valid_nodes=x[14]/x[1],
        cache_hit=(x[6]+x[7])/x[1],source_weights=(x[5:8]/x[1]).tolist(),
        clean_source_weights=(x[8:11]/x[3]).tolist(),
        clean_conditional_al=[x[11+i]/x[8+i] if x[8+i] else None for i in range(3)])


def compare(left,right,eligible=None,seed=20261008):
    """left-right AL with paired questions; repeated processes stay clustered."""
    a=np.sum([np.array(r['per_question']) for r in left],axis=0)
    b=np.sum([np.array(r['per_question']) for r in right],axis=0)
    if a.shape!=b.shape:raise ValueError('Mismatched question count')
    if eligible is not None:a=a[eligible];b=b[eligible]
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(a),(4000,len(a)))
    aa=a[:,[2,3]][ix].sum(1);bb=b[:,[2,3]][ix].sum(1)
    delta=aa[:,0]/aa[:,1]-bb[:,0]/bb[:,1]
    al=aggregate(a);bl=aggregate(b)
    return dict(questions=len(a),left=al,right=bl,
        delta_al=al['boundary_excluded_al']-bl['boundary_excluded_al'],
        delta_al_ci95=np.quantile(delta,[.025,.975]).tolist(),
        al_relative_pct=100*(al['boundary_excluded_al']/bl['boundary_excluded_al']-1),
        note='Question bootstrap; repeated process samples kept in each question cluster. Not a latency confidence interval.')
