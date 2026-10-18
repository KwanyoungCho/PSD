"""Validate one complete full-dataset seed before rolling aggregation."""
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean,stdev
import csv,hashlib,importlib.util,json,math,subprocess,sys,os

HERE=Path(os.environ['VER15_REPORT'])
RAW=Path(os.environ['VER15_RAW'])
CODE=Path(os.environ['VER15_CHECKOUT'])
sys.path.insert(0,str(HERE))
from question_mean import POLICY as QUESTION_POLICY, summarize as question_summary
sys.path.insert(0,str(CODE/'ssd/bench'))
from mlsys_request_reporting import classify
from mlsys_measurement import performance_eligible
spec=importlib.util.spec_from_file_location('frozen_performance_policy',CODE/'ssd/ssd/utils/performance.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
TAGS=['seed42','seed123','seed2026'];LABELS={1:'P1 hit',2:'P2 hit',0:'Miss','all':'Overall'}


def dump(name,value):
    path=HERE/name;temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


def population(rows,required):
    by_turn=defaultdict(list)
    for r in rows:by_turn[r['uid']].append(r)
    result={}
    for q,expected in required.items():
        if not all(by_turn.get(uid) for uid in expected):continue
        values=[]
        for uid in sorted(expected):
            rr=by_turn[uid];secs=sum(r['seconds'] for r in rr)
            values.append(dict(tps=sum(r['emitted'] for r in rr)/secs,al=mean(r['al'] for r in rr),
                hit=mean(r['source'] in (1,2) for r in rr),step_ms=secs*1000/len(rr),steps=len(rr)))
        result[q]={k:mean(v[k] for v in values) for k in ('tps','al','hit','step_ms')}
        result[q].update(turns=len(values),steps=sum(v['steps'] for v in values))
    return result


def source_stats(rows):
    result={}
    for src in (1,2,0,'all'):
        rr=[r for r in rows if src=='all' or r['source']==src]
        secs=sum(r['seconds'] for r in rr);n=len(rr)
        result[str(src)]=dict(label=LABELS[src],n=n,share_pct=100*n/len(rows) if rows else None,
            hit_pct=100*sum(r['source'] in (1,2) for r in rr)/n if n else None,
            al=mean(r['al'] for r in rr) if n else None,step_ms=1000*secs/n if n else None,
            tps=sum(r['emitted'] for r in rr)/secs if n else None,
            target_call_and_accept_ms=mean(r['verify_ms'] for r in rr) if n else None,
            accepted_length_survival={str(x):sum(r['al']>=x for r in rr)/n if n else None for x in range(1,10)})
    return result


def validate(tag,expected,source_reference=None):
    folder=RAW/'runs'/tag
    status=json.loads((folder/'status.json').read_text())
    assert status['status']=='complete' and status['exit_code']==0 and status['source_unchanged'],status
    data=json.loads((folder/'run.json').read_text())
    assert data['status']=='complete' and data['paper_commit']==json.loads((RAW/'RELEASE.json').read_text())['commit']
    assert not data['paper_dirty']
    assert data['args']['seed']==int(tag[4:]) and data['args']['passes']==1
    assert data['env']['SSD_SEED']==data['env']['PAPER_SEED']==str(data['args']['seed'])
    job=json.loads((RAW/'PLAN.json').read_text())['job_template']
    config=dict(job['config'],seed=int(tag[4:]))
    contract=dict(config, batch=1, p1_nodes=config['p1_verify'], p2_nodes=config['p2_verify'])
    contract.pop('prompts')
    for key,value in contract.items():
        assert data['args'][key]==value,(tag,key,data['args'][key])
    assert data['kwargs']['dtype']=='bfloat16'
    assert all(data['env'][key]=='0' for key in ('SSD_PROFILE_DUET','SSD_PROFILE','SSD_PROFILE_DRAFT','SSD_DUET_EXIT_REPLICA','SSD_PROXY_STREAM'))
    assert data['env']['SSD_ASYNC_PROXY_SEND']=='1'
    assert len(data['groups'])==557 and len(data['warm_groups'])==3 and len(data['input_metadata'])==560
    assert data['experiment_protocol']['warmup_scope']=='within_dataset'
    assert data['experiment_protocol']['requested_seeds']==[42,123,2026] and data['experiment_protocol']['allow_nonnegative_continuation'] is True
    assert data['experiment_protocol']==json.loads((RAW/'PROTOCOL.json').read_text())
    frozen_helpers=json.loads((RAW/'ADAPTER_SHA256.json').read_text())
    assert all(sha256(RAW/path)==digest for path,digest in frozen_helpers.items())
    assert all(frozen_helpers['adapter/ssd/bench/'+name]==digest for name,digest in data['adapter_hashes'].items())
    assert [m['uid'] for m in data['input_metadata']]==[m['uid'] for m in expected]
    assert data['prompt_ids']==[m['input_ids'] for m in expected]
    records=data['request_records'];assert len(records)==557
    assert [r['paper_uid'] for r in records]==[m['uid'] for m in expected[3:]]
    warm_records=[r for g in data['warm_groups'] for r in g['request_records']]
    assert [r['paper_uid'] for r in warm_records+records]==[m['uid'] for m in expected]
    assert [g['index'] for g in data['warm_groups']+data['groups']]==list(range(560))
    categories=Counter(classify(r) for r in warm_records+records)
    for m,r in zip(expected,warm_records+records):
        assert r['effective_output_cap']==min(1024,2048-len(m['input_ids']))
        assert r['completion_tokens']<=r['effective_output_cap']
        assert r['target_temperature']==r['draft_temperature']==.7 and not r['ignore_eos']
    assert all(r['completion_tokens']<=1024 for r in records)
    assert all(len(r['output_ids'])==r['completion_tokens'] for r in records)
    assert sum(len(g['outputs'][0]) for g in data['groups'])==sum(r['completion_tokens'] for r in records)
    assert data['source_hashes']==source_reference if source_reference else True
    assert all(sha256(CODE/path)==digest for path,digest in data['source_hashes'].items())
    # Rebuild the independently recorded group/step mapping and invoke the
    # frozen existing policy on the native records; no new cutoff is added.
    native=[];events=[];rows=[];cap_only=[];flat=[];required=defaultdict(set)
    for m in expected:required[str(m['question_id'])].add(m['uid'])
    for scope,(m,g,r) in enumerate(zip(expected[3:],data['groups'],records)):
        assert g['n']==1 and g['phase']=='measured_0'
        assert g['request_records'][0]['paper_uid']==m['uid']
        assert g['outputs'][0]==r['output_ids']
        required[str(m['question_id'])].add(m['uid'])
        context=len(m['input_ids'])
        for local,s in enumerate(g['steps'],1):
            assert s['batch']==1 and len(s['events'])==1
            assert math.isfinite(s['seconds']) and s['seconds']>0
            e=s['events'][0];assert e['step_id']==local
            assert e['verification']=='target_sample_match'
            assert e['source'] in (0,1,2) and bool(e['cache_hit'])==(e['source'] in (1,2))
            assert len(s['emitted_by_request'])==1 and s['emitted_by_request'][0]==s['emitted']
            assert 0<=s['emitted']<=e['accepted_len']
            start=len(events);events.extend(s['events'])
            native.append(dict(seconds=s['seconds'],emitted_tokens=s['emitted'],batch_size=s['batch'],
                event_start=start,event_end=len(events),output_cap_reached=s['cap'],clipped=s['clipped'],duration_scope=scope))
            row=dict(uid=m['uid'],question=str(m['question_id']),group=m['group'],seconds=s['seconds'],
                emitted=s['emitted'],al=e['accepted_len'],source=e['source'],verify_ms=sum(s['verify'])*1000,context=context)
            if s['batch']==1 and not s['cap'] and not s['clipped']:cap_only.append(row)
            if performance_eligible(s,1):rows.append(row)
            flat.append(s);context+=s['emitted']
        assert context==r['end_context_length']
    assert len(required)==480 and len(flat)==len(data['steps'])
    assert all(a['seconds']==b['seconds'] and a['events']==b['events'] and a['emitted']==b['emitted'] for a,b in zip(flat,data['steps']))
    selected=policy.performance_summary(dict(decode_steps=native,phase_events=events),1)
    indices=[i for i,s in enumerate(flat) if performance_eligible(s,1)]
    assert indices==selected['included_step_indices']==data['measurement']['included_step_indices']
    assert selected['duration_filter']['median_factor']==2.0
    for key,ref in (('tps','tps'),('al','al'),('step_ms','step_ms'),('cache_hit','hit')):
        assert math.isclose(selected[key],data['summary'][ref],rel_tol=1e-12,abs_tol=1e-12),(tag,key)
    stats=source_stats(rows)
    assert math.isclose(stats['all']['tps'],data['summary']['tps'],rel_tol=1e-12)
    guards=[r['paper_uid'] for r in records if r['classification']=='context_guard_no_decode']
    guard_ends=[r['paper_uid'] for r in records if r['finish_reason']=='context_limit_guard']
    question_result=question_summary(data['groups'])
    raw_question_result=question_summary(data['groups'],filtered=False)
    valid_uids=set(question_result['valid_turn_uids'])
    population_stats={}
    for q,rr in question_result['by_question'].items():
        if not required[q]<=valid_uids:continue
        values=rr['sources']['all']
        population_stats[q]=dict(tps=values['tps'],al=values['al'],hit=values['hit_pct']/100,
            step_ms=values['step_ms'],turns=rr['turns'],steps=rr['steps'])
    info=dict(seed=int(tag[4:]),gpus=status['gpus'],questions=480,turns=560,warmup_turns=3,performance_input_turns=557,
              warmup_uids=[r['paper_uid'] for r in warm_records],
              request_classifications=dict(categories),zero_decode_uids=guards,context_guard_ended_uids=guard_ends,
              output_cap_turns=sum(r['completion_tokens']==r['effective_output_cap'] for r in records),
              output_1024_turns=sum(r['completion_tokens']==1024 for r in records),
              actual_output_tokens=sum(r['completion_tokens'] for r in records),
              reporting_policy=QUESTION_POLICY,sources=question_result['sources'],
              question_mean=question_result,raw_question_mean=raw_question_result,
              step_pooled_sources=stats,native_summary=data['summary'],raw_cap_clip_summary=data['summary_unfiltered'],
              cap_only_without_long_filter=source_stats(cap_only),
              exclusion_counts=dict(all_steps=len(flat),included_steps=len(rows),
                  cap_or_clip=sum(s['cap'] or s['clipped'] for s in flat),
                  long_step=selected['duration_filter']['excluded_steps'],
                  excluded_long_seconds=selected['duration_filter']['excluded_seconds'],
                  excluded_long_tokens=selected['duration_filter']['excluded_emitted_tokens']),
              filtered_question_population=dict(questions=len(population_stats),
                  missing_question_ids=sorted(set(required)-set(population_stats)),
                  turns=sum(v['turns'] for v in population_stats.values())),
              raw_request_population=data['request_population'])
    checks=dict(status='complete',seed=int(tag[4:]),expected_questions=480,expected_turns=560,
                all_input_ids_untruncated=True,ordered_uids_match=True,strict_request_accounting=True,explicit_sampler_seed_matches=True,first_three_excluded_without_duplicate_inputs=True,per_request_cap_and_temperatures_checked=True,
                source_and_hit_consistent=True,actual_output_tokens_reconciled=True,performance_reaggregation_equal=True,
                same_filtered_population_for_tps_al_hit=True,raw_source_hashes_match=True,
                zero_decode_classification_explicit=True,runtime_clean_at_run=True,frozen_benchmark_helpers_unchanged=True,
                reporting_policy=QUESTION_POLICY,question_mean_source_sha256=sha256(Path(__file__).with_name('question_mean.py')),
                run_json_sha256=sha256(folder/'run.json'),provenance_sha256=sha256(folder/'PROVENANCE.json'))
    info['parameters']=config
    capacity=json.loads((folder/'DRAFT_CAPACITY.json').read_text())
    assert capacity['fraction']==.35 and capacity['allocated_blocks']>=capacity['required_blocks_with_4x_margin']
    assert not (folder/'UNEXPECTED_PREEMPTION.json').exists()
    info['draft_capacity']=capacity
    checks['uniform_draft_capacity_and_no_preemption']=True
    return info,population_stats,rows,checks,data['source_hashes']


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()



if __name__=='__main__':
    tag=sys.argv[1]
    expected=json.loads((RAW/'INPUTS_FULL.json').read_text())
    info, population_stats, rows, checks, hashes=validate(tag, expected)
    (HERE/'validated').mkdir(exist_ok=True)
    dump('validated/'+tag+'.json', dict(info=info,checks=checks,questions=population_stats))
    print(json.dumps(dict(tag=tag,checks=checks,overall=info['sources']['all'])),flush=True)
