"""CPU reporting: turn ratios, then equal question weights, then seed means."""
from collections import Counter, defaultdict
from statistics import mean

POLICY='turn_then_question_mean_existing_step_filter_v1'
KEYS=('share_pct','hit_pct','al','step_ms','tps','target_call_and_accept_ms')
LABELS={1:'P1 hit',2:'P2 hit',0:'Miss','all':'Overall'}


def average(values):
    values=[value for value in values if value is not None]
    return mean(values) if values else None


def summarize(groups, *, filtered=True):
    by_question=defaultdict(list)
    excluded=[];seen=set();steps_by_source=Counter();turns=[]
    for group in groups:
        assert group['n']==1 and len(group['request_records'])==1
        record=group['request_records'][0]
        uid=record['paper_uid'];qid=str(record['question_id'])
        assert uid not in seen
        seen.add(uid)
        if record['completion_tokens']<2 or record['decode_total_time']<=0:
            excluded.append(dict(uid=uid,question_id=qid,
                reason='output_below_two' if record['completion_tokens']<2 else 'no_decode'))
            continue
        original=group['steps']
        if filtered:
            assert all('performance_included' in step for step in original)
        selected=[(i,step) for i,step in enumerate(original)
                  if not filtered or step['performance_included']]
        if not selected:
            excluded.append(dict(uid=uid,question_id=qid,reason='no_eligible_steps'))
            continue
        # This engine emits the first target token on its first decode step.
        # Remove that token only if the step containing it was retained.
        first_emission=next((i for i,s in enumerate(original) if s['emitted']>0),None)
        metrics={};counts={}
        for source in (1,2,0,'all'):
            rr=[(i,s) for i,s in selected if source=='all' or s['events'][0]['source']==source]
            n=len(rr);counts[str(source)]=n
            steps_by_source[str(source)]+=n
            seconds=sum(s['seconds'] for _,s in rr)
            tokens=sum(s['emitted'] for _,s in rr)-int(any(i==first_emission for i,_ in rr))
            assert tokens>=0
            events=[s['events'][0] for _,s in rr]
            metrics[str(source)]=dict(
                share_pct=100*n/len(selected),
                hit_pct=100*sum(e['cache_hit'] for e in events)/n if n else None,
                al=mean(e['accepted_len'] for e in events) if n else None,
                step_ms=1000*seconds/n if n else None,
                tps=tokens/seconds if seconds else None,
                target_call_and_accept_ms=1000*sum(sum(s['verify']) for _,s in rr)/n if n else None,
                accepted_length_survival={str(x):sum(e['accepted_len']>=x for e in events)/n if n else None
                                          for x in range(1,10)})
        turn=dict(uid=uid,question_id=qid,sources=metrics,steps=counts)
        by_question[qid].append(turn);turns.append(turn)
    questions={}
    for qid,tt in by_question.items():
        sources={}
        for source in ('1','2','0','all'):
            sources[source]={key:average(t['sources'][source][key] for t in tt) for key in KEYS}
            sources[source]['accepted_length_survival']={str(x):average(
                t['sources'][source]['accepted_length_survival'][str(x)] for t in tt) for x in range(1,10)}
        questions[qid]=dict(sources=sources,uids=[t['uid'] for t in tt],turns=len(tt),
            steps=sum(t['steps']['all'] for t in tt))
    sources={}
    for source in ('1','2','0','all'):
        summary={key:average(q['sources'][source][key] for q in questions.values()) for key in KEYS}
        summary.update(label=LABELS[int(source)] if source!='all' else LABELS['all'],
            n=steps_by_source[source],
            metric_questions={key:sum(q['sources'][source][key] is not None for q in questions.values()) for key in KEYS},
            metric_turns={key:sum(t['sources'][source][key] is not None for t in turns) for key in KEYS},
            accepted_length_survival={str(x):average(q['sources'][source]['accepted_length_survival'][str(x)]
                for q in questions.values()) for x in range(1,10)})
        sources[source]=summary
    if questions:
        assert abs(sum(sources[s]['share_pct'] for s in ('1','2','0'))-100)<1e-10
        assert abs(sources['all']['hit_pct']-sum(sources[s]['share_pct'] for s in ('1','2')))<1e-10
    return dict(policy=POLICY if filtered else 'turn_then_question_mean_unfiltered_v1',
        filtered=filtered,sources=sources,questions=len(questions),turns=len(turns),
        measured_turns=len(groups),excluded_turns=excluded,
        excluded_turn_reasons=dict(Counter(r['reason'] for r in excluded)),
        valid_turn_uids=[t['uid'] for t in turns],by_question=questions,
        definitions=dict(aggregation='turn ratios -> mean valid turns within question -> equal-weight question mean; across-seed mean is computed separately',
            tps='post-first-token emitted tokens / retained decode seconds; subtract the first emitted token only if its step is retained',
            conditional='P1/P2/Miss AL and latency average only turns/questions containing that source; absent-source rate is zero, absent-source AL/latency is undefined',
            selection='Existing cap/clip, full-batch and >2x grouped-median step exclusions preserved; output<2 or no-decode turns excluded',
            mean_identity='Mean AL is not mean TPS times mean latency; conditional means may have different valid question populations'))
