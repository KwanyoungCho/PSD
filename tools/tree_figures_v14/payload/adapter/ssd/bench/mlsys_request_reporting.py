"""Benchmark-only termination validation and explicit question populations.

Request decode time is the sum of shared batch-step times while that request
is active. Request TPS is NOT aggregate serving throughput. Engine runtime,
context guards and historical raw files are never modified by this module.
"""
import math
from collections import defaultdict

METRICS = ('request_decode_tps', 'al', 'hit', 'p1_al', 'p2_al', 'step_ms')


def request_reporting_supported(mode):
    # AR emits its first token during prefill; synchronous SD does not emit
    # the async per-request verification events used by this schema.
    return mode in ('tree', 'chain', 'ssd')


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def safe_tps(tokens, seconds, steps):
    if not math.isfinite(seconds) or seconds < 0 or tokens < 0 or steps < 0:
        raise ValueError('Invalid decode counts or time')
    if steps == 0:
        if tokens or seconds:
            raise ValueError('Nonempty counts without decode steps')
        return None
    if seconds == 0:
        raise ValueError('Decode steps must have positive time')
    return tokens / seconds


def classify(record):
    """Accept zero time only for an explicitly observed context guard."""
    seconds = record['decode_total_time']
    output = record['output_ids']
    n = record['completion_tokens']
    prompt = record['input_tokens']
    end = record['end_context_length']
    if (not math.isfinite(seconds) or seconds < 0 or n != len(output)
            or n < 0 or prompt <= 0 or end != prompt + n
            or end > record['max_model_len']):
        raise ValueError(f"Invalid request accounting: {record['uid']}")
    if record['decode_tokens'] not in (None, n):
        raise ValueError('Decode/output token mismatch')
    steps = record['decode_steps']
    if seconds == 0:
        if (record.get('finish_reason') != 'context_limit_guard'
                or n or output or steps
                or record.get('verification_count') not in (None, 0)
                or record.get('decode_tokens') not in (None, 0)
                or end != prompt):
            raise ValueError(f"Unexplained zero decode time: {record['uid']}")
        return 'context_guard_no_decode'
    if not steps:
        raise ValueError('Positive decode time without decode steps')
    if any(not math.isfinite(s['seconds']) or s['seconds'] <= 0 for s in steps):
        raise ValueError('Invalid individual step time')
    if not math.isclose(sum(s['seconds'] for s in steps), seconds,
                        rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Request/step time mismatch')
    if sum(s['emitted'] for s in steps) != n:
        raise ValueError('Request/step token mismatch')
    verifications = sum(s['event'] is not None for s in steps)
    if record.get('verification_count') != verifications:
        raise ValueError('Verification event count mismatch')
    return 'decoded'


def request_metrics(record):
    category = classify(record)
    if category == 'context_guard_no_decode':
        return dict.fromkeys(METRICS)
    steps = record['decode_steps']
    events = [s['event'] for s in steps if s['event'] is not None]
    return dict(request_decode_tps=record['completion_tokens'] / record['decode_total_time'],
        al=mean([e['accepted_len'] for e in events]),
        hit=mean([e.get('cache_hit') for e in events]),
        p1_al=mean([e['accepted_len'] for e in events
                    if e.get('cache_hit') and e.get('source') == 1]),
        p2_al=mean([e['accepted_len'] for e in events
                    if e.get('cache_hit') and e.get('source') == 2]),
        step_ms=1000 * record['decode_total_time'] / len(steps))


def build_requests(sequences, steps, terminal_reasons, question_ids, outputs,
                   pass_index, max_model_len, speculative):
    ids = sorted(sequences)
    if len(ids) != len(outputs) or len(ids) != len(question_ids):
        raise ValueError('Request/output population mismatch')
    by_id = {sid: [] for sid in ids}
    for s in steps:
        if (len(s['seq_ids']) != s['batch']
                or len(set(s['seq_ids'])) != s['batch']
                or len(s['emitted_by_request']) != s['batch']
                or len(s['events']) != (s['batch'] if speculative else 0)):
            raise ValueError('Step request/event population mismatch')
        for i, sid in enumerate(s['seq_ids']):
            by_id[sid].append(dict(seconds=s['seconds'],
                emitted=s['emitted_by_request'][i],
                event=s['events'][i] if speculative else None))
    rows = []
    for sid, qid, output in zip(ids, question_ids, outputs):
        seq = sequences[sid]
        ss = by_id[sid]
        row = dict(uid=f'{pass_index}:{qid}_t0', question_id=str(qid),
            turn_index=0, pass_index=pass_index,
            finish_reason=terminal_reasons.get(sid, 'completed'),
            input_tokens=seq.initial_num_prompt_tokens,
            end_context_length=seq.num_tokens, max_model_len=max_model_len,
            completion_tokens=len(output), output_ids=list(output),
            decode_tokens=sum(s['emitted'] for s in ss),
            verification_count=sum(s['event'] is not None for s in ss),
            decode_total_time=sum(s['seconds'] for s in ss), decode_steps=ss)
        row['classification'] = classify(row)
        row['metrics'] = request_metrics(row)
        rows.append(row)
    return rows


def question_summary(records, selected_questions=None):
    seen = set()
    by_question = defaultdict(list)
    guards = []
    for r in records:
        if r['uid'] in seen:
            raise ValueError('Duplicate request UID')
        seen.add(r['uid'])
        kind = classify(r)
        qid = str(r['question_id'])
        if selected_questions is not None and qid not in selected_questions:
            continue
        by_question[qid].append(request_metrics(r))
        if kind == 'context_guard_no_decode':
            guards.append(r['uid'])
    means, counts, turn_counts = {}, {}, {}
    for metric in METRICS:
        values = [mean([r[metric] for r in rows]) for rows in by_question.values()]
        means[metric] = mean(values)
        counts[metric] = sum(v is not None for v in values)
        turn_counts[metric] = sum(r[metric] is not None
            for rows in by_question.values() for r in rows)
    return dict(questions=len(by_question), turn_records=sum(map(len, by_question.values())),
        metrics=means, metric_questions=counts, metric_turn_records=turn_counts,
        context_guard_no_decode_uids=guards,
        policy='mean of valid turns/passes within question, then mean across questions',
        time_definition='shared batch decode time while request active; not aggregate TPS')


def common_question_summary(arms):
    """Use only questions whose every recorded turn/pass decoded in every arm.

    The evaluated request identities must match; an unrecorded turn cannot be
    silently interpreted as a success. Callers supply completed arms only.
    """
    if not arms:
        return dict(question_ids=[], arms={})
    identities = None
    eligible = []
    for records in arms.values():
        # Also checks duplicate UIDs, including empty-arm cases.
        question_summary(records)
        keys = {(str(r['question_id']), r['turn_index'], r['pass_index']) for r in records}
        if len(keys) != len(records):
            raise ValueError('Duplicate question/turn/pass identity')
        if identities is not None and keys != identities:
            raise ValueError('Arms have different evaluated request identities')
        identities = keys
        all_q = {str(r['question_id']) for r in records}
        invalid = {str(r['question_id']) for r in records if classify(r) != 'decoded'}
        eligible.append(all_q - invalid)
    common = set.intersection(*eligible)
    return dict(question_ids=sorted(common),
        arms={name: question_summary(rows, common) for name, rows in arms.items()})
