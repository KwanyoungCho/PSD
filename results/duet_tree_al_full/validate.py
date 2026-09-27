"""Independent full-record audit: corpus, history, stopping, budgets and provenance."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--completed-only', action='store_true')
    args = ap.parse_args()
    from transformers import AutoTokenizer
    plan = json.loads((HERE / 'plan.json').read_text())
    questions = json.loads((HERE / 'questions.json').read_text())
    require(sha(HERE / 'questions.json') == plan['questions_sha256'], 'Question file changed')
    require(sha(Path(plan['calibration_path'])) == plan['calibration_sha256'], 'Calibration changed')
    require(sha(HERE / 'upstream_question.jsonl') == plan['source']['downloaded_sha256'], 'Original corpus changed')
    tok = AutoTokenizer.from_pretrained(
        '/home/chokwans99/awq_calibrated/layerskip_llama2_70b', local_files_only=True)
    expected = [f'{q["question_id"]}_t{t}' for q in questions for t in range(len(q['turns']))]
    checks = []
    common_command = common_env = None
    for job in plan['jobs']:
        folder = HERE / 'runs' / f"s{job['seed']}_{job['policy']}"
        if args.completed_only and not (folder / 'completion.json').exists():
            continue
        completion = json.loads((folder / 'completion.json').read_text())
        valid = json.loads((folder / 'validated.json').read_text())
        require(completion['exit_code'] == 0 and valid['questions'] == 480 and valid['turns'] == 560,
                f'Incomplete run: {folder.name}')
        require(sha(folder / 'records.jsonl') == valid['records_sha256'], f'Raw data changed: {folder.name}')
        rows = [json.loads(line) for line in (folder / 'records.jsonl').read_text().splitlines()]
        require([r['uid'] for r in rows] == expected, f'Coverage/order mismatch: {folder.name}')
        command = json.loads((folder / 'command.json').read_text())
        argv = list(command['command'])
        require(int(argv[argv.index('--seed') + 1]) == job['seed'], 'Command seed mismatch')
        argv[argv.index('--seed') + 1] = '<seed>'
        env = {k: v for k, v in command['env'].items()
               if k not in ('DUET_TREE_SCORE_MODE', 'DUET_FULL_OUT', 'SSD_DIST_PORT')}
        require(command['env']['DUET_TREE_SCORE_MODE'] == job['policy'], 'Command policy mismatch')
        if common_command is None:
            common_command, common_env = argv, env
        require(argv == common_command and env == common_env, 'Uncontrolled command/env difference')
        log = (folder / 'run.log').read_text()
        config_lines = [line for line in log.splitlines() if line.startswith('[Config] DUET-SSD enabled:')]
        shape_lines = [line.split(' in ', 1)[0] for line in log.splitlines()
                       if line.startswith('[DUET tree] warmed P1 executors contexts=')]
        require(bool(config_lines) and len(set(config_lines)) == 1 and len(set(shape_lines)) == 1,
                'Missing or inconsistent resolved configuration')
        hook_message = '[tree score hook] phase_sibling_q_bin; calibration SHA256=' + plan['calibration_sha256']
        require((hook_message in log) == (job['policy'] == 'phase_sibling_q_bin'), 'Score hook installation mismatch')
        histories = {}
        terminal_overcount = immediate_eos = zero_events = steps = hit_steps = 0
        for q in questions:
            histories[q['question_id']] = []
        lookup = {q['question_id']: q for q in questions}
        for r in rows:
            require(r['policy'] == job['policy'] and r['seed'] == job['seed'], 'Record policy mismatch')
            q = lookup[r['question_id']]
            require(r['group'] == q['group'] and r['category'] == q['category'], 'Task mapping changed')
            current = tok.encode(q['turns'][r['turn']], add_special_tokens=False)
            prompt = current if r['turn'] == 0 else histories[r['question_id']] + tok.encode(
                '\n\n', add_special_tokens=False) + current
            require(len(prompt) == r['input_tokens'], 'Input length mismatch')
            require(hashlib.sha256(json.dumps(prompt).encode()).hexdigest() == r['input_sha256'],
                    'Input text/history mismatch')
            require(len(prompt) + plan['max_new_tokens'] + 128 <= plan['max_model_len'], 'Context overflow')
            ids = r['output_ids']
            require(len(ids) == r['output_tokens'] and 1 <= len(ids) <= plan['max_new_tokens'], 'Output length')
            require(hashlib.sha256(json.dumps(ids).encode()).hexdigest() == r['output_sha256'], 'Output checksum')
            require(tok.eos_token_id not in ids[:-1], 'Generation continued after EOS')
            require(r['stop'] == ('eos' if ids[-1] == tok.eos_token_id else 'cap'), 'Stop label')
            require(r['stop'] == 'eos' or len(ids) == plan['max_new_tokens'], 'Unexplained short generation')
            histories[r['question_id']] = prompt + ids
            events = r['metrics']['phase_events']
            steps += len(events)
            immediate_eos += ids == [tok.eos_token_id]
            if not events:
                require(ids == [tok.eos_token_id], 'Missing verification events')
                zero_events += 1
            else:
                # Async prefill keeps its token pending. Each suffix starts
                # with that step's already sampled correction/root token;
                # it is not an extra token outside the event lengths.
                before_final = sum(e['accepted_len'] for e in events[:-1])
                before_clipping = before_final + events[-1]['accepted_len']
                require(before_final < len(ids) <= before_clipping, 'Suffix/output accounting mismatch')
                terminal_overcount += before_clipping > len(ids)
            for e in events:
                require(e['accepted_len'] == e['accepted_spec_len'] + 1, 'AL unit mismatch')
                require(e['cache_hit'] == int(e['source'] in (1, 2)), 'Hit/phase mismatch')
                if e['cache_hit']:
                    phase = e['source']
                    require(0 <= e['accepted_spec_len'] <= {1: 4, 2: 2}[phase], 'Depth exceeds budget')
                    require(e['accepted_spec_len'] <= e['valid_k'] <= {1: 8, 2: 6}[phase], 'Verify node budget')
                    hit_steps += 1
            require([e['accepted_len'] for e in events] == r['metrics']['accepted_suffix_lens_with_recovery'],
                    'Independent accepted-length metric mismatch')
        checks.append(dict(**job, questions=len(questions), turns=len(rows),
            records_sha256=valid['records_sha256'], stop_counts=dict(Counter(r['stop'] for r in rows)),
            input_max=max(r['input_tokens'] for r in rows),
            output_min=min(r['output_tokens'] for r in rows),
            output_median=statistics.median(r['output_tokens'] for r in rows),
            output_max=max(r['output_tokens'] for r in rows),
            immediate_eos=immediate_eos, zero_event_requests=zero_events,
            clipped_final_events=terminal_overcount,
            verification_events=steps, cache_hit_events=hit_steps, wall_s=completion['wall_s'],
            started_unix=json.loads((folder/'process.json').read_text())['started'],
            resolved_config=config_lines[0], p1_context_shapes=shape_lines[0]))
    require(len({c['resolved_config'] for c in checks}) == 1 and
            len({c['p1_context_shapes'] for c in checks}) == 1,
            'Resolved model configuration or P1 shapes differ across runs')
    require(bool(checks), 'No complete jobs to audit')
    result = dict(passed=True, complete=len(checks)==len(plan['jobs']),
        total_turns=sum(r['turns'] for r in checks), jobs=checks,
        checks=['exact full corpus/order', 'no input truncation', 'actual multi-turn history',
                'same command/environment except policy, seed and process coordinates',
                'frozen calibration and correctly installed policy hook', 'natural EOS/cap',
                'output and suffix accounting', 'AL units, phase and node budgets', 'raw data checksums'],
        limits=['This is an execution/measurement audit, not an end-to-end distribution-preservation proof.',
                'Forward counts and generated nodes for every unhit root were not instrumented.'])
    filename = 'audit_partial.json' if args.completed_only else 'audit.json'
    (HERE / filename).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
