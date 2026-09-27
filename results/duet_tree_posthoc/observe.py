"""Run the existing full-corpus harness with a passive post-verification observer.

All served trees receive compact exact labels; 1/32 receive raw p/q audit files.
Observer overhead changes asynchronous timing, so this is NOT an AL/TPS trial.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
FULL = HERE.parent / 'duet_tree_al_full'
sys.path.insert(0, str(FULL))
import run_full  # Installs the same worker hooks as the original study.


def main():
    import torch
    import bench
    from ssd.engine.verifier import Verifier
    import ssd.engine.helpers.p2_tree as p2
    from features import extract
    dest = Path(os.environ['DUET_FULL_OUT'])
    (dest / 'raw').mkdir(parents=True, exist_ok=True)
    questions = json.loads((FULL / 'questions.json').read_text())
    if os.environ.get('DUET_FULL_SMOKE') == '1':
        plan = json.loads((FULL / 'plan.json').read_text())
        selected = [questions[0]] + [next(q for q in questions if q['group'] == g) for g in plan['groups'][1:]]
        selected.append(max(questions, key=lambda q: sum(len(s) for s in q['turns'])))
        questions = list({q['question_id']: q for q in selected}.values())
    requests = [dict(uid=f'{q["question_id"]}_t{t}', question_id=q['question_id'],
        group=q['group'], turn=t) for q in questions for t in range(len(q['turns']))]
    state = dict(call=-1, active=False, phase=0, rows=[], serial=0)
    original_method = Verifier._tree_verify_walk
    def method(self, result, *args, **kwargs):
        state['phase'] = int(result.phase_source[0])
        return original_method(self, result, *args, **kwargs)
    Verifier._tree_verify_walk = method
    original_walk = p2.tree_verify_walk_tensor
    def walk(ti, p_logits, q_probs, temp, coin_fn, mult_fn):
        answer = original_walk(ti, p_logits, q_probs, temp, coin_fn, mult_fn)
        if not state['active']:
            return answer
        n = int(ti['valid'])
        ref = ti['parent_q_ref'][:n].long().to(q_probs.device)
        p = torch.softmax(p_logits[:n+1].float() / float(temp), dim=-1).detach().cpu().numpy()
        q = q_probs.index_select(0, ref).detach().float().cpu().numpy()
        arrays = dict(p=p, q=q, par=np.asarray(ti['parent_local'][:n]),
            sib=np.asarray(ti['sib_order'][:n]), tok=np.asarray(ti['tok'][:n]),
            path=np.asarray(answer[0], dtype=np.int64))
        row = dict(**requests[state['call']], phase=state['phase'],
            hit_index=len(state['rows']), serial=state['serial'], temperature=float(temp))
        if state['serial'] % 32 == 0:
            filename = f'raw/tree{state["serial"]:06d}.npz'
            np.savez_compressed(dest / filename, **arrays)
            row.update(raw_file=filename, raw_sha256=hashlib.sha256((dest / filename).read_bytes()).hexdigest())
        row.update(extract(**arrays))
        state['rows'].append(row); state['serial'] += 1
        return answer
    p2.tree_verify_walk_tensor = walk
    original_run = bench.run_benchmark
    trace = (dest / 'trees.jsonl').open('x', buffering=1)
    def run(args, llm, prompts, params):
        # First call is the original harness warmup, excluded from all labels.
        state['active'] = state['call'] >= 0
        state['rows'] = []
        result = original_run(args, llm, prompts, params)
        if state['active']:
            events = result[2]['phase_events']
            hit_events = [(i, e) for i, e in enumerate(events) if e['cache_hit']]
            if len(hit_events) != len(state['rows']):
                raise ValueError('Observer/hit event count mismatch')
            for row, (i, e) in zip(state['rows'], hit_events):
                if row['observed_al'] != e['accepted_spec_len'] or row['phase'] != e['source'] or row['n'] != e['valid_k']:
                    raise ValueError('Observer/event label mismatch')
                row.update(event_index=i, is_final_event=i == len(events)-1,
                    policy=os.environ['DUET_TREE_SCORE_MODE'])
                trace.write(json.dumps(row, allow_nan=False) + '\n')
            trace.flush()
        state['call'] += 1
        return result
    bench.run_benchmark = run
    try:
        run_full.main()
    finally:
        trace.close()


if __name__ == '__main__':
    main()
