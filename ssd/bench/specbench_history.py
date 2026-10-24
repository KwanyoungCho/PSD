"""Base-model Spec-Bench history, with exact generated-token continuation.

This is an explicit experimental protocol, NOT an official chat template.
Turn 0 preserves supplied raw input IDs (after vocabulary/text validation),
or uses target_tokenizer.encode(raw_question, add_special_tokens=True).
Turn n is previous_input + previous_answer_without_terminal_EOS +
target_tokenizer.encode("\n\n" + raw_question, add_special_tokens=False).
No previous content is truncated or decoded/re-encoded. Single turns stay raw.
The target tokenizer artifact must also be shared by all reference engines.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path

PROTOCOL = 'specbench_base_token_history_v1'
SEPARATOR = '\n\n'
IDENTITY = ('uid', 'question_id', 'question_index', 'group', 'turn')
TASKS = ('mt_bench', 'translation', 'summarization', 'qa', 'math_reasoning', 'rag')
MT_CATEGORIES = {'writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
        separators=(',', ':'), sort_keys=True).encode('utf-8')).hexdigest()


def load_turns(path, limit=0, task='all'):
    """Preserve dataset order; limit counts complete questions, never flat turns."""
    source = json.loads(Path(path).read_text())
    if not source or not isinstance(source, list) or limit < 0 or task not in ('all', *TASKS):
        raise ValueError('Expected a nonempty JSON list and a nonnegative question limit')
    rows = []
    for qi, item in enumerate(source):
        if 'prompt' in item:
            rows.append({k: item[k] for k in (*IDENTITY, 'prompt', 'input_ids', 'vocab_sha256') if k in item})
        else:
            category = item.get('category', 'unknown')
            for ti, prompt in enumerate(item['turns']):
                rows.append(dict(uid=f"{item['question_id']}_t{ti}",
                    question_id=item['question_id'], question_index=qi,
                    group=item.get('group', 'mt_bench' if category in MT_CATEGORIES else category),
                    turn=ti, prompt=prompt))
    seen_uids = set()
    seen_questions = set()
    current = None
    expected_turn = 0
    question_order = []
    for row in rows:
        if any(key not in row for key in (*IDENTITY, 'prompt')):
            raise ValueError('Incomplete input identity')
        qid = str(row['question_id'])
        if qid != current:
            if qid in seen_questions:
                raise ValueError('Question turns must be contiguous')
            seen_questions.add(qid)
            question_order.append(qid)
            current, expected_turn = qid, 0
        if row['turn'] != expected_turn or not isinstance(row['prompt'], str):
            raise ValueError('Missing/out-of-order turn or non-text prompt')
        if row['uid'] in seen_uids:
            raise ValueError('Duplicate turn UID')
        seen_uids.add(row['uid'])
        expected_turn += 1
    if task != 'all':
        rows = [r for r in rows if r['group'] == task]
        question_order = list(dict.fromkeys(str(r['question_id']) for r in rows))
    if not rows:
        raise ValueError('No questions match the requested task')
    chosen = set(question_order[:limit] if limit else question_order)
    return [r for r in rows if str(r['question_id']) in chosen]


class BaseTokenHistory:
    def __init__(self, rows, tokenizer, context=2048, output_cap=1024):
        if context <= 0 or output_cap <= 0:
            raise ValueError('Context and output cap must be positive')
        if getattr(tokenizer, 'chat_template', None):
            raise ValueError('This protocol is for base models without a chat template')
        self.rows, self.tokenizer = rows, tokenizer
        self.context, self.output_cap = context, output_cap
        self.cursor = 0
        self.pending = None
        self.previous = None
        self.initial_ids = {}
        if any('input_ids' in r for r in rows):
            vocab = tokenizer.get_vocab()
            valid_ids = set(vocab.values())
            vocab_hash = hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest()
            for i, row in enumerate(rows):
                if row['turn'] != 0 or 'input_ids' not in row:
                    continue
                ids = row['input_ids']
                if (row.get('vocab_sha256') != vocab_hash or not ids
                        or any(type(t) is not int or t not in valid_ids for t in ids)):
                    raise ValueError('Frozen input IDs must use the target vocabulary')
                decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if decoded != row['prompt']:
                    raise ValueError('Frozen input IDs do not reproduce the raw prompt')
                self.initial_ids[i] = list(ids)

    def prepare(self, index):
        if index != self.cursor or self.pending is not None:
            raise ValueError('Prepare/complete each turn exactly once in dataset order')
        row = self.rows[index]
        previous_uid = None
        previous_hash = None
        blocked = False
        if row['turn'] == 0:
            ids = self.initial_ids[index] if index in self.initial_ids else self.tokenizer.encode(
                row['prompt'], add_special_tokens=True)
        else:
            prev = self.previous
            if (prev is None or str(prev['question_id']) != str(row['question_id'])
                    or prev['turn'] + 1 != row['turn']):
                raise ValueError('A follow-up requires this run\'s previous answer')
            previous_uid = prev['uid']
            previous_hash = prev['output_ids_sha256']
            blocked = prev['unavailable']
            ids = prev['input_ids'] + prev['answer_ids'] + self.tokenizer.encode(
                SEPARATOR + row['prompt'], add_special_tokens=False)
        cap = max(0, min(self.output_cap, self.context - len(ids)))
        reason = 'history_unavailable' if blocked else ('context_limit_input' if cap == 0 else None)
        self.pending = dict(**{k: row[k] for k in IDENTITY},
            input_ids=list(ids), input_tokens=len(ids), input_ids_sha256=digest(ids),
            previous_uid=previous_uid, previous_output_ids_sha256=previous_hash,
            effective_output_cap=cap, context_limit=self.context,
            admission_reason=reason)
        return dict(self.pending)

    def complete(self, output_ids, scheduler_reason=None):
        if self.pending is None:
            raise ValueError('Complete requires a prepared turn')
        record = dict(self.pending)
        output_ids = list(output_ids)
        if record['admission_reason'] and output_ids:
            raise ValueError('A non-admitted turn cannot have generated output')
        if len(output_ids) > record['effective_output_cap']:
            raise ValueError('Generated output exceeds the declared output cap')
        eos = self.tokenizer.eos_token_id
        if eos is not None and eos in output_ids:
            at = output_ids.index(eos)
            if at != len(output_ids) - 1:
                raise ValueError('Generation must stop at the first EOS')
            answer_ids = output_ids[:at]
        else:
            answer_ids = output_ids
        if record['admission_reason']:
            reason = record['admission_reason']
        elif eos is not None and eos in output_ids:
            reason = 'eos'
        elif scheduler_reason == 'context_limit_guard':
            reason = scheduler_reason
        elif len(output_ids) == record['effective_output_cap']:
            reason = ('output_limit' if record['effective_output_cap'] == self.output_cap
                      else 'context_limit_output')
        else:
            raise ValueError('Generation ended without EOS, cap, or an observed context guard')
        record.update(output_ids=output_ids, output_ids_sha256=digest(output_ids),
            completion_tokens=len(output_ids), finish_reason=reason,
            answer_ids=list(answer_ids), unavailable=bool(record['admission_reason']))
        self.previous = record
        self.cursor += 1
        self.pending = None
        return record


def manifest(target, tokenizer, source, rows):
    files = {}
    for name in ('tokenizer.model', 'tokenizer.json', 'tokenizer_config.json',
                 'special_tokens_map.json', 'added_tokens.json', 'config.json'):
        path = Path(target) / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    return dict(protocol=PROTOCOL, target=str(target),
        dataset_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        questions=len({str(r['question_id']) for r in rows}), turns=len(rows),
        groups=dict(Counter(r['group'] for r in rows)), tokenizer_files=files,
        tokenizer_class=type(tokenizer).__name__,
        packages={name: importlib.metadata.version(name) for name in ('transformers', 'tokenizers')},
        backend_sha256=hashlib.sha256(backend.to_str().encode()).hexdigest() if backend else None,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        add_bos_token=getattr(tokenizer, 'add_bos_token', None),
        add_eos_token=getattr(tokenizer, 'add_eos_token', None),
        chat_template=None, separator=SEPARATOR,
        history='exact prior input IDs + generated answer IDs excluding only terminal EOS; no decode/re-encode',
        new_turn='encode(separator + question, add_special_tokens=False)',
        first_turn='preserve supplied input_ids after target vocabulary and exact raw-text validation; otherwise encode(raw question, add_special_tokens=True)',
        frozen_first_turns=sum(r['turn']==0 and 'input_ids' in r for r in rows),
        truncation=False, model_context=2048, requested_output_cap=1024,
        overflow='record context_limit_input, do not call model or assign a TPS',
        scope='Spec-Bench dataset, base-model conversation protocol; not official MT-Bench chat-quality evaluation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--task', choices=('all', *TASKS), default='all')
    parser.add_argument('--export', type=Path, required=True,
        help='CPU-only tokenizer/protocol fixtures, not performance results')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    token = AutoTokenizer.from_pretrained(args.target, use_fast=True)
    rows = load_turns(args.prompts, task=args.task)
    pair = next(i for i in range(len(rows)-1)
        if rows[i]['turn'] == 0 and rows[i+1]['turn'] == 1)
    demo = BaseTokenHistory(rows[pair:pair+2], token)
    first = demo.prepare(0)
    answer = token.encode(' This is a fixed answer for protocol verification.', add_special_tokens=False)
    completed = demo.complete(answer + [token.eos_token_id])
    second = demo.prepare(1)
    result = manifest(args.target, token, args.prompts, rows)
    result['fixture'] = dict(questions=[r['prompt'] for r in rows[pair:pair+2]],
        first_input_ids=first['input_ids'], first_output_ids=completed['output_ids'],
        second_input_ids=second['input_ids'], second_input_ids_sha256=second['input_ids_sha256'],
        note='Synthetic fixed answer for input-format checks only; not a model result')
    args.export.parent.mkdir(parents=True, exist_ok=True)
    with args.export.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f"{PROTOCOL}: {result['questions']} questions / {result['turns']} turns; CPU fixture: {args.export}")


if __name__ == '__main__':
    main()
