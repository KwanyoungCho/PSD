"""CPU-only regressions for generated history, warmup and context accounting."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from specbench_history import BaseTokenHistory, load_turns


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    chat_template = None

    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [ord(c)+3 for c in text]

    def get_vocab(self):
        return {str(i): i for i in range(300)}

    def decode(self, ids, **kwargs):
        # Token 299 deliberately decodes as the same raw text as token 123.
        return ''.join('x' if t==299 else chr(t-3) for t in ids if t>2)


def row(qid, turn, prompt='x'):
    return dict(uid=f'{qid}_t{turn}', question_id=qid, question_index=qid,
        group='mt_bench', turn=turn, prompt=prompt)


class HistoryTests(unittest.TestCase):
    def source(self, data):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)/'input.json'
        path.write_text(json.dumps(data))
        return path

    def test_answer_tokens_are_preserved_and_only_terminal_eos_is_removed(self):
        tok = Tokenizer()
        history = BaseTokenHistory([row(1,0), row(1,1,'follow-up')],tok)
        first = history.prepare(0)
        # Includes a token with a deliberately non-canonical text encoding.
        history.complete([299,44,2])
        second = history.prepare(1)
        self.assertEqual(second['input_ids'],first['input_ids']+[299,44]+tok.encode('\n\nfollow-up',False))
        self.assertEqual(second['input_ids'].count(1),1)
        self.assertNotIn(2,second['input_ids'])
        self.assertEqual(second['previous_uid'],'1_t0')

    def test_first_three_warmup_turns_still_feed_measured_followup(self):
        rows = [row(1,0),row(1,1),row(2,0),row(2,1)]
        h = BaseTokenHistory(rows,Tokenizer())
        output = {}
        for phase, indexes in [('warm', range(3)), ('measured_0', range(3,4))]:
            for i in indexes:
                prepared = h.prepare(i)
                output[i]=h.complete([200+i,2])
        self.assertEqual(prepared['input_ids'],output[2]['input_ids']+[202]+Tokenizer().encode('\n\nx',False))
        self.assertEqual(h.cursor,4)

    def test_new_question_resets_history(self):
        tok=Tokenizer();h=BaseTokenHistory([row(1,0),row(2,0,'new')],tok)
        h.prepare(0);h.complete([299,2])
        self.assertEqual(h.prepare(1)['input_ids'],tok.encode('new'))

    def test_context_cap_preserves_all_answer_tokens(self):
        h=BaseTokenHistory([row(1,0),row(1,1)],Tokenizer(),context=12,output_cap=8)
        h.prepare(0);h.complete([200]*6+[2])
        prepared=h.prepare(1)
        self.assertEqual(prepared['input_tokens'],11)
        self.assertEqual(prepared['effective_output_cap'],1)
        self.assertEqual(h.complete([200])['finish_reason'],'context_limit_output')

    def test_overflow_is_explicit_and_does_not_truncate(self):
        h=BaseTokenHistory([row(1,0),row(1,1,'long')],Tokenizer(),context=10,output_cap=8)
        h.prepare(0);h.complete([200]*7+[2])
        prepared=h.prepare(1)
        self.assertGreater(prepared['input_tokens'],10)
        self.assertEqual(prepared['effective_output_cap'],0)
        self.assertEqual(h.complete([])['finish_reason'],'context_limit_input')

    def test_completion_reasons_and_empty_answer(self):
        for output, guard, expected in [([2],None,'eos'),([200]*3,None,'output_limit'),
                ([], 'context_limit_guard','context_limit_guard')]:
            with self.subTest(expected=expected):
                h=BaseTokenHistory([row(1,0),row(1,1)],Tokenizer(),output_cap=3)
                h.prepare(0)
                self.assertEqual(h.complete(output,guard)['finish_reason'],expected)
                self.assertIsNone(h.prepare(1)['admission_reason'])

    def test_missing_history_and_repeated_turn_rejected(self):
        h=BaseTokenHistory([row(1,0),row(1,1)],Tokenizer())
        with self.assertRaises(ValueError):h.prepare(1)
        h.prepare(0)
        with self.assertRaises(ValueError):h.prepare(0)
        h.complete([2])
        with self.assertRaises(ValueError):h.complete([2])

    def test_unexplained_stop_and_tokens_after_eos_rejected(self):
        for output in ([200],[2,200]):
            h=BaseTokenHistory([row(1,0)],Tokenizer());h.prepare(0)
            with self.assertRaises(ValueError):h.complete(output)

    def test_template_is_not_silently_overridden(self):
        tok=Tokenizer();tok.chat_template='chat'
        with self.assertRaisesRegex(ValueError,'base models'):BaseTokenHistory([row(1,0)],tok)

    def test_limit_selects_complete_questions(self):
        data=[row(1,0),row(1,1),row(2,0)]
        self.assertEqual([r['uid'] for r in load_turns(self.source(data),1)],['1_t0','1_t1'])
        for invalid in ([row(1,1)], [row(1,0),row(2,0),row(1,1)]):
            with self.assertRaises(ValueError):load_turns(self.source(invalid))

    def test_preserve_frozen_raw_input_ids(self):
        tok=Tokenizer();r=row(1,0)
        r.update(input_ids=[1,299],vocab_sha256=hashlib.sha256(json.dumps(tok.get_vocab(),sort_keys=True).encode()).hexdigest())
        rows=load_turns(self.source([r]))
        self.assertEqual(BaseTokenHistory(rows,tok).prepare(0)['input_ids'],[1,299])
        rows[0]['prompt']='different'
        with self.assertRaisesRegex(ValueError,'raw prompt'):BaseTokenHistory(rows,tok)

    def test_task_filter_preserves_whole_dialogue_and_original_identity(self):
        data=[row(1,0),row(1,1),dict(row(2,0),group='translation'),row(3,0),row(3,1)]
        rows=load_turns(self.source(data),task='mt_bench')
        self.assertEqual([r['uid'] for r in rows],['1_t0','1_t1','3_t0','3_t1'])
        self.assertEqual(rows[-1]['question_index'],3)
        self.assertEqual(len(load_turns(self.source(data),limit=1,task='mt_bench')),2)
        with self.assertRaisesRegex(ValueError,'No questions'):load_turns(self.source(data),task='rag')

    def test_source_question_categories_are_grouped_as_mt(self):
        data=[dict(question_id=81,category='writing',turns=['hello','again'])]
        self.assertEqual([r['group'] for r in load_turns(self.source(data),task='mt_bench')],['mt_bench']*2)

    def test_nonadmitted_first_turn_blocks_later_turn_then_resets(self):
        h=BaseTokenHistory([row(1,0,'too long'),row(1,1),row(2,0)],Tokenizer(),context=3,output_cap=2)
        h.prepare(0);self.assertEqual(h.complete([])['finish_reason'],'context_limit_input')
        self.assertEqual(h.prepare(1)['admission_reason'],'history_unavailable')
        h.complete([])
        self.assertIsNone(h.prepare(2)['admission_reason'])


if __name__ == '__main__':
    unittest.main()
