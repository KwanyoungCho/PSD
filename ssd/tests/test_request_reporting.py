"""Zero-decode handover: strict validation and disclosed populations."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from mlsys_request_reporting import (classify, request_metrics, question_summary,
    common_question_summary, safe_tps, build_requests, request_reporting_supported)
from mlsys_tree_results import compare


class RequestReporting(unittest.TestCase):
    def test_async_record_schema_does_not_reinterpret_ar_or_sync_sd(self):
        self.assertTrue(all(request_reporting_supported(m) for m in ('tree','chain','ssd')))
        self.assertFalse(request_reporting_supported('ar'))
        self.assertFalse(request_reporting_supported('sd'))

    def record(self, q='a', turn=0, seconds=1., count=2, guard=False):
        event = dict(accepted_len=count, cache_hit=1, source=1)
        steps = [] if guard else [dict(seconds=seconds, emitted=count, event=event)]
        return dict(uid=f'0:{q}_t{turn}', question_id=q, turn_index=turn, pass_index=0,
            input_tokens=1734, end_context_length=1734+count, max_model_len=2048,
            completion_tokens=count, output_ids=[1]*count, decode_tokens=count,
            verification_count=len(steps), decode_steps=steps, decode_total_time=seconds,
            finish_reason='context_limit_guard' if guard else 'completed')

    def test_explicit_guard_is_undefined_not_zero(self):
        r = self.record(seconds=0, count=0, guard=True)
        self.assertEqual(classify(r), 'context_guard_no_decode')
        self.assertTrue(all(v is None for v in request_metrics(r).values()))
        r['verification_count'] = r['decode_tokens'] = None
        self.assertEqual(classify(r), 'context_guard_no_decode')

    def test_reject_other_zero_or_contradictory_guard(self):
        base = self.record(seconds=0, count=0, guard=True)
        changes = [dict(finish_reason='completed'), dict(output_ids=[1]),
            dict(completion_tokens=1), dict(verification_count=1), dict(decode_tokens=1),
            dict(decode_steps=[dict(seconds=0)]), dict(end_context_length=1735),
            dict(input_tokens=2049, end_context_length=2049)]
        for delta in changes:
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                classify(dict(base, **delta))

    def test_reject_invalid_time_positive_and_guard(self):
        for guard in (False, True):
            for seconds in (-1, float('nan'), float('inf'), -float('inf')):
                with self.subTest(guard=guard, seconds=seconds), self.assertRaises(ValueError):
                    classify(self.record(seconds=seconds, count=0 if guard else 2, guard=guard))

    def test_positive_accounting_and_mismatch(self):
        r = self.record(seconds=2, count=4)
        self.assertEqual(request_metrics(r)['request_decode_tps'], 2)
        self.assertEqual(request_metrics(r)['al'], 4)
        for delta in (dict(decode_total_time=1), dict(decode_steps=[]),
                      dict(decode_tokens=3), dict(verification_count=0)):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                classify(dict(r, **delta))

    def test_question_mean_not_independent_mt_turns(self):
        rows = [self.record('mt', 0, count=2), self.record('mt', 1, count=6),
                self.record('single', count=10),
                self.record('guard', seconds=0, count=0, guard=True)]
        out = question_summary(rows)
        self.assertEqual(out['questions'], 3)
        self.assertEqual(out['turn_records'], 4)
        self.assertEqual(out['metrics']['request_decode_tps'], 7)
        self.assertEqual(out['metric_questions']['request_decode_tps'], 2)
        self.assertEqual(out['metric_turn_records']['request_decode_tps'], 3)
        self.assertEqual(out['context_guard_no_decode_uids'], ['0:guard_t0'])

    def test_common_requires_all_turns_decoded(self):
        a = [self.record('mt'), self.record('mt', 1), self.record('single')]
        b = [self.record('mt'), self.record('mt', 1, seconds=0, count=0, guard=True),
             self.record('single')]
        out = common_question_summary(dict(a=a, b=b))
        self.assertEqual(out['question_ids'], ['single'])
        self.assertEqual(out['arms']['a']['questions'], 1)
        # Same input does not require a P2 hit in either arm.
        self.assertIsNone(out['arms']['a']['metrics']['p2_al'])
        with self.assertRaises(ValueError):
            common_question_summary(dict(a=a, b=b[:-1]))

    def test_duplicate_and_empty(self):
        r = self.record()
        with self.assertRaises(ValueError): question_summary([r, r])
        out = question_summary([])
        self.assertEqual(out['questions'], 0)
        self.assertTrue(all(v is None for v in out['metrics'].values()))
        self.assertEqual(common_question_summary({})['question_ids'], [])

    def test_safe_tps_keeps_empty_distinct_from_bad_steps(self):
        self.assertIsNone(safe_tps(0, 0, 0))
        self.assertEqual(safe_tps(6, 2, 1), 3)
        for args in ((1,0,0), (0,1,0), (0,0,1), (1,-1,1), (1,float('nan'),1)):
            with self.subTest(args=args), self.assertRaises(ValueError): safe_tps(*args)

    def test_batch_mapping_discloses_guard_without_fake_step(self):
        seqs = {11: SimpleNamespace(initial_num_prompt_tokens=1734, num_tokens=1734),
                12: SimpleNamespace(initial_num_prompt_tokens=5, num_tokens=7)}
        steps = [dict(seq_ids=[12], batch=1, seconds=1, emitted_by_request=[2],
                      events=[dict(accepted_len=2, cache_hit=0, source=0)])]
        rows = build_requests(seqs, steps, {11:'context_limit_guard'}, ['a','b'],
                              [[],[4,5]], 0, 2048, True)
        self.assertEqual(rows[0]['classification'], 'context_guard_no_decode')
        self.assertEqual(rows[1]['metrics']['request_decode_tps'], 2)
        with self.assertRaises(ValueError):
            build_requests(seqs, steps, {}, ['a','b'], [[],[4,5]], 0, 2048, True)

    def test_empty_comparison_is_undefined(self):
        raw = dict(args=dict(batch=1), groups=[], outputs=[],
                   summary=dict(tps=None, verify_ms=None))
        out = compare(raw, copy.deepcopy(raw))
        self.assertIsNone(out['tps_change'])
        self.assertIsNone(out['verify_change_ci95'])
        self.assertEqual(out['valid_tps_bootstraps'], 0)


if __name__ == '__main__': unittest.main()
