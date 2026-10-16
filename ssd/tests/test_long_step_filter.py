"""Upper-tail decode selection without profiling or generation mutation."""
import copy
import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from ssd.utils.performance import duration_step_filter, performance_summary


class LongSteps(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,SSD_PROFILE_DUET='0',SSD_PROFILE='0',SSD_PROFILE_DRAFT='0')
        self.env.start();self.addCleanup(self.env.stop)

    def row(self,seconds=.01,events=None,batch=1,scope=None):
        return dict(seconds=seconds,batch_size=batch,
                    events=events if events is not None else [dict(cache_hit=1,source=1,accepted_len=2)],
                    duration_scope=scope)

    def test_only_upper_tail_and_strict_boundary(self):
        rows=[self.row() for _ in range(5)]+[self.row(.02),self.row(.020001),self.row(.001)]
        before=copy.deepcopy(rows);out=duration_step_filter(rows)
        self.assertEqual([r['index'] for r in out['excluded']],[6])
        self.assertEqual(out['included_indices'],[0,1,2,3,4,5,7])
        self.assertEqual(rows,before)

    def test_normal_misses_are_not_compared_to_hits(self):
        miss=[dict(cache_hit=0,source=0,accepted_len=1)]
        rows=[self.row() for _ in range(5)]+[self.row(.2,miss) for _ in range(5)]+[self.row(2.,miss)]
        out=duration_step_filter(rows)
        self.assertEqual([r['index'] for r in out['excluded']],[10])
        self.assertEqual(out['groups'][1]['median_seconds'],.2)

    def test_batch_mix_is_order_invariant_and_counts_are_separate(self):
        hit=dict(cache_hit=1,source=2,accepted_len=2);miss=dict(cache_hit=0,source=0,accepted_len=1)
        rows=[self.row(.1,[hit,miss] if i%2 else [miss,hit],2) for i in range(5)]
        rows += [self.row(.5,[hit,miss],2)]+[self.row(.5,[miss,miss],2) for _ in range(5)]
        out=duration_step_filter(rows)
        self.assertEqual([r['index'] for r in out['excluded']],[5])
        self.assertEqual(len(out['groups']),2)

    def test_sparse_groups_retained_and_request_scopes_separate(self):
        rows=[self.row() for _ in range(3)]+[self.row(10.)]
        self.assertEqual(duration_step_filter(rows)['excluded_steps'],0)
        rows=[self.row(scope=0) for _ in range(5)]+[self.row(1.,scope=1) for _ in range(5)]
        self.assertEqual(duration_step_filter(rows)['excluded_steps'],0)

    def test_disabled_profile_does_not_remove_steps(self):
        rows=[self.row() for _ in range(5)]+[self.row(1.)]
        self.assertEqual(duration_step_filter(rows,enabled=False)['included_indices'],list(range(6)))
        metrics=dict(phase_events=[],decode_steps=[dict(seconds=r['seconds'],batch_size=1,
            emitted_tokens=1,event_start=0,event_end=0) for r in rows])
        with patch.dict(os.environ,SSD_PROFILE_DUET='1'):
            result=performance_summary(metrics,1)
        self.assertEqual(result['included_steps'],6)
        self.assertFalse(result['duration_filter']['enabled'])

    def test_native_summary_keeps_time_tokens_events_and_indices_together(self):
        events=[];steps=[]
        for seconds,tokens,cap,batch in [(10.,10,True,2)]+[(.01,4,False,2)]*5+[(1.,18,False,2),(.01,2,False,1)]:
            start=len(events);events.extend([dict(accepted_len=tokens/batch,cache_hit=1,source=1)]*batch)
            steps.append(dict(seconds=seconds,emitted_tokens=tokens,batch_size=batch,
                event_start=start,event_end=len(events),output_cap_reached=cap))
        data=dict(phase_events=events,decode_steps=steps);before=copy.deepcopy(data)
        out=performance_summary(data,2)
        self.assertEqual(out['included_step_indices'],list(range(1,6)))
        self.assertEqual(out['duration_filter']['excluded'][0]['index'],6)
        self.assertEqual(out['duration_filter']['excluded_emitted_tokens'],18)
        self.assertEqual(out['emitted_tokens'],20)
        self.assertAlmostEqual(out['seconds'],.05)
        self.assertEqual(out['al'],2)
        self.assertEqual(data,before)

    def test_ar_and_invalid_durations(self):
        rows=[self.row(events=[],batch=2) for _ in range(5)]+[self.row(1.,[],2)]
        self.assertEqual(duration_step_filter(rows)['excluded_steps'],1)
        for seconds in (0,-1,float('nan'),float('inf'),-float('inf')):
            with self.subTest(seconds=seconds),self.assertRaises(ValueError):
                duration_step_filter([self.row(seconds)])

    def test_wandb_reports_filtered_step_time_and_removed_count(self):
        # Load only the logger: no model, W&B connection or CUDA work.
        source=Path(__file__).resolve().parents[1]/'bench'/'bench.py'
        function=next(n for n in ast.parse(source.read_text()).body
                      if isinstance(n,ast.FunctionDef) and n.name=='log_wandb_metrics')
        logged=[]
        namespace={'wandb':SimpleNamespace(log=logged.append,Histogram=tuple)}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
        events=[dict(accepted_len=n,cache_hit=1,source=1) for n in [2]*5+[9]]
        steps=[dict(seconds=t,emitted_tokens=e['accepted_len'],batch_size=1,event_start=i,event_end=i+1)
               for i,(t,e) in enumerate(zip([.01]*5+[1.],events))]
        metrics=dict(phase_events=events,decode_steps=steps,target_step_times=[.01]*5+[1.],
            accepted_suffix_lens_on_hit=[2]*5+[9],accepted_suffix_lens_with_recovery=[2]*5+[9])
        metrics['performance']=performance_summary(metrics,1)
        namespace['log_wandb_metrics'](SimpleNamespace(wandb=True),metrics,19,1.05,19/1.05,'test','test','test')
        row=logged[0]
        self.assertEqual(row['metrics_excluded_long_steps'],1)
        self.assertAlmostEqual(row['metrics_avg_target_step_time_ms'],10.)
        self.assertEqual(row['metrics_avg_accepted_suffix_lens_on_hit'],2)
        self.assertEqual(row['metrics_accepted_suffix_lens_on_hit_histogram'],(2,)*5)
        self.assertEqual(row['raw_accepted_suffix_lens_on_hit_histogram'],tuple([2]*5+[9]))


if __name__=='__main__':unittest.main()
