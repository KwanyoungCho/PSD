import sys
import unittest
import json
import os
import tempfile
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from mlsys_measurement import overlap_ns,annotate,summarize,capture_intervals,finalize_measurement


class MeasurementTests(unittest.TestCase):
    def step(self,start,end,events,seconds=1.):
        return dict(audit_start_ns=start,audit_end_ns=end,seconds=seconds,
            batch=len(events),cap=False,clipped=False,
            emitted=sum(e['accepted_len'] for e in events),events=events,verify=[.1])

    def event(self,hit,n,step=2):
        return dict(cache_hit=hit,valid_k=n,accepted_len=n+1,step_id=step)

    def test_union_includes_capture_started_in_prior_step(self):
        cs=[dict(start_ns=1,end_ns=12),dict(start_ns=5,end_ns=16)]
        self.assertEqual(overlap_ns(10,20,cs),6)
        self.assertEqual(overlap_ns(16,20,cs),0)
        self.assertEqual(overlap_ns(10,20,[dict(start_ns=5)]),10)

    def test_prepare_excludes_tokens_time_al_and_hits_together(self):
        rows=[self.step(0,10,[self.event(0,2)]*2,10),
              self.step(10,20,[self.event(1,1)]*2,1)]
        annotate(rows,[dict(start_ns=2,end_ns=9)])
        out=summarize(rows,2,[[1,2]])
        self.assertEqual(out['summary']['seconds'],1)
        self.assertEqual(out['summary']['tokens'],4)
        self.assertEqual(out['summary']['tps'],4)
        self.assertEqual(out['summary']['hit'],1)
        self.assertEqual(out['preparation_full_batch']['affected_step_seconds'],10)

    def test_miss_frequency_distinct_from_actual_fallback(self):
        rows=[self.step(0,1,[self.event(1,2),self.event(1,2)]),
              self.step(1,2,[self.event(1,2),self.event(0,0)]),
              self.step(2,3,[self.event(0,2,1),self.event(0,2,1)])]
        annotate(rows,[]);out=summarize(rows,2,[]);m=out['batch_metrics']
        self.assertAlmostEqual(m['request_hit'],.5)
        self.assertAlmostEqual(m['any_miss'],2/3)
        self.assertAlmostEqual(m['actual_jit_step'],1/3)
        self.assertAlmostEqual(m['miss_ar_request'],1/6)
        self.assertEqual(out['batch_metrics_excluding_group_startup']['actual_jit_step'],0)

    def test_missing_capture_clock_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):annotate([dict(seconds=1)],[])

    def test_start_end_records_merge_and_open_capture_remains_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            start=dict(pid=1,label='capture',start_ns=2)
            end=dict(start,end_ns=7)
            pending=dict(pid=2,label='capture',start_ns=9)
            Path(folder,'capture_1.jsonl').write_text('\n'.join(map(json.dumps,[start,end,pending])))
            rows=capture_intervals(folder)
            self.assertEqual(len(rows),2)
            self.assertEqual(overlap_ns(0,12,rows),8)

    def test_finalizer_preserves_raw_summary_and_updates_group_copies(self):
        rows=[self.step(0,10,[self.event(0,2)],10),
              self.step(10,20,[self.event(1,1)],1)]
        report=dict(steps=rows,groups=[dict(steps=rows)],args=dict(batch=1),
                    outputs=[[1,2]],summary=dict(tps=99))
        report=json.loads(json.dumps(report))
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'capture_1.jsonl').write_text(json.dumps(
                dict(pid=1,label='capture',start_ns=2,end_ns=9))+'\n')
            with patch.dict(os.environ,DUET_REVIEW_CAPTURE_AUDIT=folder,SSD_PROFILE_DUET='0'):
                finalize_measurement(report)
        self.assertEqual(report['summary_with_preparation']['tps'],99)
        self.assertEqual(report['summary']['tps'],2)
        self.assertTrue(report['groups'][0]['steps'][0]['graph_preparation'])
        self.assertEqual(report['steps'][0]['seconds'],10)


if __name__=='__main__':unittest.main()
