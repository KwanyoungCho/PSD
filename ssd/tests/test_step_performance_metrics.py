import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from ssd.utils.performance import performance_summary


class StepMetrics(unittest.TestCase):
    def test_whole_step_union_keeps_token_time_al_and_hit_populations_equal(self):
        events = []
        steps = []
        for index,(batch,cap,clip,tokens,seconds) in enumerate([(2,False,False,6,2),
                (2,True,False,3,8),(1,False,False,9,7),(1,True,True,1,4),
                (2,False,True,4,5),(2,False,False,8,1)]):
            start=len(events)
            events.extend([dict(accepted_len=tokens/batch,cache_hit=int(index!=5))]*batch)
            steps.append(dict(batch_size=batch,event_start=start,event_end=len(events),
                              output_cap_reached=cap,clipped=clip,
                              emitted_tokens=tokens,seconds=seconds))
        out=performance_summary(dict(phase_events=events,decode_steps=steps),2)
        self.assertEqual(out['included_step_indices'],[0,5])
        self.assertEqual(out['emitted_tokens'],14)
        self.assertEqual(out['seconds'],3)
        self.assertEqual(out['al'],3.5)
        self.assertEqual(out['tps'],14/3)
        self.assertEqual(out['cache_hit'],.5)
        self.assertEqual(out['exclusion_reasons'],dict(cap_or_clip=3,partial_batch=2,both=1))
        self.assertEqual(out['excluded_steps'],4)

    def test_ar_and_empty_population(self):
        row=dict(batch_size=2,event_start=0,event_end=0,output_cap_reached=False,
                 clipped=False,emitted_tokens=2,seconds=.5)
        out=performance_summary(dict(decode_steps=[row]),2)
        self.assertEqual(out['tps'],4)
        self.assertIsNone(out['al'])
        row['output_cap_reached']=True
        self.assertIsNone(performance_summary(dict(decode_steps=[row]),2)['tps'])

    def test_primary_log_labels_use_same_population(self):
        import os
        os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
        from types import SimpleNamespace
        from contextlib import redirect_stdout
        from io import StringIO
        from ssd.engine.llm_engine import LLMEngine,METRICS
        saved=METRICS.copy()
        try:
            for k,v in METRICS.items():METRICS[k]=[] if isinstance(v,list) else 0
            METRICS.update(prefill_total_time=1.,prefill_total_tokens=100,
                decode_total_time=2.,decode_total_tokens=6,
                accepted_suffix_lens_with_recovery=[2,2,10,1],
                cache_hits=[1,1,0,0],target_step_times=[1.,1.],
                phase_events=[dict(accepted_len=n,cache_hit=h) for n,h in zip([2,2,10,1],[1,1,0,0])],
                decode_steps=[dict(seconds=1.,emitted_tokens=4,batch_size=2,event_start=0,event_end=2),
                              dict(seconds=1.,emitted_tokens=2,batch_size=2,event_start=2,event_end=4,output_cap_reached=True)])
            stub=SimpleNamespace(config=SimpleNamespace(max_num_seqs=2,speculate=True,
                duet_phase1_k=None,speculate_k=2,draft_async=True,duet_enabled=False))
            out=StringIO()
            with redirect_stdout(out):LLMEngine.log_metrics(stub)
            self.assertIn('Final Decode Throughput: 4.00tok/s',out.getvalue())
            self.assertIn('[metrics] Avg Tokens per step (incl recovery): 2.00',out.getvalue())
            self.assertIn('[metrics] Avg Cache Hits: 1.00',out.getvalue())
            self.assertIn('[metrics] Excluded long decode steps: 0',out.getvalue())
            self.assertIn('[raw metrics] All-event accepted length: 3.75',out.getvalue())
        finally:METRICS.clear();METRICS.update(saved)

    def test_historical_speculative_rows_and_unknown_ar(self):
        event=dict(accepted_len=3,cache_hit=0)
        row=dict(event_start=0,event_end=1,emitted_tokens=3,seconds=1)
        self.assertEqual(performance_summary(dict(phase_events=[event],decode_steps=[row]),1)['tps'],3)
        with self.assertRaises(ValueError):
            performance_summary(dict(decode_steps=[dict(row,event_end=0)]),1)


if __name__=='__main__': unittest.main()
