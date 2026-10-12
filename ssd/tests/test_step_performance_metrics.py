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

    def test_historical_speculative_rows_and_unknown_ar(self):
        event=dict(accepted_len=3,cache_hit=0)
        row=dict(event_start=0,event_end=1,emitted_tokens=3,seconds=1)
        self.assertEqual(performance_summary(dict(phase_events=[event],decode_steps=[row]),1)['tps'],3)
        with self.assertRaises(ValueError):
            performance_summary(dict(decode_steps=[dict(row,event_end=0)]),1)


if __name__=='__main__': unittest.main()
