import unittest
import numpy as np
from final_analysis import paired_summary, collect_run


class FinalAnalysisTests(unittest.TestCase):
    def test_paired_difference_resamples_questions_not_independent_arms(self):
        a={str(i):float(i) for i in range(20)}
        b={k:v+2 for k,v in a.items()}
        result=paired_summary(a,b,list(a))
        self.assertEqual(result['questions'],20)
        self.assertEqual(result['ci95'],[2.,2.])

    def test_exact_source_contributions_and_equal_turn_weights(self):
        def group(uid, question, events):
            return dict(n=1,phase='measured_0',request_records=[dict(seq_id=uid,
                question_id=question,paper_uid=str(uid),completion_tokens=10,
                finish_reason='completed')],steps=[dict(cap=False,clipped=False,
                batch=1,seq_ids=[uid],events=[dict(source=s,cache_hit=int(s>0),accepted_len=a,
                tree_max_depth=3)]) for s,a in events])
        r=dict(args=dict(siblings=3,p1_verify=32),groups=[
            group(1,'a',[(1,2),(2,4)]),group(2,'a',[(0,5)]),group(3,'b',[(2,3)])],
            input_metadata=[dict(question_id='a',group='mt_bench'),dict(question_id='b',group='qa')])
        x=collect_run(r)
        np.testing.assert_allclose(x['contributions']['a'],[2.5,.5,1.])
        self.assertEqual(x['qstats']['a'][0],4.)
        self.assertEqual(np.mean([x['qstats'][q][0] for q in ['a','b']]),3.5)


if __name__=='__main__':unittest.main()
