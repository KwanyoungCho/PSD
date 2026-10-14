import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
p=Path(__file__).with_name('analyze.py');spec=importlib.util.spec_from_file_location('analysis',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def step(sid,al,source=0,**kw):
    return dict(batch=1,cap=False,clipped=False,seq_ids=[sid],events=[dict(accepted_len=al,cache_hit=int(source>0),source=source)],**kw)

class AnalysisTests(unittest.TestCase):
    def test_equal_weight_per_question_after_turn_average(self):
        r={'groups':[dict(phase='measured_0',request_records=[dict(seq_id=i,question_id=q)],steps=[step(i,al)])
            for i,q,al in [(1,'a',1),(2,'a',3),(3,'b',6)]]}
        x=a.question_stats(r)
        self.assertEqual(x['a'][0],2);self.assertEqual(x['b'][0],6)
        self.assertEqual(a.mean([x['a'][0],x['b'][0]]),4) # not 10/3
    def test_cap_clip_excluded_graph_included(self):
        ss=[step(1,2,1,graph_preparation=True),step(1,6,2),step(1,9),step(1,8)]
        ss[2]['cap']=True;ss[3]['clipped']=True
        r={'groups':[dict(phase='measured_0',request_records=[dict(seq_id=1,question_id='a')],steps=ss)]}
        x=a.question_stats(r)['a'];self.assertEqual(x[0],4);self.assertEqual(x[1],100)
        self.assertEqual(x[3],2);self.assertEqual(x[4],6)
    def test_gzip_budget_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'equal_forest.jsonl'
            rows=[dict(phase=1,root=i,generated=24,processed=8,served=24,generated_depth=8,
                physical_rows=64,roots=8,seq=1,step=1,C=3) for i in range(8)]
            with gzip.open(str(p)+'.gz','wt') as f:
                for r in rows:f.write(json.dumps(r)+'\n')
            x=a.forest(p);self.assertEqual(x['full_active_calls'],1);self.assertEqual(x['inconsistent_generated'],0)
            with gzip.open(str(p)+'.gz','at') as f:
                r=copy.copy(rows[0]);r['physical_rows']=63;f.write(json.dumps(r)+'\n')
            with self.assertRaises(AssertionError):a.forest(p)

if __name__=='__main__':unittest.main(verbosity=2)
