"""Paper input/turn identity regressions, including intentional non-truncation."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from mlsys_equal_dataset import load,attach
class Tokenizer:
 def encode(self,text):return list(range(1,len(text)+1))
 def get_vocab(self):return {str(i):i for i in range(20)}
class Inputs(unittest.TestCase):
 def save(self,data):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);p=Path(t.name)/'data.json';p.write_text(json.dumps(data));return p
 def row(self,turn):return dict(uid='81_t'+str(turn),question_id=81,question_index=0,group='mt_bench',turn=turn,prompt='long text')
 def test_no_cap_and_legacy_cap(self):
  p=self.save([self.row(0),self.row(1)])
  full,ix,m=load(p,Tokenizer(),input_cap=0)
  short,_,_=load(p,Tokenizer(),input_cap=3)
  self.assertEqual(len(full[0]),9);self.assertEqual(len(short[0]),3)
  out=[{},{}];attach(out,m,'measured_0')
  self.assertEqual(out[0]['question_id'],out[1]['question_id']);self.assertNotEqual(out[0]['uid'],out[1]['uid'])
  self.assertEqual([r['turn_index'] for r in out],[0,1])
 def test_preserve_paper_ids_when_default_tokenizer_differs(self):
  r=self.row(0);r.update(input_ids=[3,2,4],vocab_sha256=hashlib.sha256(json.dumps(Tokenizer().get_vocab(),sort_keys=True).encode()).hexdigest())
  self.assertEqual(load(self.save([r]),Tokenizer())[0],[[3,2,4]])
  r['vocab_sha256']='wrong'
  with self.assertRaisesRegex(ValueError,'vocabulary'):load(self.save([r]),Tokenizer())
 def test_reject_duplicate_or_incomplete_identity(self):
  with self.assertRaisesRegex(ValueError,'Duplicate'):load(self.save([self.row(0)]*2),Tokenizer())
  r=self.row(0);del r['turn']
  with self.assertRaisesRegex(ValueError,'Incomplete'):load(self.save([r]),Tokenizer())
 def test_legacy_first_turn_unchanged(self):
  x,ix,m=load(self.save([dict(turns=['abc','longer'])]),Tokenizer(),input_cap=2)
  self.assertEqual(x,[[1,2]]);self.assertEqual(ix,[0]);self.assertEqual(m,[])
if __name__=='__main__':unittest.main()
