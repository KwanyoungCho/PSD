"""Use the same repaired fixtures against before/experiment runtimes."""
import argparse,json,os,sys,unittest
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
os.environ['SSD_FORCE_SPLIT_K1K2']='1'
os.environ['SSD_DUET_JIT_SHORT']='1'
sys.path.insert(0,str(a.checkout/'ssd'))
import ssd.engine.draft_runner as runtime
fixtures=Path(__file__).resolve().parents[3]/'ssd'
sys.path.insert(0,str(fixtures))
names=['tests.test_b_gt1_m2','tests.test_b_gt1_jit_subset']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
r=unittest.TextTestRunner(verbosity=2).run(suite)
a.output.write_text(json.dumps(dict(runtime_file=runtime.__file__,tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped)),indent=2)+'\n')
raise SystemExit(not r.wasSuccessful())
