"""Reproduce four paper bugs, then run the same checks with the narrow port."""
import json,os,sys,unittest
from pathlib import Path
sys.path.insert(0,'/home/chokwans99/PSD-paper-review/ssd')
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
os.environ['PAPER_COLD_PREFIX']='0';os.environ.setdefault('SSD_SHM_NAME','paper7_kv_test')
from mlsys_paper_hooks import install_common,install_kv,load
install_common()
here=Path(__file__).resolve().parents[1]
prefix=load('paper_prefix_tests',here/'tests/test_cached_prefill.py')
state=load('paper_state_tests',here/'tests/test_output_accounting.py')
cases=[(prefix.TestAdmission,'test_fully_cached_prompt_keeps_a_query_token'),
       (prefix.TestCachedPrefill,'test_mixed_cached_uncached_rows_match_dense_attention'),
       (state.TestOutputAccounting,'test_completed_block_hash_points_to_its_own_physical_page'),
       (state.TestOutputAccounting,'test_preemption_preserves_output_and_remaining_budget')]
rows=[]
for arm in ('base','kv'):
    if arm=='kv':install_kv()
    for cls,name in cases:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([cls(name)]))
        rows.append(dict(arm=arm,test=name,success=result.wasSuccessful(),
                         failures=[str(x[1]) for x in result.failures],errors=[str(x[1]) for x in result.errors]))
Path(sys.argv[1]).write_text(json.dumps(rows,indent=2))
if any(not x['success'] for x in rows if x['arm']=='kv'):raise SystemExit(1)
if any(x['success'] for x in rows if x['arm']=='base'):raise SystemExit('Expected baseline defect was not reproduced')
