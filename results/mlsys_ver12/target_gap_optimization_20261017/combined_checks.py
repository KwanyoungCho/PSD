import json, os, sys, unittest
from pathlib import Path
root=Path('/data2/chokwans99/target-gap-20261017')
out=root/'checks_combined';out.mkdir(exist_ok=True)
sys.path.insert(0,str(root/'combined/ssd/tests'))
names=['test_deferred_tree_restore','test_response_snapshot','test_single_verify_inputs','test_common_port','test_ver01_review','test_tree_accept_port','test_b_gt1_m6_verify_window','test_step_performance_metrics','test_long_step_filter','test_mlsys_measurement']
rows=[]
for name in names:
 with (out/(name+'.log')).open('w') as f:
  result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(name))
 rows.append(dict(name=name,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped)))
 print(rows[-1],flush=True)
 (out/'CHECKS.json').write_text(json.dumps(rows,indent=2)+'\n')
raise SystemExit(any(r['failures'] or r['errors'] or r['skipped'] for r in rows))
