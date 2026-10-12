import json, os, sys, unittest
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parents[2]/'ssd'))
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
out=[]
for name in ('test_common_port','test_output_accounting','test_step_performance_metrics','test_cached_prefill','test_ver01_review','test_tree_accept_port','test_p1_dynamic_tree','test_p2_tree_alloc','test_p2_executor_parity','test_tree_verify_planless','test_b_gt1_m6_verify_window'):
    suite=unittest.defaultTestLoader.discover(str(R.parents[2]/'ssd/tests'),pattern=name+'.py')
    with (R/(name+'.suite.log')).open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    row=dict(name=name,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
    out.append(row);print(row,flush=True);(R/'CHECKS.json').write_text(json.dumps(out,indent=2))
    if not result.wasSuccessful() or result.skipped:raise SystemExit(1)
