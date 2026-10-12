"""Run the same public bug reproductions against the clean pre-common checkout."""
import json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;root=R.parents[2]
env=dict(os.environ,PYTHONPATH='/home/chokwans99/PSD-ver01-common-base/ssd',SSD_HF_CACHE='/tmp',SSD_DATASET_DIR='/tmp')
rows=[]
for name in ('test_output_accounting','test_cached_prefill'):
    with (R/(name+'.before.log')).open('w') as log:
        proc=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'ssd/tests'),'-p',name+'.py','-v'],env=env,stdout=log,stderr=subprocess.STDOUT)
    rows.append(dict(suite=name,returncode=proc.returncode,expected_failure=True))
    assert proc.returncode != 0
(R/'OLD_REPRODUCTIONS.json').write_text(json.dumps(rows,indent=2))
