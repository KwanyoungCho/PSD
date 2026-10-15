import json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
marker=root/'checks/CHECKS_PROVENANCE.json'
while not marker.exists():time.sleep(2)
p=json.loads(marker.read_text());rows=json.loads((root/'checks/CHECKS.json').read_text())
assert p['source_unchanged'] and len(rows)==40 and all(r['returncode']==0 for r in rows)
print('Final regression passed; starting isolated GPU 4/5 campaign',flush=True)
subprocess.run(['/home/chokwans99/PSD/ssd/.venv/bin/python',str(root/'run_campaign.py')],check=True)
