"""Finalize frozen experiment artifacts after every controller exits cleanly."""
import json
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
while True:
    pending = []
    for plan in R.glob('*/plan.json'):
        manifest = plan.with_name('campaign.json')
        recorded = {j['name']: j for j in json.loads(manifest.read_text())} if manifest.exists() else {}
        for job in json.loads(plan.read_text()):
            path = plan.with_name(job['name'] + '.json')
            record = recorded.get(job['name'], {})
            state = json.loads(path.read_text()) if path.exists() else {}
            if state.get('status') == 'failed':
                raise RuntimeError('Model execution failed: ' + job['name'])
            if state.get('status') != 'complete' or record.get('status') != 'complete' or record.get('exit_code') != 0:
                pending.append(job['name'])
    if not pending:
        break
    print('Pending:', ', '.join(pending), flush=True)
    time.sleep(30)
for script in ('analyze.py', 'stratify.py', 'profile_analysis.py', 'make_figures.py',
               'archive.py', 'make_report.py', 'update_handoff.py'):
    print('FINALIZE', script, flush=True)
    subprocess.run([sys.executable, str(R / script)], check=True)
# Raw files remain available locally; Git stores verified compressed versions.
manifest = json.loads((R / 'ARCHIVE_MANIFEST.json').read_text())
ignored = ['# Rebuild these exact raw bytes with archive.py --restore.', '__pycache__/', '*.pyc']
ignored.extend('/' + row['raw'] for row in manifest)
(R / '.gitignore').write_text('\n'.join(ignored) + '\n')
print('Final report, plots, verified archives and handoff complete', flush=True)
