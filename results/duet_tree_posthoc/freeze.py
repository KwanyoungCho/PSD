"""Freeze reviewed research artifacts, preserving historical experiment files."""
import hashlib
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    for name in ['audit.json','math_audit.json','statistics_checks.json','sibling_bound.json','prior_results_audit.json']:
        assert json.loads((HERE/name).read_text())['passed'],name
    assert (HERE/'FINDINGS.md').exists()
    completed=json.loads((HERE/'analysis_completed.json').read_text());assert completed['complete']
    completed.update(reviewed=True,reviewed_unix=time.time(),
        scope='Data collection, audits, figures and interpretation review complete.')
    (HERE/'analysis_completed.json').write_text(json.dumps(completed,indent=2))
    ignored={'final_manifest.json'}
    files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()
        and p.name not in ignored and '__pycache__' not in p.parts and not p.name.endswith('.tmp')}
    (HERE/'final_manifest.json').write_text(json.dumps(dict(created_unix=time.time(),files=files),indent=2))
    print('frozen',len(files),'files')


if __name__=='__main__':main()
