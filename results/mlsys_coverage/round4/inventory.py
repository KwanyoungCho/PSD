"""Record all actual campaigns, telemetry contamination and artifact hashes."""
from pathlib import Path
import csv,json,hashlib
from collections import Counter
HERE=Path(__file__).resolve().parent
rows=[]
for path in sorted(HERE.glob('*/campaign.json')):
 for run in json.loads(path.read_text()):
  raw=path.parent/(run['name']+'.json');report=json.loads(raw.read_text()) if raw.exists() else {}
  r=dict(stage=path.parent.name,name=run['name'],status=run['status'],returncode=run['returncode'],
   seconds=run['ended']-run['started'],questions=len(report.get('question_indexes',[])),
   cells=len(report.get('cells',[])),batches=report.get('args',{}).get('batches'),
   temperatures=report.get('args',{}).get('temperatures'),external_pids=run.get('external_pids',[]),
   raw_path=str(raw.relative_to(HERE)),raw_bytes=raw.stat().st_size if raw.exists() else 0,
   sha256=hashlib.sha256(raw.read_bytes()).hexdigest() if raw.exists() else None)
  rows.append(r)
(HERE/'RUN_INVENTORY.json').write_text(json.dumps(rows,indent=2))
if rows:
 with (HERE/'RUN_INVENTORY.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(jobs=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
             full480_jobs=sum(r['questions']==480 for r in rows),
             full480_passes=sum(r['cells'] for r in rows if r['questions']==480),
             external_pid_jobs=[r['stage']+'/'+r['name'] for r in rows if r['external_pids']],
             planned_unexecuted_files=['optimized_full_plan.json','llama2_optimized_plan.json','llama3_optimized_plan.json'],
             note='The three planning-only files were superseded by follow plans; they are not extra experiments.')
(HERE/'INVENTORY_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(summary)
# Snapshot inventory is useful when transferring the optional local NPZ bundle.
items=[]
for p in sorted((HERE/'dense_calibration').rglob('*.npz')):
 items.append(dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(HERE/'LOCAL_SNAPSHOTS.json').write_text(json.dumps(items,indent=2))
