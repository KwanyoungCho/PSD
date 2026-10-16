"""Keep every run's excluded-step denominator visible."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent;rows=[]
for folder in sorted((P/'raw/runs').iterdir()):
 if not (folder/'status.json').exists():continue
 status=json.loads((folder/'status.json').read_text())
 if status['status']!='complete':continue
 j=json.loads((folder/'run.json').read_text());s=j['summary']
 rows.append(dict(tag=folder.name,profile=status['profile'],total_measured_steps=s['all_steps'],
  included_steps=s.get('steps'),excluded_long_steps=s.get('excluded_long_steps',0),
  cap_clip_partial_steps=sum(bool(x['cap'] or x['clipped'] or x['batch']!=1) for x in j['steps']),
  duration_filter_enabled=not status['profile'],
  measurement_exclusion_reasons=j.get('measurement',{}).get('exclusion_reasons')))
(P/'EXCLUSIONS.json').write_text(json.dumps(dict(runs=rows,completed_runs=len(rows),note='Counts are per repeated measurement run, not independent questions. Profile-ON durations are never 2x-filtered. Raw records retained.'),indent=2)+'\n')
print('Recorded exclusion counts for',len(rows),'runs')
