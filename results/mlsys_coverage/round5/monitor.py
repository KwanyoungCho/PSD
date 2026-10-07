"""Read-only progress for the final/optimization lanes; stop on new failures."""
import argparse,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent


def snapshot():
    lines=[]
    for d in sorted(HERE.iterdir()):
        if not d.is_dir() or not(d.name.startswith('final_') or any(s in d.name for s in ('_opt_l','_stream_l','_postopt_l','_stability_l'))):continue
        pp=d/'plan.json';mp=d/'campaign.json'
        if not pp.exists():continue
        try:jobs=json.loads(pp.read_text());rows=json.loads(mp.read_text()) if mp.exists() else []
        except json.JSONDecodeError:continue
        if any(r['status']!='complete' for r in rows):raise RuntimeError(f'Campaign failed: {d.name}')
        done={r['name'] for r in rows};pending=[j['name'] for j in jobs if j['name'] not in done]
        if pending:lines.append(f'{d.name}: {len(rows)}/{len(jobs)}, {pending[0]}')
    complete=all((HERE/f'{m}_{stage}_l{lane}_DONE.json').exists() for m in ('llama2','llama3') for lane in (0,1) for stage in ('postopt','stability'))
    return '\n'.join(lines) or 'Between stages',complete


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args();previous=None
    while True:
        s,complete=snapshot()
        if s!=previous or not a.watch:print(time.strftime('%H:%M:%S'),s,sep='\n',flush=True);previous=s
        if complete:print('ALL_LANES_COMPLETE',flush=True);return
        if not a.watch:return
        time.sleep(30)

if __name__=='__main__':main()
