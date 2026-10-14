"""Summarize measured phase deadlines; do not infer quality from latency."""
import argparse,json,statistics
from pathlib import Path
from collections import defaultdict


def analyze(folder):
    rows=[]
    for f in Path(folder).glob('duet*.json'):
        group=0;last=0
        for x in json.loads(f.read_text()):
            if x['label']=='_anchor':continue
            sid=x.get('step_id')
            if sid is not None:
                if sid<last:group+=1
                last=sid
            rows.append(dict(x,group=group))
    by_step=defaultdict(dict)
    for r in rows:
        if r.get('step_id') is not None:by_step[(r['group'],r['step_id'])][r['label']]=r
    bad={(r['group'],r['step_id']) for r in rows if 'capture' in r['label']}
    costs=defaultdict(list);timeline=[]
    for key,rs in by_step.items():
        group,sid=key
        if key in bad or sid<4 or group==0:continue
        for name,r in rs.items():costs[name].append(r['cuda_ms'])
        needed=('batch_target_pre','batch_p1_total','batch_p2_total','batch_proxy_receive')
        if not all(k in rs for k in needed):continue
        nxt=by_step.get((group,sid+1),{}).get('target_send_request')
        if not nxt or (group,sid+1) in bad:continue
        origin=rs['batch_target_pre']['wall_start_ns']
        ms=lambda x:(x-origin)/1e6
        row=dict(step=sid,group=group,target_start=0.,proxy_ready=ms(rs['batch_proxy_receive']['wall_end_ns']),
                 p1_start=ms(rs['batch_p1_total']['wall_start_ns']),p1_end=ms(rs['batch_p1_total']['wall_end_ns']),
                 p2_start=ms(rs['batch_p2_total']['wall_start_ns']),p2_end=ms(rs['batch_p2_total']['wall_end_ns']),
                 next_request=ms(nxt['cpu_dispatch_start_ns']))
        row['p1_margin']=row['proxy_ready']-row['p1_end']
        row['p2_margin']=row['next_request']-row['p2_end']
        timeline.append(row)
    return dict(folder=str(folder),costs={k:dict(n=len(v),median_ms=statistics.median(v),mean_ms=statistics.mean(v)) for k,v in costs.items()},
        timing_medians={k:statistics.median(x[k] for x in timeline) for k in timeline[0] if k!='step'} if timeline else {},
        p2_late_fraction=sum(x['p2_margin']<0 for x in timeline)/len(timeline) if timeline else None,
        steps=timeline,excluded_capture_step_ids=sorted(bad))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=[]
    for folder in sorted(args.root.rglob('*_profile')):
        if list(folder.glob('duet*.json')):out.append(analyze(folder))
    args.output.write_text(json.dumps(out,indent=2))
    names=['batch_target_pre','batch_target_post','batch_p1_total','batch_p2_total','target_spec_wait']
    for row in out:
        print(row['folder'].replace(str(args.root),''),{n:round(row['costs'][n]['median_ms'],2) for n in names if n in row['costs']},
              'margins',{k:round(v,2) for k,v in row['timing_medians'].items() if 'margin' in k})

if __name__=='__main__':main()
