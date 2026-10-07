"""Match slow diagnostic steps to recorded lazy captures; no timing deletion."""
import numpy as np
from make_plans import HERE,save
from result_metrics import read
from analyze_screen import analyze,grouped


def main():
    out=[]
    for directory in sorted(HERE.glob('*_profile')):
        for path in sorted(directory.glob('llama*.json')):
            row=analyze(path)
            if not row or not row.get('profile_paths'):continue
            raw=read(path);cell=raw['cells'][-1];events=cell['metrics']['phase_events']
            capture={}
            for p,marker in zip(row['profile_paths'],('target_send_request','batch_miss_draft')):
                for sid,labels in grouped(HERE/p,marker).items():
                    for key,spans in labels.items():
                        if 'capture' in key:
                            capture.setdefault(sid,[]).append(dict(label=key,
                                ms=sum(e['cuda_ms'] for e in spans)))
            clean=[s for s in cell['metrics']['decode_steps'] if not(s['output_cap_reached'] or s['clipped'])]
            median=float(np.median([s['seconds'] for s in clean]));slow=[]
            for s in clean:
                if s['seconds']<=5*median:continue
                sid=events[s['event_start']]['step_id']
                slow.append(dict(step=sid,ms=s['seconds']*1000,
                    same_step_captures=capture.get(sid,[]),previous_step_captures=capture.get(sid-1,[])))
            out.append(dict(path=row['path'],median_clean_step_ms=median*1000,
                slow_steps=slow,total_slow=len(slow),
                capture_linked_slow=sum(bool(s['same_step_captures'] or s['previous_step_captures']) for s in slow),
                capture_steps=len(capture)))
    save('CAPTURE_TAIL_AUDIT.json',dict(note='Diagnostic runs only. Matching same/previous-step capture is an association, not a label for separate uninstrumented warm runs. No reported TPS is filtered.',runs=out))
    print('Profile slow steps',sum(r['total_slow'] for r in out),
        'with same/previous capture',sum(r['capture_linked_slow'] for r in out))


if __name__=='__main__':main()
