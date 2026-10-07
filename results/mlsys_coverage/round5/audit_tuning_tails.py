"""Descriptive timing-tail audit; never changes any reported TPS counter."""
from make_plans import HERE,save
from result_metrics import read
from continue_stability import tail


def main():
    records=[]
    for directory in sorted(HERE.glob('*warm')):
        if not directory.is_dir() or not(directory/'plan.json').exists():continue
        for job in read(directory/'plan.json'):
            path=directory/(job['name']+'.json')
            if not path.exists():continue
            raw=read(path)
            if raw.get('status')!='complete' or raw['env'].get('SSD_PROFILE_DUET','0')=='1':continue
            records.append(dict(path=str(path.relative_to(HERE)),args=raw['args'],
                passes=[dict(seed=c['seed'],**tail(c)) for c in raw['cells']]))
    save('TUNING_TAIL_AUDIT.json',dict(definition='Descriptive slow step = duration > 5 times median clean step duration within a pass. Does not identify the cause. No step deleted from TPS*.',runs=records))
    lines=['# Tuning timing-tail audit','',
        'These are diagnostics only: all actual TPS* values include slow steps. A large tail can dominate a short selection run. '
        'The B8 amendment repeats complete tuning passes, then uses their median TPS*. No per-step trimming is used. '
        'Lazy shape capture is a possible explanation; these uninstrumented records alone do not prove it.','',
        '| Run | Pass | Seed | TPS* | Median / max step (ms) | >5×median steps | Tail time share |',
        '|---|---:|---:|---:|---|---:|---:|']
    for r in records:
        for i,s in enumerate(r['passes']):
            if i==0:continue
            lines.append(f"| {r['path']} | {i+1} | {s['seed']} | {s['tps']:.1f} | {s['median_ms']:.2f} / {s['max_ms']:.2f} | {s['over_5median_steps']} | {s['over_5median_time_fraction']:.1%} |")
    (HERE/'TUNING_TAIL_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print('Audited tuning runs',len(records))


if __name__=='__main__':main()
