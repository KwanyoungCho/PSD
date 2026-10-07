"""Paired-run GPU telemetry; clocks/temperature are descriptive controls."""
import json,collections
import numpy as np
from make_plans import HERE,save


def stat(x):
    return None if not x else dict(n=len(x),mean=float(np.mean(x)),p50=float(np.median(x)),min=float(min(x)),max=float(max(x)))


def main():
    out=[]
    for campaign in sorted(HERE.glob('*/campaign.json')):
        for run in json.loads(campaign.read_text()):
            path=campaign.parent/(run['name']+'.telemetry.jsonl')
            if not path.exists():continue
            per=collections.defaultdict(lambda:collections.defaultdict(list))
            for line in path.read_text().splitlines():
                record=json.loads(line)
                for row in record['gpu'].splitlines():
                    values=[x.strip() for x in row.split(',')]
                    gpu=int(values[0]);mem,util,clock,power,temp=map(float,values[1:])
                    per[gpu]['all_temperature'].append(temp)
                    per[gpu]['utilization'].append(util)
                    if util>=50:
                        for k,v in [('active_clock',clock),('active_power',power),('active_temperature',temp),('active_memory',mem)]:per[gpu][k].append(v)
            out.append(dict(run=run['name'],directory=campaign.parent.name,status=run['status'],
                seconds=run['ended']-run['started'],external_pids=run['external_pids'],
                devices={str(g):{k:stat(v) for k,v in d.items()} for g,d in per.items()}))
    save('TELEMETRY.json',out)
    print('GPU histories',len(out),'overlap records',sum(bool(r['external_pids']) for r in out))

if __name__=='__main__':main()
