"""Generate the next finite neighborhood from measured phase deadlines."""
import argparse,json
from make_plans import HERE,job,save
from analyze_screen import analyze

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--batch',type=int,required=True)
a=p.parse_args();m=a.model;b=a.batch
rows=[analyze(p) for p in (HERE/f'{m}_b{b}_duet_anchor').glob(f'{m}*.json')]
rows=[r for r in rows if r]
if len(rows)!=4:raise RuntimeError('All four exit anchors are required')
best=max(rows,key=lambda r:r['summary']['boundary_excluded_al'])
e=best['args']['exit_layer'];pr=best['profile']
p1=pr['p1_slack_ms']['p50'];p2=pr['p2_slack_ms']['p50']
k1s=(2,3,4,6) if p1<0 else (4,6,8,12)
k2s=(1,2) if p2<0 else (2,3,4)
jobs=[];seen={(r['args']['exit_layer'],4,2,2) for r in rows}

def add(exit,k1,k2,roots=2,reason='phase neighborhood'):
    key=(exit,k1,k2,roots)
    if key in seen or k2>k1:return
    seen.add(key)
    j=job(m,b,f'refine_e{exit}_k{k1}_{k2}_r{roots}',extra=[
        '--exit-layer',str(exit),'--k1',str(k1),'--k2',str(k2),
        '--p1-nodes',str(min(16,k1*2)),'--p2-nodes',str(k2*2),'--p1-roots',str(roots)])
    j['reason']=reason;jobs.append(j)

if p1<0:
    for ex in sorted({e,21}):
        add(ex,4,2,1,'P1 misses proxy deadline: halve P1 roots at fixed K/N; measure lost hit versus cheaper forwards')
        add(ex,3,2,1,'P1 root/depth joint budget; earlier exit may restore P2 window')
for k1 in k1s:
    for k2 in k2s:add(e,k1,k2,reason=f'anchor P1 slack {p1:.3f} ms, P2 slack {p2:.3f} ms; test depth neighbors')
center=3 if p1<0 else 6
for ex in sorted({max(1,e-4),min(31,e+2)}):
    for k2 in (1,2) if p2<0 else (2,3):
        add(ex,center,k2,reason='Re-evaluate exit after changing draft phase duration')
save(f'{m}_b{b}_refine_plan.json',jobs)
save(f'{m}_b{b}_refine_decision.json',dict(anchor=best['name'],p1_slack_ms=p1,p2_slack_ms=p2,
     k1_values=k1s,k2_values=k2s,jobs=len(jobs),selection='AL-best anchor; AL/time Pareto retained in later selection'))
print(m,b,best['name'],p1,p2,len(jobs))
