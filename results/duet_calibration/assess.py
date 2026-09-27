"""Evaluate frozen predictions without revising them after validation."""
import json
from pathlib import Path
import numpy as np
from calibrator import HERE, Calibrator, load_run, metrics


def spearman(x,y):
    def ranks(z):return np.array([(np.sum(z<v)+np.sum(z<=v)-1)/2 for v in z])
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])


def summary():
    frozen=json.loads((HERE/'frozen.json').read_text());plan=json.loads((HERE/'plan.json').read_text())
    model=Calibrator(frozen['model']);rows=[]
    for p in (HERE/'runs/validation').glob('*/complete.json'):
        run=load_run(p.parent);m=metrics(run)
        # Input lengths are known before inference; quality/cost model is frozen.
        pred=model.predict(run['config'],[r['prompt_tokens'] for r in run['records']])
        actual_cycle=np.mean([x for r in run['records'] for x in r['metrics']['target_step_times'][3:-1]])*1000
        rows.append(dict(tag=run['config']['tag'],config=run['config'],actual=m,predicted=pred,
                         actual_cycle_ms=actual_cycle))
    rows.sort(key=lambda r:r['actual']['tps'],reverse=True)
    if not rows:return
    bytag={r['tag']:r for r in rows};best=rows[0]['actual']['tps'];sel=frozen['selection']
    scored={}
    for name,tag in sel.items():
        if isinstance(tag,list):
            candidates=[bytag[t] for t in tag if t in bytag]
            if not candidates:continue
            chosen=max(candidates,key=lambda r:r['actual']['tps']);tag=chosen['tag']
        if tag in bytag:
            scored[name]=dict(tag=tag,tps=bytag[tag]['actual']['tps'],regret_pct=100*(1-bytag[tag]['actual']['tps']/best))
    x=np.array([r['predicted']['tps'] for r in rows]);y=np.array([r['actual']['tps'] for r in rows])
    report=dict(complete=len(rows)==len(plan['validation']),n=len(rows),expected=len(plan['validation']),rows=rows,
                selection=scored,oracle=rows[0]['tag'],rank_spearman=spearman(x,y) if len(rows)>1 else None,
                tps_mape_pct=float(np.mean(abs(x-y)/y)*100),
                cycle_mape_pct=float(np.mean([abs(r['predicted']['cycle_ms']-r['actual_cycle_ms'])/r['actual_cycle_ms'] for r in rows])*100),
                reward_mape_pct=float(np.mean([abs(r['predicted']['tokens_per_step']-r['actual']['u'])/r['actual']['u'] for r in rows])*100),
                calibration_total_s=frozen['calibration_cost_s'],validation_total_s=sum(r['actual']['elapsed_s'] for r in rows))
    (HERE/'validation_summary.json').write_text(json.dumps(report,indent=2))
    lines=['# Frozen model validation','',f'Completed {len(rows)}/{len(plan["validation"])} configurations.',
           '', '| config | actual output TPS | predicted TPS | actual steady cycle ms | predicted cycle ms | actual tokens/step | predicted tokens/step |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["tag"]} | {r["actual"]["tps"]:.3f} | {r["predicted"]["tps"]:.3f} | {r["actual_cycle_ms"]:.3f} | {r["predicted"]["cycle_ms"]:.3f} | {r["actual"]["u"]:.3f} | {r["predicted"]["tokens_per_step"]:.3f} |')
    (HERE/'VALIDATION.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':summary()
