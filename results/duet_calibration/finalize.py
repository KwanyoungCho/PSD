"""Summaries and prompt-level paired uncertainty for completed experiments."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shlex
import numpy as np
from campaign import HERE
from calibrator import load_run,metrics,profile_rows
from adapters import cli_arguments


def confirmation():
    ref=json.loads((HERE/'refinement_frozen.json').read_text());plan=json.loads((HERE/'plan.json').read_text())
    datasets=[m['dataset'] for m in plan['datasets']['confirmation']['mapping']]
    groups=[np.flatnonzero(np.array(datasets)==name) for name in dict.fromkeys(datasets)]
    runs={};times={};rows=[]
    for cfg in ref['confirmation']:
        key=cfg['tag'];rr=[]
        for seed in ref['seeds']:rr.append(load_run(HERE/'runs/confirmation'/f'{key}_s{seed}'))
        runs[key]=rr;times[key]=np.array([[r['wall_s'] for r in x['records']] for x in rr])
        ev=[e for x in rr for r in x['records'] for e in r['metrics']['phase_events']]
        pi=[np.mean([e['source']==c for e in ev]) for c in range(3)]
        al=[np.mean([e['accepted_spec_len'] for e in ev if e['source']==c]) for c in range(3)]
        tokens=sum(r['output_tokens'] for x in rr for r in x['records'])
        rows.append(dict(tag=key,config=cfg,tps=tokens/times[key].sum(),per_seed_tps=[metrics(r)['tps'] for r in rr],
                         tokens=tokens,wall_s=float(times[key].sum()),phase_probability=pi,accepted_length=al))
    baseline='chain_e56_k8_4_w15'
    rng=np.random.default_rng(913);resamples=np.array([np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(4000)])
    for row in rows:
        key=row['tag'];gain=(times[baseline][:,resamples].sum(axis=(0,2))/times[key][:,resamples].sum(axis=(0,2))-1)*100
        row['gain_vs_default_pct']=100*(times[baseline].sum()/times[key].sum()-1)
        row['gain_vs_default_ci95_pct']=np.quantile(gain,[.025,.975]).tolist()
        row['dataset_tps']={name:256*len(ref['seeds'])*len(group)/float(times[key][:,group].sum())
                            for name,group in zip(dict.fromkeys(datasets),groups)}
    rows.sort(key=lambda r:r['tps'],reverse=True)
    selected=next(r for r in rows if r['tag']==ref['selected']['tag'])
    best=rows[0];gain=(times[best['tag']][:,resamples].sum(axis=(0,2))/times[selected['tag']][:,resamples].sum(axis=(0,2))-1)*100
    frozen=json.loads((HERE/'frozen.json').read_text())
    selection_cost=sum(load_run(HERE/'runs/selection'/f'{c["config"]["tag"]}_s2413')['meta']['elapsed_s'] for c in ref['selection'])
    cost=frozen['calibration_cost_s']+selection_cost
    default=next(r for r in rows if r['tag']==baseline)
    saving=1/default['tps']-1/selected['tps']
    validation=json.loads((HERE/'validation_summary.json').read_text())
    oracle=validation['oracle']
    oracle_gain=(times[oracle][:,resamples].sum(axis=(0,2))/times[selected['tag']][:,resamples].sum(axis=(0,2))-1)*100
    comparisons=[]
    for reference in rows:
        tag=reference['tag']
        if tag==selected['tag']:continue
        paired=(times[tag][:,resamples].sum(axis=(0,2))/times[selected['tag']][:,resamples].sum(axis=(0,2))-1)*100
        comparisons.append(dict(reference=tag,selected_gain_pct=100*(times[tag].sum()/times[selected['tag']].sum()-1),
                                selected_gain_ci95_pct=np.quantile(paired,[.025,.975]).tolist()))
    out=dict(rows=rows,secondary_selected=selected['tag'],best_observed=best['tag'],
             selected_pairwise_comparisons=comparisons,
             selected_regret_pct=100*(1-selected['tps']/best['tps']),selected_vs_best_ci95_pct=np.quantile(gain,[.025,.975]).tolist(),
             calibration_plus_selection_cost_s=cost,selection_cost_s=selection_cost,
             amortization_tokens_vs_default=cost/saving if saving>0 else None,
             validation_oracle=oracle,selected_vs_validation_oracle_pct=100*(times[oracle].sum()/times[selected['tag']].sum()-1),
             same_configuration_as_validation_oracle=(selected['tag']==oracle),
             selected_vs_validation_oracle_ci95_pct=np.quantile(oracle_gain,[.025,.975]).tolist(),
             validation_exhaustive_cost_s=validation['validation_total_s'],
             recommended_recipe_config_count=5+len(ref['selection']),validation_grid_config_count=len(plan['validation']),
             inference_count=sum(len(x['records']) for rr in runs.values() for x in rr),
             uncertainty='4000 dataset-stratified paired prompt bootstrap; both seeds retained together. Best observed arm is not a global optimum.')
    (HERE/'confirmation_summary.json').write_text(json.dumps(out,indent=2))
    lines=['# Independent confirmation','',out['uncertainty'],'',
           '| config | output TPS | seed 2913 TPS | seed 3913 TPS | vs default (%) | paired 95% CI (%) |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["tag"]} | {r["tps"]:.3f} | {r["per_seed_tps"][0]:.3f} | {r["per_seed_tps"][1]:.3f} | {r["gain_vs_default_pct"]:+.2f} | {r["gain_vs_default_ci95_pct"]} |')
    (HERE/'CONFIRMATION.md').write_text('\n'.join(lines)+'\n')
    recommendation=dict(config=ref['selected'],cli_args=cli_arguments(ref['selected']),
                        scope='Measured 80-layer LayerSkip70B AWQ TP4 + TinyLlama BF16 TP1, RTX4090, B1 T0.7, chain full P1+P2; re-calibrate for another model/hardware/shape.',
                        primary_frozen_model=frozen['selection'],confirmation=selected,
                        method=f'5 measured anchors + {len(ref["selection"])} actual shortlisted candidates, including a secondary lower-bound expansion. No grid-wide quality fitting.')
    recommendation['runtime_contract']=dict(target='/home/chokwans99/awq_calibrated/layerskip_llama2_70b',
        awq_artifact='/home/chokwans99/awq_artifacts/layerskip70b_awq_tp4',
        draft='/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0',target_tp=4,draft_tp=1,batch=1,
        temperature=.7,max_model_len=2048,input_tokens=[16,768],measured_output_tokens=[128,256],
        jit_short=True,exit_replica=True,async_proxy_send=True,chain_proxy_graph=True,
        candidate_source='legacy residual, raw-logit candidate temperature 1.0',
        public_cost_prediction_scope='ChainAdapter guards domain. The frozen experimental Calibrator is a recorded v1 estimator, not an unchecked production configuration API.')
    (HERE/'recommended.json').write_text(json.dumps(recommendation,indent=2))
    (HERE/'recommended_args.txt').write_text(shlex.join(recommendation['cli_args'])+'\n')
    print(json.dumps(out,indent=2))


def extensions():
    rows=[]
    for p in (HERE/'runs/extensions').glob('*/complete.json'):
        r=load_run(p.parent);m=metrics(r);phase=[]
        for c in range(3):
            vals=[t*1000 for rec in r['records'] for e,t in zip(rec['metrics']['phase_events'][2:],rec['metrics']['target_verify_times'][2:]) if e['source']==c]
            phase.append(dict(n=len(vals),mean_ms=float(np.mean(vals)) if vals else None))
        item=dict(config=r['config'],metrics=m,phase_verify=phase)
        if r['config']['mode']=='chain':
            trace,diag=profile_rows(r);item['profile_diagnostics']=diag
            item['components']={k:float(np.mean([x[k] for x in trace])) for k in ['F','P','x','D1','D2','S','gap1','gap2']}
        rows.append(item)
    failures=[dict(path=str(p),error=p.read_text()) for p in (HERE/'runs/extensions').glob('*/failure.txt')]
    failures += [dict(path=str(p),error='Initial tree setup overflowed the 128 MiB FlashInfer verify workspace.',
                      resolved_by='Re-run with 256 MiB verify workspace; see adjacent complete.json.')
                 for p in (HERE/'runs/extensions').glob('*/failed_workspace128.log')]
    (HERE/'extension_summary.json').write_text(json.dumps(dict(rows=rows,failures=failures,scope='Profiled shape/quality diagnostics; not final throughput comparisons.'),indent=2))


def cost_audit():
    stages=defaultdict(lambda:dict(config_runs=0,prompts=0,output_tokens=0,generate_s=0.,launch_through_completion_s=0.))
    for path in (HERE/'runs').glob('*/*/complete.json'):
        run=load_run(path.parent);s=stages[run['meta']['stage']]
        s['config_runs']+=1;s['prompts']+=len(run['records'])
        s['output_tokens']+=sum(r['output_tokens'] for r in run['records'])
        s['generate_s']+=sum(r['wall_s'] for r in run['records'])
        s['launch_through_completion_s']+=run['meta']['elapsed_s']
    discarded=[dict(path=str(p.relative_to(HERE)),elapsed_s=json.loads(p.read_text())['elapsed_s'])
               for p in (HERE/'failed_attempts').rglob('complete.json')]
    (HERE/'cost_audit.json').write_text(json.dumps(dict(stages=stages,
        successful_run_elapsed_s=sum(s['launch_through_completion_s'] for s in stages.values()),
        discarded_completed_runs=discarded,discarded_completed_run_elapsed_s=sum(r['elapsed_s'] for r in discarded),
        note='Counts warmup/load in launch time; excludes failed setup attempts, idle time, CPU analysis. Probe time includes instrumentation and is not inference throughput.'),indent=2))


if __name__=='__main__':
    extensions()
    confirmation()
    cost_audit()
