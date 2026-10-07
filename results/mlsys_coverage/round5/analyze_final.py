"""Frozen-config full480 comparison, paired GPU/process repeats and AL CI."""
import csv,json,re
from collections import defaultdict
import numpy as np
from make_plans import HERE,save
from result_metrics import read,collect,compare,aggregate,assert_paired_runs


def main():
    source=read(HERE.parent/'questions.json');split=read(HERE/'SPLIT.json')
    records={};inventory=[]
    for directory in HERE.glob('final_*'):
        if not directory.is_dir():continue
        plan=read(directory/'plan.json');manifest=read(directory/'campaign.json')
        if len(plan)!=len(manifest) or any(r['status']!='complete' for r in manifest):raise RuntimeError(f'Incomplete {directory}')
        byname={r['name']:r for r in manifest}
        for item in plan:
            if item.get('stage')!='full480':continue
            name=item['name'];match=re.fullmatch(r'(llama[23])_b(1|8)_final_(\w+)_r([01])',name)
            if not match:raise ValueError(name)
            model,b,role,rep=match.groups();b=int(b);rep=int(rep)
            path=directory/(name+'.json');report=read(path)
            if report['status']!='complete' or report['env'].get('SSD_PROFILE_DUET','0')!='0':raise ValueError('Invalid final timing')
            if report['question_indexes']!=list(range(480)):raise ValueError('Incomplete full corpus')
            run=byname[name]
            if run['external_pids']:raise ValueError('Final GPUs were shared')
            row=collect(report)
            if row['pass_count']<2:raise ValueError('Missing warm pass')
            row.update(model=model,batch=b,role=role,replicate=rep,path=str(path.relative_to(HERE)),
                gpus=report['env']['CUDA_VISIBLE_DEVICES'],git_commit=report['git_commit'],
                model_parameters={k:report['args'].get(k) for k in ['k1','k2','draft_fan_out','proxy_fan_out','exit_layer',
                    'p1_nodes','p2_nodes','p1_roots','p2_budget','tree_width','tree_beta','tree_proxy_threshold','tree_conf_threshold']})
            records[model,b,role,rep]=row;inventory.append(row)
    comparisons=[]
    for model in ('llama2','llama3'):
        frozen=read(HERE/f'{model}_FROZEN.json')
        for b in (1,8):
            presets=frozen['presets'][str(b)]
            def role(name):return presets[name].get('alias',name)
            for name,l,r in [('fast','duet_fast','ssd_fast'),('al','duet_al','ssd_al'),
                             ('al_vs_fast','duet_al','ssd_fast')]:
                left=[records[model,b,role(l),i] for i in (0,1)]
                right=[records[model,b,role(r),i] for i in (0,1)]
                paired=[]
                for a,z in zip(left,right):
                    assert_paired_runs(a,z)
                    if a['gpus']!=z['gpus'] or a['prompt_sha256']!=z['prompt_sha256']:raise ValueError('Unpaired hardware/input')
                    paired.append(dict(gpus=a['gpus'],replicate=a['replicate'],
                        duet_tps=a['boundary_excluded_step_tps'],ssd_tps=z['boundary_excluded_step_tps'],
                        ratio=a['boundary_excluded_step_tps']/z['boundary_excluded_step_tps'],
                        duet_returned_tps=a['decode_tps'],ssd_returned_tps=z['decode_tps']))
                groups={g:compare(left,right,[i for i,q in enumerate(source) if q['group']==g])
                        for g in sorted({q['group'] for q in source})}
                result=dict(model=model,batch=b,comparison=name,left_role=role(l),right_role=role(r),
                    full480=compare(left,right),heldout432=compare(left,right,split['confirmation']),
                    groups=groups,throughput=paired,
                    geometric_mean_speedup=float(np.exp(np.mean(np.log([p['ratio'] for p in paired])))),
                    speedup_range=[min(p['ratio'] for p in paired),max(p['ratio'] for p in paired)])
                comparisons.append(result)
    save('FINAL_RESULTS.json',dict(runs=inventory,comparisons=comparisons))
    lines=['# Frozen full480 comparison','',
        'AL* excludes cap/clipping terminal events. TPS* excludes the complete boundary batch step. '
        'Measured last passes only; raw returned-token TPS is retained in FINAL_RESULTS.json. '
        'Two process seeds use paired GPU groups; the ratio range is not a confidence interval.','',
        '| Model | B | comparison | DUET AL* | SSD AL* | Delta AL* 95% CI | heldout432 delta CI | TPS* ratio by paired repeat |',
        '|---|---:|---|---:|---:|---|---|---|']
    for r in comparisons:
        f=r['full480'];h=r['heldout432'];pair=r['throughput']
        lines.append(f"| {r['model']} | {r['batch']} | {r['comparison']} | {f['left']['boundary_excluded_al']:.4f} | {f['right']['boundary_excluded_al']:.4f} | {f['delta_al_ci95']} | {h['delta_al_ci95']} | "+', '.join(f"GPU {p['gpus']}: {p['ratio']:.3f}x" for p in pair)+' |')
    (HERE/'FINAL_TABLES.md').write_text('\n'.join(lines)+'\n')
    print('Audited full runs',len(inventory),'comparisons',len(comparisons))

if __name__=='__main__':main()
