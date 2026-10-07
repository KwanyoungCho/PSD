"""Final recommended TPS points after tuning-only deadline revalidation."""
from make_plans import HERE,save
from result_metrics import read,collect,compare,assert_paired_runs


def main():
    original=read(HERE/'FINAL_RESULTS.json');split=read(HERE/'SPLIT.json');out=[];presets=[]
    for model in ('llama2','llama3'):
        frozen=read(HERE/f'{model}_POSTOPT_FROZEN.json')['presets']
        for lane in (0,1):
            if read(HERE/f'{model}_postopt_l{lane}_DONE.json')['status']!='complete':raise ValueError('Postopt incomplete')
        for b in (1,8):
            point=(read(HERE/f'{model}_STABILITY_FROZEN.json')['preset'] if b==8 else frozen[str(b)])
            duet=[];ssd=[];speed=[]
            for rep in (0,1):
                lane=0 if ((b==1)==(rep==0)) else 1
                kind=point['selected_existing']
                if kind=='stream':
                    directory=HERE/f'{model}_stream_l{lane}_full';name=f'{model}_b{b}_stream_fast_r{rep}'
                elif kind=='trim':
                    directory=HERE/f'{model}_opt_l{lane}_full';name=f'{model}_b{b}_opt_fast_r{rep}'
                elif kind=='postopt' or b==1:
                    directory=HERE/f'{model}_postopt_l{lane}_full';name=f'{model}_b{b}_postopt_final_r{rep}'
                else:
                    directory=HERE/f'{model}_stability_l{lane}_full';name=f'{model}_b{b}_stability_final_r{rep}'
                manifest=read(directory/'campaign.json');record=next(r for r in manifest if r['name']==name)
                if record['status']!='complete' or record['external_pids']:raise ValueError('Invalid postopt measurement')
                raw=read(directory/(name+'.json'));row=collect(raw)
                if rep==0:
                    job=next(j for j in read(directory/'plan.json') if j['name']==name)
                    presets.append(dict(model=model,batch=b,role='duet_fast',measured_path=str((directory/(name+'.json')).relative_to(HERE)),
                        job=job,engine_kwargs=raw['engine_kwargs'],effective_features=raw['effective_features']))
                if row['pass_count']!=2 or raw['env'].get('SSD_PROFILE_DUET','0')!='0':raise ValueError('Invalid final pass')
                row.update(gpus=raw['env']['CUDA_VISIBLE_DEVICES'],path=str((directory/(name+'.json')).relative_to(HERE)))
                control=next(r for r in original['runs'] if (r['model'],r['batch'],r['role'],r['replicate'])==(model,b,'ssd_fast',rep))
                assert_paired_runs(row,control)
                if row['gpus']!=control['gpus'] or row['prompt_sha256']!=control['prompt_sha256']:raise ValueError('Unpaired control')
                duet.append(row);ssd.append(control)
                speed.append(dict(gpus=row['gpus'],replicate=rep,
                    duet_tps=row['boundary_excluded_step_tps'],ssd_tps=control['boundary_excluded_step_tps'],
                    ratio=row['boundary_excluded_step_tps']/control['boundary_excluded_step_tps'],
                    duet_returned_tps=row['decode_tps'],ssd_returned_tps=control['decode_tps'],
                    emitted_per_step_ratio=row['timed_step_mean_tokens']/control['timed_step_mean_tokens'],
                    step_time_ratio=row['timed_step_mean_ms']/control['timed_step_mean_ms']))
            out.append(dict(model=model,batch=b,preset=point,duet=duet,ssd=ssd,
                full480=compare(duet,ssd),heldout432=compare(duet,ssd,split['confirmation']),throughput=speed))
    save('POSTOPT_RESULTS.json',out)
    for model in ('llama2','llama3'):
        for b in (1,8):
            points=read(HERE/f'{model}_b{b}_FROZEN.json')['presets']
            directory=HERE/f'final_{model}_b{b}_r0'
            for role in ('ssd_fast','ssd_al','duet_al'):
                actual=points[role].get('alias',role);name=f'{model}_b{b}_final_{actual}_r0'
                raw=read(directory/(name+'.json'));job=next(j for j in read(directory/'plan.json') if j['name']==name)
                presets.append(dict(model=model,batch=b,role=role,measured_path=str((directory/(name+'.json')).relative_to(HERE)),
                    job=job,engine_kwargs=raw['engine_kwargs'],effective_features=raw['effective_features']))
    save('FINAL_PRESETS.json',presets)
    for model in ('llama2','llama3'):
        for b in (1,8):
            save(f'{model}_b{b}_RECOMMENDED_PLAN.json',[p['job'] for p in presets if p['model']==model and p['batch']==b and p['role'] in ('duet_fast','ssd_fast')])
    lines=['# Final post-optimization parameter check','',
        'Selection uses tuning48 only. B1: four phase/exit neighbors were profiled; top two plus stream/trim controls were warm-checked. '
        'A change requires at least2% over the stream baseline in that warm check; this is a noise guard, not a significance test. '
        'Unchanged settings reuse their existing full480 paired controls; new settings receive both full480 repeats. SSD controls retain the same seeds and GPU IDs. '
        'B8 supersedes that provisional choice with a stability amendment: three warm passes then three measured passes per candidate; select median whole-pass TPS* with a 2% change guard against the original trim control. No slow steps are deleted. '
        'The finite neighborhood is not a proof of global optimality.','',
        '| Model | B | K1/K2 | exit | stream | DUET AL* | SSD AL* | Full480 ΔAL*95% CI | Selection-excluded432 ΔAL*95% CI | TPS* ratios |',
        '|---|---:|---|---:|---:|---:|---:|---|---|---|']
    for r in out:
        a=r['preset']['args'];f=r['full480'];stream=r['preset']['env'].get('SSD_BATCH_TREE_PROXY_STREAM','0')
        lines.append(f"| {r['model']} | {r['batch']} | {a['k1']}/{a['k2']} | {a['exit_layer']} | {stream} | {f['left']['boundary_excluded_al']:.4f} | {f['right']['boundary_excluded_al']:.4f} | {f['delta_al_ci95']} | {r['heldout432']['delta_al_ci95']} | "+' / '.join(f"{p['ratio']:.3f}x" for p in r['throughput'])+' |')
    (HERE/'POSTOPT_TABLES.md').write_text('\n'.join(lines)+'\n')
    print('Final post-optimization comparisons',len(out))

if __name__=='__main__':main()
