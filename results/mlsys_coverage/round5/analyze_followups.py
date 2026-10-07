"""Audit transferred batches, long contexts and fixed-algorithm optimizations."""
from collections import defaultdict
import numpy as np
from make_plans import HERE,save
from result_metrics import read,collect,compare,assert_paired_runs
from continue_opt import parity
from analyze_screen import analyze


def complete(directory):
    jobs=read(directory/'plan.json');rows=read(directory/'campaign.json')
    if len(jobs)!=len(rows) or any(r['status']!='complete' or r['external_pids'] for r in rows):
        raise ValueError(f'Incomplete/shared campaign: {directory}')
    return jobs


def load(directory,job):
    path=directory/(job['name']+'.json');raw=read(path);row=collect(raw)
    row.update(path=str(path.relative_to(HERE)),gpus=raw['env']['CUDA_VISIBLE_DEVICES'])
    return raw,row


def baseline(model,b,role,rep):
    frozen=read(HERE/f'{model}_b{b}_FROZEN.json')['presets']
    role=frozen[role].get('alias',role)
    path=HERE/f'final_{model}_b{b}_r{rep}/{model}_b{b}_final_{role}_r{rep}.json'
    return read(path),dict(collect(read(path)),path=str(path.relative_to(HERE)))


def speed(left,right):
    return dict(boundary_tps_ratio=left['boundary_excluded_step_tps']/right['boundary_excluded_step_tps'],
                returned_tps_ratio=left['decode_tps']/right['decode_tps'])


def main():
    transferred=[];impl=[];trim=[];profiles=[];streams=[]
    for model in ('llama2','llama3'):
        directory=HERE/f'final_{model}_b8_r0';jobs=complete(directory);groups=defaultdict(dict)
        for job in jobs:
            stage=job['stage']
            if stage in ('batch_transfer','long_transfer'):
                raw,row=load(directory,job)
                groups[stage,row['batch']][job['role']]=(raw,row)
            if stage=='implementation_control':
                off,offrow=load(directory,job);on,onrow=baseline(model,8,'duet_fast',0)
                assert_paired_runs(onrow,offrow)
                check=parity(on,off)
                impl.append(dict(model=model,enabled=onrow,disabled=offrow,parity=check,**speed(onrow,offrow)))
        for (stage,b),pair in groups.items():
            left,right=pair['duet'][1],pair['ssd'][1]
            assert_paired_runs(left,right)
            if left['gpus']!=right['gpus']:raise ValueError('Unpaired hardware')
            transferred.append(dict(model=model,stage=stage,batch=b,duet=left,ssd=right,
                al_comparison=compare([left],[right]),**speed(left,right)))
        for lane in (0,1):
            directory=HERE/f'{model}_opt_l{lane}_full'
            for job in complete(directory):
                b,rep,role=job['batch'],job['replicate'],job['role']
                optimized,optrow=load(directory,job);control,base=baseline(model,b,role,rep)
                assert_paired_runs(optrow,base)
                if optimized['env']['CUDA_VISIBLE_DEVICES']!=control['env']['CUDA_VISIBLE_DEVICES']:
                    raise ValueError('Unpaired optimization hardware')
                check=parity(optimized,control)
                trim.append(dict(model=model,batch=b,role=role,replicate=rep,optimized=optrow,
                    control=base,parity=check,**speed(optrow,base)))
            directory=HERE/f'{model}_opt_l{lane}_profile';jobs=complete(directory)
            rows=[analyze(directory/(j['name']+'.json')) for j in jobs]
            check=parity(read(directory/(jobs[0]['name']+'.json')),read(directory/(jobs[1]['name']+'.json')))
            profiles.append(dict(model=model,lane=lane,parity=check,control=rows[0],optimized=rows[1]))
            directory=HERE/f'{model}_stream_l{lane}_full'
            for job in complete(directory):
                b,rep=job['batch'],job['replicate']
                optimized,optrow=load(directory,job)
                control_path=HERE/f'{model}_opt_l{lane}_full/{model}_b{b}_opt_fast_r{rep}.json'
                control=read(control_path);base=collect(control)
                assert_paired_runs(optrow,base)
                if optimized['env']['CUDA_VISIBLE_DEVICES']!=control['env']['CUDA_VISIBLE_DEVICES']:
                    raise ValueError('Unpaired stream hardware')
                streams.append(dict(model=model,batch=b,replicate=rep,optimized=optrow,control=base,
                    parity=parity(optimized,control),**speed(optrow,base)))
            directory=HERE/f'{model}_stream_l{lane}_profile'
            jobs=complete(directory)
            for b in (1,8):
                pair=[j for j in jobs if j['stage']=='stream_profile' and j['batch']==b]
                profiles.append(dict(model=model,lane=lane,batch=b,kind='stream',
                    parity=parity(read(directory/(pair[0]['name']+'.json')),read(directory/(pair[1]['name']+'.json'))),
                    control=analyze(directory/(pair[0]['name']+'.json')),
                    optimized=analyze(directory/(pair[1]['name']+'.json'))))
    save('FOLLOWUP_RESULTS.json',dict(transferred=transferred,implementation=impl,ladder_trim=trim,proxy_stream=streams,profiles=profiles))
    lines=['# Supplementary comparisons','',
        'All full480 measured last passes. Transfer settings are not independently tuned B2/B4/long optima. '
        'Optimization ratios compare identical parameters, seeds and GPU IDs; both passes must preserve token outputs and proposal/acceptance events.','',
        '| Model | Test | B | DUET AL* | SSD AL* | TPS* DUET/SSD |',
        '|---|---|---:|---:|---:|---:|']
    for r in transferred:
        lines.append(f"| {r['model']} | {r['stage']} | {r['batch']} | {r['duet']['boundary_excluded_al']:.4f} | {r['ssd']['boundary_excluded_al']:.4f} | {r['boundary_tps_ratio']:.3f}x |")
    lines+=['','| Model | Optimization | B | Role/repeat | TPS* on/off | Raw returned TPS on/off |','|---|---|---:|---|---:|---:|']
    for r in impl:
        lines.append(f"| {r['model']} | fused+bulk | 8 | fast/0 | {r['boundary_tps_ratio']:.3f}x | {r['returned_tps_ratio']:.3f}x |")
    for r in trim:
        lines.append(f"| {r['model']} | ladder trim | {r['batch']} | {r['role']}/{r['replicate']} | {r['boundary_tps_ratio']:.3f}x | {r['returned_tps_ratio']:.3f}x |")
    for r in streams:
        lines.append(f"| {r['model']} | proxy stream after trim | {r['batch']} | fast/{r['replicate']} | {r['boundary_tps_ratio']:.3f}x | {r['returned_tps_ratio']:.3f}x |")
    (HERE/'FOLLOWUP_TABLES.md').write_text('\n'.join(lines)+'\n')
    print('Audited supplements',len(transferred),'implementation pairs',len(impl),'trim pairs',len(trim),'stream pairs',len(streams))

if __name__=='__main__':main()
