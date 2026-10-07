"""Final manifest/counter/source audit. Never treats failed workers as results."""
import collections,hashlib,json,subprocess,sys,platform,os,re
import importlib.metadata
from pathlib import Path
from make_plans import HERE,ROOT,save
from result_metrics import read,collect


def main():
    records=[];counts=collections.Counter();unexecuted=[]
    for directory in sorted(HERE.iterdir()):
        if not directory.is_dir() or not (directory/'plan.json').exists():continue
        jobs=read(directory/'plan.json');rows=read(directory/'campaign.json')
        names={r['name'] for r in rows}
        missing=[j['name'] for j in jobs if j['name'] not in names]
        # A historical campaign stopped at its first failure. Its remaining
        # jobs must be completed in a retry directory and are audited below.
        if missing and not any(r['status']!='complete' for r in rows):
            raise RuntimeError(f'Unfinished campaign {directory}: {missing}')
        unexecuted.extend(dict(directory=directory.name,name=n) for n in missing)
        for run in rows:
            counts[run['status']]+=1;path=directory/(run['name']+'.json')
            record=dict(directory=directory.name,name=run['name'],status=run['status'],
                        external_pids=run['external_pids'],seconds=run['ended']-run['started'])
            if path.exists():
                raw=read(path);record['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                record['raw_status']=raw.get('status')
                record['git_commit']=raw.get('git_commit')
                if run['status']=='complete':
                    if raw.get('status')!='complete':raise ValueError('Manifest/report mismatch')
                    checks=[collect(raw,i)['checks'] for i in range(len(raw['cells']))]
                    record.update(cells=len(checks),questions=len(raw['question_indexes']),all_counters_pass=True)
            elif run['status']=='complete':raise FileNotFoundError(path)
            records.append(record)
    completed={r['name'] for r in records if r['status']=='complete'}
    for item in unexecuted:
        if item['name'] not in completed:raise RuntimeError(f'Unexecuted failed-campaign remainder: {item}')
    regressions=read(HERE/'FINAL_REGRESSIONS.json')
    if len(regressions)!=2 or any(r['status']!='passed' for r in regressions):raise ValueError('Final regressions incomplete')
    stream_regressions=read(HERE/'FINAL_STREAM_REGRESSIONS.json')
    if len(stream_regressions)!=1 or stream_regressions[0]['status']!='passed':raise ValueError('Stream regressions incomplete')
    for rows,stream in ((regressions,'0'),(stream_regressions,'1')):
        for row in rows:
            log=(HERE/row['log']).read_text()
            matches=re.findall(r'Ran (\d+) tests? in ([0-9.]+)s',log)
            if not matches or not re.search(r'^OK(?: \(skipped=\d+\))?$',log,re.M):
                raise ValueError(f'Incomplete unittest log: {row["log"]}')
            skipped=re.findall(r'^OK \(skipped=(\d+)\)$',log,re.M)
            row.update(tests=int(matches[-1][0]),seconds=float(matches[-1][1]),
                       skipped=int(skipped[-1]) if skipped else 0,
                       env=dict(SSD_TREE_LADDER_TRIM=row['trim'],SSD_BATCH_TREE_PROXY_STREAM=stream,
                                SSD_TREE_FUSED_MATH='1',SSD_BATCH_TREE_BULK_EXPORT='1',SSD_TREE_PARALLEL_INSERT='0'))
    regressions+=stream_regressions
    for model in ('llama2','llama3'):
        for lane in (0,1):
            read(HERE/f'{model}_opt_l{lane}_PARITY.json')
            read(HERE/f'{model}_stream_l{lane}_PARITY.json')
            read(HERE/f'{model}_stream_l{lane}_FULL_PARITY.json')
            if read(HERE/f'{model}_postopt_l{lane}_DONE.json')['status']!='complete':raise ValueError('Postopt incomplete')
            if read(HERE/f'{model}_stability_l{lane}_DONE.json')['status']!='complete':raise ValueError('Stability revalidation incomplete')
    # Six-pass tuning rechecks also cover new narrow-tree graph shapes.
    # Compare only parameter-identical stream on/off pairs; different
    # algorithm settings are never expected to emit the same random output.
    from make_tree_plan import kwargs_from
    from continue_opt import parity
    stability_parity=[]
    for model in ('llama2','llama3'):
        groups=collections.defaultdict(dict)
        for point in read(HERE/f'{model}_STABILITY_FROZEN.json')['candidates']:
            if point['args']['mode']!='duet-tree':continue
            key=json.dumps(dict(params=kwargs_from(point),trim=point['env'].get('SSD_TREE_LADDER_TRIM','0')),sort_keys=True)
            groups[key][point['env'].get('SSD_BATCH_TREE_PROXY_STREAM','0')]=point
        for pair in groups.values():
            if set(pair)!={'0','1'}:continue
            control,stream=(read(HERE/pair[k]['origin']) for k in ('0','1'))
            if control['env']['CUDA_VISIBLE_DEVICES']!=stream['env']['CUDA_VISIBLE_DEVICES']:raise ValueError('Stability parity hardware differs')
            check=parity(control,stream)
            stability_parity.append(dict(model=model,control=pair['0']['origin'],stream=pair['1']['origin'],**check))
    if len(stability_parity)!=4:raise ValueError('Expected four six-pass stability parity pairs')
    save('STABILITY_PARITY.json',stability_parity)
    source={}
    for path in sorted((ROOT/'ssd/ssd').rglob('*.py')):
        source[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    save('SOURCE_MANIFEST.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        last_runtime_commit=subprocess.check_output(['git','log','-1','--format=%H','--','ssd/ssd'],cwd=ROOT,text=True).strip(),
        runtime_sha256=source,model_identity_reference='../round4/MODEL_MANIFEST.json',
        calibration={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (HERE.parent/'round4/calibration').glob('*.json')}))
    save('INVENTORY.json',dict(counts=dict(counts),runs=records,retried_unexecuted=unexecuted,
        regressions=regressions,stability_parity=stability_parity,exclusions=read(HERE/'EXCLUSIONS.json')))
    packages={}
    for name in ('torch','triton','numpy','matplotlib','transformers','safetensors','flashinfer-python'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    save('ANALYSIS_ENV.json',dict(python=sys.version,executable=sys.executable,
        platform=platform.platform(),packages=packages,logical_cpu_count=os.cpu_count(),
        affinity=sorted(os.sched_getaffinity(0)),note='No experiment-specific CPU affinity pinning. GPU driver/topology: GPU_TOPOLOGY.txt.'))
    print(dict(counts),'counter-audited complete runs',len(completed))

if __name__=='__main__':main()
