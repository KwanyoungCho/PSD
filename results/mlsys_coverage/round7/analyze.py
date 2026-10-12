"""Audit original-paper controls and summarize completed paired contrasts."""
import csv,hashlib,json,math,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def mean(x):return statistics.mean(x) if x else None
def arg(args,key,default=None):return args[args.index(key)+1] if key in args else default

rows=[];incomplete=[];pairs=[];audit=[]
for plan in sorted(ROOT.glob('*/plan.json')):
    jobs=json.loads(plan.read_text());manifest=plan.with_name('campaign.json')
    recorded={x['name']:x for x in json.loads(manifest.read_text())} if manifest.exists() else {}
    for job in jobs:
        path=plan.with_name(job['name']+'.json')
        if not path.exists():incomplete.append(dict(name=job['name'],status='not_started'));continue
        d=json.loads(path.read_text())
        if d['status']!='complete':incomplete.append(dict(name=job['name'],status=d['status'],error=d.get('error')));continue
        if 'summary' not in d:continue
        ss=d['summary'];nominal=d['args']['batch']
        ss['included_step_fraction']=ss['steps']/ss['all_steps']
        ss['raw_decode_tps']=sum(s['emitted'] for s in d['steps'])/sum(s['seconds'] for s in d['steps'])
        eligible=[s for s in d['steps'] if s['batch']==nominal and not s['cap'] and not s['clipped']]
        ev=[e for s in eligible for e in s['events']]
        tokens=sum(s['emitted'] for s in eligible);seconds=sum(s['seconds'] for s in eligible)
        checks=dict(all_prompts=len(d['outputs'])==len(d['prompt_ids']),
            same_population=ss['steps']==len(eligible) and ss['tokens']==tokens,
            tps=math.isclose(ss['tps'],tokens/seconds,rel_tol=1e-12),
            batch_limit=all(s['batch']<=nominal for s in d['steps']),
            no_external_gpu=not recorded.get(job['name'],{}).get('external_pids',[]),
            successful_process=recorded.get(job['name'],{}).get('status')=='complete' and recorded.get(job['name'],{}).get('exit_code')==0)
        if d['args']['mode']!='ar':
            checks['actual_tokens']=sum(map(len,d['outputs']))==sum(s['emitted'] for s in d['steps'])
            checks['al_tps_identity']=math.isclose(ss['tps'],nominal*ss['al']/(ss['step_ms']/1000),rel_tol=1e-10)
        if not all(checks.values()):raise AssertionError((job['name'],checks))
        audit.append(dict(name=job['name'],checks=checks))
        arm=job['arm'];scope=next((x for x in ('full','profile','smoke') if '_'+x+'_' in job['name']),'other')
        rep=int(re.search(r'_r(\d+)$',job['name']).group(1)) if re.search(r'_r(\d+)$',job['name']) else 0
        contrast=job['name'].rsplit('_'+arm+'_',1)[0]
        row=dict(name=job['name'],file=str(path.relative_to(ROOT)),contrast=contrast,arm=arm,scope=scope,rep=rep,
            model='llama3' if 'llama3' in d['args']['target'] else 'llama2',batch=nominal,mode=d['args']['mode'],
            temperature=d['args']['temperature'],prompts=len(d['prompt_ids']),engine_commit=d['paper_commit'],
            parameters={k:v for k,v in d['args'].items() if k not in ('output','prompts','limit')},engine_kwargs=d['kwargs'],
            input_hash=hashlib.sha256(json.dumps(d['prompt_ids']).encode()).hexdigest(),summary=ss,
            gpu_pair=recorded.get(job['name'],{}).get('gpu_pair'),control=job.get('control'),
            adapter_arm=d['arm'],implementation=job.get('implementation','benchmark_arm'),
            output_hash=hashlib.sha256(json.dumps(d['outputs']).encode()).hexdigest(),
            warm_hash=hashlib.sha256(json.dumps([g['outputs'] for g in d['warm_groups']]).encode()).hexdigest())
        rows.append(row)
lookup={(x['contrast'],x['scope'],x['rep'],x['arm']):x for x in rows}
by_name={x['name']:x for x in rows}
for row in rows:
    if row['arm']=='base':continue
    base=by_name.get(row['control']) if row['control'] else lookup.get((row['contrast'],row['scope'],row['rep'],'base'))
    if not base:continue
    if base['input_hash']!=row['input_hash'] or base['gpu_pair']!=row['gpu_pair'] or base['parameters']!=row['parameters'] or base['engine_kwargs']!=row['engine_kwargs']:raise AssertionError('Mismatched A/B controls')
    a,b=base['summary'],row['summary']
    pairs.append(dict(contrast=row['contrast'],scope=row['scope'],rep=row['rep'],arm=row['arm'],model=row['model'],
        batch=row['batch'],gpu_pair=row['gpu_pair'],base=base['name'],candidate=row['name'],shared_control=bool(row['control']),
        outputs_equal=base['output_hash']==row['output_hash'],warm_outputs_equal=base['warm_hash']==row['warm_hash'],
        verify_base_ms=a['verify_ms'],verify_candidate_ms=b['verify_ms'],
        verify_change_pct=100*(b['verify_ms']/a['verify_ms']-1) if a['verify_ms'] else None,
        tps_base=a['tps'],tps_candidate=b['tps'],tps_change_pct=100*(b['tps']/a['tps']-1),
        raw_tps_base=a['raw_decode_tps'],raw_tps_candidate=b['raw_decode_tps'],
        included_fraction_base=a['included_step_fraction'],included_fraction_candidate=b['included_step_fraction'],
        al_base=a['al'],al_candidate=b['al'],hit_base=a['hit'],hit_candidate=b['hit']))
aggregates=[]
for key in sorted({(p['contrast'],p['scope'],p['arm']) for p in pairs}):
    entries=[p for p in pairs if (p['contrast'],p['scope'],p['arm'])==key]
    item=dict(contrast=key[0],scope=key[1],arm=key[2],repeats=len(entries),
              outputs_equal=all(x['outputs_equal'] for x in entries),warm_outputs_equal=all(x['warm_outputs_equal'] for x in entries))
    for metric in ('verify_base_ms','verify_candidate_ms','verify_change_pct','tps_base','tps_candidate','tps_change_pct','raw_tps_base','raw_tps_candidate','included_fraction_base','included_fraction_candidate','al_base','al_candidate','hit_base','hit_candidate'):
        values=[x[metric] for x in entries if x[metric] is not None]
        item[metric]=mean(values)
        if metric.endswith('_pct') and values:item[metric+'_range']=[min(values),max(values)]
    aggregates.append(item)
out=dict(status='complete' if not incomplete else 'partial',completed_jobs=len(rows),incomplete=incomplete,
         rows=rows,pairs=pairs,aggregates=aggregates,audit=audit)
(ROOT/'ANALYSIS.json').write_text(json.dumps(out,indent=2))
fields=('name','contrast','arm','scope','model','batch','prompts','temperature','gpu_pair','engine_commit','control','verify_ms','step_ms','al','hit','tps','raw_decode_tps','included_step_fraction','steps','all_steps','actual_output_tokens')
with (ROOT/'RUN_INDEX.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for row in rows:
        values=dict(row,**row['summary']);writer.writerow({key:values.get(key) for key in fields})
print(json.dumps(dict(status=out['status'],completed=len(rows),incomplete=len(incomplete),aggregates=aggregates),indent=2))
