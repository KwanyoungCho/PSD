"""Summarize no-probe measurements without pretending token rows are IID runs."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def main():
    folder=HERE/'live';plan=json.loads((folder/'plan.json').read_text());records=[]
    if (folder/'original_residual_plan.json').exists():
        plan['secondary_original_residual']=json.loads((folder/'original_residual_plan.json').read_text())
        plan['policies']['original_residual']='legacy'
    for path in sorted(folder.glob('*.complete.json')):
        meta=json.loads(path.read_text());d=json.loads(path.with_name(path.name.replace('.complete.json','.json')).read_text());m=d['metrics']
        if d['policy']!=plan['policies'][meta['arm']] or not d['candidate_calls_by_K']:
            raise RuntimeError('Invalid policy or inactive hook')
        assert d['prompts']==16 and d['temperature']==.7
        assert d['candidate_temperature']==(1. if d['policy']=='legacy' else .7)
        records.append({'arm':meta['arm'],'seed':meta['seed'],'decode_tokens':m['decode_total_tokens'],
            'decode_s':m['decode_total_time'],'decode_tps':m['decode_total_tokens']/m['decode_total_time'],
            'wall_tps':d['output_tokens']/d['wall_s'],
            'steps':len(m['cache_hits']),'cache_hit':float(np.mean(m['cache_hits'])),
            'tokens_per_step':float(np.mean(m['accepted_suffix_lens_with_recovery'])),
            'target_step_ms':float(np.mean(m['target_step_times']))*1000,
            'tokens_per_hit_step':float(np.mean(m['accepted_suffix_lens_on_hit'])),
            'tokens_per_miss_step':float(np.mean(m['accepted_suffix_lens_on_miss'])),
            'candidate_calls_by_K':d['candidate_calls_by_K']})
    assert len(records)==2*len(plan['policies'])
    summaries={}
    for arm in plan['policies']:
        a=[d for d in records if d['arm']==arm];assert len(a)==2
        summaries[arm]={'pooled_decode_tps':sum(d['decode_tokens'] for d in a)/sum(d['decode_s'] for d in a),
          'mean_run_decode_tps':float(np.mean([d['decode_tps'] for d in a])),
          'run_decode_tps':[d['decode_tps'] for d in a],
          'pooled_cache_hit':sum(d['cache_hit']*d['steps'] for d in a)/sum(d['steps'] for d in a),
          'mean_tokens_per_step':float(np.mean([d['tokens_per_step'] for d in a]))}
    for arm,value in summaries.items():
        value['paired_tps_gain_percent_vs_legacy']=[100*(d['decode_tps']/next(r['decode_tps'] for r in records if r['arm']=='legacy' and r['seed']==d['seed'])-1) for d in records if d['arm']==arm]
    out={'plan':plan,'records':records,'arms':summaries,
         'limitation':'Exploratory pilot: some generation intervals overlapped GPU2 analysis. The original residual control was added afterward. Two seeds on the same 16 prompts; first four arms used reverse order. Use LIVE_CLEAN.md for the isolated allocation comparison. Do not bootstrap individual steps as independent throughput replications or call this a full Mirror-SD benchmark.'}
    (HERE/'live_summary.json').write_text(json.dumps(out,indent=2))
    lines=['# No-probe runtime screen','', 'T=0.7, same 16 prompts × two seeds; each arm uses the same prompt file and output limit 256.',
           '', '| Arm | Decode TPS, seed 801 | Decode TPS, seed 802 | Pooled decode TPS | Pooled cache hit |',
           '|---|---:|---:|---:|---:|']
    for arm,v in summaries.items():
        runs={r['seed']:r['decode_tps'] for r in records if r['arm']==arm}
        lines.append(f'| {arm} | {runs[801]:.3f} | {runs[802]:.3f} | {v["pooled_decode_tps"]:.3f} | {100*v["pooled_cache_hit"]:.3f}% |')
    lines+=['',out['limitation']]
    (HERE/'LIVE.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
