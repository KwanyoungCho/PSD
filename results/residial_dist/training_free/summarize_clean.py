"""Summarize isolated live runs using emitted-token wall throughput."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def main():
    folder=HERE/'live_clean';plan=json.loads((folder/'plan.json').read_text());rows=[]
    for path in sorted(folder.glob('*.complete.json')):
        meta=json.loads(path.read_text());p=path.with_name(path.name.replace('.complete.json','.json'))
        d=json.loads(p.read_text());m=d['metrics']
        assert d['policy']==plan['policies'][meta['arm']] and d['candidate_temperature']==.7
        assert d['prompts']==16 and d['output_tokens']==4096 and d['candidate_calls_by_K']
        assert meta['no_concurrent_analysis'] and meta['gpu2_before']=='0, 0'
        assert meta['runtime_policy_sha256']==hashlib.sha256((HERE/'runtime_policy.py').read_bytes()).hexdigest()
        rows.append({'arm':meta['arm'],'seed':meta['seed'],'output_tokens':d['output_tokens'],
           'generate_s':d['wall_s'],'output_tps':d['output_tokens']/d['wall_s'],
           'decode_tps':m['decode_total_tokens']/m['decode_total_time'],
           'decode_tokens':m['decode_total_tokens'],'decode_s':m['decode_total_time'],
           'cache_hit':float(np.mean(m['cache_hits'])),'steps':len(m['cache_hits']),
           'tokens_per_step':float(np.mean(m['accepted_suffix_lens_with_recovery'])),
           'tokens_on_hit':float(np.mean(m['accepted_suffix_lens_on_hit'])),
           'tokens_on_miss':float(np.mean(m['accepted_suffix_lens_on_miss'])),
           'generate_start_unix':p.stat().st_mtime-d['wall_s'],'generate_end_unix':p.stat().st_mtime})
    assert len(rows)==4
    last_analysis=max((HERE/n).stat().st_mtime for n in ['runtime_checks.json','runtime_snapshot_checks.json','numerical_case.json','budget_checks.json'])
    assert all(r['generate_start_unix']>last_analysis for r in rows)
    arms={}
    for arm in plan['policies']:
        a=[r for r in rows if r['arm']==arm]
        arms[arm]={'pooled_output_tps':sum(r['output_tokens'] for r in a)/sum(r['generate_s'] for r in a),
                   'pooled_decode_tps':sum(r['decode_tokens'] for r in a)/sum(r['decode_s'] for r in a),
                   'pooled_cache_hit':sum(r['cache_hit']*r['steps'] for r in a)/sum(r['steps'] for r in a)}
    paired=[]
    for seed in [903,904]:
        b=next(r for r in rows if r['seed']==seed and r['arm']=='matched_proxy')
        p=next(r for r in rows if r['seed']==seed and r['arm']=='optimized_proxy')
        paired.append({'seed':seed,'output_tps_gain_percent':100*(p['output_tps']/b['output_tps']-1),
                       'decode_tps_gain_percent':100*(p['decode_tps']/b['decode_tps']-1),
                       'cache_hit_gain_pp':100*(p['cache_hit']-b['cache_hit'])})
    result={'passed':True,'plan':plan,'rows':rows,'arms':arms,'paired':paired,
            'pooled_output_tps_gain_percent':100*(arms['optimized_proxy']['pooled_output_tps']/arms['matched_proxy']['pooled_output_tps']-1),
            'last_gpu2_analysis_completed_unix':last_analysis,
            'primary_timing_metric':'4096 actually emitted tokens / generate wall time; includes prefill, excludes model loading.',
            'limitation':'Two seeds, same 16 prompts, reverse arm order; no concurrent GPU2 analysis. Limited runtime screen, not a large-sample speedup claim or full Mirror-SD benchmark.'}
    (HERE/'live_clean_summary.json').write_text(json.dumps(result,indent=2))
    lines=['# Isolated runtime comparison','', result['primary_timing_metric'],'',
      '| Seed | Arm | Output TPS | Engine decode TPS | Cache hit (%) | Output tokens |',
      '|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["seed"]} | {r["arm"]} | {r["output_tps"]:.3f} | {r["decode_tps"]:.3f} | {100*r["cache_hit"]:.3f} | {r["output_tokens"]} |')
    lines+=['',f'Pooled output-throughput change: {result["pooled_output_tps_gain_percent"]:+.3f}%.', '',result['limitation']]
    (HERE/'LIVE_CLEAN.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
