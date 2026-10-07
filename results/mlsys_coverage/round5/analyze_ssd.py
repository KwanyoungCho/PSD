"""Keep the independent SSD search and its timing reasons reviewable."""
from make_plans import HERE,save
from result_metrics import read


def main():
    screen=read(HERE/'SCREEN_RESULTS.json');out=[]
    lines=['# Independently tuned SSD baseline','',
        'Each model/batch has its own K/fanout search and warm uninstrumented selection on tuning48. '
        'Profile timings below are diagnostics, not final TPS. Cache-build slack is target postprocess completion '
        'minus future-cache population completion in the same measured step; negative values mean that build missed this boundary. '
        'All-hit rate is measured over complete batches, not estimated by raising average hit rate to B. '
        'AL-priority and TPS-priority settings are both retained.','']
    for model in ('llama2','llama3'):
        for b in (1,8):
            frozen=read(HERE/f'{model}_b{b}_FROZEN.json')['presets']
            points={role:frozen[frozen[role].get('alias',role)] for role in ('ssd_fast','ssd_al')}
            rows=[r for r in screen if r['name'].startswith(f'{model}_b{b}_') and
                  r['args']['mode']=='ssd' and r.get('profile_paths') and
                  r['args']['max_new_tokens']==96 and r['args']['limit']==0 and
                  not r['timing_excluded']]
            rows.sort(key=lambda r:(r['args']['k1'],r['args']['draft_fan_out'],r['name']))
            selected={role:[r['name'] for r in rows if all(r['args'][k]==point['args'][k]
                      for k in ('k1','draft_fan_out'))] for role,point in points.items()}
            if any(not names for names in selected.values()):raise ValueError('Missing chosen SSD diagnostic')
            out.append(dict(model=model,batch=b,chosen=points,chosen_profiles=selected,profiles=rows))
            lines += [f'## {model}, B={b}','',
                'Chosen '+', '.join(f"{role}: K={p['args']['k1']}, F={p['args']['draft_fan_out']}" for role,p in points.items()),'',
                '| Diagnostic | K/F | AL* | hit | all-hit | target verify p50 ms | draft build/decode p50 ms | build slack p50 ms | target wait p50 ms |',
                '|---|---|---:|---:|---:|---:|---|---:|---:|']
            for r in rows:
                p=r['profile'];fmt=lambda key: '-' if not p.get(key) else f"{p[key]['p50']:.2f}"
                s=r['summary'];a=r['args']
                lines.append(f"| {r['name']} | {a['k1']}/{a['draft_fan_out']} | {s['boundary_excluded_al']:.3f} | "
                    f"{s['cache_hit']:.3f} | {r['all_hit_rate']:.3f} | {fmt('verify_replay')} | "
                    f"{fmt('ssd_build_tree')}/{fmt('ssd_decode_tree')} | {fmt('ssd_build_slack_ms')} | {fmt('target_spec_wait')} |")
            lines.append('')
    save('SSD_BREAKDOWN.json',out)
    (HERE/'SSD_BREAKDOWN.md').write_text('\n'.join(lines).rstrip()+'\n')
    print('SSD search cells',len(out),'profiled candidates',sum(len(r['profiles']) for r in out))


if __name__=='__main__':main()
