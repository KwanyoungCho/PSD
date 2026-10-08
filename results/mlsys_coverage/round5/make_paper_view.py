"""Build all Round5 tables/figures offline; no inference or experiment launch.

Usage: python results/mlsys_coverage/round5/make_paper_view.py
       --rebuild-data    reread and validate every original run
       --only NAME      render selected previews; do not write completion manifest
       --skip-figures   regenerate text/indexes using already rendered figures
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from paper_view_data import HERE, OUT, build_data, dump, read
from paper_view_figures import config_label, profile_figure, save_figure, summary_figure, wall_figure

CSS = """
body{font-family:system-ui,sans-serif;color:#26364c;margin:24px auto;max-width:1500px;padding:0 22px;line-height:1.55;background:#f7f9fc}
h1{font-size:25px}h2{font-size:20px;margin-top:28px}a{color:#155c94}table{border-collapse:collapse;width:100%;background:white;font-size:13px}
td,th{border-bottom:1px solid #dce3ea;padding:7px 10px;text-align:left}th{background:#26364c;color:white;position:sticky;top:0}
tr.duet{background:#eff6fb}.box{background:white;border:1px solid #dce3ea;padding:16px;margin:16px 0;border-radius:8px}
.muted{color:#5e6a79;font-size:13px}.bad{color:#a32424;font-weight:bold}img{width:100%;height:auto;background:white}
input,select{padding:8px;margin:5px;border:1px solid #a5b3c3;border-radius:4px}input{min-width:320px}
code{font-size:12px;word-break:break-all}pre{white-space:pre-wrap;font-size:12px;max-height:500px;overflow:auto}
.scroll{overflow:auto}.toolbar{position:sticky;top:0;background:#f7f9fc;z-index:2;padding:8px}.tag{background:#e4ebf3;border-radius:4px;padding:2px 6px}
"""


def esc(v):
    return html.escape(str(v))


def document(title, body):
    return '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+f'<title>{esc(title)}</title><style>{CSS}</style><body>{body}</body></html>'


def fmt(v, digits=3):
    return '—' if v is None else f'{v:.{digits}f}'


def csv_out(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def curated(data):
    bypath={r['path']:r for r in data['rows']}
    results=[]

    def append(section, model, batch, method, agg, runs):
        row=bypath[runs[0]['path']]
        results.append(dict(section=section,model=model,batch=batch,method=method,
                            al=agg['boundary_excluded_al'],cache_hit=agg['cache_hit'],
                            p1_hit=agg['source_weights'][1] if method=='DUET' else None,
                            p2_hit=agg['source_weights'][2] if method=='DUET' else None,
                            tps=[r['boundary_excluded_step_tps'] for r in runs],
                            raw_tps=[r['decode_tps'] for r in runs],
                            names=[Path(r['path']).stem for r in runs],paths=[r['path'] for r in runs],
                            gpus=[r['gpus'] for r in runs],parameters=row['parameters'],
                            companion=row['companion_profiles'][0] if row['companion_profiles'] else None))
    for r in read(HERE/'POSTOPT_RESULTS.json'):
        for method,key,side in [('SSD','ssd','right'),('DUET','duet','left')]:
            append('Final TPS-priority',r['model'],r['batch'],method,r['full480'][side],r[key])
    original=read(HERE/'FINAL_RESULTS.json')
    for r in original['comparisons']:
        if r['comparison']!='al':continue
        for method,role,side in [('SSD',r['right_role'],'right'),('DUET',r['left_role'],'left')]:
            runs=sorted([x for x in original['runs'] if (x['model'],x['batch'],x['role'])==(r['model'],r['batch'],role)],key=lambda x:x['replicate'])
            assert len(runs)==2
            append('AL-priority (original frozen)',r['model'],r['batch'],method,r['full480'][side],runs)
    for r in read(HERE/'FOLLOWUP_RESULTS.json')['transferred']:
        for method,key in [('SSD','ssd'),('DUET','duet')]:
            append(r['stage'],r['model'],r['batch'],method,r[key],[r[key]])
    assert len(results)==28
    for r in results:
        if r['section']=='Final TPS-priority':assert r['companion'],r
    return results


def metrics_md(rows, links=True):
    lines=['| 모델 | Batch | 방법 | AL* | Cache hit | TPS* (반복별) | 실제 반환 TPS (반복별) |',
           '|---|---:|---|---:|---:|---|---|']
    for r in rows:
        label=f"[{r['method']}](runs/{r['names'][0]}.html)" if links else r['method']
        lines.append(f"| {r['model']} | {r['batch']} | {label} | {r['al']:.3f} | {r['cache_hit']*100:.1f}% | "+
                     ' / '.join(f'{v:.2f}' for v in r['tps'])+' | '+' / '.join(f'{v:.2f}' for v in r['raw_tps'])+' |')
    return '\n'.join(lines)


def metrics_html(rows):
    table='<table><thead><tr>'+''.join(f'<th>{x}</th>' for x in ['Model','Batch','Method','AL*','Cache hit','TPS* r0 / r1','Raw TPS r0 / r1','Breakdown'])+'</tr></thead><tbody>'
    for r in rows:
        link=f"<a href='runs/{r['names'][0]}.html'>{r['method']}</a>"
        graph=f"<a href='runs/{r['companion']}.html'>같은 설정의 trace</a>" if r['companion'] else '별도 phase trace 없음'
        vals=[r['model'],r['batch'],link,f"{r['al']:.3f}",f"{r['cache_hit']*100:.1f}%",' / '.join(f'{v:.2f}' for v in r['tps']),
              ' / '.join(f'{v:.2f}' for v in r['raw_tps']),graph]
        table+=f"<tr class={'duet' if r['method']=='DUET' else 'ssd'}>"+''.join(f'<td>{v}</td>' for v in vals)+'</tr>'
    return table+'</tbody></table>'


def tables(data, summaries):
    csv_out('SUMMARY.csv',summaries)
    passes=[]
    for p in data['passes']:
        passes.append({k:v for k,v in p.items() if k not in ['checks','source_sha256','prompt_sha256','output_caps_sha256']})
    csv_out('ALL_PASSES.csv',passes)
    metrics=[];parameters=[];components=[]
    for r in data['rows']:
        m=r['last'];p=r['parameters']
        metrics.append(dict(name=r['name'],model=p['model'],batch=m['batch'],method=p['method'],
                            questions=m['questions'],temperature=m['temperature'],input_cap=p['input_cap'],output_cap=p['output_cap'],
                            AL=m['boundary_excluded_al'],cache_hit=m['cache_hit'],TPS=m['boundary_excluded_step_tps'],
                            raw_TPS=m['decode_tps'],instrumented=m['instrumented'],timing_excluded=m['timing_excluded'],
                            passes=len(r['summaries']),plot=f"figs/{r['name']}.png",raw_path=r['path'],
                            companion_profiles=r['companion_profiles']))
        parameters.append(dict(name=r['name'],batch=m['batch'],temperature=m['temperature'],**p))
        if r['detail']:
            for g in r['detail']['groups']:
                for c in g['components']:
                    components.append(dict(name=r['name'],status=g['status'],**c,
                                           target_window_mean_ms=g['mean_target_ms']))
    csv_out('ALL_EXPERIMENTS.csv',metrics)
    csv_out('FULL480.csv',[r for r in metrics if r['questions']==480])
    csv_out('ALL_PARAMETERS.csv',parameters)
    csv_out('BREAKDOWN_COMPONENTS.csv',components)
    paramlines=['# 사용 파라미터','',
                '최종 처리량 우선점과 AL 우선점을 분리한다. Exit는 **0-based**다. N은 root마다 보존하는 continuation 노드 상한, M은 검증 노드 상한, C는 sibling capacity, U는 P1 위치당 root 수, W는 P2 총 root/forward budget이다. '
                '실제 매 step의 유효 노드 수는 이 상한보다 작을 수 있다. SSD K/F는 draft 길이/fanout이다.','',
                '공통 평가: full32-layer target, target1GPU+draft1GPU, T=0.7, 입력/출력 cap512/128, 자연 EOS. '
                'AL 우선점은 최초 동결 설정을 유지했으므로 최종 처리량 우선점과 trim/stream 옵션도 다를 수 있다.','']
    for section in ['Final TPS-priority','AL-priority (original frozen)']:
        paramlines += [f'## {section}','',
                       '| 모델 | B | 방법 | K1/K2 또는 SSD K/F | Exit | C | N1/N2 | M1/M2 | U | DFO/PFO | W | trim/stream |',
                       '|---|---:|---|---|---:|---:|---|---|---:|---|---:|---|']
        for r in summaries:
            if r['section']!=section:continue
            p=r['parameters'];duet=p['method']=='DUET'
            vals=[r['model'],str(r['batch']),r['method'],f"{p['K1']}/{p['K2']}" if duet else f"{p['SSD_K']}/{p['SSD_F']}",
                  str(p['exit']) if duet else '—',str(p['C']) if duet else '—',f"{p['N1']}/{p['N2']}" if duet else '—',
                  f"{p['M1']}/{p['M2']}" if duet else '—',str(p['P1_roots']) if duet else '—',
                  f"{p['draft_fanout']}/{p['proxy_fanout']}" if duet else '—',str(p['W']) if duet else '—',
                  f"{p['trim']}/{p['stream']}" if duet else '—']
            paramlines.append('| '+' | '.join(vals)+' |')
        paramlines += ['']
    paramlines += ['## 공통 DUET 수식·tree·구현','',
                   '- 내부 context root: 실제 온도의 `e(1-q)`, leaf는 `e`. P1은 draft 기반 점수.',
                   '- 위치: token-conditioned terminal mass 0.75 + overlap terminal mass 0.25. 두 ladder를 각각 계산한 후 혼합.',
                   '- 전체 허용 vocabulary 정규화. Verification은 실제 proposal q를 유지.',
                   '- `reach_gain_frontier`, 기존 Round4의 reach/gain calibration 고정. 이번 데이터로 재학습하지 않음.',
                   '- Miss chain4, P2 proxy/confidence floor=0.01/0.03, P1 start/confidence floor=0/0.',
                   '- Fused math=ON, bulk export=ON, parallel insert=OFF. C=1은 root별 continuation이 chain인 경우.',
                   '- beta=0.5는 기록되어 있지만 선택한 정책에서는 비활성. 새로운 최적화 결과로 해석하지 않음.',
                   '- SSD에도 공통 correctness/fast-verifier/CUDA-graph 경로 적용. SSD 길이/fanout은 독립 튜닝.',
                   '', '## 전체 실험 설정', '',
                   '[ALL_PARAMETERS.csv](ALL_PARAMETERS.csv)에 487개 실험의 설정을 보존했다. '
                   '각 실험 HTML 하단에는 원본 parsed args, engine kwargs, env와 effective flags, source hash를 함께 넣었다. '
                   '반복별 seed/B/T는 [ALL_PASSES.csv](ALL_PASSES.csv)에서 확인한다. '
                   'B2/B4·장문 실험은 최초 B8 설정 이전이며 개별 최적점이 아니다.']
    (OUT/'PARAMETERS.md').write_text('\n'.join(paramlines)+'\n')
    lines=['# Round5 — 논문 형식의 성능 표와 전체 breakdown','',
           '성능 지표와 파라미터를 분리하고, 기존 논문의 status별 평균 breakdown 및 target/draft 정렬 timeline 표시 방법을 적용했다. '
           '이 문서는 기존 완료 실험을 재분석한 것이다. 새 GPU 실험은 실행하지 않았다.','',
           '- [검색 가능한 전체 실험·그래프 index](index.html)',
           '- [파라미터 표](PARAMETERS.md) · [전체 설정 CSV](ALL_PARAMETERS.csv)',
           '- [전체 487개 실험 CSV](ALL_EXPERIMENTS.csv) · [691개 pass CSV](ALL_PASSES.csv) · [full480 실험 70개 CSV](FULL480.csv)',
           '- [최종 비교 breakdown 모음 PDF](FINAL_BREAKDOWNS.pdf) · [모든 실험 그림 PDF](ALL_BREAKDOWNS.pdf)',
           '- [수치·페이지·링크 검증 기록](AUDIT.json)',
           '', '## 지표 정의와 비교 범위','',
           '- **Batch:** 실행 설정의 최대 동시 요청 수. 종료 구간의 실제 batch는 더 작을 수 있다.',
           '- **AL\\*:** 수락 draft + recovery/bonus 토큰 수. 출력cap/clipping이 발생한 마지막 verification event 제외.',
           '- **Cache hit:** 전체 verification event 중 hit 비율. 기존 보고서 counter와 같은 분모이며 AL* 제외 집합과 다르다. B>1 all-hit batch 비율과도 다르다.',
           '- **TPS\\*:** boundary 요청이 들어간 batch step의 시간과 토큰을 모두 제외. 실제 반환 토큰 TPS도 병기한다. TPS는 token/s이며 batch 전체 처리량이다.',
           '- AL·hit은 반복별 event 수를 합쳐 계산. TPS는 GPU쌍/반복을 섞어 평균내지 않고 r0/r1 각각 표시. 두 값은 신뢰구간이 아니다.',
           '- 최종 표는 비계측 full480 measured pass. Profile48의 시간/AL/TPS는 별도 진단 자료다.',
           '', '## 최종 처리량 우선 설정','',metrics_md([r for r in summaries if r['section']=='Final TPS-priority']),
           '', '![최종 비교 표](summary.png)','',
           '각 DUET/SSD 쌍은 같은 GPU에서 비교했다. 두 반복의 GPU쌍은 달라질 수 있다. 48개로 설정을 선택했고 이번 선택에서 제외한432개 분석은 기존 REPORT에 있다. '
           '기존 보유480 first-turn 입력이며 원 SpecBench 전체나 70B/B6000 논문 환경 재현이 아니다.','',
           '## AL 우선 설정 — 최초 동결점','',metrics_md([r for r in summaries if r['section']=='AL-priority (original frozen)']),
           '', '## B2/B4 및 긴 입력·출력 — 설정 이전 실험','',metrics_md([r for r in summaries if r['section'] in ['batch_transfer','long_transfer']]),
           '', 'batch_transfer는 입력512/출력128, long_transfer는 입력1024/출력256이다. 각 조건 1회 measured pass이며 최종 B8 재선택 이전의 설정을 이전했다.','',
           '## 최종 설정의 세부 breakdown','',
           '성능 측정(full480)과 세부 trace(tuning48)는 별도 실행이다. 아래 연결은 모델·B·T·알고리즘/노드/구현 옵션을 맞춘 진단이다. '
           '입력 집합·출력cap·seed가 같다는 뜻이 아니며, 해당 값을 각 그림에 표시했다.','',
           '| 목적 | 모델 | B | 방법 | 원본 trace와 그림 |','|---|---|---:|---|---|']
    for r in summaries:
        if r['section'] not in ['Final TPS-priority','AL-priority (original frozen)']:continue
        name=r['companion'];assert name
        lines.append(f"| {r['section']} | {r['model']} | {r['batch']} | {r['method']} | [{name}](runs/{name}.html) · [PNG](figs/{name}.png) · [PDF](figs/{name}.pdf) |")
    lines += ['', '## 그림을 읽는 방법','',
              '- 세부 trace337개: 위쪽은 status별 target/draft stage 평균(ms), 아래 왼쪽은 논문 색상의 평균 정렬 schematic(%), 오른쪽은 실제 기록된 대표 step(ms).',
              '- 대표 step은 각 상태에서 target request→ready 시간이 중앙값에 가장 가까운 관측값이다. Target/draft는 같은 step ID로 연결하며, 현재 wait와 겹치는 직전 P2/cache-build 구간도 오른쪽에 표시한다.',
              '- 정규화 schematic은 **동일 status 모집단**의 평균 시작/끝 offset을 평균 target window로 나눈 값이다. 서로 다른 component median을 이어붙이지 않는다. 평균 그림은 실제 단일 step이 아니다. 모든 sample에 있는 구간만 평균 timeline에 표시하며, 일부 sample에만 있는 구간도 위쪽 막대에서는 없는 sample을 0으로 포함해 집계한다.',
              '- 기존 그림의 공통 hit/miss target 비용 가정과 고정 퍼센트는 재사용하지 않았다. 이번에는 verification 폭이 다를 수 있으므로 실제 상태별 비용과 shape를 보존한다.',
              '- Wait/sync에는 통신 및 노출된 draft 대기가 포함된다. 측정 없이 순수 sync와 miss stall을 분리하지 않았다. 작은 빈 구간은 unlabelled gap이며 CPU 연산이라고 단정하지 않는다.',
              '- P1/P2 total과 replay child를 중복 합산하지 않는다. Proxy side stream은 별도 행으로 표시하고 target 직렬 막대에 더하지 않는다. `batch_proxy_receive`의 긴 대기 구간을 proxy 연산 시간으로 해석하지 않는다.',
              '- B1은 P1 hit/P2 hit/miss. B>1은 P1-only hit, P2-only hit, mixed hits, mixed hit/miss, all miss를 구분한다. B>1의 hit 표시는 batch 내 전 요청이 그 상태인 경우다.',
              '- 진단 그림은 초기19개 step, 알려진 capture와 직후 step, 불완전 batch, 출력경계 batch, 불완전 trace를 제외한다. 제외 수와 각 상태 n을 저장한다. 초기 trace에 없던 capture label은 사후 복원하지 못한다.',
              '- 비계측150개: phase별 GPU 시간을 복원할 수 없어 실제 wall time을 cache 상태별로 분해하고 source별 AL·전체 pass TPS를 그렸다. 느린 step은 임의로 제거하지 않았다.',
              '- 모든 487개 실험에 그림이 있고 691개 pass는 전부 표에 남겼다. 실험 그림은 마지막 pass 기준이며 비계측 그림의 오른쪽 아래에는 모든 pass를 표시한다. 실패4개는 별도 목록에 보존한다.',
              '- 알려진 오염 timing 1개는 EXCLUDED로 표시한다. 이는 유효한 속도 비교에 쓰지 않는다.','',
              '## 원본 표시 방법과 재현','',
              '- 논문 schematic: `ssd/tools/duet_timeline/plot_paper_fig4_schematic_pct.py` (색상 직접 import).',
              '- Status 평균: `ssd/experiments/paper_baselines/new_exp/mesa_k1_7_k2_5_dfo2_pfo1_exit56_profile1_seed42_20260521/plot_breakdown_by_status.py`.',
              '- 실제 정렬 원칙: `ssd/bench/plot_duet_aligned_timeline.py`.',
              '', '그림 도구 의존성: numpy, matplotlib, pypdf (이번 PDF 병합은 pypdf 6.19.0 사용).',
              '', '```bash','python results/mlsys_coverage/round5/archive_results.py --restore',
              'python results/mlsys_coverage/round5/make_paper_view.py --rebuild-data','```','',
              '`python results/mlsys_coverage/round5/audit_paper_view.py`로 coverage·TPS 재집계·구간 합·PDF 페이지·링크를 검사한다.',
              '', '모든 분석·그림 생성은 CPU만 사용한다. 표의 원본 path, source SHA, step IDs, component 평균은 DATA.json과 BREAKDOWN_COMPONENTS.csv에 있다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run_pages(data):
    (OUT/'runs').mkdir(exist_ok=True)
    for r in data['rows']:
        p=r['parameters'];m=r['last'];name=r['name']
        body=f"<a href='../index.html'>← 전체 실험</a><h1>{esc(name)}</h1><p>{esc(config_label(p))}</p>"
        if m['timing_excluded']:body+='<p class="bad">TIMING EXCLUDED — 동시 실행 오염. 속도 비교에서 제외.</p>'
        body+='<div class="box"><b>모든 저장 pass</b><table><tr>'+''.join(f'<th>{x}</th>' for x in
                    ['Pass','용도','N','B','T','Seed','AL*','Cache hit','TPS*','Raw TPS'])+'</tr>'
        for s in r['summaries']:
            body+='<tr>'+''.join(f'<td>{x}</td>' for x in [s['pass_index'],s['pass_role'],s['questions'],s['batch'],s['temperature'],s['target_seed'],
                       fmt(s['boundary_excluded_al']),f"{s['cache_hit']*100:.2f}%",fmt(s['boundary_excluded_step_tps'],2),fmt(s['decode_tps'],2)])+'</tr>'
        body+='</table></div>'
        body+=f"<p>Input/output cap: {p['input_cap']}/{p['output_cap']} · GPUs: {p['gpus']} · draft seed: {m['draft_seed']} · source: <code>{p['git_commit']}</code></p>"
        body+=f"<p><a href='../figs/{name}.png'>PNG</a> · <a href='../figs/{name}.pdf'>벡터 PDF</a> · <a href='../../{r['path']}'>원본 결과 JSON</a></p>"
        if not r['detail']:
            body+='<p class="muted">이 실행에는 phase trace가 없습니다. 아래 그림은 실제 step wall time·source AL 분해입니다.</p>'
            if r['companion_profiles']:
                body+='<p>같은 알고리즘·B·T·구현 옵션의 별도 진단: '+', '.join(f"<a href='{x}.html'>{x}</a>" for x in r['companion_profiles'])+'</p><p class="muted">진단의 입력집합·cap·seed·실행 source revision은 이 성능 실행과 다를 수 있습니다.</p>'
            else:body+='<p class="muted">같은 설정의 phase 진단도 기록되어 있지 않습니다.</p>'
        else:
            detail=r['detail'];body+=f"<p>세부 trace: {detail['eligible_steps']} eligible steps. <a href='../../{detail['trace_paths'][0]}'>Target 원본</a> · <a href='../../{detail['trace_paths'][1]}'>Draft 원본</a></p>"
            body+='<p class="muted">제외 수: '+esc(json.dumps(detail['excluded']))+'</p>'
            body+='<p class="muted">왼쪽 아래는 상태별 평균 정렬 schematic, 오른쪽 아래는 실제 대표 step입니다. %의 분모는 각 상태의 평균 target request→ready 시간이며 phase overlap은 별도 lane에 표시합니다.</p>'
        body+=f"<img src='../figs/{name}.png' alt='Breakdown for {esc(name)}'>"
        body+='<details class="box"><summary>전체 파라미터·환경·source 정보</summary><pre>'+esc(json.dumps(r['full_parameters'],ensure_ascii=False,indent=2))+'</pre></details>'
        if r['detail']:body+='<details class="box"><summary>그림 수치·step IDs·대표 step 원본</summary><pre>'+esc(json.dumps(r['detail'],ensure_ascii=False,indent=2))+'</pre></details>'
        (OUT/'runs'/f'{name}.html').write_text(document(name,body))


def index(data, summaries):
    body='<h1>Round5 — 실험 결과와 breakdown 전체 보기</h1>'
    body+='<p>487개 완료 실험 · 691개 pass · full480 실험70개 · phase trace337개 · wall/source breakdown150개</p>'
    body+='<p><a href="REPORT.md">보고서</a> · <a href="PARAMETERS.md">파라미터</a> · <a href="ALL_PARAMETERS.csv">전체 설정 CSV</a> · <a href="ALL_EXPERIMENTS.csv">전체 결과 CSV</a> · <a href="ALL_PASSES.csv">전체 pass CSV</a> · <a href="FINAL_BREAKDOWNS.pdf">최종 설정 그림 PDF</a> · <a href="ALL_BREAKDOWNS.pdf">전체 그림 PDF</a></p>'
    body+='<div class="box">AL*은 cap/clipping 검증 event 제외, TPS*은 해당 batch step의 토큰과 시간을 함께 제외합니다. Cache hit은 전체 검증 event의 비율입니다. AL·hit은 반복 event를 합산하고 TPS는 r0/r1 각각 표시합니다. 단위는 batch 전체 token/s입니다.<br>성능표는 비계측 full480, phase 그림은 별도 계측 진단입니다. 새 GPU 실험은 실행하지 않았습니다.</div>'
    for section in dict.fromkeys(r['section'] for r in summaries):
        body+=f'<h2>{section}</h2>'+metrics_html([r for r in summaries if r['section']==section])
    body+='<h2>전체 실험</h2><div class="toolbar"><input id="query" placeholder="실험명, 파라미터 검색"><select id="model"><option value="">모든 모델</option><option>llama2</option><option>llama3</option></select><select id="batch"><option value="">모든 batch</option><option>1</option><option>2</option><option>4</option><option>8</option></select><select id="kind"><option value="">전체</option><option value="full">full480</option><option value="profile">phase trace 있음</option><option value="wall">wall/source만 있음</option></select><span id="count"></span></div>'
    body+='<div class="scroll"><table id="runs"><thead><tr>'+''.join(f'<th>{x}</th>' for x in ['Experiment (마지막 pass)','Model','B','N','T','AL*','Cache hit','TPS*','분류','Parameters'])+'</tr></thead><tbody>'
    for r in data['rows']:
        m=r['last'];p=r['parameters'];kind='profile' if r['detail'] else 'wall'
        tags=kind+(' full' if m['questions']==480 else '')
        marker='EXCLUDED' if m['timing_excluded'] else '계측' if r['detail'] else m['pass_role']
        vals=[f"<a href='runs/{r['name']}.html'>{r['name']}</a>",p['model'],m['batch'],m['questions'],m['temperature'],
              fmt(m['boundary_excluded_al']),f"{m['cache_hit']*100:.1f}%",fmt(m['boundary_excluded_step_tps'],2),marker,esc(config_label(p))]
        body+=f"<tr data-model='{p['model']}' data-batch='{m['batch']}' data-kind='{tags}'>"+''.join(f'<td>{v}</td>' for v in vals)+'</tr>'
    body+='</tbody></table></div><h2>실패한 시도 — 성능 결과 없음</h2><ul>'
    body+=''.join(f"<li>{esc(r['directory'])}/{esc(r['name'])}: {esc(r['status'])}</li>" for r in data['failures'])+'</ul>'
    body+='''<script>
    const controls=['query','model','batch','kind'].map(x=>document.getElementById(x));
    function filter(){const [q,m,b,k]=controls.map(x=>x.value.toLowerCase());let n=0;
      document.querySelectorAll('#runs tbody tr').forEach(r=>{const show=(!q||r.textContent.toLowerCase().includes(q))&&(!m||r.dataset.model===m)&&(!b||r.dataset.batch===b)&&(!k||r.dataset.kind.split(' ').includes(k));r.hidden=!show;if(show)n++;});
      document.getElementById('count').textContent=n+' / 487 experiments';}
    controls.forEach(x=>x.addEventListener('input',filter));filter();</script>'''
    (OUT/'index.html').write_text(document('Round5 paper-style results',body))


def combine_pdfs(data,summaries):
    from pypdf import PdfWriter
    paths=[OUT/'figs'/f"{r['name']}.pdf" for r in data['rows']]
    if not all(p.exists() for p in paths):raise ValueError('Missing a per-experiment figure')
    writer=PdfWriter()
    for r,p in zip(data['rows'],paths):writer.append(str(p),outline_item=r['name'])
    with (OUT/'ALL_BREAKDOWNS.pdf').open('wb') as f:writer.write(f)
    writer.close()
    writer=PdfWriter();writer.append(str(OUT/'summary.pdf'),outline_item='Final performance table')
    for r in summaries:
        if r['section'] not in ['Final TPS-priority','AL-priority (original frozen)']:continue
        label=f"{r['section']} | {r['model']} B{r['batch']} {r['method']}"
        writer.append(str(OUT/'figs'/f"{r['companion']}.pdf"),outline_item=label)
    with (OUT/'FINAL_BREAKDOWNS.pdf').open('wb') as f:writer.write(f)
    writer.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rebuild-data',action='store_true')
    ap.add_argument('--skip-figures',action='store_true');ap.add_argument('--only',action='append')
    ap.add_argument('--force',action='store_true');args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    data=build_data() if args.rebuild_data or not (OUT/'DATA.json').exists() else read(OUT/'DATA.json')
    summaries=curated(data)
    tables(data,summaries);dump(OUT/'CURATED.json',summaries)
    if not args.skip_figures:
        save_figure(summary_figure(summaries),OUT/'summary')
        for i,r in enumerate(data['rows']):
            if args.only and r['name'] not in args.only:continue
            stem=OUT/'figs'/r['name']
            if not args.force and stem.with_suffix('.pdf').exists() and stem.with_suffix('.png').exists():continue
            fig=profile_figure(r) if r['detail'] else wall_figure(r)
            save_figure(fig,stem)
            if (i+1)%15==0 or args.only:print(f"Figure {i+1}/487: {r['name']}",flush=True)
    if args.only:return
    run_pages(data);index(data,summaries);combine_pdfs(data,summaries)
    manifest=dict(complete=True,jobs=len(data['rows']),passes=len(data['passes']),
                  detailed_traces=sum(bool(r['detail']) for r in data['rows']),
                  wall_only=sum(not bool(r['detail']) for r in data['rows']),
                  full480_jobs=sum(r['last']['questions']==480 for r in data['rows']),
                  excluded_timing_jobs=[r['name'] for r in data['rows'] if r['last']['timing_excluded']],
                  failures=len(data['failures']),pngs=len(list((OUT/'figs').glob('*.png'))),
                  pdfs=len(list((OUT/'figs').glob('*.pdf'))),gpu_jobs_launched=0,
                  source_sha256=data['source_sha256'])
    dump(OUT/'COMPLETE.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
