#!/usr/bin/env python3
"""Explain PS preparation gaps and the next-command wait using frozen traces.

No runtime changes. Command waits inherit the previous step ID, so associate
them by CPU chronology between same-request anchors, never by step ID alone.
Derived gaps are elapsed intervals, not measurements of GPU idle time.
"""
from __future__ import annotations
import argparse
from bisect import bisect_left, bisect_right
import html
import json
from pathlib import Path

from plot_model_breakdowns import (load, sha, read_arm, one, require, percentile,
                                    write_csv, subtitle, STATES, plt)
from plot_model_native_events import native_rows, DETAIL_STYLES, native
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch


def command_index(raw):
    rows=sorted((e for e in raw if e['label']=='draft_recv_cmd'),
                key=lambda e:e['cpu_dispatch_start_ns'])
    return [e['cpu_dispatch_start_ns'] for e in rows],rows


def diagnose(arm,row,index):
    key=(row['epoch'],row['step_id']);groups=arm['grouped']
    draft=groups['draft'][key]
    following=groups['draft'].get((key[0],key[1]+1))
    require(following is not None,'No consecutive next step in this request')
    merge=one(draft,'merge_cache');nxt=one(following,'draft_recv_request')
    start=merge['cpu_dispatch_end_ns'];end=nxt['cpu_dispatch_start_ns']
    starts,commands=index
    found=[e for e in commands[bisect_left(starts,start):bisect_right(starts,end)]
           if e['cpu_dispatch_end_ns']<=end]
    require(len(found)==1,f'Expected one command wait at {key}, got {len(found)}')
    cmd=found[0]
    build=one(draft,'phase2_build')
    preps=sorted((e for e in draft if e['label']=='p2_prepare'
                  and e['cpu_dispatch_start_ns']>=build['cpu_dispatch_end_ns']),
                 key=lambda e:e['cpu_dispatch_start_ns'])
    require(bool(preps),f'Missing PS prepare after phase2_build: {key}')
    first=preps[0]
    bounds=[merge['wall_end_ns'],cmd['wall_start_ns'],cmd['wall_end_ns'],nxt['wall_start_ns']]
    require(all(b>=a for a,b in zip(bounds,bounds[1:])),f'Nonmonotone draft tail: {key}')
    require(first['wall_start_ns']>=build['wall_end_ns'],f'Overlapping PS setup: {key}')
    ms=lambda a,b:(b-a)/1e6
    metrics=dict(model=arm['key'],uid=row['uid'],epoch=row['epoch'],step_id=row['step_id'],
        state=row['state'],next_step_id=key[1]+1,command_raw_step_id=cmd['step_id'],
        phase2_build_ms=ms(build['wall_start_ns'],build['wall_end_ns']),
        ps_setup_gap_ms=ms(build['wall_end_ns'],first['wall_start_ns']),
        ps_setup_cpu_gap_ms=ms(build['cpu_dispatch_end_ns'],first['cpu_dispatch_start_ns']),
        ps_total_ms=row['D2'],ps_graph_ms=row['ps_graph'],ps_outside_graph_ms=row['ps_other'],
        ps_outside_graph_fraction=row['ps_other']/row['D2'],
        post_ps_unprofiled_ms=ms(bounds[0],bounds[1]),
        command_wait_ms=ms(bounds[1],bounds[2]),
        command_to_request_gap_ms=ms(bounds[2],bounds[3]),
        ps_end_to_next_request_ms=ms(bounds[0],bounds[3]))
    require(abs(sum(metrics[k] for k in ('post_ps_unprofiled_ms','command_wait_ms',
                'command_to_request_gap_ms'))-metrics['ps_end_to_next_request_ms'])<1e-8,
            'Tail partition does not sum')
    return dict(metrics=metrics,build=build,first_prepare=first,merge=merge,
                command=cmd,next_request=nxt,prepares=preps)


def derived(label,a,b,sid):
    require(b>=a,'Negative derived gap')
    return dict(label=label,wall_start_ns=a,wall_end_ns=b,step_id=sid,
                parent_label=None,derived=True)


GAP_STYLES={'PS dispatch gap (unprofiled)':('#eeeeee','////'),
            'Post-PS gap (unprofiled)':('#f3e5ce','////'),
            'Command-to-request gap (unprofiled)':('#eeeeee','////'),
            'PS prepare A: roots / visibility':('#fdae6b','//'),
            'PS prepare B: graph input buffers':('#fd8d3c','//'),
            'Next request receive (not idle)':('#9ecae1','..')}


def render(arm,row,diag,pdf,png):
    t,d=native_rows(arm,row);m=diag['metrics'];sid=row['step_id']
    b=diag['build'];p=diag['first_prepare'];c=diag['command'];n=diag['next_request'];merge=diag['merge']
    gaps=[derived('PS dispatch gap (unprofiled)',b['wall_end_ns'],p['wall_start_ns'],sid),
          derived('Post-PS gap (unprofiled)',merge['wall_end_ns'],c['wall_start_ns'],sid),
          derived('Command-to-request gap (unprofiled)',c['wall_end_ns'],n['wall_start_ns'],sid)]
    # Retain raw records unmodified; only draw three explicitly derived spans.
    # Display aliases distinguish the two same-named prepares and the next
    # request receive. Raw events in the audit remain untouched.
    aliases={id(e):f'PS prepare {chr(65+i)}: '+('roots / visibility' if i==0 else 'graph input buffers')
             for i,e in enumerate(diag['prepares'])}
    extended=[dict(e,label=aliases[id(e)]) if id(e) in aliases else e for e in d]
    extended += [c,dict(n,label='Next request receive (not idle)')]+gaps
    origin=min(e['wall_start_ns'] for e in t+d)
    ms=lambda x:(x-origin)/1e6
    fig=plt.figure(figsize=(19,8.3));ax=fig.add_axes([.07,.40,.90,.42])
    for events,y in ((t,1),(extended,0)):
        native._draw_row(ax,events,y,y,origin,.55,.40)
    forward_counts={'p1_graph_replay':arm['result']['config']['k1'],
                    'p2_graph_replay':arm['result']['config']['k2']}
    dividers=native.draw_draft_forward_dividers(ax,d,forward_counts,origin)
    native._draw_causality_arrow(ax,t,d,1,0,origin)
    ax.set(yticks=[0,1],yticklabels=['draft','target_rank0'],
           xlim=(-.5,max(ms(e['wall_end_ns']) for e in t+extended)+.5),ylim=(-.95,1.6),
           xlabel='ms from current step origin (no clock shift)')
    ax.grid(axis='x',alpha=.25,linestyle=':')
    # Number markers have a descriptive key below, avoiding narrow-bar text.
    markers=[(1,(b['wall_start_ns']+b['wall_end_ns'])/2,.65),
             (2,(b['wall_end_ns']+p['wall_start_ns'])/2,-.65),
             (3,(merge['wall_end_ns']+c['wall_start_ns'])/2,.65),
             (4,(c['wall_start_ns']+c['wall_end_ns'])/2,-.65)]
    for number,x,y in markers:
        ax.annotate(str(number),xy=(ms(x),.28 if y>0 else -.28),xytext=(ms(x),y),
                    ha='center',va='center',fontsize=11,fontweight='bold',
                    bbox=dict(boxstyle='circle,pad=.25',fc='white',ec='#444444'),
                    arrowprops=dict(arrowstyle='-',color='#333333'),zorder=10)
    ax.axvline(ms(n['wall_start_ns']),ls='--',color='#555555',lw=.8)
    ax.text(ms(n['wall_start_ns']),1.45,f"next request / step {sid+1}",ha='right',fontsize=9)
    next_target=arm['grouped']['target'][(row['epoch'],sid+1)]
    target_end=one(t,'target_postprocess')['wall_end_ns']
    next_send=one(next_target,'target_send_request')['wall_start_ns']
    next_wait=one(next_target,'target_recv_response_wait')
    ax.axvline(ms(target_end),ls=':',color='#b22222',lw=.9)
    fig.suptitle(f"{arm['name']}\n{STATES[row['state']]} / p50 / UID {row['uid']} / request {row['epoch']} / step {sid}",
                 y=.98,fontsize=14)
    fig.text(.5,.895,subtitle(arm),ha='center',fontsize=10)
    fig.text(.5,.865,f"Measured v1.0 {arm['result']['commit'][:8]} | dotted DS/PS slots show forward COUNTS only; equal widths are NOT measured forward times",ha='center',fontsize=9)
    fig.text(.07,.33,f"1  PS roots / layout: {m['phase2_build_ms']:.3f} ms          "
             f"2  Unprofiled PS dispatch: {m['ps_setup_gap_ms']:.3f} ms (CPU gap {m['ps_setup_cpu_gap_ms']:.3f} ms)",fontsize=11)
    fig.text(.07,.295,f"3  Post-PS unprofiled: {m['post_ps_unprofiled_ms']:.3f} ms          "
             f"4  Recorded command receive: {m['command_wait_ms']:.3f} ms          "
             f"PS end -> next receive start: {m['ps_end_to_next_request_ms']:.3f} ms",fontsize=11)
    fig.text(.07,.26,f"PS total {m['ps_total_ms']:.3f} ms = graph {m['ps_graph_ms']:.3f} + outside graph {m['ps_outside_graph_ms']:.3f} ms.  "
             f"Command -> request gap: {m['command_to_request_gap_ms']:.3f} ms.",fontsize=10)
    fig.text(.07,.225,f"Target step ends {ms(target_end):.3f} ms (red dotted line); next send {ms(next_send):.3f} ms. "
             f"Next response wait: {ms(next_wait['wall_start_ns']):.3f}-{ms(next_wait['wall_end_ns']):.3f} ms (extends beyond this view).",fontsize=9)
    labels=sorted({e['label'] for e in t+extended})
    handles=[Patch(facecolor=native._style(label)[0],hatch=native._style(label)[1],
                   edgecolor='black',linewidth=.3,label=label) for label in labels]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.035),ncol=6,fontsize=7,
               frameon=True,columnspacing=1.3)
    fig.text(.07,.018,'Hatched gaps are inferred elapsed intervals, NOT proven GPU idle. Command receive includes communication/wait. Nested spans are not additive. Raw bars retain legacy 0.02 ms display floor.',fontsize=8)
    pdf.savefig(fig);fig.savefig(png,dpi=155,bbox_inches='tight');plt.close(fig)
    return dict(origin_ns=origin,metrics=m,derived_gaps=gaps,command=diag['command'],
                forward_dividers=dividers,prepare_events=diag['prepares'],
                target_end_ns=target_end,next_target_send_ns=next_send,next_target_response_wait=next_wait,
                next_request=diag['next_request'],original_target_events=len(t),original_draft_events=len(d))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();source=args.source.resolve();out=args.output.resolve()
    require(out!=source and source not in out.parents,'Output must be outside source')
    out.mkdir(parents=True,exist_ok=True)
    selected=load(source/'selected.json');selections=list(selected.items())
    selections.append(('code34b_throughput',selected['code34b']['throughput_alternative']))
    native.COLORS.update(DETAIL_STYLES);native.COLORS.update(GAP_STYLES)
    records=[];models=[];panels=[];cards=[]
    with PdfPages(out/'NATIVE_EVENTS_WITH_WAITS.pdf') as pdf:
        for key,selection in selections:
            arm=read_arm(source,key,selection)
            raw=load(next((arm['folder']/'profile').glob('*draft*')));idx=command_index(raw)
            diagnoses={}
            for row in arm['rows']:
                diag=diagnose(arm,row,idx);records.append(diag['metrics'])
                diagnoses[(row['epoch'],row['step_id'])]=diag
            statistics={}
            for state in ('all',1,2,0):
                group=[x['metrics'] for x in diagnoses.values() if state=='all' or x['metrics']['state']==state]
                if group:
                    statistics[str(state)]={k:percentile([r[k] for r in group]) for k in group[0]
                                           if k.endswith('_ms') or k.endswith('_fraction')}
            dest=out/key/'with_waits';dest.mkdir(parents=True,exist_ok=True)
            links=[]
            for state in (1,2,0):
                group=sorted([r for r in arm['rows'] if r['state']==state],
                             key=lambda r:(r['target_step_span'],r['epoch'],r['step_id']))
                if not group:continue
                row=group[round(.5*(len(group)-1))];diag=diagnoses[(row['epoch'],row['step_id'])]
                rel=Path(key)/'with_waits'/f"waits_{state}_p50_req{row['epoch']}_step{row['step_id']}.png"
                panel=render(arm,row,diag,pdf,out/rel);panel.update(model=key,image=str(rel));panels.append(panel)
                links.append(f'<h3>{STATES[state]}</h3><a href="{rel}"><img src="{rel}"></a>')
            for path,h in arm['input_hashes'].items():require(sha(Path(path))==h,'Source changed')
            models.append(dict(model=key,profile=arm['folder'].name,commit=arm['result']['commit'],
                               eligible_steps=len(arm['rows']),statistics=statistics,source_hashes=arm['input_hashes']))
            cards.append('<section><h2>'+html.escape(arm['name'])+'</h2>'+''.join(links)+'</section>')
            print(key,len(arm['rows']),'steps checked',flush=True)
    write_csv(out/'PS_GAPS_AND_WAITS.csv',records)
    (out/'PS_GAPS_AND_WAITS.json').write_text(json.dumps(dict(status='complete',source_unchanged=True,
        adapter_sha256=sha(Path(__file__)),renderer_sha256=sha(Path(native.__file__)),
        association='same-request consecutive anchors; command CPU dispatch strictly after merge and before next receive',
        interpretation='GPU-event elapsed gaps are not pure CPU work or proven GPU idle; no shift and no runtime mutation',
        models=models,panels=panels),indent=2)+'\n')
    (out/'with_waits.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>DUET PS gaps and command waits</title>
<style>body{font:16px system-ui;max-width:1900px;margin:30px auto;padding:20px;color:#243443;background:#f5f7fa}section{background:white;padding:24px;margin:25px 0;border:1px solid #ddd}img{width:100%}p{line-height:1.7}</style>
<h1>PS 준비·미계측 구간·다음 명령 대기</h1><p><a href="NATIVE_EVENTS_WITH_WAITS.pdf">전체14쪽 PDF</a> · <a href="PS_GAPS.md">해석</a> · <a href="PS_GAPS_AND_WAITS.csv">모든1,309개 step 수치</a></p>
<p>① PS root/layout 구성 ② 이름 없는 PS dispatch 준비 구간 ③ PS merge 이후 미계측 후처리 ④ 다음 명령 수신 대기. 세로 점선 이후에는 다음 step의 요청 수신을 표시했습니다.
DS/PS graph 안은 forward 횟수만큼 흰 점선으로 등분했으며 실제 forward 시간 경계가 아닙니다.
주황색 PS prepare는 A(root/가시성)와 B(graph 입력 버퍼)로 나눴고, 마지막의 다음 요청 수신은 파란색으로 구분했습니다.
미계측 구간을 GPU idle로 단정하지 않습니다. 원래 막대/색/무늬는 유지하고 관측된 command receive와 다음 request receive를 추가했습니다. 원본 raw와 기존 그림은 보존했습니다.</p>'''+''.join(cards)+'</html>')


if __name__=='__main__':main()
