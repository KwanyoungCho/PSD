#!/usr/bin/env python3
"""Two main lanes, one model/page, using the established aligned plotter.

Read-only adapter for the same strict request-anchor-v2 source/representatives
as plot_model_breakdowns.py. Keeps original detailed files intact.
"""
from __future__ import annotations
import argparse
from collections import Counter
import html
import json
from pathlib import Path

from plot_model_breakdowns import (SSD, load, sha, read_arm, representatives,
                                    one, require, write_csv, subtitle, STATES,
                                    STATE_COLOR, plt, PdfPages)
from plot_duet_aligned_timeline import draw_phase_lanes
from matplotlib.patches import Patch

STYLE={
    'cache':('#f6cd8b','Cache wait'), 'prep':('#b7b9bf','Prepare'),
    'pre':('#8b1a1a','Target pre-exit'), 'post':('#8073ac','Target post-exit'),
    'head':('#fee08b','LM head'), 'accept':('#fc8d59','Acceptance'),
    'proxy':('#24966b','Proxy'), 'response':('#67a9cf','Response / wait'),
    'glue':('#27857c','Glue'), 'ds':('#8dc7ec','DS'),
    'wait':('#ffd92f','Proxy wait'), 'ps':('#91cda3','PS'),
    'previous':('#c5cbd1','Previous PS tail'),
}


def stage(key,lane,start,end,graphs=()):
    require(end>=start, 'Negative grouped phase')
    color,text=STYLE[key]
    return dict(key=key,lane=lane,start_ns=start,end_ns=end,
                color=color,text=text,graphs=list(graphs))


def prepare_panel(arm,row):
    pair=(row['epoch'],row['step_id']);groups=arm['grouped']
    t=groups['target'][pair];d=groups['draft'][pair]
    start=lambda e:e['wall_start_ns'];end=lambda e:e['wall_end_ns']
    origin=start(one(t,'target_spec_wait'));stages=[]
    for label,key,lane in (
        ('target_spec_wait','cache','target'),('verify_setup','prep','target'),
        ('graph_pre','pre','target'),('exit_logits','head','target'),
        ('graph_post','post','target'),('final_logits','head','target'),
        ('verify_sample_accept','accept','target'),('exit_proxy_side','proxy','proxy')):
        for e in t:
            if e['label']==label:stages.append(stage(key,lane,start(e),end(e)))
    send=one(d,'draft_send_response')
    stages.append(stage('response','draft',start(one(d,'draft_recv_request')),end(send)))
    for e in d:
        if e['label']=='glue':stages.append(stage('glue','draft',start(e),end(e)))
    p1=one(d,row['p1_span_label']);wait=one(d,'proxy_wait');merge=one(d,'merge_cache')
    stages.append(stage('ds','draft',start(p1),start(wait),
                        [(start(e),end(e)) for e in d if e['label']=='p1_graph_replay']))
    stages.append(stage('wait','draft',start(wait),end(wait)))
    stages.append(stage('ps','draft',end(wait),end(merge),
                        [(start(e),end(e)) for e in d if e['label']=='p2_graph_replay']))
    prev=groups['draft'].get((pair[0],pair[1]-1),[])
    pw=[e for e in prev if e['label']=='proxy_wait'];pm=[e for e in prev if e['label']=='merge_cache']
    if pw and pm and end(pm[0])>origin:
        stages.append(stage('previous','previous',max(origin,end(pw[0])),end(pm[0])))
    # Match the *whole* current target step and DS/PS. No artificial shift.
    finish=max(end(one(t,'target_postprocess')),end(merge))
    next_request=one(groups['target'][(pair[0],pair[1]+1)],'target_send_request')
    markers=[(end(one(t,'graph_pre')),'Exit','#a57b1b'),
             (end(one(t,'final_logits')),'Final head','#c13e31')]
    return dict(row=row,stages=stages,origin=origin,extent=(finish-origin)/1e6,
                markers=markers,next_request_ns=start(next_request))


def draw_panel(ax,arm,panel,state,q,xmax):
    if panel is None:
        ax.set_axis_off();ax.text(.5,.55,f'{STATES[state]} — no eligible samples',
            ha='center',va='center',transform=ax.transAxes,fontsize=15,color='#777777')
        return []
    r=panel['row'];n=sum(row['state']==state for row in arm['rows'])
    records=draw_phase_lanes(ax,panel['stages'],origin_ns=panel['origin'],xmax_ms=xmax,
                              markers=panel['markers'])
    ax.set_title(f"{STATES[state]}  |  actual p{round(q*100)} step / n={n}  |  "
                 f"UID {r['uid']}, step {r['step_id']}  |  AL={r['accepted_len']}",
                 fontsize=11,loc='left',pad=10,color=STATE_COLOR[state],fontweight='bold')
    ax.text(1,1.10,f"DS {r['D1']:.2f} ms  /  PS {r['D2']:.2f} ms",transform=ax.transAxes,
            ha='right',fontsize=10)
    ax.text(1,-.25,f"PS margin to final head: {r['gap2_verify']:+.2f} ms  "
            f"|  next-request margin: {r['gap2']:+.2f} ms",transform=ax.transAxes,
            ha='right',fontsize=8,color='#515965')
    return records


def legend(fig):
    handles=[Patch(facecolor=c,edgecolor='#34414b',linewidth=.4,label=t)
             for c,t in STYLE.values()]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.035),
               ncol=7,fontsize=9,frameon=False)


def caption(fig,arm):
    fig.text(.5,.015,'Dark stripe inside DS/PS = graph replay. Full bar includes preparation / export / merge. '
             'Real measured steps, no clock shift. Blank gaps are retained.',ha='center',fontsize=9,color='#555555')
    fig.text(.5,.915,subtitle(arm),ha='center',fontsize=10)
    fig.text(.5,.895,f"Measured v1.0 {arm['result']['commit'][:8]} / eslab18 RTX3090 / B1, T0.7 / "
              f"4 profiled turns, output cap256 / {len(arm['rows'])} eligible steps",ha='center',fontsize=9,color='#555555')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();source=args.source.resolve();out=args.output.resolve()
    require(out!=source and source not in out.parents,'Output must be outside source')
    out.mkdir(parents=True,exist_ok=True)
    selected=load(source/'selected.json');selections=list(selected.items())
    selections.append(('code34b_throughput',selected['code34b']['throughput_alternative']))
    arms=[read_arm(source,k,s) for k,s in selections]
    manifest=[];display_rows=[]
    with PdfPages(out/'ALIGNED_MODELS.pdf') as pdf, PdfPages(out/'ALIGNED_REPRESENTATIVES.pdf') as fullpdf:
        for arm in arms:
            dest=out/arm['key'];dest.mkdir(exist_ok=True)
            reps=representatives(arm['rows']);panels={(s,q):prepare_panel(arm,r) for s,q,r in reps}
            # Real p50 selection even when two samples deduplicated into p25/p75.
            medians={}
            for state in (1,2,0):
                group=sorted([r for r in arm['rows'] if r['state']==state],
                             key=lambda r:(r['target_step_span'],r['epoch'],r['step_id']))
                medians[state]=prepare_panel(arm,group[round(.5*(len(group)-1))]) if group else None
            xmax=max(p['extent'] for p in medians.values() if p)*1.055
            fig,axes=plt.subplots(3,1,figsize=(17,12))
            fig.suptitle(arm['name']+'\nTarget / draft aligned timeline',fontsize=17,fontweight='bold',y=.975)
            caption(fig,arm)
            for ax,state in zip(axes,(1,2,0)):
                draw_panel(ax,arm,medians[state],state,.5,xmax)
            legend(fig);fig.subplots_adjust(left=.10,right=.97,top=.845,bottom=.135,hspace=.58)
            fig.savefig(dest/'ALIGNED.png',dpi=160);pdf.savefig(fig);plt.close(fig)
            rep_entries=[]
            with PdfPages(dest/'ALIGNED.pdf') as per_model:
                for state,q,r in reps:
                    p=panels[(state,q)];fig,ax=plt.subplots(figsize=(17,5.8))
                    fig.suptitle(arm['name'],fontsize=16,fontweight='bold',y=.98)
                    caption(fig,arm)
                    shown=draw_panel(ax,arm,p,state,q,max(xmax,p['extent']*1.055))
                    legend(fig);fig.subplots_adjust(left=.10,right=.97,top=.78,bottom=.28)
                    per_model.savefig(fig);fullpdf.savefig(fig)
                    if medians[state]['row'] is r:
                        fig.savefig(dest/f'ALIGNED_{state}_p50.png',dpi=160)
                    plt.close(fig)
                    rep_entries.append(dict(state=state,quantile=q,uid=r['uid'],epoch=r['epoch'],step_id=r['step_id']))
                    display_rows.extend(dict(model=arm['key'],state=state,quantile=q,uid=r['uid'],
                        epoch=r['epoch'],step_id=r['step_id'],**e) for e in shown)
            manifest.append(dict(model=arm['key'],profile=arm['folder'].name,rows=len(arm['rows']),
                representative_count=len(reps),source_counts=dict(Counter(r['state'] for r in arm['rows'])),
                source_hashes=arm['input_hashes'],representatives=rep_entries,
                medians=[dict(state=s,uid=p['row']['uid'],epoch=p['row']['epoch'],step_id=p['row']['step_id'])
                         for s,p in medians.items() if p]))
            print(arm['key'], 'aligned page +',len(reps),'individual timelines',flush=True)
    for arm in arms:
        for name,h in arm['input_hashes'].items():require(sha(Path(name))==h,'Source modified')
    write_csv(out/'ALIGNED_INTERVALS.csv',display_rows)
    (out/'ALIGNED_AUDIT.json').write_text(json.dumps(dict(status='complete',main_pdf_pages=5,
        representative_pdf_pages=sum(a['representative_count'] for a in manifest),
        renderer='ssd/bench/plot_duet_aligned_timeline.py:draw_phase_lanes',
        renderer_sha256=sha(SSD/'bench/plot_duet_aligned_timeline.py'),
        adapter_sha256=sha(Path(__file__)),source_unchanged=True,clock_shift_ms=0,
        phase_lengths='Unmodified raw start/end intervals; DS/PS graphs contained in measured phase envelopes',
        models=manifest),indent=2)+'\n')
    cards=[]
    for arm in arms:
        key=arm['key']
        cards.append(f'<section><h2>{html.escape(arm["name"])}</h2><p><a href="{key}/ALIGNED.pdf">p25/p50/p75 PDF</a> · '+
                     f'<a href="{key}/PARAMETERS.json">실제 파라미터</a></p><a href="{key}/ALIGNED.png"><img src="{key}/ALIGNED.png"></a></section>')
    (out/'aligned.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>DUET aligned timelines</title>
<style>body{font:16px system-ui;max-width:1550px;margin:30px auto;padding:20px;color:#25333e;background:#f4f6f8}section{background:white;padding:20px;margin:24px 0;border:1px solid #ddd}img{width:100%}a{color:#176499}p{line-height:1.7}</style>
<h1>Target / Draft 두 줄 timeline</h1><p>모델마다 P1 hit·P2 hit·miss의 실제 중앙값 대표 step을 같은 축으로 표시했습니다.
DS/PS의 전체 막대는 준비·실행·정리까지, 막대 아래 진한 띠는 graph replay입니다. Proxy 보조 줄은 target 내부 별도 stream입니다.
회색 Previous PS tail은 현재 요청 시점에 남아 있던 직전 step의 PS입니다. 세로 점선은 exit와 final head 완료입니다.</p>
<p><b>기존 ver1.0의 설정별 4턴 profile</b>을 다시 그렸습니다. 원시 시간·대표 선택·파라미터는 유지했고 신규 inference는 없습니다.
34B 12/6은 miss 표본이 없고, 16/6은2개뿐입니다. 전체 막대를 합산하면 overlap을 이중 계산하게 됩니다.</p>
<p><a href="ALIGNED_MODELS.pdf">모델별 한 페이지 PDF (5쪽)</a> · <a href="ALIGNED_REPRESENTATIVES.pdf">모든 대표 step PDF (41쪽)</a> ·
<a href="REPORT.md">파라미터와 수치 설명</a> · <a href="index.html">이벤트별 상세 자료</a></p>'''+''.join(cards)+'</html>')


if __name__=='__main__':main()
