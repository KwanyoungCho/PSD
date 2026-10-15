#!/usr/bin/env python3
"""Render raw per-event two-lane timelines with the existing aligned plotter.

No phase aggregation. Strict request-anchor-v2 selects the same representative
steps as the detailed diagnostics. The historical renderer's 0.02 ms visibility
floor is explicitly retained and disclosed; exact durations stay in the CSV.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
from pypdf import PdfReader, PdfWriter

from plot_model_breakdowns import (SSD, load, sha, read_arm, representatives,
                                    one, require, write_csv, subtitle, STATES)
import plot_duet_aligned_timeline as native

# Latest profiles have P1-executor and wire/KV labels missing from the older
# palette. Keep every original name and interval; only supply missing styles.
DETAIL_STYLES={
    'p1_root_build':('#c6dbef','..'), 'p1_slot_prepare':('#9ecae1','//'),
    'p1_prepare':('#4292c6','xx'), 'p1_graph_replay':('#08519c',''),
    'draft_response_pack':('#9e9ac8','//'), 'draft_send_fused':('#525252','//'),
    'draft_send_chain_q':('#737373','..'), 'draft_send_parent_q_p1':('#225ea8','..'),
    'draft_send_parent_q_p2':('#238b45','..'), 'tree_kv_restore':('#1d91c0','//'),
    'tree_hit_rerank_p1':('#74a9cf','..'), 'tree_hit_rerank_p2':('#74c476','..'),
    'tree_hit_pack_served_p1':('#a6bddb','xx'), 'tree_hit_pack_served_p2':('#a1d99b','xx'),
    'tree_hit_parent_q_gather_p1':('#3690c0','//'), 'tree_hit_parent_q_gather_p2':('#41ab5d','//'),
    'target_recv_fused':('#fe9929','//'), 'target_response_meta_read':('#fec44f','..'),
    'target_recv_parent_q_p1':('#6baed6','//'), 'target_recv_parent_q_p2':('#74c476','//'),
    'tree_wire_parse_validate':('#bdbdbd','xx'), 'tree_proxy_topology_prepare':('#b2df8a','..'),
    'tree_parent_q_select':('#33a02c','xx'), 'tree_verify_meta_cpu':('#969696','//'),
    'tree_verify_mask_prepare':('#636363','xx'), 'tree_verify_attention_buffers':('#bdbdbd','..'),
    'verify_accept_prep':('#fdd49e','xx'), 'tree_accept_walk':('#f16913','//'),
    'chain_accept':('#f16913','//'), 'tree_kv_commit':('#41b6c4','xx'),
}


def native_rows(arm,row):
    """No aggregation: raw current step plus previous-step wait overlap."""
    key=(row['epoch'],row['step_id']);groups=arm['grouped']
    target=list(groups['target'][key]);draft=list(groups['draft'][key])
    wait=one(target,'target_spec_wait');lo=wait['wall_start_ns'];hi=wait['wall_end_ns']
    # Include the previous PS whose completion can delay the current response.
    for event in groups['draft'].get((key[0],key[1]-1),[]):
        if min(event['wall_end_ns'],hi)>max(event['wall_start_ns'],lo):draft.append(event)
    # Background enclosing spans first; all exact raw intervals are retained.
    order=lambda e:(-(e['wall_end_ns']-e['wall_start_ns']),e['wall_start_ns'],e['label'])
    return sorted(target,key=order),sorted(draft,key=order)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();source=args.source.resolve();out=args.output.resolve()
    require(out!=source and source not in out.parents,'Output must be outside source')
    out.mkdir(parents=True,exist_ok=True)
    selected=load(source/'selected.json');selections=list(selected.items())
    selections.append(('code34b_throughput',selected['code34b']['throughput_alternative']))
    native.COLORS.update(DETAIL_STYLES)
    all_pdf=PdfWriter();median_pdf=PdfWriter();manifest=[];events=[];cards=[]
    for key,selection in selections:
        arm=read_arm(source,key,selection);dest=out/key/'native_events';dest.mkdir(parents=True,exist_ok=True)
        forward_counts={'p1_graph_replay':arm['result']['config']['k1'],
                        'p2_graph_replay':arm['result']['config']['k2']}
        medians={}
        for state in (1,2,0):
            group=sorted([r for r in arm['rows'] if r['state']==state],
                         key=lambda r:(r['target_step_span'],r['epoch'],r['step_id']))
            if group:medians[state]=group[round(.5*(len(group)-1))]
        model_pdf=PdfWriter();rep_meta=[];links=[]
        for state,q,row in representatives(arm['rows']):
            t,d=native_rows(arm,row)
            stem=f"events_{state}_p{round(q*100):02d}_req{row['epoch']}_step{row['step_id']}"
            suffix=(f"{arm['name']}\n{STATES[state]} / p{round(q*100)} / UID {row['uid']} / request {row['epoch']}\n"
                    f"{subtitle(arm)}\nMeasured v1.0 {arm['result']['commit'][:8]} | raw events, no phase grouping / clock shift; "
                    'bars <0.02 ms use visibility floor; dotted forward slots are equal-width guides, NOT measured boundaries')
            native.plot_aligned_step(t,d,row['step_id'],dest/(stem+'.pdf'),title_suffix=suffix,
                                     draft_forward_counts=forward_counts)
            all_pdf.append(str(dest/(stem+'.pdf')))
            is_median=medians.get(state) is row
            if is_median:
                model_pdf.append(str(dest/(stem+'.pdf')));median_pdf.append(str(dest/(stem+'.pdf')))
                native.plot_aligned_step(t,d,row['step_id'],dest/(stem+'.png'),title_suffix=suffix,
                                         draft_forward_counts=forward_counts)
                links.append(dict(state=state,image=key+'/native_events/'+stem+'.png',pdf=key+'/native_events/'+stem+'.pdf'))
            origin=min(e['wall_start_ns'] for e in t+d)
            for role,rows in [('target',t),('draft',d)]:
                for e in rows:
                    events.append(dict(model=key,state=state,quantile=q,uid=row['uid'],epoch=row['epoch'],
                        selected_step=row['step_id'],event_step=e['step_id'],role=role,label=e['label'],
                        parent=e.get('parent_label'),start_ms=(e['wall_start_ns']-origin)/1e6,
                        end_ms=(e['wall_end_ns']-origin)/1e6,
                        duration_ms=(e['wall_end_ns']-e['wall_start_ns'])/1e6,
                        cuda_ms=e.get('cuda_ms'),minimum_display_width_ms=.02))
            rep_meta.append(dict(state=state,quantile=q,epoch=row['epoch'],step_id=row['step_id'],
                uid=row['uid'],is_median=is_median,target_events=len(t),draft_events=len(d),
                pdf=key+'/native_events/'+stem+'.pdf'))
        model_pdf.write(str(out/key/'NATIVE_EVENTS.pdf'));model_pdf.close()
        for p,h in arm['input_hashes'].items():require(sha(Path(p))==h,'Source modified')
        manifest.append(dict(key=key,profile=arm['folder'].name,rows=len(arm['rows']),
            representative_count=len(rep_meta),median_count=len(medians),source_hashes=arm['input_hashes'],
            forward_counts=forward_counts,representatives=rep_meta))
        cards.append(f'<section><h2>{html.escape(arm["name"])}</h2><p><a href="{key}/NATIVE_EVENTS.pdf">모델별 원시 이벤트 PDF</a></p>'+''.join(
            f'<h3>{STATES[x["state"]]}</h3><a href="{x["pdf"]}"><img src="{x["image"]}"></a>' for x in links)+'</section>')
        print(key,len(rep_meta),'raw-event representatives;',len(medians),'median panels',flush=True)
    all_pdf.write(str(out/'NATIVE_EVENTS_ALL.pdf'));all_pdf.close()
    median_pdf.write(str(out/'NATIVE_EVENTS_MODELS.pdf'));median_pdf.close()
    write_csv(out/'NATIVE_EVENT_INTERVALS.csv',events)
    (out/'NATIVE_EVENTS_AUDIT.json').write_text(json.dumps(dict(status='complete',
        renderer='ssd/bench/plot_duet_aligned_timeline.py:plot_aligned_step',
        historical_last_commit='607396672e7b2c49f992f329f559b799d67882b5',
        renderer_sha256=sha(SSD/'bench/plot_duet_aligned_timeline.py'),adapter_sha256=sha(Path(__file__)),
        style_additions=DETAIL_STYLES,source_unchanged=True,clock_shift_ms=0,phase_aggregation=False,
        minimum_display_width_ms=.02,
        forward_dividers='equal-width count guides inside measured graph envelopes; not measured per-forward boundaries',
        notes='Legacy renderer retains minimum-width visibility and parent/child overlays. Exact raw boundaries are in CSV. Shared p2_prepare/output_convert may also execute in DS.',
        median_pages=sum(a['median_count'] for a in manifest),representative_pages=sum(a['representative_count'] for a in manifest),models=manifest),indent=2)+'\n')
    (out/'native_events.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>DUET raw-event aligned timeline</title>
<style>body{font:16px system-ui;max-width:1750px;margin:25px auto;padding:20px;color:#243443;background:#f5f7fa}section{background:white;padding:24px;margin:25px 0;border:1px solid #ddd}img{width:100%}a{color:#176499}p{line-height:1.7}</style>
<h1>기존 상세 aligned timeline — 이벤트별 준비/실행/변환/merge</h1>
<p>9월8일 갱신된 기존 <code>plot_duet_aligned_timeline.py → plot_aligned_step</code>로 직접 그렸습니다.
DS/PS를 합치지 않고 원래 event 이름과 색/무늬를 유지합니다. P1 executor 등 새 label에는 식별 색을 추가했습니다.
DS/PS graph 막대는 K1/K2 횟수만큼 흰 점선으로 등분했습니다. 개별 forward 실측 경계가 아니라 횟수 안내입니다.
현재 요청과 step을 엄격히 구분하고, cache wait와 겹친 직전 draft step도 포함했습니다.</p>
<p><a href="NATIVE_EVENTS_MODELS.pdf">모델별 P1/P2/miss 중앙 대표 PDF (14쪽)</a> ·
<a href="NATIVE_EVENTS_ALL.pdf">p25/p50/p75 전체 (41쪽)</a> · <a href="NATIVE_EVENT_INTERVALS.csv">모든 표시 이벤트의 실제 시간</a> · <a href="REPORT.md">설명</a></p>
<p>기존 renderer의 0.02ms 미만 막대 최소 표시 폭은 유지합니다. 실제 시간은 CSV를 따르며 겹치는 부모·자식 막대를 합산하지 않습니다.
준비 과정을 숨기기 위한 phase 통합이나 시계 이동은 없습니다. 다만 profile 자체가 phase 전체를 한 CUDA graph로 기록하므로 graph 안의 forward별 시간은 알 수 없습니다.
<code>p2_prepare</code> 같은 공통 helper는 DS에서도 쓰여 같은 label이 두 번 나올 수 있습니다.
측정 자료는 앞과 동일한 ver1.0·설정당4턴입니다. Code34B 12/6은 miss 표본0개라 해당 그림이 없고, 16/6은2개뿐입니다.</p>'''+''.join(cards)+'</html>')


if __name__=='__main__':main()
