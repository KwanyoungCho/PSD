#!/usr/bin/env python3
"""Two consecutive DUET steps with observed draft readiness deadlines.

Uses the existing native-event palette, without clock shifts or minimum bar
widths. A third same-request anchor supplies the second deadline, not a third
forward. Every exported window is self-contained and can be redrawn elsewhere.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from copy import deepcopy
import html
import json
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from plot_model_breakdowns import (group_events, load, one, read_arm, require,
                                    sha, subtitle, STATES, write_csv, plt)
from plot_model_native_events import native, DETAIL_STYLES
from plot_model_wait_timelines import command_index, derived, GAP_STYLES

SCHEMA = 'duet-consecutive-window-v1'


def transition(groups, index, epoch, sid):
    """Associate a command using CPU chronology, never its stale step ID."""
    require(all((epoch, s) in groups[role] for role in ('target', 'draft')
                for s in (sid, sid+1)), 'Missing consecutive same-request anchors')
    draft = groups['draft'][(epoch, sid)]
    following = groups['draft'][(epoch, sid+1)]
    target = groups['target'][(epoch, sid+1)]
    merge = one(draft, 'merge_cache')
    recv = one(following, 'draft_recv_request')
    starts, commands = index
    lo, hi = merge['cpu_dispatch_end_ns'], recv['cpu_dispatch_start_ns']
    candidates = [e for e in commands[bisect_left(starts, lo):bisect_right(starts, hi)]
                  if e['cpu_dispatch_end_ns'] <= hi]
    require(len(candidates) == 1, f'Expected one next command after step {sid}, got {len(candidates)}')
    cmd = candidates[0]
    bounds = [merge['wall_end_ns'], cmd['wall_start_ns'], cmd['wall_end_ns'], recv['wall_start_ns']]
    require(all(b >= a for a, b in zip(bounds, bounds[1:])), 'Nonmonotone draft tail')
    send = one(target, 'target_send_request')
    wait = one(target, 'target_recv_response_wait')
    deadline = send['wall_start_ns']
    # Signed slack, including genuinely late preparation; do not clamp to zero.
    return dict(step_id=sid, next_step_id=sid+1,
        cache_ready_ns=merge['wall_end_ns'], draft_receive_ready_ns=cmd['wall_start_ns'],
        deadline_ns=deadline, cache_margin_ms=(deadline-merge['wall_end_ns'])/1e6,
        ready_margin_ms=(deadline-cmd['wall_start_ns'])/1e6,
        post_cache_cleanup_ms=(cmd['wall_start_ns']-merge['wall_end_ns'])/1e6,
        next_send_ms=(send['wall_end_ns']-send['wall_start_ns'])/1e6,
        next_response_wait_ms=(wait['wall_end_ns']-wait['wall_start_ns'])/1e6,
        command=cmd, next_receive=recv, next_send=send, next_response_wait=wait)


def make_window(groups, commands, epoch, sid, *, title, metadata, k1, k2):
    """Keep original records plus following anchors, with no trace mutation."""
    require(k1 >= 1 and k2 >= 1, 'Positive graph forward counts required')
    idx = command_index(commands)
    transitions = [transition(groups, idx, epoch, s) for s in (sid, sid+1)]
    target = [e for s in (sid, sid+1) for e in groups['target'][(epoch, s)]]
    draft = [e for s in (sid, sid+1) for e in groups['draft'][(epoch, s)]]
    # Show the preceding PS tail if it overlaps the first request handshake.
    wait = one(groups['target'][(epoch, sid)], 'target_spec_wait')
    for e in groups['draft'].get((epoch, sid-1), []):
        if min(e['wall_end_ns'], wait['wall_end_ns']) > max(e['wall_start_ns'], wait['wall_start_ns']):
            draft.append(e)
    chain=any(e['label']=='phase1_replay' for e in draft)
    return deepcopy(dict(schema=SCHEMA, title=title, metadata=metadata, epoch=epoch,
        step_ids=[sid, sid+1], forward_counts={} if chain else {'p1_graph_replay':k1, 'p2_graph_replay':k2},
        native_chain_forward_counts={'phase1_replay':k1,'phase2_replay':k2} if chain else {},
        target_events=target, draft_events=draft, transitions=transitions,
        deadline_definition='start of next target_send_request; observed readiness reference, not guaranteed spare compute budget',
        source_clock='raw wall timestamps mapped from per-process CUDA anchors; no clock shift',
        limitations=['Margins cross devices and inherit timestamp-alignment uncertainty.',
                    'Response wait includes cache lookup, miss drafting and communication.',
                    'Command receive is not entirely idle or freely usable draft time.',
                    'Equal graph divisions show counts, not measured forward boundaries.']))


def display_rows(window):
    """Alias the two PS prepares and explicitly draw only known missing spans."""
    target = window['target_events']; draft = window['draft_events']; shown = []
    aliases = {}; gaps = []
    for sid in window['step_ids']:
        events = [e for e in draft if e.get('step_id') == sid]
        build = one(events, 'phase2_build')
        preps = sorted((e for e in events if e['label'] == 'p2_prepare'
                        and e['cpu_dispatch_start_ns'] >= build['cpu_dispatch_end_ns']),
                       key=lambda e:e['cpu_dispatch_start_ns'])
        if any(e['label']=='phase2_replay' for e in events):
            # Native chain already records every per-round phase2_prepare
            # and phase2_replay. Retain them, without tree-only aliases.
            continue
        require(bool(preps), 'No PS preparation events')
        for i, e in enumerate(preps):
            aliases[id(e)] = f'PS prepare {chr(65+i)}: ' + ('roots / visibility' if i == 0 else 'graph input buffers')
        gaps.append(derived('PS dispatch gap (unprofiled)', build['wall_end_ns'], preps[0]['wall_start_ns'], sid))
    shown.extend(dict(e, label=aliases[id(e)]) if id(e) in aliases else dict(e) for e in draft)
    for tr in window['transitions']:
        c = tr['command']; r = tr['next_receive']; sid = tr['step_id']
        shown.append(dict(c))
        gaps.extend([derived('Post-PS gap (unprofiled)', tr['cache_ready_ns'], c['wall_start_ns'], sid),
                     derived('Command-to-request gap (unprofiled)', c['wall_end_ns'], r['wall_start_ns'], sid)])
    # The second following receive belongs to the third step: keep its raw
    # event in the JSON but do not draw another forward/request lane here.
    shown.extend(gaps)
    order = lambda e:(-(e['wall_end_ns']-e['wall_start_ns']), e['wall_start_ns'], e['label'])
    return sorted(target, key=order), sorted(shown, key=order), gaps


def render(window, pdf, png, phase_summary=None):
    require(window['schema'] == SCHEMA, 'Unsupported window schema')
    target, draft, gaps = display_rows(window)
    native.COLORS.update(DETAIL_STYLES); native.COLORS.update(GAP_STYLES)
    origin = min(e['wall_start_ns'] for e in target+draft)
    ms = lambda ns:(ns-origin)/1e6
    end = max([e['wall_end_ns'] for e in target+draft] + [t['deadline_ns'] for t in window['transitions']])
    fig = plt.figure(figsize=(20, 12) if phase_summary is not None else (20,9.5))
    ax = fig.add_axes([.065, .59, .91, .24] if phase_summary is not None else [.065, .48, .91, .35])
    for rows, y in ((target, 1), (draft, 0)):
        native._draw_row(ax, rows, y, y, origin, .53, .38, min_width_ms=0)
    divisions = native.draw_draft_forward_dividers(ax, window['draft_events'], window['forward_counts'], origin)
    for label,count in window.get('native_chain_forward_counts',{}).items():
        for sid in window['step_ids']:
            events=sorted((e for e in window['draft_events'] if e['label']==label and e['step_id']==sid),key=lambda e:e['wall_start_ns'])
            require(len(events)==count,'Native chain forward count mismatch')
            for i,e in enumerate(events):
                ax.text((ms(e['wall_start_ns'])+ms(e['wall_end_ns']))/2,0,str(i+1),
                        ha='center',va='center',color='white',fontsize=6,zorder=8)
            divisions.append(dict(label=label,step_id=sid,count=count,measured_boundaries=True,
                intervals_ns=[[e['wall_start_ns'],e['wall_end_ns']] for e in events]))
    for i, tr in enumerate(window['transitions']):
        deadline = ms(tr['deadline_ns']); ready = ms(tr['draft_receive_ready_ns'])
        cache = ms(tr['cache_ready_ns']); positive = tr['ready_margin_ms'] >= 0
        ax.axvline(deadline, color='#b2182b', lw=1.6, ls='--', zorder=8)
        ax.text(deadline, 1.57, f"D{i+1}: request {tr['next_step_id']} starts\n{deadline:.3f} ms",
                ha='right', va='top', color='#a50f15', fontsize=10, fontweight='bold')
        ax.plot(cache, -.29, marker='o', color='#238b45', ms=6, zorder=9)
        ax.plot(ready, -.29, marker='^', color='#01665e', ms=7, zorder=9)
        ax.annotate('', xy=(deadline, -.60), xytext=(ready, -.60),
                    arrowprops=dict(arrowstyle='<->', color='#01665e' if positive else '#a50f15', lw=1.5))
        ax.text(deadline if i else (ready+deadline)/2, -.85, f"ready margin {tr['ready_margin_ms']:+.3f} ms",
                ha='right' if i else 'center', fontsize=9, color='#01665e' if positive else '#a50f15')
        # Next send/response wait can be long even when previous PS is on time.
        if i == 0:
            w = tr['next_response_wait']
            ax.plot([ms(w['wall_start_ns']),ms(w['wall_end_ns'])], [1.33,1.33], color='#b35806',lw=3)
            ax.text(ms(w['wall_start_ns']),1.70,f"response wait {tr['next_response_wait_ms']:.3f} ms",ha='left',fontsize=8,color='#8c510a')
    for sid in window['step_ids']:
        rows=[e for e in target if e['step_id']==sid]
        start=one(rows,'target_send_request')['wall_start_ns']
        finish=one(rows,'target_postprocess')['wall_end_ns']
        ax.text((ms(start)+ms(finish))/2,1.85,f'Target step {sid}',ha='center',fontsize=12)
    ax.set(yticks=[0,1],yticklabels=['draft','target_rank0'],ylim=(-1.05,2.1),
           xlim=(-.4,ms(end)+.6),xlabel='ms from first displayed event (continuous time; no clock shift)')
    ax.grid(axis='x',alpha=.25,ls=':')
    fig.suptitle(window['title'],y=.98,fontsize=14)
    fig.text(.5,.94,window['metadata'].get('description',''),ha='center',fontsize=10)
    fig.text(.5,.908,'Red dashed line = next target request starts (observed deadline reference).  Green circle = PS cache ready; triangle = draft ready to receive.',ha='center',fontsize=10)
    fig.text(.5,.883,'Both target steps and their DS/PS work share one time axis. Only the following request boundary is used for the second deadline.',ha='center',fontsize=9)
    if phase_summary is not None and phase_summary.get('proxy_overlap', {}).get('overlapping_steps') == 0:
        fig.text(.5,.858,'In this measurement: no observed proxy computation / target graph_post overlap. Send enqueue does not measure transfer completion.',ha='center',fontsize=9,color='#8c510a')
    elif phase_summary is not None and phase_summary.get('proxy_overlap'):
        overlap=phase_summary['proxy_overlap']
        fig.text(.5,.858,f'Observed proxy / target graph_post overlap: {overlap["overlapping_steps"]}/{overlap["steps"]} steps; mean {overlap["mean_overlap_ms"]:.3f} ms over all steps. Send completion not measured.',ha='center',fontsize=9,color='#01665e')
    rows=[]; metrics=[]
    for tr in window['transitions']:
        item={k:v for k,v in tr.items() if not isinstance(v,dict)}
        item.update(epoch=window['epoch'],title=window['title'],origin_ns=origin)
        metrics.append(item)
        rows.append([str(tr['step_id']), f"{ms(tr['cache_ready_ns']):.3f}",f"{ms(tr['draft_receive_ready_ns']):.3f}",
                     f"{ms(tr['deadline_ns']):.3f}",f"{tr['cache_margin_ms']:+.3f}",f"{tr['ready_margin_ms']:+.3f}",
                     f"{tr['next_send_ms']:.3f}",f"{tr['next_response_wait_ms']:.3f}"])
    if phase_summary is None:
        tab=fig.add_axes([.065,.315,.91,.105]);tab.axis('off')
        table=tab.table(cellText=rows,colLabels=['Prepared in step','Cache ready (ms)','Receive-ready (ms)','Next send (ms)',
                     'Cache margin (ms)','Ready margin (ms)','Next send span (ms)','Next response wait (ms)'],loc='center',cellLoc='center')
        table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.6)
        fig.text(.065,.29,'Margin = next send start minus readiness. Positive means ready before the observed request; negative means preparation ran past it.',fontsize=9)
        fig.text(.065,.266,'This is not a causal stall estimate or a promise that all margin can be used for more forwards. The second next-response wait is outside the two-step view.',fontsize=9)
    else:
        from phase_averages import table_headers, table_cells, PERFORMANCE_HEADERS, performance_cells
        for y,headers,cells in ((.395,PERFORMANCE_HEADERS,performance_cells(phase_summary['rows'])),
                                (.245,table_headers(phase_summary['rows']),table_cells(phase_summary['rows']))):
            tab=fig.add_axes([.065,y,.91,.12]);tab.axis('off')
            table=tab.table(cellText=cells,colLabels=headers,loc='center',cellLoc='center')
            table.auto_set_font_size(False);table.set_fontsize(9.5);table.scale(1,1.65)
            for (r,c),cell in table.get_celld().items():
                if r==0:cell.set_facecolor('#e1eaf0');cell.set_text_props(weight='bold')
                elif r==4:cell.set_facecolor('#edf4ec');cell.set_text_props(weight='bold')
        fig.text(.065,.532,'ALL ELIGIBLE STEPS FOR THIS SETTING (not the two example steps above); Overall is step-weighted.',fontsize=10,weight='bold')
        fig.text(.065,.382,'Rate = source / all steps; Overall rate = total hit (P1 + P2) / all. AL includes recovery/bonus. Profile TPS = emitted tokens / summed step seconds.',fontsize=9)
        fig.text(.065,.222,'Step = current target request start -> next request start. *DS spare = proxy send-ready - DS end; PS spare = next request start - draft receive-ready.',fontsize=9)
        if 'ds_forward_once_ms' in phase_summary['rows'][0]:
            note=('Forward = native model + LM head graph replay (output stores included). '
                  if window.get('native_chain_forward_counts') else
                  'Forward = measured model + LM head (markers included). ')
            note+='Compute* = forward sum + spare, with current non-forward work fixed; not a safe larger-K guarantee.'
        else:
            note='Positive spare = early; negative = late. Graph/K includes sampling/selection, not isolated transformer latency. These are profiled-subset TPS, not full evaluation TPS.'
        fig.text(.065,.202,note,fontsize=9)
    labels=sorted({e['label'] for e in target+draft})
    handles=[Patch(facecolor=native._style(l)[0],hatch=native._style(l)[1],edgecolor='black',linewidth=.3,label=l) for l in labels]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.047),ncol=6,fontsize=6.8,columnspacing=1.0)
    divider_note=('Numbered blue/green bars are separately measured chain forwards.' if window.get('native_chain_forward_counts') else
                  'White graph dividers show counts, NOT per-forward measurements.')
    fig.text(.065,.025,'Hatched gaps = unprofiled elapsed intervals, NOT proven GPU idle. Nested spans are not additive. '+divider_note,fontsize=8)
    fig.text(.065,.009,'Raw intervals retained, no 0.02 ms visibility floor. Cross-device margins inherit CUDA-to-host clock alignment uncertainty; exact raw events and source hashes accompany this figure.',fontsize=8)
    pdf.savefig(fig);fig.savefig(png,dpi=155);plt.close(fig)
    return dict(metrics=metrics,derived_gaps=gaps,forward_dividers=divisions,phase_summary=phase_summary,
                minimum_bar_width_ms=0,clock_shift_ns=0)


def read_profiles(folder):
    paths={role:sorted(folder.glob(f'duet_profile_{prefix}*.json'))
           for role,prefix in (('target','target_rank0'),('draft','draft'))}
    require(all(paths.values()), 'Expected target_rank0 and draft profile JSON files')
    events={role:[e for p in files for e in load(p) if e['label']!='_anchor']
            for role,files in paths.items()}
    groups={role:group_events(events[role],anchor) for role,anchor in
            (('target','target_send_request'),('draft','draft_recv_request'))}
    require(set(groups['target'])==set(groups['draft']), 'Target/draft request anchors do not match')
    return groups,events['draft'],{str(p):sha(p) for files in paths.values() for p in files}


def model_windows(source):
    selected=load(source/'selected.json');selections=list(selected.items())
    selections.append(('code34b_throughput',selected['code34b']['throughput_alternative']))
    for key,selection in selections:
        arm=read_arm(source,key,selection)
        raw=[e for p in sorted((arm['folder']/'profile').glob('*draft*.json')) for e in load(p) if e['label']!='_anchor']
        for state in (1,2,0):
            rows=sorted([r for r in arm['rows'] if r['state']==state and
                         all((r['epoch'],r['step_id']+i) in arm['grouped']['target'] for i in (1,2))],
                        key=lambda r:(r['target_step_span'],r['epoch'],r['step_id']))
            if not rows:continue
            row=rows[round(.5*(len(rows)-1))]
            config=arm['result']['config']
            window=make_window(arm['grouped'],raw,row['epoch'],row['step_id'],
                title=f"{arm['name']} | {STATES[state]} -> next step | UID {row['uid']} | request {row['epoch']}",
                metadata=dict(description=f"Historical v1.0 {arm['result']['commit'][:8]} | "+subtitle(arm),
                              source_hashes=arm['input_hashes'],selection='median eligible target step span within source',
                              current_runtime_measurement=False,uid=row['uid'],state=state),
                k1=config['k1'],k2=config['k2'])
            yield f"{key}_source{state}_req{row['epoch']}_step{row['step_id']}",window


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--window',type=Path,help='Portable exported raw window; no original server paths needed')
    mode.add_argument('--source',type=Path,help='Historical model handoff with selected.json')
    mode.add_argument('--profile-dir',type=Path,help='Directory with target_rank0/draft profile JSONs')
    parser.add_argument('--request-epoch',type=int);parser.add_argument('--step-id',type=int)
    parser.add_argument('--k1',type=int);parser.add_argument('--k2',type=int)
    parser.add_argument('--title',default='DUET consecutive steps')
    parser.add_argument('--description',default='Raw profile; explicitly supplied forward counts')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    inp=(args.window or args.source or args.profile_dir).resolve()
    require(out!=inp and inp not in out.parents,'Output must be outside source')
    if args.window:
        windows=[(args.window.stem,load(args.window))]
    elif args.source:
        windows=model_windows(args.source)
    else:
        require(all(x is not None for x in (args.request_epoch,args.step_id,args.k1,args.k2)),
                'profile-dir requires request-epoch, step-id, k1 and k2')
        groups,commands,hashes=read_profiles(args.profile_dir)
        window=make_window(groups,commands,args.request_epoch,args.step_id,title=args.title,
            metadata=dict(description=args.description,source_hashes=hashes),k1=args.k1,k2=args.k2)
        windows=[(f'req{args.request_epoch}_step{args.step_id}',window)]
    out.mkdir(parents=True,exist_ok=True);panels=[];metrics=[];cards=[]
    with PdfPages(out/'CONSECUTIVE_STEPS.pdf') as pdf:
        for stem,window in windows:
            # Fresh extraction must leave the measured source unchanged.
            if not args.window:
                for p,h in window['metadata'].get('source_hashes',{}).items():
                    require(sha(Path(p))==h,'Source changed while extracting timeline')
            dest=out/(stem+'.json');dest.write_text(json.dumps(window,indent=2)+'\n')
            panel=render(window,pdf,out/(stem+'.png'));panel.update(window=dest.name,window_sha256=sha(dest))
            panels.append(panel);metrics.extend(panel['metrics'])
            cards.append(f'<section><h2>{html.escape(window["title"])}</h2><a href="{stem}.png"><img src="{stem}.png"></a><p><a href="{stem}.json">Portable raw window</a></p></section>')
            print(stem,'ready margins:',[round(t['ready_margin_ms'],3) for t in window['transitions']],flush=True)
    write_csv(out/'DEADLINES.csv',metrics)
    (out/'AUDIT.json').write_text(json.dumps(dict(schema=SCHEMA,panels=panels,
        tool_sha256=sha(Path(__file__)),renderer_sha256=sha(Path(native.__file__)),
        pages=len(panels),source_unchanged=True,portable_replay=bool(args.window)),indent=2)+'\n')
    (out/'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>DUET two-step deadlines</title>
<style>body{font:16px system-ui;max-width:2000px;margin:30px auto;padding:20px;background:#f6f8fa}section{background:white;padding:20px;margin:20px 0}img{width:100%}p{line-height:1.7}</style>
<h1>연속 두 step과 draft 마감</h1><p><a href="CONSECUTIVE_STEPS.pdf">전체 PDF</a> · <a href="DEADLINES.csv">정확한 수치</a></p>
<p>빨간 점선은 다음 target 요청 시작 시점입니다. 초록 원은 PS cache 완성, 삼각형은 draft가 다음 명령을 받을 준비가 된 시점입니다.
여유 시간은 관측된 차이이며 모두 추가 계산에 써도 된다는 뜻은 아닙니다. 회색 수신 구간과 target 응답 대기를 PS 지연으로 단정하지 않습니다.
각 페이지의 두 step은 같은 요청의 연속 기록이며, 다음 요청의 경계만 두 번째 마감 계산에 사용합니다. 과거 모델 trace와 새 최적화 trace는 제목으로 구분합니다.</p>'''+''.join(cards)+'</html>')


if __name__=='__main__':main()
