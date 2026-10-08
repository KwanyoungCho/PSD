"""Paper palette + status means + real aligned steps; no GPU/model imports."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

from paper_view_data import HERE, OUT, STATUS_ORDER

PAPER_SCRIPT = HERE.parents[2] / 'ssd/tools/duet_timeline/plot_paper_fig4_schematic_pct.py'
spec = importlib.util.spec_from_file_location('original_paper_palette', PAPER_SCRIPT)
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)
P = paper.COLORS
COLORS = {
    'Wait / sync': P['sync'], 'Prepare': '#949494',
    'Target pre': P['verify_pre'], 'Target post': P['verify_post'],
    'Target verify': P['verify_pre'], 'Proxy / launch': P['proxy'],
    'Proxy compute': P['proxy'], 'Final logits': '#e7c17b',
    'Accept': P['sample'], 'Accept prepare': '#bdbdbd', 'Commit': '#875d75',
    'Postprocess': '#656565', 'Unlabelled gap': '#eeeeee',
    'Miss draft': P['redraft'], 'Miss response': P['redraft'],
    'Mixed response': '#d97966', 'Context': P['context'], 'P1': P['draft'],
    'P2': P['proxy_draft'], 'Proxy wait': '#f2d775',
    'Request': '#bbbbbb', 'Cache response': '#7dac83', 'Send response': '#777777',
    'SSD build': '#889fc2', 'SSD decode': P['draft'], 'Cache insert': '#77ad6c',
    'Previous P2': '#9dd9e5', 'Previous SSD decode': '#a9bfdf', 'Previous cache insert': '#9dba96',
}
STATUS_COLORS = {'P1 hit':P['draft'], 'P2 hit':P['proxy_draft'], 'Hit':P['draft'],
                 'Mixed hits':'#608e86', 'Mixed hit/miss':'#e8a15a','Miss':P['redraft']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})


def config_label(p):
    if p['method'] == 'SSD':
        return f"SSD: K={p['SSD_K']}, F={p['SSD_F']}"
    return (f"DUET: K1/K2={p['K1']}/{p['K2']}, exit={p['exit']} (0-based), C={p['C']}, "
            f"N1/N2={p['N1']}/{p['N2']}, U={p['P1_roots']}, W={p['W']}, "
            f"trim/stream={p['trim']}/{p['stream']}")


def label_color(label):
    return COLORS.get(label, '#cccccc')


def stacked(ax, groups, lane):
    labels = list(dict.fromkeys(c['label'] for g in groups for c in g['components'] if c['lane']==lane))
    if lane == 'Target':
        preferred=['Wait / sync','Prepare','Target pre','Proxy / launch','Target post',
                   'Target verify','Final logits','Accept prepare','Accept','Commit','Postprocess','Unlabelled gap']
    else:
        preferred=['Request','Cache response','Miss response','Mixed response','Miss draft',
                   'Send response','Context','P1','Proxy wait','P2','SSD build','SSD decode','Cache insert']
    labels = [s for s in preferred if s in labels] + [s for s in labels if s not in preferred]
    bottom = np.zeros(len(groups))
    for label in labels:
        values = [sum(c['mean_ms'] for c in g['components'] if c['lane']==lane and c['label']==label) for g in groups]
        ax.bar(range(len(groups)), values, bottom=bottom, color=label_color(label), edgecolor='white', lw=.4, label=label)
        bottom += values
    for i,v in enumerate(bottom):
        ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(range(len(groups)),[g['status']+'\nn='+str(g['n']) for g in groups],fontsize=9)
    ax.set(ylabel='Mean stage duration (ms)',title=f'{lane}: status means on the same eligible steps')
    ax.set_ylim(0,max(bottom)*1.18 if len(bottom) and max(bottom)>0 else 1)
    ax.grid(axis='y',alpha=.18)
    return labels


def timeline(ax, groups, normalized):
    y=0
    ticks=[];ticklabels=[];extents=[0.,100. if normalized else 0.]
    seen=set()
    for g in groups:
        r=g['representative']
        vs=g['mean_intervals'] if normalized else r['intervals']+r['previous']
        denominator=g['mean_target_ms'] if normalized else 1.
        scale=100/denominator if normalized else 1.
        lanes=['Target']
        if any(v['lane']=='Proxy stream' for v in vs):lanes.append('Proxy stream')
        if not normalized and any(v['lane']=='Draft previous' for v in vs):lanes.append('Draft previous')
        lanes.append('Draft current')
        group_top=y
        for lane in lanes:
            intervals=[v for v in vs if v['lane']==lane]
            for v in intervals:
                start,end=v['start']*scale,v['end']*scale
                if end<=start:continue
                ax.broken_barh([(start,end-start)],(y-.30,.6),facecolors=label_color(v['label']),
                               edgecolors='white',linewidth=.35,
                               hatch='//' if lane=='Draft previous' else None)
                extents.extend([start,end]);seen.add(v['label'])
                if normalized and end-start>=10:
                    ax.text((start+end)/2,y,f'{end-start:.0f}%',ha='center',va='center',fontsize=8,
                            color='white' if v['label'] in ['Target pre','Target post','P1','SSD decode','Target verify'] else '#222222')
            ticks.append(y)
            ticklabels.append(lane.replace(' current','').replace(' previous',' (prev.)'))
            y+=1
        markers=g['mean_markers'] if normalized else r['markers']
        for label,key,color,style in [('Ready','Target ready','#333333','--'),
                                      ('Proxy','Proxy received',P['proxy'],':')]:
            if key in markers:
                x=markers[key]*scale
                ax.vlines(x,group_top-.45,y-.5,color=color,linestyle=style,lw=1)
                extents.append(x)
        info=(f"{g['status']} | n={g['n']} | mean target={g['mean_target_ms']:.2f} ms" if normalized else
              f"{g['status']} | step {r['step_id']} | target={r['target_ms']:.2f} ms")
        ax.text(.005,group_top-.70,info,transform=ax.get_yaxis_transform(),fontsize=9,fontweight='bold',va='bottom')
        y+=1.7
    ax.set_yticks(ticks,ticklabels,fontsize=8)
    ax.set_ylim(y-1,-1.4)
    span=max(extents)-min(extents)
    ax.set_xlim(min(0,min(extents))-.01*span,max(extents)+.035*span)
    ax.grid(axis='x',alpha=.18)
    ax.set_xlabel('Time / mean target request-to-ready window (%)' if normalized else
                  'Time from target request start (ms)')
    ax.set_title('Paper-style mean aligned schematic' if normalized else 'Observed representative steps (near median target window)',pad=18)
    return seen


def profile_figure(row, title=None):
    groups=row['detail']['groups']; p=row['parameters']; m=row['last']
    if not groups:
        return wall_figure(row, note='GPU trace exists, but no step satisfies the steady diagnostic filters.')
    n=len(groups)
    fig=plt.figure(figsize=(17,max(10,6+1.7*n)))
    gs=fig.add_gridspec(2,2,height_ratios=[2.5,max(4,n*1.6)],hspace=.45,wspace=.24)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1])
    labels=set(stacked(a,groups,'Target')+stacked(b,groups,'Draft current'))
    a=fig.add_subplot(gs[1,0]);b=fig.add_subplot(gs[1,1])
    labels|=timeline(a,groups,True);labels|=timeline(b,groups,False)
    head=title or row['name']
    if m['timing_excluded']:
        head='TIMING EXCLUDED (concurrent workload) | '+head
    fig.suptitle(head+'\n'+config_label(p)+
                 f"\nB={m['batch']}, T={m['temperature']}, {m['questions']} prompts, cap={p['input_cap']}/{p['output_cap']} | "
                 f"instrumented diagnostic; {row['detail']['eligible_steps']} eligible steps",fontsize=12,y=.985)
    handles=[Patch(facecolor=label_color(k),label=k) for k in COLORS if k in labels]
    handles += [Line2D([0],[0],color='#333333',ls='--',label='Target ready'),
                Line2D([0],[0],color=P['proxy'],ls=':',label='Proxy received')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.023),ncol=7,fontsize=8,frameon=False)
    fig.text(.5,.009,'Stage means are diagnostics, not final TPS. Proxy side-stream work is a separate lane, never added to the target stack.\n'
             'Blank timeline gaps are retained; no common hit/miss compute cost is imposed. Cross-GPU times use recorded CPU/CUDA anchors.',
             ha='center',va='bottom',fontsize=8,color='#444444')
    fig.subplots_adjust(top=.87,bottom=.14,left=.095,right=.975)
    return fig


def wall_figure(row,note=None):
    m=row['last']; p=row['parameters']; groups=row['wall']['groups']; statuses=[g['status'] for g in groups]
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    ax=axes[0,0];left=0
    for g in groups:
        width=g['time_share']*100
        ax.barh([0],[width],left=left,color=STATUS_COLORS[g['status']],label=g['status'])
        if width>7:ax.text(left+width/2,0,f'{width:.1f}%',ha='center',va='center',fontsize=10)
        left+=width
    ax.set(xlim=(0,100),yticks=[],xlabel='Share of included decode time (%)',title='Measured wall time by batch cache status')
    ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.24),ncol=3)
    ax=axes[0,1]
    ax.bar(range(len(groups)),[g['mean_ms'] for g in groups],color=[STATUS_COLORS[s] for s in statuses])
    ax.set_xticks(range(len(groups)),[s+'\nn='+str(g['steps']) for s,g in zip(statuses,groups)],fontsize=8)
    for i,g in enumerate(groups):ax.text(i,g['mean_ms'],f"{g['mean_ms']:.2f}",ha='center',va='bottom',fontsize=9)
    ax.set(ylabel='Mean wall time / batch step (ms)',title='Boundary-free steps, including slow steps');ax.grid(axis='y',alpha=.2)
    ax=axes[1,0];w=m['clean_source_weights'];al=m['clean_conditional_al']; names=['Miss','Hit'] if p['method']=='SSD' else ['Miss','P1 hit','P2 hit']
    for i,s in enumerate(names):
        v=al[i] or 0
        ax.bar(i,v,color=STATUS_COLORS[s]);ax.text(i,v,f'{v:.3f}\nshare {w[i]*100:.1f}%',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(range(len(names)),names);ax.set(ylabel='AL* (includes recovery/bonus)',title='Conditional AL* and source share')
    ax.set_ylim(0,max(x or 0 for x in al)*1.32);ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];ss=row['summaries']
    xs=list(range(len(ss)));vs=[x['boundary_excluded_step_tps'] for x in ss]
    ax.bar(xs,vs,color=['#bdbdbd' if x['pass_role']=='warmup' else '#2f6db5' for x in ss])
    ax.set_xticks(xs,[f"Pass {x['pass_index']}\n{x['pass_role']}\nB{x['batch']}, T{x['temperature']}" for x in ss],fontsize=8)
    for i,v in enumerate(vs):ax.text(i,v,f'{v:.1f}',ha='center',va='bottom',fontsize=9)
    ax.set(ylabel='TPS* (tokens/s)',title='All saved passes; gray = warmup');ax.grid(axis='y',alpha=.2)
    fig.suptitle(row['name']+'\n'+config_label(p)+
                 f"\nLast pass: B={m['batch']}, T={m['temperature']}, N={m['questions']} | AL*={m['boundary_excluded_al']:.3f}, "
                 f"hit={m['cache_hit']*100:.1f}%, TPS*={m['boundary_excluded_step_tps']:.2f}",fontsize=12)
    text=note or 'No per-phase GPU trace was recorded for this run. These are observed whole-step/source breakdowns, not inferred phase times.'
    if row['last']['timing_excluded']:text='TIMING EXCLUDED: concurrent workload contaminated this run. Retained only for audit.\n'+text
    fig.text(.5,.025,text+'\nTop and lower-left panels use the last saved pass. AL* and TPS* use different boundary filters; partial batches remain in TPS*.',
             ha='center',fontsize=8,color='#555555')
    fig.subplots_adjust(top=.83,bottom=.13,hspace=.6,wspace=.25)
    return fig


def save_figure(fig,stem):
    stem.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(stem.with_suffix('.png'),dpi=145)
    fig.savefig(stem.with_suffix('.pdf'))
    plt.close(fig)


def summary_figure(summaries):
    rs=[r for r in summaries if r['section']=='Final TPS-priority']
    columns=['Model','B','Method','AL*','Cache hit','TPS* r0 / r1']
    cells=[[r['model'].upper(),r['batch'],r['method'],f"{r['al']:.3f}",f"{100*r['cache_hit']:.1f}%",
            ' / '.join(f'{x:.2f}' for x in r['tps'])] for r in rs]
    fig,ax=plt.subplots(figsize=(11.5,5.4));ax.axis('off')
    table=ax.table(cellText=cells,colLabels=columns,cellLoc='center',bbox=[0,.13,1,.73],colWidths=[.16,.07,.12,.14,.18,.33])
    table.auto_set_font_size(False);table.set_fontsize(12)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if i==0:cell.set_facecolor('#26364c');cell.set_text_props(color='white',weight='bold')
        elif rs[i-1]['method']=='DUET':cell.set_facecolor('#eef6fb')
    ax.set_title('Final selected configurations: DUET and independently tuned SSD',fontsize=14,pad=10)
    fig.text(.5,.06,'Full 480 prompts | T=0.7 | RTX 4090 | input/output cap 512/128\n'
             'AL and hit pool event counts across two repeats; TPS is shown separately for each paired repeat.\n'
             '* Output-cap/clipping boundary exclusions. Raw returned TPS is preserved in the accompanying tables.',
             ha='center',fontsize=9)
    return fig
