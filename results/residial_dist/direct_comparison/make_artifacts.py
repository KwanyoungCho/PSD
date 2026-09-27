"""Render completed results without reselecting any policy."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'shared_review'))
from aggregate import load, estimate


def main():
    stats={phase:json.loads((HERE/(phase+'_summary.json')).read_text()) for phase in ('development','confirmation')}
    extra={}
    for phase in stats:
        d,_=load(HERE/phase);mask=np.ones(len(d['step']),bool)
        pairs={f'repair_{g:g}':(f'repair::{g:g}_residual',f'repair::{g:g}_proxy') for g in (0,.1,.25,.5,.75,1)}
        pairs.update(true_h=('global::residual_true_h','global::proxy_true_h'),
                     oracle_source_proxy_h=('global::oracle_residual_proxy_h','global::oracle_target_proxy_h'))
        extra[phase]={k:estimate(d,d[a]-d[b],mask) for k,(a,b) in pairs.items()}
    (HERE/'paired_diagnostics.json').write_text(json.dumps(extra,indent=2))
    rows=[]
    for k in stats['development']['policies']:
        row=dict(policy=k)
        for phase,s in stats.items():
            p=s['policies'][k];row[phase]=p['coverage']['mean']
            row[phase+'_gain_vs_proxy']=p['gain_vs_proxy']['mean']
            row[phase+'_gain_lo'],row[phase+'_gain_hi']=p['gain_vs_proxy']['ci95']
        rows.append(row)
    with (HERE/'policies.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    names=['Residual','Proxy only','Frozen 10% mix','Oracle switch','Exact ceiling']
    keys=['policy::residual','policy::proxy','best_static','global::oracle_switch','global::ceiling']
    s=stats['confirmation']['policies'];vals=np.array([s[k]['coverage']['mean'] for k in keys])*100
    intervals=np.array([s[k]['coverage']['ci95'] for k in keys]).T*100
    axes[0].barh(names,vals,color=['#356fa3','#ce6d41','#499780','#999999','#b9b9b9'])
    axes[0].errorbar(vals,np.arange(len(vals)),xerr=abs(intervals-vals),fmt='none',color='black',capsize=3)
    for j,v in enumerate(vals):axes[0].text(v+2,j,f'{v:.2f}',va='center',fontsize=9)
    axes[0].invert_yaxis();axes[0].set(xlim=(0,100),xlabel='Global coverage (%)',title='New prompts: paired comparison')
    gammas=np.array([0,.1,.25,.5,.75,1])
    for name,color in [('residual','#356fa3'),('proxy','#ce6d41')]:
        y=[s[f'repair::{g:g}_{name}']['coverage']['mean']*100 for g in gammas]
        axes[1].plot(gammas,y,'-o',color=color,label=name)
    axes[1].set(xlabel='Oracle repair fraction g',ylabel='Global coverage (%)',title='Both policies use the SAME repaired proxy')
    axes[1].text(.03,.97,'e(g) = (1-g)e + gp\nUses true target: not deployable',transform=axes[1].transAxes,va='top',fontsize=9)
    axes[1].legend(loc='lower right')
    groups=[stats[phase]['event_groups']['all']['means'] for phase in stats]
    x=np.arange(2);width=.24
    for shift,key,label,color in [(-width,'available_gain','Oracle headroom over proxy','#499780'),
                                 (0,'support_loss','Loss from removed support','#c37254'),
                                 (width,'rank_loss','Loss from ranking','#ba9a52')]:
        ys=[g[key]*100 for g in groups]
        axes[2].bar(x+shift,ys,width,label=label,color=color)
    axes[2].set(xticks=x,xticklabels=['Development','New prompts'],ylim=(0,20),ylabel='Local top-3 correction coverage (pp)',title='Residual wins if headroom > both losses')
    axes[2].legend(fontsize=8)
    fig.savefig(HERE/'comparison.png',dpi=180);fig.savefig(HERE/'comparison.pdf')
    plt.close(fig)

    # A compact report fragment keeps numerical tables reproducible.
    lines=['**새 프롬프트의 전역 후보 coverage**','',
           '| 정책 | Coverage | only-proxy 대비 차이 (95% CI), %p |',
           '|---|---:|---:|']
    for name,k in [('기존 residual','policy::residual'),('only-proxy','policy::proxy'),
                   ('기존 데이터에서 선택한 10% residual 혼합','best_static'),
                   ('단일 임계값 선택기','stump'),('Ridge 선택기','ridge'),
                   ('두 정책 중 정답 선택기 — oracle','global::oracle_switch')]:
        p=s[k];a=p['gain_vs_proxy'];lo,hi=a['ci95']
        lines.append(f'| {name} | {100*p["coverage"]["mean"]:.3f}% | {100*a["mean"]:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}] |')
    lines+=['','**새 프롬프트: 실제 proxy 오차를 target 방향으로 줄인 진단**','',
            '| 제거한 proxy 오차 비율 γ | Residual | Only-proxy | 차이 (95% CI), %p |',
            '|---:|---:|---:|---:|']
    for g in gammas:
        r=s[f'repair::{g:g}_residual']['coverage']['mean'];e=s[f'repair::{g:g}_proxy']['coverage']['mean']
        diff=extra['confirmation'][f'repair_{g:g}'];lo,hi=diff['ci95']
        lines.append(f'| {g:.0%} | {r:.3%} | {e:.3%} | {100*diff["mean"]:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}] |')
    lines+=['','**25개 고정 점수 정책: 탐색 결과를 모두 공개**','',
            '| 정책 | 기존 데이터 | 새 프롬프트 | 새 데이터에서 only-proxy 대비, %p |',
            '|---|---:|---:|---:|']
    for k in s:
        if k.startswith('policy::'):
            p=s[k];old=stats['development']['policies'][k]
            lines.append(f'| `{k[8:]}` | {old["coverage"]["mean"]:.3%} | {p["coverage"]["mean"]:.3%} | {100*p["gain_vs_proxy"]["mean"]:+.3f} |')
    (HERE/'TABLES.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:28]))


if __name__=='__main__':main()
