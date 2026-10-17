"""Separate profile-OFF effects from profile-ON gap attribution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
VARIANTS=('send','early','prepack','hostlist')
LABELS=('Async send','Early tokens + async','Ahead-of-request topology','Host integer parsing')


def main():
    data=json.loads((HERE/'STEP_COMPARISONS.json').read_text())['comparisons']
    keyed={(r['model'],r['variant']):r for r in data if r['cohort'].startswith('2x')}
    assert len(keyed)==7 and ('13b','hostlist') not in keyed,'Wait for all valid repeated comparisons'
    fig,axes=plt.subplots(2,1,figsize=(10.5,7.5),sharex=True)
    for ax,model in zip(axes,('7b','13b')):
        for x,variant in enumerate(VARIANTS):
            if (model,variant) not in keyed:
                ax.plot(x,0,'x',color='#aa3333',markersize=8)
                ax.annotate('Excluded:\noutput/event mismatch',(x,0),xytext=(0,14),
                            textcoords='offset points',ha='center',fontsize=9,color='#aa3333')
                continue
            r=keyed[model,variant];lo,hi=r['reduction_pct_ci95'];v=r['reduction_pct']
            ax.errorbar(x,v,yerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',capsize=6,
                        color='#176b4c' if lo>0 and all(z['reduction_pct']>0 for z in r['repeats']) else '#666666',markersize=7)
            ax.annotate(f'{r["before_ms"]:.3f} → {r["after_ms"]:.3f} ms\n{v:+.2f}%',
                        (x,v),xytext=(0,14),textcoords='offset points',ha='center',fontsize=9)
        ax.axhline(0,color='black',linewidth=.9);ax.grid(axis='y',alpha=.25)
        ax.set_xlim(-.5,3.5)
        ax.set_ylabel('Step latency decrease (%)')
        ax.set_title('LayerSkip Llama2-'+('7B' if model=='7b' else '13B')+' + llama-68m, B=1')
        ymin,ymax=ax.get_ylim();ax.set_ylim(ymin-.3,ymax+.7)
    axes[-1].set_xticks(range(4),LABELS,fontsize=9)
    fig.suptitle('Profile OFF: complete decode-step latency, exact output/AL/hit parity',fontsize=12)
    fig.text(.04,.01,'Two process repetitions, 12 fixed inputs × 2 passes/process. Error bars: UID-cluster bootstrap 95% CI.\n'
             'Cap/clip and 2× median filtering applied symmetrically; matched retained steps. Host parsing has its own ABBA baseline.',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,.95]);fig.savefig(HERE/'LATENCY_COMPARISON.pdf');fig.savefig(HERE/'LATENCY_COMPARISON.png',dpi=170);plt.close(fig)

    profiles=json.loads((HERE/'PROFILES.json').read_text())['runs']
    keyed={r['tag']:next(s for s in r['summary'] if s['state']=='all') for r in profiles}
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,model in zip(axes,('7b','13b')):
        rows=[keyed[f'{v}_{model}_profile'] for v in ('before',*VARIANTS)]
        response=np.array([r['response_ms'] for r in rows]);after=np.array([r['response_to_forward_ms'] for r in rows])
        before=np.array([r['whole_gap_ms'] for r in rows])-response-after
        x=np.arange(5)
        ax.bar(x,before,label='Previous logits → request send end',color='#536a8b')
        ax.bar(x,response,bottom=before,label='Response wait',color='#df9d42')
        ax.bar(x,after,bottom=before+response,label='Response → next forward',color='#5d9a80')
        for xx,r in zip(x,rows):ax.text(xx,r['whole_gap_ms']+.07,f'{r["whole_gap_ms"]:.2f}',ha='center',fontsize=9)
        ax.set_xticks(x,['ver1.3','Async','Early','Prepack','Host ints' if model=='7b' else 'Host ints*'],rotation=20)
        ax.set_title('Llama2-'+('7B' if model=='7b' else '13B')+' +68m')
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Target forward gap (ms), profile ON')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('Native timeline diagnosis; *13B host parsing failed full-run parity (not an adoption claim)')
    fig.tight_layout(rect=[0,.08,1,.94]);fig.savefig(HERE/'PROFILE_BREAKDOWN.pdf');fig.savefig(HERE/'PROFILE_BREAKDOWN.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
