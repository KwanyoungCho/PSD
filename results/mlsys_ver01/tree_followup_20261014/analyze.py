"""Regenerate review summaries/figures from immutable campaign outputs."""
import argparse,json,sys,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'ssd/bench'))
from mlsys_tree_results import summarize,compare


def completed(folder):
    out={}
    for f in folder.glob('*.json'):
        j=json.loads(f.read_text())
        if isinstance(j,dict) and j.get('status')=='complete' and 'summary' in j:out[f.stem]=j
    return out


def main(root):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    all_results={}
    for path in sorted(root.rglob('campaign.json')):
        d=completed(path.parent)
        if not d:continue
        key=str(path.parent.relative_to(root))
        all_results[key]={'arms':{k:summarize(j) for k,j in d.items()},'comparisons':{}}
        pairs=[]
        if 'base' in d:pairs.extend(('base',k) for k in d if k not in ('base',) and not k.startswith('greedy'))
        for a,b in [('tuned_wait','tuned_m1'),('long_miss_wait','long_miss_m2'),('c2','c2_systems'),('c2_systems','c2_mixed'),('numpy','mixed'),('flat_prob','mixed'),('greedy_base','greedy_candidate'),('tp2_base_t0.0','tp2_candidate_t0.0'),('tp2_base_t0.7','tp2_candidate_t0.7')]:
            if a in d and b in d:pairs.append((a,b))
        for a,b in pairs:all_results[key]['comparisons'][a+' -> '+b]=compare(d[a],d[b])
    (HERE/'ALL_SUMMARIES.json').write_text(json.dumps(all_results,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    raw=json.loads((root/'remote/input_component_r2.json').read_text())
    rows=[r for r in raw['cells'] if r['batch']==8]
    x=np.arange(len(rows))
    for i,k in enumerate(['before','after']):
        vals=[statistics.median(r['samples_ms'][k]) for r in rows]
        axes[0].bar(x+(i-.5)*.32,vals,width=.32,label=k,color=['#8a98aa','#218c8d'][i])
    axes[0].set_xticks(x,[f"{r['contexts']} contexts" for r in rows]);axes[0].set_title('Draft input preparation\nRTX 3090, B=8');axes[0].set_ylabel('Function latency (ms)');axes[0].legend()
    raw=json.loads((root/'flat_component_bf16.json').read_text())
    row=next(r for r in raw['cells'] if r['vocab']==128256 and r['shape']=='half_short')
    for i,k in enumerate(['dense','flat']):
        vals=[row['median'][k+'_'+name]['cuda_ms'] for name in ['proxy','accept']]
        axes[1].bar(np.arange(2)+(i-.5)*.32,vals,width=.32,label=k,color=['#8a98aa','#218c8d'][i])
    axes[1].set_xticks([0,1],['Proxy score','Acceptance']);axes[1].set_title('Probability work only\nRTX 4090, BF16, B=8, Q: 72 → 48');axes[1].set_ylabel('CUDA graph latency (ms)');axes[1].legend()
    rows=[]
    for folder,tag in [('factorial','Llama2 / 4090'),('remote/factorial_l3_v2','Llama3 / 3090')]:
        vals={k:[] for k in ['gather','replica_stream']}
        for f in (root/folder).glob('*b8.json'):
            j=json.loads(f.read_text())
            for c in j.get('component',{}).get('cells',[]):
                if c['shape']=='uniform' and c['arm'] in vals:vals[c['arm']].append(c['median_ms'])
        if all(vals.values()):rows.append((tag,vals))
    for i,k in enumerate(['gather','replica_stream']):
        axes[2].bar(np.arange(len(rows))+(i-.5)*.32,[statistics.mean(r[1][k]) for r in rows],width=.32,label=k,color=['#8a98aa','#218c8d'][i])
    axes[2].set_xticks(range(len(rows)),[r[0] for r in rows]);axes[2].set_title('Full head + side stream\nTP=2, B=8, fixed inputs');axes[2].set_ylabel('Target + proxy latency (ms)');axes[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(HERE/'COMPONENTS.png',dpi=180);fig.savefig(HERE/'COMPONENTS.pdf');plt.close(fig)
    # Deliberately display request ownership, not a global cross-request mask.
    fig,ax=plt.subplots(figsize=(11,4));ax.set_xlim(0,11);ax.set_ylim(-1,4);ax.axis('off')
    colors=['#3e7cb1','#d4863d','#218c8d'];lengths=[4,2,1];offset=0
    ax.text(.1,3.5,'Packed query axis (one contiguous tensor)',fontsize=14,weight='bold')
    for req,(n,col,pages) in enumerate(zip(lengths,colors,['[5, 9]','[2, 18]','[11, 3]'])):
        label=chr(65+req)
        for i in range(n):
            ax.add_patch(plt.Rectangle((offset*.58,2.5),.54,.55,color=col));ax.text(offset*.58+.27,2.77,f'{label}{i}',ha='center',va='center',color='white');offset+=1
        y=1.6-req*.85
        ax.text(.1,y,f'Request {label}',color=col,weight='bold');ax.text(2,y,f'KV pages {pages}',color=col)
        ax.text(5,y,f'Mask: {label} prefix + ancestors in {label} tree',color=col)
    ax.text(.1,-.9,'Query storage is shared; attention ownership, KV pages, and RoPE positions remain request-specific.',fontsize=10)
    fig.tight_layout();fig.savefig(HERE/'PACKED_OWNERSHIP.png',dpi=180);plt.close(fig)
    print('campaigns',len(all_results),'complete runs',sum(len(v['arms']) for v in all_results.values()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,default=Path('/data/chokwans99/duet-tree-20261014'));main(p.parse_args().raw)
