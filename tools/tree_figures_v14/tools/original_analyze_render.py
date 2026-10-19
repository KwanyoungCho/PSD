"""Reproducible single-panel figures; exact population/counts kept in JSON."""
from collections import defaultdict,Counter
from pathlib import Path
from statistics import mean
import argparse,gzip,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parents[1]
RAW=Path('/data/chokwans99/duet-v14-tree-figures-20261019')
ARMS=('chain','equal_tree','duet')


def gz(path):
    if not path.exists():return []
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]


def question_means(by_turn,identity):
    grouped=defaultdict(list)
    for seq,values in by_turn.items():
        grouped[identity[seq]].append(np.mean(values,axis=0))
    return {q:np.mean(turns,axis=0).tolist() for q,turns in grouped.items()}


def summarize(tag):
    folder=RAW/'runs'/tag
    status=json.loads((folder/'status.json').read_text())
    if status['status']!='complete' or not status['source_unchanged']:
        raise RuntimeError('Unvalidated inference cannot enter figures')
    d=json.loads((folder/'run.json').read_text())
    if d['paper_commit']!='dedc85ccb299df2d76fcf34d50f05f2d8747a32a' or d['paper_dirty']:
        raise RuntimeError('Native v1.4 provenance mismatch')
    if d['args']['output_tokens']==1024:
        if len(d['warm_groups'])+len(d['groups'])!=560:raise RuntimeError('Not full Spec-Bench')
        inputs=json.loads((RAW/'INPUTS_FULL.json').read_text())
        if d['prompt_ids']!=[x['input_ids'] for x in inputs]:raise RuntimeError('Input IDs/order changed')
        baseline=Path('/data/chokwans99/duet-v14-residual-review-20261019/runs')/f"full_native_v14_s{d['args']['seed']}_v3/run.json"
        reference=json.loads(baseline.read_text())
        ga=reference['warm_groups']+reference['groups'];gb=d['warm_groups']+d['groups']
        if [g['outputs'] for g in ga]!=[g['outputs'] for g in gb]:
            raise RuntimeError('Full-corpus shadow observation changed native outputs')
        signature=lambda gs:[[(e['source'],e['accepted_len']) for s in g['steps'] for e in s['events']] for g in gs]
        if signature(ga)!=signature(gb):raise RuntimeError('Full-corpus native source/AL parity differs')
    if not (d['kwargs']['dtype']=='bfloat16' and d['args']['batch']==1 and d['args']['temperature']==.7):
        raise RuntimeError('Model/temperature contract changed')
    identity={};eligible={};steps=0;prefixes={};native_al={}
    for group in d['groups']:
        record=group['request_records'][0]
        seq=record['seq_id'];identity[seq]=str(record['question_id'])
        emitted=0
        for s in group['steps']:
            e=s['events'][0];key=(seq,e['step_id']);steps+=1
            enough=record['effective_output_cap']-emitted>=11
            prefixes[key]=record['input_tokens']+emitted
            native_al[key]=e['accepted_len']
            eligible[key]=not(s['cap'] or s['clipped']) and enough
            emitted+=s['emitted']
    allocation=gz(folder/'allocation.jsonl.gz')
    choices=gz(folder/'choices.jsonl.gz');paired=gz(folder/'paired.jsonl.gz')
    result=dict(tag=tag,seed=d['args']['seed'],full=d['args']['output_tokens']==1024,
                input_turns=len(d['warm_groups'])+len(d['groups']),
                measured_questions=len(set(identity.values())),native_steps=steps,phases={})
    for phase in (1,2):
        rank=defaultdict(list);contrast=defaultdict(list);als=defaultdict(list);raw_al=defaultdict(list)
        survival=defaultdict(list);coverage=defaultdict(list);nforests=0;nhits=0;native_hits=0;reasons=Counter()
        for row in allocation:
            seq=row['seq']
            if row['phase']!=phase or seq not in identity or not row['eligible']:continue
            rr=[r for _,r in sorted(enumerate(row['roots']),key=lambda pair:(-pair[1]['score'],pair[0]))]
            if sum(r['inputs'] for r in rr)!=100:raise RuntimeError('Not 100 valid draft inputs')
            rank[seq].append([r['inputs'] for r in rr]);nforests+=1
        for row in choices:
            key=row['seq'],row['step']
            if row['source']!=phase or key not in eligible or not eligible[key]:continue
            native_hits+=1
            if row.get('reason')!='not_sampled_generation':
                coverage[row['seq']].append(float(row['eligible']))
            if not row['eligible']:reasons[row.get('reason','unknown')]+=1
        for row in paired:
            seq=row['seq'];key=seq,row['step']
            if key in prefixes and (row['prefix']!=prefixes[key] or row['native_al']!=native_al[key]):
                raise RuntimeError('Shadow prefix/native event join differs')
            if row['phase']!=phase or key not in eligible or not eligible[key]:continue
            b=row['allocation']['roots'][row['root']]['inputs']
            contrast[seq].append([b,(100-b)/9])
            als[seq].append([row['outcomes'][a]['expected_usable_al'] for a in ARMS])
            raw_al[seq].append([row['outcomes'][a]['expected_al'] for a in ARMS])
            survival[seq].append([row['outcomes'][a]['survival'] for a in ARMS]);nhits+=1
        result['phases'][str(phase)]=dict(
            forests=nforests,paired_hits=nhits,native_hit_steps=native_hits,
            paired_hit_coverage=nhits/native_hits if native_hits else None,
            excluded_reasons=dict(reasons),rank=question_means(rank,identity),
            sampled_generation_hits=native_hits-reasons.get('not_sampled_generation',0),
            coverage=question_means(coverage,identity),
            contrast=question_means(contrast,identity),al=question_means(als,identity),
            raw_al=question_means(raw_al,identity),survival=question_means(survival,identity))
        result['phases'][str(phase)]['paired_delta']={
            q:[v[2]-v[1],v[2]-v[0]] for q,v in result['phases'][str(phase)]['al'].items()}
    (HERE/(tag+'_SUMMARY.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def estimate(results,phase,key):
    maps=[r['phases'][str(phase)][key] for r in results]
    maps=[x for x in maps if x]
    if not maps:return None
    questions=sorted(set().union(*(x.keys() for x in maps)))
    shape=np.asarray(next(iter(maps[0].values()))).shape
    a=np.full((len(maps),len(questions),*shape),np.nan)
    for s,values in enumerate(maps):
        for j,q in enumerate(questions):
            if q in values:a[s,j]=values[q]
    point=np.nanmean(a,axis=1).mean(axis=0)
    rng=np.random.default_rng(90371)
    boot=[]
    for _ in range(1000):
        idx=rng.integers(0,len(questions),len(questions))
        v=np.nanmean(a[:,idx],axis=1)
        if np.isfinite(v).all():boot.append(v.mean(axis=0))
    lo,hi=np.quantile(boot,[.025,.975],axis=0)
    return dict(mean=point.tolist(),lo=lo.tolist(),hi=hi.tolist(),questions=[len(x) for x in maps])


def render(results):
    full=len(results)==3 and all(r['full'] for r in results)
    out=HERE/('paper' if full else 'preview');out.mkdir(exist_ok=True)
    stats={str(p):{k:estimate(results,p,k) for k in ('rank','contrast','al','raw_al','paired_delta','coverage')} for p in (1,2)}
    metadata=dict(state='complete' if full else 'preliminary',seeds=[r['seed'] for r in results],
        full_dataset=all(r['full'] for r in results),samples=results,statistics=stats,
        native_verifier='original ordered WOR + recursive residual',
        counterfactual='same prefix/root, acceptance coins integrated exactly; natural EOS stops usable AL',
        aggregation='steps within turn, turns within question, questions within seed, equal seeds',
        uncertainty='paired question bootstrap, same question draws carried across seeds; not a hardware/TPS measurement')
    (out/'NUMBERS.json').write_text(json.dumps(metadata,indent=2)+'\n')
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':330})
    title='' if full else 'PRELIMINARY'
    note=('Full Spec-Bench; 3 seeds.' if full else
          f'Full Spec-Bench; {len(results)}/3 seeds, preliminary.' if all(r['full'] for r in results)
          else 'Small preflight sample; not final results.')
    colors={1:'#2166ac',2:'#b2182b'};figures=[]
    def create():
        fig,ax=plt.subplots(figsize=(3.8,3.05));ax.grid(axis='y',color='#ddd',lw=.45,zorder=0)
        return fig,ax
    def save(fig,name):
        fig.text(.5,.025,note+' R=W=K=10; no pruning.',ha='center',fontsize=6.8)
        fig.tight_layout(rect=(0,.075,1,1));fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.png'));figures.append(fig)
    fig,ax=create()
    for p in (1,2):
        s=stats[str(p)]['rank']
        if s is None:continue
        x=np.arange(1,11);ax.plot(x,s['mean'],marker='o' if p==1 else 's',markersize=3,color=colors[p],label=f'P{p}')
        ax.fill_between(x,s['lo'],s['hi'],alpha=.12,color=colors[p])
    ax.axhline(10,color='#777',ls='--',label='Equal allocation')
    ax.set(xlabel='Candidate rank by root score',ylabel='Draft input budget (%)',xticks=[1,2,4,6,8,10],title=title)
    ax.legend(frameon=False,fontsize=8);save(fig,'FIG1_rank_budget')
    fig,ax=create()
    for p,x in ((1,0),(2,1)):
        s=stats[str(p)]['contrast']
        if s is None:continue
        for j,shift in ((0,-.17),(1,.17)):
            value=s['mean'][j]
            ax.bar(x+shift,value,.30,color=colors[p] if j==0 else 'white',
                   edgecolor=colors[p],hatch=None if j==0 else '////',zorder=3)
            ax.errorbar(x+shift,value,yerr=[[value-s['lo'][j]],[s['hi'][j]-value]],fmt='none',c='black',capsize=2,lw=.8)
            ax.text(x+shift,max(value,s['hi'][j])+.6,f'{value:.1f}',ha='center',fontsize=8)
    ax.axhline(10,color='#777',ls='--',lw=1)
    ax.set(xticks=[0,1],xticklabels=['P1','P2'],ylabel='Draft input rows per root',title=title)
    ax.legend(handles=[Patch(facecolor='#444',label='Actual hit root'),Patch(facecolor='white',edgecolor='#444',hatch='////',label='Other roots (mean)')],frameon=False,fontsize=7.3)
    ax.margins(y=.2);save(fig,'FIG2_hit_budget')
    fig,ax=create();palette=['#c9c9c9','#6baed6','#d35f5f']
    for p,x in ((1,0),(2,1)):
        s=stats[str(p)]['al']
        if s is None:continue
        for j,shift in enumerate((-.24,0,.24)):
            value=s['mean'][j];ax.bar(x+shift,value,.215,color=palette[j],edgecolor='#444',lw=.5,zorder=3)
            ax.errorbar(x+shift,value,yerr=[[max(0,value-s['lo'][j])],[max(0,s['hi'][j]-value)]],fmt='none',c='black',capsize=2,lw=.8)
            ax.text(x+shift,max(value,s['hi'][j])+.07,f'{value:.2f}',ha='center',fontsize=7.5)
    ax.set(xticks=[0,1],xticklabels=['P1 hit','P2 hit'],ylabel='Conditional expected AL (tokens)',title=title)
    ax.legend(handles=[Patch(facecolor=c,label=label) for c,label in zip(palette,('Chain','Equal-budget tree','DUET tree'))],frameon=False,fontsize=7.2)
    ax.margins(y=.22);save(fig,'FIG3_hit_AL')
    with PdfPages(out/'FIGURES_1_2_3.pdf') as pdf:
        for fig in figures:pdf.savefig(fig)
    plt.close('all')
    lines=['# ver1.4 tree 예산 배분 실험','',f"상태: **{metadata['state']}**, 완료 seed: {metadata['seeds']}",'',
        '기존 native B1 q/WOR·residual 검증 유지. Shadow 비교는 R=W=K10, tree C3, G=M300으로 상한·깊이 예약 비활성. '
        'P1은 native 후보 중 위치 균등 quota, P2는 native DS 중복 제거 후 점수 상위10을 hit 전에 고정한다. '
        '서로 다른 native 후보 모집단에서 뽑은 조건부 비교이며 cache hit 개선을 주장하는 실험이 아니다.','',
        '[그림1·2·3 PDF]('+('paper' if full else 'preview')+'/FIGURES_1_2_3.pdf)','',
        '| Phase | 준비 forest | 비교 hit | 문항 평균 포함률 | Chain AL | 균등 Tree AL | DUET AL |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in (1,2):
        rr=[r['phases'][str(p)] for r in results];s=stats[str(p)]['al']
        hit=sum(x['paired_hits'] for x in rr);native=sum(x['sampled_generation_hits'] for x in rr)
        av=' | '.join(f'{x:.3f}' for x in s['mean']) if s else '— | — | —'
        cov=stats[str(p)]['coverage']
        lines.append(f'| P{p} | {sum(x["forests"] for x in rr)} | {hit} | {100*cov["mean"] if cov else 0:.1f}% | {av} |')
    lines+=['','| Phase | DUET−균등 Tree (95% CI) | DUET−Chain (95% CI) |',
            '|---|---:|---:|']
    for p in (1,2):
        s=stats[str(p)]['paired_delta']
        if s:
            cells=[f"{s['mean'][j]:+.3f} [{s['lo'][j]:+.3f}, {s['hi'][j]:+.3f}]" for j in range(2)]
            lines.append(f'| P{p} | '+ ' | '.join(cells)+' |')
    lines+=['','AL은 동일 문맥·실제 hit root에서 원래 residual verifier의 검증 난수를 적분한 값이다. '
        '자연 EOS에서 끝나는 usable AL을 주값으로 표시하며 원래 walk 전체 AL도 JSON에 보존한다. '
        'cap/clip 및 출력 잔여11토큰 미만은 공통 제외, latency outlier는 남긴다. '
        '긴 입력의 공통 scratch 부적합 및 후보10개 밖의 hit를 이유별 공개한다. '
        '품질은 턴→문항→seed 평균이며, CI는 문항을 seed 사이 함께 재추출한 bootstrap이다. '
        '계측 시간이 포함되므로 TPS/latency 성능으로 해석하지 않는다.','',
        '전체 결과 확정 전에는 PRELIMINARY 그림이며, 원본 실행은 Spec-Bench480문항/560턴·출력1024를 유지한다. '
        '관측할 generation step 간격은 실행 provenance와 PROTOCOL.json에 고정한다.','']
    tmp=HERE/'REPORT.analysis.tmp';tmp.write_text('\n'.join(lines));tmp.replace(HERE/'REPORT.md')
    print(json.dumps(dict(state=metadata['state'],output=str(out))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('tags',nargs='+');args=parser.parse_args()
    render([summarize(tag) for tag in args.tags])
