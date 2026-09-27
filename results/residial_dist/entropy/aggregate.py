"""Reproduce entropy tables and figures from per-context NPZ observations."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def quantiles(x, w, qs=(.05, .10, .25, .50, .75, .90, .95)):
    order = np.argsort(x)
    sx, sw = x[order], w[order]
    good = sw > 0
    sx, sw = sx[good], sw[good]
    c = np.cumsum(sw, dtype=np.float64)
    return sx[np.minimum(np.searchsorted(c, np.asarray(qs)*c[-1]), len(sx)-1)]


def stats(x, w):
    total = w.sum()
    avg = lambda v: float(np.dot(v.astype(np.float64), w) / total)
    q = quantiles(x, w)
    out = dict(mean=avg(x), median=float(q[3]), mean_abs=avg(np.abs(x)),
               std=float(np.sqrt(max(0, avg(x*x)-avg(x)**2))),
               positive=avg(x > 0), negative=avg(x < 0),
               flatter_005=avg(x > .05), similar_005=avg(np.abs(x) <= .05),
               sharper_005=avg(x < -.05))
    out.update(above_one_nat=avg(x > 1), below_minus_one_nat=avg(x < -1))
    aq=quantiles(np.abs(x),w,qs=(.5,.9))
    out.update(median_abs=float(aq[0]),p90_abs=float(aq[1]))
    for mask,name in [(x>0,"mean_when_positive"),(x<0,"mean_when_negative")]:
        out[name] = float(np.average(x[mask],weights=w[mask])) if w[mask].sum()>0 else None
    out.update({f"p{int(p*100):02d}": float(v)
                for p, v in zip((.05,.10,.25,.50,.75,.90,.95), q)})
    return out


def load_run(done):
    record = json.loads(done.read_text())
    path = done.parent / record["manifest"]
    meta = json.loads(path.read_text())
    parts = []
    for c in meta["chunks"]:
        with np.load(path.parent / c["file"]) as z:
            parts.append({k: z[k] for k in z.files})
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert len(d["step"]) == meta["n_rows"]
    assert np.array_equal(np.unique(d["step"]), np.arange(meta["n_steps"]))
    hs = np.bincount(d["step"], weights=d["hazard"])
    np.testing.assert_allclose(hs, 1, atol=1e-6)
    reject = ~d["is_bonus"]
    rh = d["hazard"] * reject
    rs = np.bincount(d["step"], weights=rh)
    cw = np.divide(rh, rs[d["step"]], out=np.zeros_like(rh), where=rs[d["step"]]>0)
    d["reject_conditional_weight"] = cw
    d["delta"] = d["entropy_proxy"] - d["entropy_target"][:, None]
    d["dataset"] = np.full(len(reject), record["dataset"])
    d["seed"] = np.full(len(reject), record["seed"])
    return record, meta, d


def weights(d):
    reject = ~d["is_bonus"]
    return dict(positions=reject.astype(float),
                all_contexts=np.ones(len(reject),dtype=float),
                root=(d["position"]==0).astype(float),
                reject_event=(d["hazard"]*reject).astype(float),
                reject_conditional=d["reject_conditional_weight"].astype(float),
                bonus=d["is_bonus"].astype(float),
                recovery=d["hazard"].astype(float))


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--source", type=Path, default=root/"ssd/experiments/proxy_source_ablation/probe_entropy_20260910/out")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    done = sorted(args.source.glob("*.complete.json"))
    if not done:
        raise SystemExit("no validated complete runs")
    runs = [load_run(p) for p in done]
    layers = runs[0][1]["layers"]
    assert all(m["layers"] == layers and m["temperature"] == 1 for _,m,_ in runs)
    d = {k: np.concatenate([x[k] for _,_,x in runs]) for k in runs[0][2]}
    ws = weights(d)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scope in ["ALL", *sorted(set(d["dataset"]))]:
        mask = np.ones(len(d["step"]), dtype=bool) if scope=="ALL" else d["dataset"]==scope
        for name, weight in ws.items():
            w = weight[mask]
            for j,l in enumerate(layers):
                x = d["delta"][mask,j]
                s = stats(x, w)
                s.update(dataset=scope, weighting=name, layer=l,
                         n_rows=int(np.count_nonzero(w)), weight_sum=float(w.sum()),
                         entropy_target=float(np.average(d["entropy_target"][mask],weights=w)),
                         entropy_proxy=float(np.average(d["entropy_proxy"][mask,j],weights=w)))
                rows.append(s)
    with (out/"summary.csv").open("w") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lookup={(r['dataset'],r['weighting'],r['layer']):r for r in rows}
    # Per-run summaries expose the seed variation rather than treating rows
    # from one autoregressive sequence as independent repetitions.
    per_run=[]
    clusters={}
    for rec, meta, x in runs:
        j=layers.index(56); w=weights(x)['positions']
        s=stats(x['delta'][:,j],w)
        per_run.append(dict(dataset=rec['dataset'],seed=rec['seed'],**s))
        for seq in np.unique(x['seq_id']):
            mask=(x['seq_id']==seq)&(~x['is_bonus'])
            clusters.setdefault((rec['dataset'],int(seq)),[]).append(float(x['delta'][mask,j].mean()))
    # Descriptive cluster bootstrap: resample prompt IDs within each dataset,
    # preserving all available seeds per prompt. Equal dataset weighting.
    rng=np.random.default_rng(20260910)
    samples=[]
    for ds in sorted(set(d['dataset'])):
        a=np.array([np.mean(v) for (dataset,_),v in clusters.items() if dataset==ds])
        samples.append(rng.choice(a,size=(2000,len(a)),replace=True).mean(1))
    ci=np.quantile(np.mean(samples,axis=0),[.025,.975])
    prompt_macro=float(np.mean([np.mean([np.mean(v) for (dataset,_),v in clusters.items() if dataset==ds]) for ds in sorted(set(d['dataset']))]))
    checks=dict(runs=len(runs),sequences=sum(r['numseqs'] for r,_,_ in runs),
                distinct_dataset_prompt_ids=len(clusters),
                steps=sum(m['n_steps'] for _,m,_ in runs), rows=len(d['step']),
                reject_positions=int((~d['is_bonus']).sum()),
                observed_chain_lengths={str(int(k)): int(n) for k,n in
                    zip(*np.unique(d['position'][d['is_bonus']],return_counts=True))},
                max_prob_sum_error=max(m['max_prob_sum_error'] for _,m,_ in runs),
                prompt_macro_mean_l56=prompt_macro,prompt_bootstrap95_l56=ci.tolist(),
                n_zero_reject_steps=int(sum(np.count_nonzero(np.bincount(x['step'],weights=x['hazard']*(~x['is_bonus']))==0) for _,_,x in runs)),
                per_run_l56=per_run)
    if 79 in layers:
        delta_same_head=d['entropy_proxy'][:,layers.index(56)]-d['entropy_proxy'][:,layers.index(79)]
        checks['same_head_l56']={name:stats(delta_same_head,ws[name])
                                 for name in ['positions','root','reject_event']}
    (out/'validation.json').write_text(json.dumps(checks,indent=2))
    plt.rcParams.update({'font.size':11,'figure.facecolor':'white'})
    fig,(ax,zoom)=plt.subplots(1,2,figsize=(12,4.8))
    for weight,color,label in [('positions','#1f6feb','All draft positions'),('root','#b45309','Committed-prefix root'),('reject_event','#c0392b','True rejection weighted')]:
        a=[lookup['ALL',weight,l] for l in layers]
        ax.plot(layers,[v['mean'] for v in a],label=label,color=color,lw=2)
        zoom.plot(layers,[v['mean'] for v in a],label=label,color=color,lw=2)
    a=[lookup['ALL','positions',l] for l in layers]
    ax.fill_between(layers,[v['p10'] for v in a],[v['p90'] for v in a],color='#1f6feb',alpha=.12,label='Position P10-P90')
    ax.axhline(0,color='black',lw=1);ax.axvline(56,color='gray',ls=':')
    ax.set(xlabel='Proxy layer (zero-based)',ylabel='H(proxy) - H(target), nats',title='Entropy difference across proxy layers')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    zoom.axhline(0,color='black',lw=1);zoom.axvline(56,color='gray',ls=':')
    late=[lookup['ALL',w,l]['mean'] for w in ['positions','root','reject_event'] for l in layers if l>=48]
    zoom.set(xlim=(48,79),ylim=(min(late)-.08,max(late)+.08),
             xlabel='Proxy layer (zero-based)',ylabel='Mean entropy difference, nats',
             title='Late layers: mean difference can change sign')
    zoom.grid(alpha=.2);zoom.legend(fontsize=8);fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(out/f'01_layers.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    j=layers.index(56)
    for weight,color,label in [('positions','#1f6feb','All draft positions'),('root','#b45309','Committed-prefix root'),('reject_event','#c0392b','True rejection weighted')]:
        x=d['delta'][:,j];w=ws[weight];valid=w>0
        order=np.argsort(x[valid]);sx=x[valid][order];sw=w[valid][order]
        axes[0].plot(sx,np.cumsum(sw)/sw.sum(),color=color,label=label)
    axes[0].axvline(0,color='black',lw=1);axes[0].set(xlim=(-3,3),xlabel='Entropy difference, nats',ylabel='Cumulative share',title='Layer 56: full empirical CDF (central range)')
    axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
    datasets=sorted(set(d['dataset']))
    bottom=np.zeros(len(datasets))
    for key,color,label in [('sharper_005','#2563eb','Sharper (< -0.05)'),('similar_005','#9ca3af','Within +/-0.05'),('flatter_005','#d97706','Flatter (> +0.05)')]:
        vals=np.array([lookup[ds,'positions',56][key] for ds in datasets])
        axes[1].bar(datasets,vals,bottom=bottom,color=color,label=label)
        for k,v in enumerate(vals):
            if v>.06: axes[1].text(k,bottom[k]+v/2,f'{100*v:.1f}%',ha='center',va='center',color='white',fontsize=10)
        bottom+=vals
    axes[1].set(ylabel='Share of draft positions',ylim=(0,1),title='Layer 56: both signs occur')
    axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=1)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'02_distribution.{ext}',dpi=160)
    plt.close(fig)
    def row(r):
        return f"{r['mean']:+.4f} | {r['median']:+.4f} | {r['p10']:+.4f} ~ {r['p90']:+.4f} | {r['mean_abs']:.4f} / {r['median_abs']:.4f} | {100*r['sharper_005']:.1f}% | {100*r['similar_005']:.1f}% | {100*r['flatter_005']:.1f}%"
    primary=lookup['ALL','positions',56]
    text=["# Proxy–target entropy difference: per-context measurement", "",
          "`ΔH = H(pE) − H(pT)`, 자연로그 기준 nats. 양수는 proxy의 엔트로피가 더 큼.","",
          f"검증 완료 {len(runs)}런, {checks['sequences']}개 생성, {len(clusters)}개 dataset–prompt ID, {checks['steps']:,} step, {checks['rows']:,}개 context.",
          "70B AWQ(calibrated) + TinyLlama, B=1, T=1, only-proxy chain, exit=56, K1/K2=8/4, P2 budget=15. 모든 80개 layer를 같은 context에서 비교.","",
          f"실제 관찰된 verify chain 길이 K와 step 수: `{checks['observed_chain_lengths']}`. K1 설정값과 각 step의 실제 K는 구분한다.","",
          f"Layer 56에서 draft 위치를 동일 가중하면 target의 평균 엔트로피는 {primary['entropy_target']:.4f}, proxy는 {primary['entropy_proxy']:.4f} nats다. 평균 차이는 **{primary['mean']:+.4f} nats**, 절대차 중앙값은 **{primary['median_abs']:.4f} nats**다. 평균적인 평탄화 방향과 개별 위치에서의 변화 크기는 구분해야 한다.","",
          f"엄격한 부호 기준으로 증가 위치는 {100*primary['positive']:.1f}%이며 그 안의 평균 차이는 {primary['mean_when_positive']:+.4f} nats, 감소 위치는 {100*primary['negative']:.1f}%이며 그 안의 평균 차이는 {primary['mean_when_negative']:+.4f} nats다. 증가와 감소가 섞여 있고, 증가량이 더 커서 전체 평균이 양수가 된다.","",
          "## Layer 56", "",
          "행을 합쳐 집계한 경험분포. `positions`는 bonus를 제외한 모든 draft 위치를 동일 가중하며, 실제 거절 이후의 반사실적 draft context도 포함한다. `root`는 각 step의 확정된 prefix 뒤 첫 위치만 사용한다.","",
          "| 집계 | 평균 | 중앙값 | P10 ~ P90 | 절대차 평균 / 중앙값 | 더 뾰족 | 유사 | 더 평평 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ['positions','all_contexts','root','reject_event','reject_conditional','bonus','recovery']:
        text.append(f"| {name} | {row(lookup['ALL',name,56])} |")
    text += ["", "유사 구간은 `|ΔH|≤0.05 nats`라는 서술용 기준이며 유의성 검정이 아니다. 엄격한 양수/음수 비율은 summary.csv에 함께 기록.",
             "`all_contexts`는 bonus까지 포함한 모든 위치를 동일 가중. `reject_event`는 실제 첫 기각 확률 h_i로 가중. `reject_conditional`은 step별 기각확률을 먼저 정규화한 뒤 유효 가중치 합으로 나눔. `recovery`는 bonus까지 포함한 h_i 가중.","",
             f"프롬프트별 평균을 낸 뒤 seed를 평균하고 dataset을 동일 가중한 layer-56 평균: **{prompt_macro:+.4f} nats**. Dataset 내 prompt 단위 bootstrap 95% 구간: **[{ci[0]:+.4f}, {ci[1]:+.4f}]** (2,000회). 토큰을 독립 표본으로 bootstrap하지 않음.","",
             "## 데이터셋별 layer 56 (positions)","",
             "| 데이터셋 | 평균 | 중앙값 | P10 ~ P90 | 절대차 평균 / 중앙값 | 더 뾰족 | 유사 | 더 평평 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    text += [f"| {ds} | {row(lookup[ds,'positions',56])} |" for ds in datasets]
    text += ["", "## 레이어별 (positions)","",
             "| Layer | 평균 | 중앙값 | P10 ~ P90 | 절대차 평균 / 중앙값 | 더 뾰족 | 유사 | 더 평평 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    text += [f"| {l} | {row(lookup['ALL','positions',l])} |" for l in [0,16,32,40,48,56,64,72,74,75,76,77,78,79] if l in layers]
    text += ["", "![](01_layers.png)", "", "![](02_distribution.png)", "", "## 검증 및 범위", "",
             "- 전체 어휘 32,000개를 사용. 토큰 제외, top-k 절단, residual 정규화 전의 pE/pT 자체를 비교.",
             "- rank0 full lm_head replica로 pE 계산; 실제 TP 최종 logits로 pT 계산. layer 56 hidden tap의 엔진 일치 확인.",
             "- uniform/delta 분포의 ±log(V), 동일 분포의 0, 가변 K, RNG 불변, 마지막 부분 chunk 저장을 검사.",
             f"- 확률합 최대 오차 {checks['max_prob_sum_error']:.3g}; 모든 step의 h_i 합≈1 및 contiguous step 확인. 기각질량 0인 step {checks['n_zero_reject_steps']:,}개는 기각 가중 집계의 분모에서도 제외.",
             "- layer 79 차이는 head 연산 경로의 수치 차이를 평가하는 기준. 숫자가 작아도 엄격한 부호 비율은 0이 아닐 수 있음.",
             f"- 각 dataset 파일의 앞 {'/'.join(map(str,sorted({r['numseqs'] for r,_,_ in runs})))}개 프롬프트를 사용하며 seed는 생성 난수만 바꿈. 이 모델·temperature·only-proxy 궤적에 대한 결과이며 champion/TPS나 전체 언어 모델로 일반화하지 않음.",
             "- 모든 개별 scalar 관측치는 원시 NPZ에 보존. 종료 시 최종 부분 chunk도 flush. complete.json이 있는 검증된 런만 집계.","",
             "분위수는 가중 경험 CDF의 역함수로 계산한다. summary.csv의 ALL은 모든 관측 행을 합친 결과이며, prompt bootstrap의 동일 프롬프트·dataset 가중 평균과 분모가 다르다.","",
             "재생성: `OPENBLAS_NUM_THREADS=1 ssd/.venv/bin/python results/residial_dist/entropy/aggregate.py`", ""]
    if 'same_head_l56' in checks:
        text += ["## 동일 head 연산 경로를 사용한 통제", "",
                 "추가 통제는 layer 56과 layer 79의 replica-head 엔트로피를 직접 빼서, 실제 TP head와 replica head 사이의 수치 차이를 제거한다. 주 결과의 참조는 여전히 실제 엔진의 최종 분포이며, 아래는 통제용이다.","",
                 "| 참조 및 집계 | 평균 | 중앙값 | P10 ~ P90 | 절대차 평균 / 중앙값 | 더 뾰족 | 유사 | 더 평평 |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in ['positions','root','reject_event']:
            text.append(f"| TP final, {name} | {row(lookup['ALL',name,56])} |")
            text.append(f"| Replica final, {name} | {row(checks['same_head_l56'][name])} |")
        text.append("")
    (out/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(checks,indent=2))
    print('L56 positions',json.dumps(lookup['ALL','positions',56]))
    print('report',out/'REPORT.md')


if __name__ == '__main__':
    main()
