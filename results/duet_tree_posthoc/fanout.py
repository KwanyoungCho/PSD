"""Exact expected benefit of 1/2 WOR siblings, integrating sampled tokens.

This uses full p/q from the deterministic 1/32 raw audit sample. No Monte Carlo,
no offline token-dependent pruning, and no claim that target p is free online.
"""
from collections import defaultdict
import itertools
import argparse
import json
from pathlib import Path
import numpy as np
from features import groups
from analyze import macro, interval
from calibrate import frozen_predict, from_edges
from features import extract

HERE=Path(__file__).resolve().parent


def gains(p,q):
    p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
    p=p/p.sum();q=q/q.sum()
    g1=float(np.minimum(p,q).sum());z=1-g1
    if z<1e-14:return g1,g1
    residual=np.maximum(p-q,0);residual/=residual.sum()
    support=q>0
    ratios=residual[support]/q[support]
    order=np.argsort(ratios);ratios=ratios[order]
    qq=q[support][order];rr=residual[support][order]
    rprefix=np.r_[0.,np.cumsum(rr)]
    # Suffix sums avoid losing tiny q tails via 1-cumsum cancellation.
    qsuffix=np.r_[np.cumsum(qq[::-1])[::-1],0.]
    ids=np.flatnonzero(q>p);weights=q[ids]-p[ids]
    denom=1-q[ids];normal=denom>1e-12
    overlap=np.zeros(len(ids))
    scale=1/denom[normal]
    pos=np.searchsorted(ratios,scale,side='right')
    overlap[normal]=rprefix[pos]+scale*qsuffix[pos]
    for ix in np.flatnonzero(~normal):
        d=q.copy();d[ids[ix]]=0;mass=d.sum()
        if mass>0:overlap[ix]=np.minimum(residual,d/mass).sum()
    g2=g1+float(np.dot(weights,np.clip(overlap,0,1)))
    if not g1-1e-10<=g2<=1+1e-10:raise ValueError('Invalid gain')
    return g1,min(1.,g2)


def brute(p,q):
    total=0.
    for first,second in itertools.permutations(range(len(p)),2):
        proposal=q[first]*q[second]/(1-q[first])
        a=min(1.,p[first]/q[first]);r=np.maximum(p-q,0)
        if r.sum()>0:r/=r.sum()
        d=q.copy();d[first]=0;d/=d.sum()
        b=min(1.,r[second]/d[second])
        total+=proposal*(a+(1-a)*b)
    return total


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only-policy',choices=['q_path','phase_sibling_q_bin']);args=ap.parse_args()
    rng=np.random.default_rng(25323);error=0.
    for vocab in range(2,9):
        for _ in range(50):
            p=rng.dirichlet(np.ones(vocab));q=rng.dirichlet(np.ones(vocab))
            error=max(error,abs(gains(p,q)[1]-brute(p,q)))
    assert error<1e-12
    assert np.allclose(gains([0,1],[.9,.1]),[.1,1.])
    # Exact synthetic counterexample: first-child breadth allocation gets .26;
    # allocating both to parent A gets .8, under the same two-new-node budget.
    a=gains([0,1],[.9,.1]);b=gains([0,1],[.1,.9])
    breadth=.8*a[0]+.2*b[0];opt=max(.8*a[1],breadth,.2*b[1])
    assert np.isclose(breadth,.26) and np.isclose(opt,.8)
    # Low-cost, deployable-input diagnostic: gain curves from the ORIGINAL
    # eight calibration prompts, with no new full-corpus fitting or neural head.
    old=HERE.parent/'duet_tree_analysis';manifest=json.loads((old/'probe/manifest.json').read_text())
    frozen=json.loads((old/'calibration_frozen.json').read_text());stats=defaultdict(lambda:np.zeros(3))
    for meta in manifest['snapshots']:
        if manifest['plan']['mapping'][meta['prompt']]['split']!='calibration':continue
        with np.load(old/'probe'/meta['file']) as z:
            t=extract(**dict(z))
            for parent,children in groups(t['par'],t['sib']).items():
                if parent<0:continue
                w=t['reach'][parent];g1,g2=gains(z['p'][parent+1],z['q'][children[0]])
                stats[meta['phase']]+=np.array([w*g1,w*g2,w])
    curves={phase:(v[:2]/v[2]).tolist() for phase,v in stats.items()}
    rows=[];allocation=[];phase_counts=defaultdict(int)
    for policy in ([args.only_policy] if args.only_policy else ['q_path','phase_sibling_q_bin']):
        folder=HERE/'runs'/policy
        for line in (folder/'trees.jsonl').open():
            t=json.loads(line)
            if t['is_final_event'] or 'raw_file' not in t:continue
            fr,_=from_edges(t,frozen_predict(t,frozen))
            with np.load(folder/t['raw_file']) as z:
                local={};bydepth=defaultdict(list)
                for parent,children in groups(t['par'],t['sib']).items():
                    g1,g2=gains(z['p'][parent+1],z['q'][children[0]])
                    rho=1. if parent<0 else t['reach'][parent]
                    depth=0 if parent<0 else t['depth'][parent]
                    meta={k:t[k] for k in ['question_id','group','phase','policy','serial','uid']}
                    row=dict(meta,parent=parent,depth=depth,rho=rho,g1=g1,g2=g2,
                        second_increment=g2-g1,nonconcave=float(g2>2*g1+1e-10),
                        weighted_first=rho*g1,weighted_second=rho*(g2-g1))
                    rows.append(row);local[parent]=row;bydepth[depth].append(parent)
                    phase_counts[t['phase']]+=1
                for depth,parents in bydepth.items():
                    if len(parents)<2:continue
                    # Compare each observed pair with exactly two new nodes and
                    # up to two parent forwards; no successor-depth advantage.
                    for u,v in itertools.combinations(parents,2):
                        ru,rv=local[u],local[v]
                        one_each=ru['rho']*ru['g1']+rv['rho']*rv['g1']
                        optimum=max(one_each,ru['rho']*ru['g2'],rv['rho']*rv['g2'])
                        # Choices = (1,1), (2,0), (0,2). All are selected before
                        # observing new child token identities.
                        truth=[one_each,ru['rho']*ru['g2'],rv['rho']*rv['g2']]
                        c1,c2=curves[t['phase']]
                        alternatives={
                            'q_curve':[(t['q_path'][u]+t['q_path'][v])*c1,t['q_path'][u]*c2,t['q_path'][v]*c2],
                            'frozen_curve':[(fr[u]+fr[v])*c1,fr[u]*c2,fr[v]*c2],
                            'rho_curve':[(ru['rho']+rv['rho'])*c1,ru['rho']*c2,rv['rho']*c2],
                            'frozen_oracle_g':[fr[u]*ru['g1']+fr[v]*rv['g1'],fr[u]*ru['g2'],fr[v]*rv['g2']]}
                        resultrow=dict(meta,depth=depth,parents=[u,v],breadth=one_each,oracle=optimum,
                            gap=optimum-one_each,changed=float(optimum>one_each+1e-10))
                        for name,values in alternatives.items():
                            choice=int(np.argmax(values));resultrow[name]=truth[choice]
                            resultrow[name+'_delta']=truth[choice]-one_each
                            resultrow[name+'_changed']=float(choice!=0)
                        allocation.append(resultrow)
    result=dict(scope='Exact prospective fanout diagnostic on 1/32 raw snapshots; known expanded contexts only.',
        verification=dict(random_small_vocab_cases=350,max_enumeration_error=error,
            two_node_example=dict(breadth=breadth,optimum=opt)),
        contexts=len(rows),questions=len({r['question_id'] for r in rows}),phase_counts=dict(phase_counts),
        original_8_prompt_gain_curves=curves,summary={})
    for phase in ['all',1,2]:
        rs=[r for r in rows if phase=='all' or r['phase']==phase]
        aa=[r for r in allocation if phase=='all' or r['phase']==phase]
        result['summary'][str(phase)]=dict(contexts=len(rs),
            means={k:macro(rs,k) for k in ['g1','g2','second_increment','nonconcave','weighted_first','weighted_second']},
            pooled_reach_weighted_nonconcave=sum(r['rho']*r['nonconcave'] for r in rs)/max(sum(r['rho'] for r in rs),1e-14),
            pair_comparisons=len(aa),allocation_means={k:macro(aa,k) for k in ['breadth','oracle','gap','changed',
                'q_curve','frozen_curve','rho_curve','frozen_oracle_g','frozen_curve_changed']},
            gap_ci=interval(aa,'gap'),curve_vs_breadth={k:interval(aa,k,'breadth')
                for k in ['q_curve','frozen_curve','rho_curve','frozen_oracle_g']})
    prefix=args.only_policy+'_' if args.only_policy else ''
    (HERE/(prefix+'fanout_analysis.json')).write_text(json.dumps(result,indent=2))
    (HERE/(prefix+'fanout_contexts.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (HERE/(prefix+'fanout_allocation.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in allocation))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
