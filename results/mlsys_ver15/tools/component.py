"""Fixed-work latency checks, not an AL/TPS comparison between verifiers."""
import argparse,json,math,random,time
from pathlib import Path
from statistics import median
from types import SimpleNamespace as NS
import torch
from ssd.engine.helpers.score_tree import select_tree
from ssd.engine.helpers import score_tree
from ssd.engine.helpers.batch_tree_draft import BatchedDuetDraft
from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy,BatchedTreeAccept


def summary(values):
    values=sorted(values)
    return dict(median_ms=median(values),p95_ms=values[min(len(values)-1,math.ceil(.95*len(values))-1)],n=len(values))


def export_fixture(roots,inputs,siblings,dev):
    rng=random.Random(812+roots);n=roots+inputs*siblings
    tok=list(range(n));par=[-1]*roots;root=list(range(roots));sib=[0]*roots
    depth=[0]*roots;q=[1.]*roots;leaf=list(range(roots))
    for j in range(inputs):
        # Deliberately uneven root budgets, including roots with >M nodes.
        weights=[1/(1+root[v])**1.4 for v in leaf]
        idx=rng.choices(range(len(leaf)),weights=weights,k=1)[0];p=leaf.pop(idx)
        for s in range(siblings):
            v=len(par);par.append(p);root.append(root[p]);sib.append(s)
            depth.append(depth[p]+1);q.append(.05+.9*rng.random());leaf.append(v)
    ar=NS(capacity=n,tok=torch.tensor(tok,device=dev),root=torch.tensor(root,device=dev),
          parent_idx=torch.tensor(par,device=dev),sib=torch.tensor(sib,device=dev),
          raw_q=torch.tensor(q,device=dev,dtype=torch.float64),
          valid=torch.ones(n,dtype=torch.bool,device=dev),depth=torch.tensor(depth,device=dev),
          cell=torch.arange(n,device=dev))
    return NS(arena=ar,R=roots,F=max(depth),cell_logits=None)


@torch.inference_mode()
def run():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=60);args=p.parse_args()
    dev='cuda:0';torch.manual_seed(391)
    service=object.__new__(BatchedDuetDraft);service.entries={}
    service.cfg=NS(duet_p1_tree_policy='on',duet_p2_tree_policy='on',
                   duet_p1_tree_verify_nodes=23,duet_p2_tree_verify_nodes=15)
    records=[]
    for phase,roots,inputs in ((1,96,600),(2,24,140)):
        ex=export_fixture(roots,inputs,3,dev)
        records.append((ex,31,torch.zeros(roots,device=dev,dtype=torch.long),
                        10000*phase+torch.arange(roots,device=dev),phase,True))
    # Both arms perform the same arena export and host reconstruction. Only
    # final selection differs. Reference is benchmark-only; never served.
    def prefix(par,q,cap):
        k=min(len(par),cap);ss=[];counts={}
        for x in par[:k]:ss.append(counts.get(x,0));counts[x]=ss[-1]+1
        return list(range(k)),par[:k],ss
    from ssd.engine.helpers.score_tree_export import ScoreTreeExport
    selectors=[ScoreTreeExport(r[0],23 if r[4]==1 else 15) for r in records]
    export={'prefix_reference':[],'cpu_score_top_m':[],'gpu_score_top_m':[]}
    try:
        for iteration in range(args.repeats+10):
            order=list(export) if iteration%2==0 else list(export)[::-1]
            for name in order:
                score_tree.select_tree=prefix if name=='prefix_reference' else select_tree
                for record,selector in zip(records,selectors):
                    if name=='gpu_score_top_m':record[0].score_export=selector
                    elif hasattr(record[0],'score_export'):del record[0].score_export
                service.entries={};torch.cuda.synchronize();start=time.perf_counter()
                service._save_entries_bulk(records)
                ms=(time.perf_counter()-start)*1000
                if iteration>=10:export[name].append(ms)
    finally:score_tree.select_tree=select_tree
    score_tree.select_tree=select_tree
    for record in records:
        if hasattr(record[0],'score_export'):del record[0].score_export
    service.entries={};service._save_entries_bulk(records);reference=service.entries
    for record,selector in zip(records,selectors):record[0].score_export=selector
    service.entries={};service._save_entries_bulk(records)
    assert set(reference)==set(service.entries)
    for key,entry in reference.items():
        for field in ('tokens','parents','siblings','cells','generated_nodes'):
            assert entry[field]==service.entries[key][field],(key,field)
    selected=sum(len(e['tokens']) for e in service.entries.values())
    result=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,
        export_note='CPU wall ms; same bounded GPU arena export+CPU reconstruction; no pending model kernels. Max serving envelope fixture, not real model TPS.',
        export={k:summary(v) for k,v in export.items()},selected_nodes=selected,
        exported_arena_rows=sum(r[0].arena.capacity for r in records),
        cpu_gpu_selection_metadata_equal=True,accept={})
    result['export']['gpu_increment_vs_prefix_ms']=median(export['gpu_score_top_m'])-median(export['prefix_reference'])
    for n in (15,23):
        proxy=BatchedTreeProxy(1,n,32000,torch.bfloat16,dev,122,n,122,c_max=3)
        parents=[(j-1)//3 if j else -1 for j in range(n)]
        counts={};siblings=[]
        for par in parents:siblings.append(counts.get(par,0));counts[par]=siblings[-1]+1
        tokens=torch.arange(1,n+1,device=dev).view(1,n)
        q=torch.randn(1,n,32000,dtype=torch.bfloat16,device=dev)
        proxy.prepare([parents],[siblings],tokens,q,[.7],[.7])
        logits=torch.randn(1,n+1,32000,dtype=torch.bfloat16,device=dev)
        graphs={name:BatchedTreeAccept(proxy,None,5,False,stochastic_only=True,target_only=target)
                for name,target in (('original_q_reference',False),('target_sample_match',True))}
        values={name:[] for name in graphs}
        for it in range(args.repeats+10):
            for name in (list(graphs) if it%2==0 else list(graphs)[::-1]):
                # Inputs are fixed. Time graph replay only: no prewarm or D2H.
                g=graphs[name];g.logits.copy_(logits);g.tt.fill_(.7);g.dt.fill_(.7)
                a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                a.record();g.graph.replay();z.record();z.synchronize()
                if it>=10:values[name].append(a.elapsed_time(z))
        result['accept'][str(n)]={k:summary(v) for k,v in values.items()}
    result['accept_note']='Same B1 BF16 logits, V32000, shapes, full tree. GPU graph time only. Different algorithms/output coupling: no AL parity or speedup inferred.'
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':run()
