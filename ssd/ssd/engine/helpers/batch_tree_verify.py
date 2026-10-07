"""Batched target tree verification and request-specific KV commits."""
from time import perf_counter
import torch
from ssd.engine.helpers.batch_tree_common import capacity
from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy, verify_batch, pack_topologies
from ssd.engine.helpers.p2_tree import parse_tree_ints, validate_tree_ints, commit_copy_plan
from ssd.engine.helpers.speculate_types import VerifyResult


@torch.inference_mode()
def verify(verifier,seqs,result):
    t0=perf_counter()
    r=verifier.target_model_runner
    cfg=r.config
    B=len(seqs)
    N=cfg.duet_response_token_width
    step_width=result.speculations.shape[1]-1
    valid=[s.verify_valid_k for s in seqs]
    wire=result.tree_ints.cpu()
    spec=result.speculations.cpu().tolist()
    phases=result.phase_source.cpu().tolist()
    hits=result.cache_hits.cpu().tolist()
    parents=[]; siblings=[]; is_tree=[]; rows=[]
    for b,s in enumerate(seqs):
        tree=int(wire[b,0])>0
        if tree:
            ti=parse_tree_ints(wire[b],cfg.duet_tree_wire_nodes)
            validate_tree_ints(ti,cfg.duet_tree_wire_nodes,cfg.hf_config.vocab_size,sibling_capacity=3)
            n=ti['valid']
            if n!=valid[b] or ti['epoch']!=2:
                raise RuntimeError('Batched tree wire version/length mismatch')
            par=ti['parent_local'][:n].tolist(); sib=ti['sib_order'][:n].tolist()
            if ti['tok'][:n].tolist()!=spec[b][1:n+1]:
                raise RuntimeError('Tree topology token payload mismatch')
        else:
            n=valid[b]; par=list(range(-1,n-1)); sib=[0]*n
        parents.append(par); siblings.append(sib); is_tree.append(tree)
        rows.append(dict(tokens=spec[b][:n+1],parents=par,
                         prefix=s.num_tokens-(step_width+1),blocks=list(s.block_table)))
    tt=[s.temperature for s in seqs]
    dt=[s.draft_temperature if s.draft_temperature is not None else s.temperature for s in seqs]
    if any(t>0 and d<=0 for t,d in zip(tt,dt)):
        raise NotImplementedError('Stochastic target with deterministic tree proposals requires a different law')
    if not hasattr(r,'_batch_tree_proxies'):
        r._batch_tree_proxies={}; r._batch_tree_proxy_pool=None
    cap=capacity(B)
    if cap not in r._batch_tree_proxies:
        p=BatchedTreeProxy(cap,N,cfg.hf_config.vocab_size,cfg.hf_config.torch_dtype,r.device,
                           cfg.duet_proxy_wire_N,max(cfg.duet_phase1_k,cfg.duet_phase2_k),
                           cfg.duet_proxy_top_k,pool=r._batch_tree_proxy_pool)
        r._batch_tree_proxies[cap]=p
        if r._batch_tree_proxy_pool is None: r._batch_tree_proxy_pool=p.graph.pool()
    proxy=r._batch_tree_proxies[cap]
    tokens=torch.zeros(B,N,dtype=torch.int64,device=r.device)
    tokens[:,:step_width].copy_(result.speculations[:,1:])
    q=torch.zeros(B,N,cfg.hf_config.vocab_size,dtype=result.logits_q.dtype,device=r.device)
    q[:,:result.logits_q.shape[1]].copy_(result.logits_q)
    proxy.prepare(parents,siblings,tokens,q)
    def callback(logits,b):
        pos,packed=proxy.replay(logits)
        verifier._send_proxy_wire(cfg,r.async_pg,r.draft_rank,pos[:B],packed[:B])
    r._duet_proxy_fn=callback
    try:
        logits=r.call('run_batched_tree',rows,N+1)
    finally:
        r._duet_proxy_fn=None
    topology={k:v[:B] for k,v in proxy.topology.items()}
    paths,recoveries,terminal=verify_batch(logits,q,tokens,topology,tt,dt,
                                          verifier.sampler_x,verifier.async_fan_out,
                                          greedy=all(t==0 for t in tt))
    result_cpu=torch.cat([paths,recoveries,terminal],1).cpu().tolist()
    suffixes=[]; recovery=[]; src=[]; dst=[]
    for b,s in enumerate(seqs):
        path=[j for j in result_cpu[b][:N] if j>=0]
        suffixes.append([spec[b][0]]+[spec[b][1+j] for j in path])
        recovery.append(result_cpu[b][N])
        s.tree_terminal_node=result_cpu[b][N+1] if is_tree[b] else None
        s.num_cached_tokens+=step_width+1
        if is_tree[b]:
            for a,z in commit_copy_plan(path,rows[b]['prefix'],s.block_table,r.block_size):
                src.append(a); dst.append(z)
    if src: r.call('commit_tree_kv',src,dst)
    m=verifier.metrics
    m['target_verify_times'].append(perf_counter()-t0)
    m['accepted_suffix_lens_with_recovery'].extend(map(len,suffixes))
    m['cache_hits'].append(sum(hits)/B)
    m['phase1_hits'].append(sum(p==1 for p in phases)/B)
    m['phase2_hits'].append(sum(p==2 for p in phases)/B)
    for b,suffix in enumerate(suffixes):
        length=len(suffix)
        m['phase_events'].append(dict(tree=is_tree[b],batch_size=B,verify_width=step_width,
            step_id=result.step_id,source=phases[b],cache_hit=int(hits[b]),
            accepted_len=length,accepted_spec_len=length-1,valid_k=valid[b]))
        m['accepted_suffix_lens_on_hit' if hits[b] else 'accepted_suffix_lens_on_miss'].append(length)
        if hits[b] and phases[b] in (1,2):
            m[f'accepted_lens_phase{phases[b]}_hit'].append(length-1)
    return VerifyResult(suffixes,recovery)
