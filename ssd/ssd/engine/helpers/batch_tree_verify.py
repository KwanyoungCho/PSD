"""Batched target tree verification and request-specific KV commits."""
from ssd.engine.helpers.root_policy import options as root_options
from time import perf_counter
import os
import torch
from ssd.engine.helpers.batch_tree_common import capacity
from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy, BatchedTreeAccept, verify_batch
from ssd.engine.helpers.p2_tree import parse_tree_ints, validate_tree_ints, commit_copy_plan
from ssd.engine.helpers.speculate_types import VerifyResult
from ssd.engine.helpers.cudagraph_helpers import duet_record, duet_close

DETAIL_TREE = os.getenv('SSD_PROFILE_DUET_DETAIL', '0') == '1'


@torch.inference_mode()
def verify(verifier,seqs,result):
    t0=perf_counter()
    prepare_event=duet_record('batch_tree_prepare') if DETAIL_TREE else None
    r=verifier.target_model_runner
    cfg=r.config
    B=len(seqs)
    step_width=result.speculations.shape[1]-1
    N=cfg.duet_response_token_width
    if os.getenv('SSD_BATCH_TREE_FIXED_VERIFY','0')!='1':
        widths={cfg.duet_phase1_k,cfg.duet_phase2_k,N}
        N=next(n for n in sorted(widths) if n>=step_width)
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
            validate_tree_ints(ti,cfg.duet_tree_wire_nodes,cfg.hf_config.vocab_size,
                               sibling_capacity=cfg.duet_tree_c_tensor)
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
    proxy_key=cap,N
    if proxy_key not in r._batch_tree_proxies:
        capture_event=duet_record('batch_proxy_capture') if DETAIL_TREE else None
        p=BatchedTreeProxy(cap,N,cfg.hf_config.vocab_size,cfg.hf_config.torch_dtype,r.device,
                           cfg.duet_proxy_wire_N,max(cfg.duet_phase1_k,cfg.duet_phase2_k, int(os.getenv("SSD_DUET_MISS_K", "0"))),
                           cfg.duet_proxy_top_k,pool=r._batch_tree_proxy_pool,
                           q_dtype=result.logits_q.dtype,c_max=cfg.duet_tree_c_tensor,policy=root_options(cfg))
        r._batch_tree_proxies[proxy_key]=p
        if r._batch_tree_proxy_pool is None: r._batch_tree_proxy_pool=p.graph.pool()
        if DETAIL_TREE:duet_close('batch_proxy_capture',capture_event)
    proxy=r._batch_tree_proxies[proxy_key]
    tokens=torch.zeros(B,N,dtype=torch.int64,device=r.device)
    tokens[:,:step_width].copy_(result.speculations[:,1:])
    q=torch.zeros(B,N,cfg.hf_config.vocab_size,dtype=result.logits_q.dtype,device=r.device)
    q[:,:result.logits_q.shape[1]].copy_(result.logits_q)
    proxy.prepare(parents,siblings,tokens,q,tt,dt)
    if DETAIL_TREE:duet_close('batch_tree_prepare',prepare_event)
    def callback(logits,b):
        pos,packed=proxy.replay(logits)
        verifier._send_proxy_wire(cfg,r.async_pg,r.draft_rank,pos[:B],packed[:B])
    r._duet_proxy_fn=callback
    try:
        logits=r.call('run_batched_tree',rows,N+1)
    finally:
        r._duet_proxy_fn=None
    topology={k:v[:B] for k,v in proxy.topology.items()}
    accept_event=duet_record('batch_tree_accept') if DETAIL_TREE else None
    greedy=all(t==0 for t in tt)
    if os.getenv('SSD_BATCH_TREE_ACCEPT_GRAPH','1')=='1':
        if not hasattr(r,'_batch_tree_accept'):
            r._batch_tree_accept={};r._batch_tree_accept_pool=None
        key=cap,N,greedy
        if key not in r._batch_tree_accept:
            capture_event=duet_record('batch_accept_capture') if DETAIL_TREE else None
            graph=BatchedTreeAccept(proxy,verifier.sampler_x,verifier.async_fan_out,
                                    greedy,pool=r._batch_tree_accept_pool)
            r._batch_tree_accept[key]=graph
            if r._batch_tree_accept_pool is None:r._batch_tree_accept_pool=graph.graph.pool()
            if DETAIL_TREE:duet_close('batch_accept_capture',capture_event)
        paths,recoveries,terminal=r._batch_tree_accept[key].replay(logits,tt,dt)
    else:
        paths,recoveries,terminal=verify_batch(logits,q,tokens,topology,tt,dt,
                                              verifier.sampler_x,verifier.async_fan_out,greedy)
    result_cpu=torch.cat([paths,recoveries,terminal],1).cpu().tolist()
    if DETAIL_TREE:duet_close('batch_tree_accept',accept_event)
    commit_event=duet_record('batch_tree_commit') if DETAIL_TREE else None
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
        physical_queries=(((sum(v+1 for v in valid)+7)//8)*8
                          if os.getenv('SSD_PACKED_TREE_VERIFY','0')=='1'
                          else cap*(N+1))
        m['phase_events'].append(dict(tree=is_tree[b],batch_size=B,verify_width=step_width,
            physical_verify_width=N,
            physical_batch_queries=physical_queries,
            step_id=result.step_id,source=phases[b],cache_hit=int(hits[b]),
            accepted_len=length,accepted_spec_len=length-1,valid_k=valid[b]))
        if os.getenv('SSD_TREE_SHAPE_METRICS','0')=='1':
            depths=[]
            for parent in parents[b]:depths.append(1 if parent<0 else depths[parent]+1)
            maximum=max(depths,default=0)
            m['phase_events'][-1].update(tree_max_depth=maximum,
                tree_sibling_width=max(siblings[b],default=-1)+1,
                reached_depth_ceiling=(length-1==maximum),
                node_utilization=(length-1)/valid[b] if valid[b] else 0.)
        m['accepted_suffix_lens_on_hit' if hits[b] else 'accepted_suffix_lens_on_miss'].append(length)
        if hits[b] and phases[b] in (1,2):
            m[f'accepted_lens_phase{phases[b]}_hit'].append(length-1)
    if DETAIL_TREE:duet_close('batch_tree_commit',commit_event)
    return VerifyResult(suffixes,recovery)
