"""Prepare finite batch/context/page graph shapes before serving requests."""
from ssd.engine.helpers.root_policy import options as root_options
import os


def page_buckets(config, draft=False, extra=0):
    from ssd.engine.helpers.batch_tree_common import capacity
    setting=os.getenv('SSD_TREE_EXEC_WARMUP','all').strip().lower()
    if setting in ('0','off','false','none'):return []
    maximum=capacity((config.max_model_len+extra+config.kvcache_block_size-1)//config.kvcache_block_size)
    buckets=[1<<i for i in range(maximum.bit_length())]
    if not draft:buckets=sorted({min(x,config.max_blocks) for x in buckets})
    if setting!='all':
        requested={int(x) for x in setting.split(',') if x.strip()}
        buckets=[x for x in buckets if x in requested]
    return buckets


def prewarm_draft(service):
    """Finite serving envelope; done before the worker's draft_ready barrier."""
    import torch,time
    from ssd.engine.helpers.batch_tree_inputs import BatchedArenaInputs
    from ssd.engine.helpers.batch_tree_roots import BatchedP1Roots
    from ssd.engine.helpers.batch_tree_common import capacity, verification_widths
    r=service.r;cfg=service.cfg;start=time.perf_counter()
    widths=[n+1 for n in verification_widths(cfg)]
    # The first forward evaluates roots across all contexts, not max_nodes
    # of one root's tree. Include its full footprint even near a length cap,
    # where execution is masked but the wrapper still needs a page bucket.
    first_round=max(max(widths)*max(cfg.duet_p1_roots_per_position,cfg.duet_draft_fan_out),
                    cfg.duet_proxy_total_budget)
    pages=page_buckets(cfg,draft=True,extra=first_round)
    if not pages:return
    batches=[1<<i for i in range(capacity(cfg.max_num_seqs).bit_length())]
    with torch.inference_mode():
        rng=torch.cuda.get_rng_state(r.device);states={}
        for width in widths:
            for batch in batches:
                root_key=batch,width
                if root_key not in service.root_graphs:
                    g=BatchedP1Roots(batch,width,r.hf_config.vocab_size,r.hf_config.torch_dtype,r.device,
                        cfg.duet_p1_roots_per_position,cfg.sampler_x,cfg.async_fan_out,pool=service.root_graph_pool)
                    service.root_graphs[root_key]=g
                    if service.root_graph_pool is None:service.root_graph_pool=g.graph.pool()
                for page in pages:
                    if page<=cfg.max_blocks:service.glue.capture(batch,width,page)
                    for phase in (1,2):
                        ex,arenas=service._forest(phase,width,batch,page)
                        for ar in arenas:
                            if id(ar) not in states:states[id(ar)]=(ar,ar.gen.get_state())
                            ar.prime_capture_inputs(page)
                        ex.capture(page,graph_pool=service.graph_pool)
                        if service.graph_pool is None:service.graph_pool=ex.graphs[page].pool()
                        key=id(ex),page
                        if key not in service.input_graphs:
                            g=BatchedArenaInputs(ex,page,cfg.max_model_len,service.input_graph_pool)
                            service.input_graphs[key]=g
                            if service.input_graph_pool is None:service.input_graph_pool=g.graph.pool()
        for ar,state in states.values():ar.gen.set_state(state)
        torch.cuda.set_rng_state(rng,r.device);torch.cuda.synchronize(r.device)
    print('[DUET prewarm] draft',len(service.executors),'executors',sum(len(x.graphs) for x in service.executors.values()),
          'graphs',round(time.perf_counter()-start,3),'seconds',flush=True)


def prewarm_target(r):
    import torch,time
    from ssd.engine.helpers.batch_tree_common import capacity, verification_widths
    from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy,BatchedTreeAccept
    cfg=r.config;start=time.perf_counter()
    pages=page_buckets(cfg)
    if not pages:return
    batches=[1<<i for i in range(capacity(cfg.max_num_seqs).bit_length())]
    widths=verification_widths(cfg)
    r._batch_tree_proxies={};r._batch_tree_proxy_pool=None
    r._batch_tree_accept={};r._batch_tree_accept_pool=None
    with torch.inference_mode():
        rng=torch.cuda.get_rng_state(r.device)
        for b in batches:
            for n in widths:
                p=BatchedTreeProxy(b,n,cfg.hf_config.vocab_size,cfg.hf_config.torch_dtype,r.device,
                    cfg.duet_proxy_wire_N,n,cfg.duet_proxy_top_k,
                    pool=r._batch_tree_proxy_pool,q_dtype=cfg.draft_hf_config.torch_dtype,c_max=cfg.duet_tree_c_tensor,policy=root_options(cfg))
                r._batch_tree_proxies[b,n]=p
                if r._batch_tree_proxy_pool is None:r._batch_tree_proxy_pool=p.graph.pool()
                a=BatchedTreeAccept(p,cfg.sampler_x,cfg.async_fan_out,cfg.greedy_only,
                    pool=r._batch_tree_accept_pool,stochastic_only=cfg.stochastic_only,target_only=True)
                r._batch_tree_accept[b,n,cfg.greedy_only,cfg.stochastic_only]=a
                if r._batch_tree_accept_pool is None:r._batch_tree_accept_pool=a.graph.pool()
                for page in pages:r.call('warm_batched_tree_forward',b,n+1,page)
                if b*(n+1)>8:
                    from ssd.engine.helpers.flat_tree_sampling import FlatProxy,FlatAccept
                    fp=None;fa=None
                    for q in range(8,b*(n+1),8):
                        if q<b:continue
                        remaining=q-b;lengths=[]
                        for _ in range(b):
                            k=min(n,remaining);lengths.append(k);remaining-=k
                        if fp is None:
                            fp=FlatProxy(b,n,cfg.hf_config.vocab_size,cfg.hf_config.torch_dtype,r.device,
                                cfg.duet_proxy_wire_N,n,cfg.duet_proxy_top_k,
                                pool=r._batch_tree_proxy_pool,q_dtype=cfg.draft_hf_config.torch_dtype,c_max=cfg.duet_tree_c_tensor,policy=root_options(cfg))
                            r._batch_tree_proxies[b,n,'flat']=fp
                        fp.prepare([list(range(-1,k-1)) for k in lengths],[[0]*k for k in lengths],
                            torch.zeros(b,n,device=r.device,dtype=torch.long),torch.zeros_like(p.q),[.7]*b,[.7]*b)
                        if fa is None:
                            fa=FlatAccept(fp,cfg.sampler_x,cfg.async_fan_out,cfg.greedy_only,
                                pool=r._batch_tree_accept_pool,stochastic_only=cfg.stochastic_only,target_only=True)
                            r._batch_tree_accept[b,n,cfg.greedy_only,cfg.stochastic_only,'flat']=fa
                        else:fa._capture()
        torch.cuda.set_rng_state(rng,r.device);torch.cuda.synchronize(r.device)
    print('[DUET prewarm] target',len(r._batch_tree_proxies),'proxy/accept shapes',round(time.perf_counter()-start,3),'seconds',flush=True)

