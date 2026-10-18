"""Isolated systems experiments; switches exist only in the review harness."""
import os


def install():
    if os.getenv('DUET_REVIEW_FACTORIAL','0')=='1':
        from ssd.engine.model_runner import ModelRunner
        def set_head_mode(self,replica,stream):
            os.environ['SSD_DUET_EXIT_REPLICA']=str(int(replica))
            os.environ['SSD_BATCH_TREE_PROXY_STREAM']=str(int(stream))
        ModelRunner.review_head_mode=set_head_mode
    if os.getenv('DUET_REVIEW_LIVE_SIBLINGS','0')=='1':
        import inspect,textwrap
        from ssd.engine.helpers import batch_tree_verify as module
        if os.getenv('DUET_REVIEW_FLAT','0')!='0':raise ValueError('Test sibling and flat changes separately first')
        source=textwrap.dedent(inspect.getsource(module.verify.__wrapped__))
        source=source.replace('proxy_key=cap,N',
            "live_c=max((max(s,default=0) for s in siblings),default=0)+1\n    proxy_key=cap,N,live_c")
        source=source.replace('c_max=cfg.duet_tree_c_tensor)','c_max=live_c)')
        source=source.replace('key=cap,N,greedy,cfg.stochastic_only','key=cap,N,greedy,cfg.stochastic_only,live_c')
        scope={};exec(compile(source,'<live-sibling-review>','exec'),module.__dict__,scope)
        module.verify=scope['verify']
    if os.getenv('DUET_REVIEW_REPLICA','0')=='1':
        from ssd.config import Config
        config_original=Config.__post_init__
        def validate(self):
            # Runtime already has independent batched replica projection;
            # bypass only its historical B==1 config gate for this experiment.
            enabled=os.environ.get('SSD_DUET_EXIT_REPLICA','0')
            if self.max_num_seqs>1 and enabled=='1':
                if not self.duet_tree_enabled:raise ValueError('Replica review requires tree')
                os.environ['SSD_DUET_EXIT_REPLICA']='0'
                try:return config_original(self)
                finally:os.environ['SSD_DUET_EXIT_REPLICA']=enabled
            return config_original(self)
        Config.__post_init__=validate
    if os.getenv('DUET_REVIEW_NUMPY_INPUT','0')=='1':
        import inspect,textwrap,numpy as np
        from ssd.engine.helpers import batch_tree_inputs as module
        source=textwrap.dedent(inspect.getsource(module.BatchedArenaInputs.replay.__wrapped__))
        replacements={
            "host=torch.zeros_like(self.meta,device='cpu')":"host=np.zeros(tuple(self.meta.shape),dtype=np.int64)",
            'host[b,:3]=torch.tensor([prefix,n,int(okay)])':'host[b,:3]=[prefix,n,int(okay)]',
            'host[b,3:3+len(bt)]=torch.tensor(bt)':'host[b,3:3+len(bt)]=bt',
            'host[b,3+self.M:3+self.M+n]=torch.tensor([0]+[d+1 for d in depths])':'host[b,3+self.M:3+self.M+n]=[0]+[d+1 for d in depths]',
            '.view(self.G,self.G)[:n,:n]=torch.as_tensor(vis)':'.reshape(self.G,self.G)[:n,:n]=vis',
            'host[b,-self.P:]=torch.tensor([bt[j] if j<live else bt[0] for j in range(self.P)])':'host[b,-self.P:]=[bt[j] if j<live else bt[0] for j in range(self.P)]',
            'self.meta.copy_(host)':'self.meta.copy_(torch.from_numpy(host))'}
        for before,after in replacements.items():
            if before not in source:raise ValueError('Input adapter no longer matches runtime')
            source=source.replace(before,after)
        scope={};exec(compile(source,'<numpy-input-review>','exec'),dict(module.__dict__,np=np),scope)
        module.BatchedArenaInputs.replay=scope['replay']
    if os.getenv('DUET_REVIEW_PREWARM','0')=='1':
        from ssd.engine.helpers.batch_tree_draft import BatchedDuetDraft
        draft_original=BatchedDuetDraft.__init__
        def initialize(self,*args,**kwargs):
            draft_original(self,*args,**kwargs)
            prewarm_draft(self)
        BatchedDuetDraft.__init__=initialize


def prewarm_draft(service):
    """Finite serving envelope; done before the worker's draft_ready barrier."""
    import torch,time
    from ssd.engine.helpers.batch_tree_inputs import BatchedArenaInputs
    from ssd.engine.helpers.batch_tree_roots import BatchedP1Roots
    from ssd.engine.helpers.batch_tree_common import capacity
    r=service.r;cfg=service.cfg;start=time.perf_counter()
    pages=[int(x) for x in os.getenv('DUET_REVIEW_PREWARM_PAGES','1,2,4').split(',')]
    batches=[1<<i for i in range(capacity(cfg.max_num_seqs).bit_length())]
    widths=sorted({cfg.duet_phase1_k+1,cfg.duet_phase2_k+1,cfg.duet_response_token_width+1})
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
                    service.glue.capture(batch,width,page)
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
    print('[review prewarm] draft',len(service.executors),'executors',sum(len(x.graphs) for x in service.executors.values()),
          'graphs',round(time.perf_counter()-start,3),'seconds',flush=True)


def prewarm_target(llm):
    import torch,time
    from ssd.engine.helpers.batch_tree_common import capacity
    from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy,BatchedTreeAccept
    r=llm.model_runner;cfg=r.config;start=time.perf_counter()
    pages=[int(x) for x in os.getenv('DUET_REVIEW_PREWARM_PAGES','1,2,4').split(',')]
    batches=[1<<i for i in range(capacity(cfg.max_num_seqs).bit_length())]
    widths=sorted({cfg.duet_phase1_k,cfg.duet_phase2_k,cfg.duet_response_token_width})
    r._batch_tree_proxies={};r._batch_tree_proxy_pool=None
    r._batch_tree_accept={};r._batch_tree_accept_pool=None
    with torch.inference_mode():
        rng=torch.cuda.get_rng_state(r.device)
        for b in batches:
            for n in widths:
                p=BatchedTreeProxy(b,n,cfg.hf_config.vocab_size,cfg.hf_config.torch_dtype,r.device,
                    cfg.duet_proxy_wire_N,max(cfg.duet_phase1_k,cfg.duet_phase2_k),cfg.duet_proxy_top_k,
                    pool=r._batch_tree_proxy_pool,q_dtype=cfg.draft_hf_config.torch_dtype,c_max=cfg.duet_tree_c_tensor)
                r._batch_tree_proxies[b,n]=p
                if r._batch_tree_proxy_pool is None:r._batch_tree_proxy_pool=p.graph.pool()
                a=BatchedTreeAccept(p,cfg.sampler_x,cfg.async_fan_out,cfg.greedy_only,
                    pool=r._batch_tree_accept_pool,stochastic_only=cfg.stochastic_only)
                r._batch_tree_accept[b,n,cfg.greedy_only,cfg.stochastic_only]=a
                if r._batch_tree_accept_pool is None:r._batch_tree_accept_pool=a.graph.pool()
                if os.getenv('DUET_REVIEW_LIVE_SIBLINGS','0')=='1':
                    r._batch_tree_proxies[b,n,cfg.duet_tree_c_tensor]=p
                    r._batch_tree_accept[b,n,cfg.greedy_only,cfg.stochastic_only,cfg.duet_tree_c_tensor]=a
                    for c in range(1,cfg.duet_tree_c_tensor):
                        sp=BatchedTreeProxy(b,n,cfg.hf_config.vocab_size,cfg.hf_config.torch_dtype,r.device,
                            cfg.duet_proxy_wire_N,max(cfg.duet_phase1_k,cfg.duet_phase2_k),cfg.duet_proxy_top_k,
                            pool=r._batch_tree_proxy_pool,q_dtype=cfg.draft_hf_config.torch_dtype,c_max=c)
                        sa=BatchedTreeAccept(sp,cfg.sampler_x,cfg.async_fan_out,cfg.greedy_only,
                            pool=r._batch_tree_accept_pool,stochastic_only=cfg.stochastic_only)
                        r._batch_tree_proxies[b,n,c]=sp
                        r._batch_tree_accept[b,n,cfg.greedy_only,cfg.stochastic_only,c]=sa
                for page in pages:
                    prefix=max(1,(page-1)*r.block_size)
                    rows=[dict(tokens=[0]*(n+1),parents=list(range(-1,n-1)),prefix=prefix,blocks=[0]*page) for _ in range(b)]
                    r.call('run_batched_tree',rows,n+1)
                if os.getenv('DUET_REVIEW_FLAT','0') in ('1','prob'):
                    from mlsys_flat_tree import FlatProxy,FlatAccept
                    fp=None;fa=None
                    for q in range(8,b*(n+1),8):
                        if q<b:continue
                        remaining=q-b;lengths=[]
                        for _ in range(b):
                            k=min(n,remaining);lengths.append(k);remaining-=k
                        if fp is None:
                            fp=FlatProxy(b,n,cfg.hf_config.vocab_size,cfg.hf_config.torch_dtype,r.device,
                                cfg.duet_proxy_wire_N,max(cfg.duet_phase1_k,cfg.duet_phase2_k),cfg.duet_proxy_top_k,
                                pool=r._batch_tree_proxy_pool,q_dtype=cfg.draft_hf_config.torch_dtype,c_max=cfg.duet_tree_c_tensor)
                            r._batch_tree_proxies[b,n,'flat']=fp
                        fp.prepare([list(range(-1,k-1)) for k in lengths],[[0]*k for k in lengths],
                            torch.zeros(b,n,device=r.device,dtype=torch.long),torch.zeros_like(p.q),[.7]*b,[.7]*b)
                        if fa is None:
                            fa=FlatAccept(fp,cfg.sampler_x,cfg.async_fan_out,cfg.greedy_only,
                                pool=r._batch_tree_accept_pool,stochastic_only=cfg.stochastic_only)
                            r._batch_tree_accept[b,n,cfg.greedy_only,cfg.stochastic_only,'flat']=fa
                        else:fa._capture()
        torch.cuda.set_rng_state(rng,r.device);torch.cuda.synchronize(r.device)
    print('[review prewarm] target',len(r._batch_tree_proxies),'proxy/accept shapes',round(time.perf_counter()-start,3),'seconds',flush=True)


def measure_target(llm,prompts,repeats):
    """Real prefills with request-owned pages, no draft or proxy wire wait."""
    import torch,time,statistics
    from ssd.engine.sequence import Sequence
    from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy
    from ssd.engine.helpers import cudagraph_helpers as h
    r=llm.model_runner;cfg=r.config;b=cfg.max_num_seqs;n=8
    with torch.inference_mode():
        # Each prefix and tree fits one physical page. Every request uses a
        # different page and its own real prompt; no shared-context assumption.
        seqs=[Sequence(prompts[i%len(prompts)][:min(128,r.block_size-n-1)]) for i in range(b)]
        for i,s in enumerate(seqs):s.block_table=[i]
        r.call('run',seqs,True)
        result=[]
        for shape in ('uniform','half_short','one_hit'):
            lens=[n]*b if shape=='uniform' else [n]*(b//2)+[2]*(b-b//2) if shape=='half_short' else [n]+[0]*(b-1)
            ps=[[-1,-1,0,0,1,2,3,4][:k] for k in lens];ss=[[0,1,0,1,0,0,0,0][:k] for k in lens]
            rows=[dict(tokens=[100+i]+list(range(200,200+k)),parents=ps[i],prefix=s.num_tokens,blocks=[i]) for i,(s,k) in enumerate(zip(seqs,lens))]
            proxy=BatchedTreeProxy(b,n,cfg.hf_config.vocab_size,cfg.hf_config.torch_dtype,r.device,25,n,13)
            tokens=torch.zeros(b,n,device=r.device,dtype=torch.long)
            for i,k in enumerate(lens):tokens[i,:k]=torch.arange(200,200+k,device=r.device)
            proxy.prepare(ps,ss,tokens,torch.zeros_like(proxy.q),[.7]*b,[.7]*b)
            r._duet_proxy_fn=lambda logits,batch:proxy.replay(logits)
            def run():return r.call('run_batched_tree',rows,n+1)
            if os.getenv('DUET_REVIEW_FACTORIAL','0')=='1':
                modes={'gather':(False,False),'replica':(True,False),'replica_stream':(True,True)}
                samples={k:[] for k in modes};outputs={}
                for name,flags in modes.items():
                    r.call('review_head_mode',*flags)
                    for _ in range(4):outputs[name]=run()
                measured_start=time.time()
                for repeat in range(repeats):
                    names=list(modes)
                    if repeat%2:names.reverse()
                    for name in names:
                        r.call('review_head_mode',*modes[name]);torch.cuda.synchronize()
                        start=time.perf_counter();run();torch.cuda.synchronize()
                        samples[name].append((time.perf_counter()-start)*1000)
                measured_end=time.time()
                for name,flags in modes.items():
                    r.call('review_head_mode',*flags);h.duet_reset();h.PROFILE_DUET=True
                    for _ in range(8):run()
                    torch.cuda.synchronize();components={}
                    for e in h._duet_events:components.setdefault(e[1],[]).append(e[3].elapsed_time(e[4]))
                    h.PROFILE_DUET=False;h.duet_reset()
                    live=torch.cat([outputs[name][i,:k+1] for i,k in enumerate(lens)])
                    result.append(dict(shape=shape,arm=name,lengths=lens,wall_ms=samples[name],
                        measured_start=measured_start,measured_end=measured_end,
                        median_ms=statistics.median(samples[name]),component_samples_ms=components,
                        component_median_ms={k:statistics.median(v) for k,v in components.items()},
                        argmax=live.argmax(-1).tolist(),finite=bool(torch.isfinite(live).all())))
                continue
            for _ in range(4):output=run()
            torch.cuda.synchronize();samples=[]
            for _ in range(repeats):
                torch.cuda.synchronize();start=time.perf_counter();run();torch.cuda.synchronize()
                samples.append((time.perf_counter()-start)*1000)
            h.duet_reset();h.PROFILE_DUET=True
            for _ in range(8):run()
            torch.cuda.synchronize();components={}
            for e in h._duet_events:components.setdefault(e[1],[]).append(e[3].elapsed_time(e[4]))
            h.PROFILE_DUET=False;h.duet_reset()
            live=torch.cat([output[i,:k+1] for i,k in enumerate(lens)])
            result.append(dict(shape=shape,lengths=lens,wall_ms=samples,median_ms=statistics.median(samples),
                component_samples_ms=components,component_median_ms={k:statistics.median(v) for k,v in components.items()},
                argmax=live.argmax(-1).tolist(),finite=bool(torch.isfinite(live).all())))
        r._duet_proxy_fn=None
    return dict(scope='same declared prefix/tree inputs; target forward and proxy scoring; no accept or draft wait',cells=result)
