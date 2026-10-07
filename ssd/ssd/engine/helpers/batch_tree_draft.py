"""Batched DUET service: request-owned cache, staged KV and phase forests."""
from copy import copy
import os
import torch
import torch.distributed as dist
from ssd.engine.helpers.batch_tree_common import capacity, restore_plan
from ssd.engine.helpers.batch_tree_forward import BatchedTreeForward
from ssd.engine.helpers.tree_host_topology import context_topology
from ssd.engine.helpers.batched_tree_executor import BatchedTreeExecutor
from ssd.engine.helpers.p1_tree import P1TreeExecutor, build_uniform_p1_roots
from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor
from ssd.engine.helpers.p2_tree import tree_wire_ints_len, rerank_tree_indices
from ssd.utils.async_helpers.nccl_pack import recv_int64


class BatchedDuetDraft:
    def __init__(self,runner):
        self.r=runner
        self.cfg=runner.config
        if self.cfg.use_eagle or self.cfg.duet_proxy_on_draft or self.cfg.duet_exit_topm_gather:
            raise NotImplementedError('Batched DUET tree currently uses dense full-vocabulary models')
        if not self.cfg.jit_speculate:
            raise ValueError('Batched tree service requires jit_speculate')
        self.entries={}
        self.staged={}
        self.forests={}
        self.executors={}
        self.root_graphs={}
        self.root_graph_pool=None
        self.input_graphs={}
        self.input_graph_pool=None
        self.graph_pool=None
        self.glue=BatchedTreeForward(runner,split=False)
        self.stats=dict(steps=0,tree_hits=0,phase1_replays=0,phase2_replays=0,captures=0)

    def invalidate(self,seq_ids):
        ids=set(seq_ids)
        self.entries={k:v for k,v in self.entries.items() if k[0] not in ids}
        for seq in ids: self.staged.pop(seq,None)

    def _flat_kv(self):
        k=self.r.kv_cache
        return k.view(k.shape[0],k.shape[1],-1,k.shape[4],k.shape[5])

    def _forest(self,phase,contexts,batch,pages):
        key=phase,contexts
        if key not in self.forests:
            cfg=copy(self.cfg)
            on=getattr(cfg,f'duet_p{phase}_tree_policy')=='on'
            if not on:
                cfg.duet_tree_c_tensor=1
                if phase==1:
                    cfg.duet_p1_tree_allocation_policy='backbone'
                    cfg.duet_p1_tree_max_nodes=cfg.duet_phase1_k
                    cfg.duet_p1_roots_per_position=cfg.duet_draft_fan_out
                else:
                    cfg.duet_tree_policy='backbone'
                    cfg.duet_p2_tree_max_nodes=cfg.duet_phase2_k
            hf=self.r.hf_config
            arenas=[]
            for b in range(capacity(self.cfg.max_num_seqs)):
                args=(self.r.model,self.r.model.compute_logits,cfg,self.r.device,
                      self.r.block_size,cfg.max_blocks,hf.vocab_size,
                      hf.num_attention_heads,hf.num_key_value_heads,hf.head_dim)
                if phase==1:
                    ex=P1TreeExecutor(*args,context_bucket=contexts,dtype=hf.torch_dtype,
                                      materialize_backbone_logits=False)
                else:
                    ex=P2TreeExecutor(*args,dtype=hf.torch_dtype,
                                      glue_width=contexts,materialize_backbone_logits=False)
                ex.gen.manual_seed((torch.initial_seed()+100003*phase+1009*contexts+b)%(2**63-1))
                arenas.append(ex)
            self.forests[key]=arenas
        arenas=self.forests[key][:batch]
        graph_key=phase,contexts,batch
        if graph_key not in self.executors:
            self.executors[graph_key]=BatchedTreeExecutor(arenas,staging_only=True,
                                                        workspace=self.glue.workspace())
        executor=self.executors[graph_key]
        if pages is not None:executor.prepare(pages)
        return executor,arenas

    def _prepare_arena(self,ex,row,page_bucket,tokens,scores,context_ids,temp,active=True):
        r=self.r
        dev=r.device
        n=len(row['tokens'])
        prefix=row['prefix']
        _,_,visible=context_topology(row['parents'])
        depths=[0]+[d+1 for d in context_topology(row['parents'])[0]]
        ex.in_root_tok.zero_(); ex.in_root_piv.zero_(); ex.in_glue.zero_()
        nr=len(tokens)
        ex.in_root_tok[:nr].copy_(tokens)
        ex.in_root_piv[:nr].copy_(scores)
        ci=context_ids.to(torch.int64).clamp(0,n-1)
        ex.in_glue[:nr,:n].copy_(torch.as_tensor(visible,device=dev)[ci])
        ex.in_rope_base.zero_()
        ex.in_rope_base[:nr].copy_(prefix+1+torch.tensor(depths,device=dev)[ci])
        ex.in_glue_w.fill_(n)
        ex.in_prefix_len.fill_(prefix)
        ex.in_temps.fill_(temp)
        bt=row['blocks']
        ex.in_block_tables.zero_()
        ex.in_block_tables[0,:len(bt)].copy_(torch.tensor(bt,dtype=torch.int32,device=dev))
        first=prefix+n
        final=first+ex.total_cells
        safe=(active and final<=r.config.max_model_len and (final-1)//r.block_size<len(bt)
              and all(p>=0 for p in bt[:(final+r.block_size-1)//r.block_size]))
        if not safe:
            # Near the model limit, emit no future cache entries. The next
            # request uses JIT. Never direct masked queries to negative pages.
            ex.in_root_piv.zero_()
        for f,w in enumerate(ex.round_widths):
            ex.in_slot[f].fill_(-1)
            if safe:
                pos=first+ex.round_offsets[f]+torch.arange(w,device=dev)
                blocks=torch.tensor(bt,dtype=torch.int32,device=dev)
                ex.in_slot[f][:w].copy_(blocks[pos//r.block_size]*r.block_size+pos%r.block_size)
            ex.in_ctx_len[f].fill_(min(final,r.config.max_model_len))
            count=page_bucket+ex.canvas_extra_pages
            live=min((final+r.block_size-1)//r.block_size,len(bt)) if safe else (prefix+n+r.block_size-1)//r.block_size
            pages=[bt[j] if j<live else bt[0] for j in range(count)]
            ex.wrappers[page_bucket][f]._paged_kv_indices_buf.copy_(
                torch.tensor(pages,dtype=torch.int32,device=dev))
        return safe

    def _save_entries(self,arena,seq,contexts,tokens,phase,safe):
        if not safe: return
        R=len(tokens)
        meta=torch.cat([arena.out_valid[:R,None],arena.view_tok[:R],arena.view_par[:R],
                        arena.view_sib[:R],arena.view_pcell[:R]],1).cpu().tolist()
        roots=torch.stack([contexts,tokens],1).cpu().tolist()
        N=arena.NV
        is_tree=getattr(self.cfg,f'duet_p{phase}_tree_policy')=='on'
        cap=(getattr(self.cfg,f'duet_p{phase}_tree_verify_nodes') if is_tree else
             getattr(self.cfg,f'duet_phase{phase}_k'))
        for root,(ctx,token) in enumerate(roots):
            n=meta[root][0]
            if n<=0: continue
            a=meta[root]
            toks=a[1:1+N][:n]; par=a[1+N:1+2*N][:n]
            sib=a[1+2*N:1+3*N][:n]; cells=a[1+3*N:1+4*N][:n]
            if n>cap:
                q=arena.view_rawq[root,:n].cpu().tolist()
                keep=rerank_tree_indices(par,sib,q,cap)
                remap={old:new for new,old in enumerate(keep)}
                par=[-1 if par[j]<0 else remap[par[j]] for j in keep]
                toks=[toks[j] for j in keep]; sib=[sib[j] for j in keep]
                cells=[cells[j] for j in keep]
            self.entries.setdefault((seq,int(ctx),int(token)),dict(
                tokens=toks,parents=par,siblings=sib,cells=cells,
                logits=arena.cell_logits,phase=phase,tree=is_tree))

    def _prepare_batch(self,executor,pages,rows,params,temps):
        from ssd.engine.helpers.batch_tree_inputs import BatchedArenaInputs
        key=id(executor),pages
        if key not in self.input_graphs:
            g=BatchedArenaInputs(executor,pages,self.cfg.max_model_len,self.input_graph_pool)
            self.input_graphs[key]=g
            if self.input_graph_pool is None:self.input_graph_pool=g.graph.pool()
        return self.input_graphs[key].replay(rows,params,temps)

    @torch.inference_mode()
    def serve(self):
        r,cfg=self.r,self.cfg
        B,K,F,step=r.recv_tensor((4,),torch.int64).tolist()
        r._request_step_id=int(step)
        from ssd.engine.helpers.cudagraph_helpers import duet_set_context
        duet_set_context(step_id=step,proc='draft')
        raw=recv_int64(r.async_pg,src=0,total_length=B*(5+cfg.max_blocks),device=r.device)
        keys=raw[:3*B].view(B,3)
        lengths=raw[3*B:4*B]
        bt=raw[4*B:4*B+B*cfg.max_blocks].view(B,cfg.max_blocks).int()
        temps=raw[-B:].int().view(torch.float32)
        host=raw.cpu().tolist()
        key_cpu=[host[j*3:j*3+3] for j in range(B)]
        length_cpu=host[3*B:4*B]
        block_cpu=[host[4*B+j*cfg.max_blocks:4*B+(j+1)*cfg.max_blocks] for j in range(B)]
        temp_cpu=temps.cpu().tolist()
        # Restore by identity, never by the current batch row. Prefill
        # explicitly invalidates the resumed sequence's old generation.
        flat=self._flat_kv()
        for b,(seq,terminal,_) in enumerate(key_cpu):
            previous=self.staged.pop(seq,None)
            if previous is not None:
                src,dst=restore_plan(previous,length_cpu[b],terminal,block_cpu[b],r.block_size)
                if src:
                    flat[:,:,torch.tensor(dst,device=r.device)]=previous['kv'][:,:,src]
        self.staged={}
        hits=[self.entries.get(tuple(k)) for k in key_cpu]
        miss=[b for b,e in enumerate(hits) if e is None]
        miss_q=torch.zeros(len(miss),K,r.hf_config.vocab_size,dtype=r.hf_config.torch_dtype,device=r.device)
        miss_t=torch.zeros(len(miss),cfg.duet_response_token_width,dtype=torch.int64,device=r.device)
        if miss:
            idx=torch.tensor(miss,dtype=torch.int64,device=r.device)
            r.jit_speculate(keys[idx],lengths[idx],miss_q,miss_t,temps[idx],bt[idx],None)
        miss_k=cfg.duet_phase2_k if os.getenv('SSD_DUET_JIT_SHORT','0')=='1' else max(cfg.duet_phase1_k,cfg.duet_phase2_k)
        width=cfg.duet_response_token_width
        nv=cfg.duet_tree_wire_nodes
        tokens=torch.zeros(B,width,dtype=torch.int64,device=r.device)
        topology=torch.zeros(B,tree_wire_ints_len(nv),dtype=torch.int64,device='cpu')
        phases=[]; valid=[]; rows=[]; qrows=[]
        mi=0
        for b,entry in enumerate(hits):
            if entry is None:
                ts=miss_t[mi,:miss_k].cpu().tolist()
                parent=list(range(-1,miss_k-1)); sibling=[0]*miss_k
                q=miss_q[mi,:miss_k]; phase=0; tree=False; mi+=1
            else:
                ts=entry['tokens']; parent=entry['parents']; sibling=entry['siblings']
                q=entry['logits'].index_select(0,torch.tensor(entry['cells'],device=r.device)).to(r.hf_config.torch_dtype)
                phase=entry['phase']; tree=entry['tree']
            n=len(ts)
            tokens[b,:n]=torch.tensor(ts,device=r.device)
            valid.append(n); phases.append(phase); qrows.append(q)
            if tree:
                topology[b,:3]=torch.tensor([n,n,2]) # v2: q row is node-local
                topology[b,3:3+n]=torch.tensor(ts)
                topology[b,3+nv:3+nv+n]=torch.tensor(parent)
                topology[b,3+2*nv:3+2*nv+n]=torch.tensor(sibling)
                first_child={}
                refs=[first_child.setdefault(p,j) for j,p in enumerate(parent)]
                topology[b,3+3*nv:3+3*nv+n]=torch.tensor(refs)
                self.stats['tree_hits']+=1
            rows.append(dict(tokens=[key_cpu[b][2]]+ts,parents=parent,siblings=sibling,
                             prefix=length_cpu[b]-1,blocks=block_cpu[b]))
        max_valid=max(valid)
        q=torch.zeros(B,max_valid,r.hf_config.vocab_size,dtype=r.hf_config.torch_dtype,device=r.device)
        for b,qr in enumerate(qrows): q[b,:valid[b]].copy_(qr)
        fused=torch.cat([torch.tensor([int(e is not None) for e in hits],device=r.device),
                         torch.tensor(phases,device=r.device),torch.tensor(valid,device=r.device),
                         tokens.flatten(),topology.to(r.device).flatten()])
        dist.send(fused,dst=0,group=r.async_pg)
        dist.send(q,dst=0,group=r.async_pg)
        work,proxy_buf=r._irecv_duet_proxy(B,K)
        # One ancestor-masked draft forward covers both tree and chain rows.
        glue_width=next(x for x in sorted({cfg.duet_phase1_k+1,cfg.duet_phase2_k+1,
                                           cfg.duet_response_token_width+1}) if x>=max_valid+1)
        glue=self.glue.run(rows,glue_width)
        for b,entry in enumerate(hits):
            if entry is None or not entry['tree']: continue
            row=rows[b]; start=row['prefix']; n=len(row['tokens'])
            slots=[row['blocks'][j//r.block_size]*r.block_size+j%r.block_size for j in range(start,start+n)]
            self.staged[key_cpu[b][0]]=dict(parents=row['parents'],length=length_cpu[b],
                                          kv=flat[:,:,torch.tensor(slots,device=r.device)].clone())
        # No view from the previous generation is retained after q and KV
        # have been copied, so arena reuse cannot mutate a served snapshot.
        self.entries={}
        self._build(rows,glue,key_cpu,temp_cpu,work,proxy_buf,B,K)
        self.stats['steps']+=1

    def _build(self,rows,glue,keys,temps,proxy_work,proxy_buf,B,K):
        cfg,r=self.cfg,self.r
        contexts=max(len(row['tokens']) for row in rows)
        cb=next(x for x in sorted({cfg.duet_phase2_k+1,cfg.duet_phase1_k+1,
                                 cfg.duet_response_token_width+1}) if x>=contexts)
        bc=capacity(B)
        padded=[]
        if not cfg.duet_only_proxy:
            # Discover round-zero width before selecting the shared page
            # canvas. All arenas use the same context bucket and top-W budget.
            exs,_=self._forest(1,cb,bc,None)
            sample=exs.first
            pages=capacity(max((row['prefix']+len(row['tokens'])+sample.round_ends[0]+r.block_size-1)//r.block_size for row in rows))
            exs,arenas=self._forest(1,cb,bc,pages)
            batched_roots=None
            if os.getenv('SSD_BATCH_TREE_ROOT_GRAPH','1')=='1':
                from ssd.engine.helpers.batch_tree_roots import BatchedP1Roots
                key=bc,cb
                if key not in self.root_graphs:
                    g=BatchedP1Roots(bc,cb,r.hf_config.vocab_size,r.hf_config.torch_dtype,
                        r.device,arenas[0].roots_per_position,cfg.sampler_x,cfg.async_fan_out,
                        pool=self.root_graph_pool)
                    self.root_graphs[key]=g
                    if self.root_graph_pool is None:self.root_graph_pool=g.graph.pool()
                roots_graph=self.root_graphs[key]
                batched_roots=roots_graph.replay(rows,glue,temps)
            params=[]
            inputs=[]
            capture_inputs=os.getenv('SSD_BATCH_TREE_INPUT_GRAPH','1')=='1'
            for b in range(bc):
                row=rows[min(b,B-1)]
                n=len(row['tokens']); pfo=arenas[b].roots_per_position
                _,_,visible=context_topology(row['parents'])
                nr=n*pfo
                if batched_roots is not None:
                    rt=batched_roots[0][b,:nr];sc=batched_roots[1][b,:nr]
                    ci=roots_graph.context_ids[:nr]
                else:
                    roots=build_uniform_p1_roots(glue[min(b,B-1),:n],
                        torch.tensor(row['tokens'],device=r.device),pfo,
                        torch.full((n,),temps[min(b,B-1)],device=r.device),
                        sampler_x=cfg.sampler_x,async_fan_out=cfg.async_fan_out,
                        context_glue_rows=torch.as_tensor(visible,device=r.device),root_width=arenas[b].R)
                    rt=roots['tokens'][:nr]; sc=roots['scores'][:nr]; ci=roots['context_ids'][:nr]
                if b>=B: sc=torch.zeros_like(sc)
                safe=True
                if not capture_inputs:
                    safe=self._prepare_arena(arenas[b],row,pages,rt,sc,ci,temps[min(b,B-1)],active=b<B)
                if b<B:
                    p=torch.full((cb,pfo),-1,dtype=torch.int64,device=r.device)
                    p[:n]=rt.view(n,pfo); padded.append(p)
                params.append((rt,ci,safe))
                inputs.append((rt,sc,ci))
            if capture_inputs:
                safe=self._prepare_batch(exs,pages,rows,inputs[:B],temps)
                params=[(rt,ci,safe[b] if b<B else False) for b,(rt,ci,_) in enumerate(params)]
            self._execute(exs,pages,1)
            for b in range(B):
                rt,ci,safe=params[b]
                self._save_entries(arenas[b],keys[b][0],ci,rt,1,safe)
        proxy_work.wait()
        proxy=r._unpack_duet_proxy(proxy_buf,B,K)
        pos,tok,score=proxy['chosen_pos'],proxy['chosen_tok'],proxy['chosen_piv']
        valid=(pos>=0)&(pos<torch.tensor([len(row['tokens']) for row in rows],device=r.device)[:,None])
        if padded:
            roots=torch.stack(padded)
            bi=torch.arange(B,device=r.device)[:,None]
            in_draft=(roots[bi,pos.clamp(0,cb-1)]==tok[:,:,None]).any(-1)
            valid&=~in_draft
        count=cfg.duet_p2_seed_count
        pick=(~valid).int().argsort(dim=1,stable=True)[:,:count]
        ci=pos.gather(1,pick).clamp(0,contexts-1)
        rt=tok.gather(1,pick)
        sc=torch.where(valid.gather(1,pick),score.gather(1,pick),0)
        exs,_=self._forest(2,cb,bc,None)
        sample=exs.first
        pages=capacity(max((row['prefix']+len(row['tokens'])+sample.round_ends[0]+r.block_size-1)//r.block_size for row in rows))
        exs,arenas=self._forest(2,cb,bc,pages)
        safe=[]
        if os.getenv('SSD_BATCH_TREE_INPUT_GRAPH','1')=='1':
            safe=self._prepare_batch(exs,pages,rows,[(rt[b],sc[b],ci[b]) for b in range(B)],temps)
        else:
            for b in range(bc):
                i=min(b,B-1)
                scores=sc[i] if b<B else torch.zeros_like(sc[i])
                safe.append(self._prepare_arena(arenas[b],rows[i],pages,rt[i],scores,ci[i],temps[i],active=b<B))
        self._execute(exs,pages,2)
        for b in range(B): self._save_entries(arenas[b],keys[b][0],ci[b],rt[b],2,safe[b])

    def _execute(self,executor,pages,phase):
        if pages not in executor.graphs:
            executor.capture(pages,graph_pool=self.graph_pool)
            if self.graph_pool is None: self.graph_pool=executor.graphs[pages].pool()
            self.stats['captures']+=1
        executor.replay(pages)
        self.stats[f'phase{phase}_replays']+=1
