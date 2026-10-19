"""Shadow native v1.4 without changing its roots, KV contents, RNG, or verifier.

Three R10/W10/K10 forests are computed from a frozen, pre-hit candidate subset.
Only compact statistics persist. Full q logits use a bounded tmpfs sidecar
for the actual hit and are deleted as soon as target reads them.
"""
from copy import copy
from pathlib import Path
import gzip
import json
import os
import torch
from figure_math import R, W, K, C, G, ARMS, merge, expected,check_native


def append(name, row):
    path=Path(os.environ['SSD_PROFILE_DIR'])/name
    with gzip.open(path, 'at', compresslevel=1) as stream:
        stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')


def flat_kv(r):
    return r.kv_cache.flatten(2,3)


def slots(r, blocks, start, count):
    pos=torch.arange(start,start+count,device=r.device)
    if start < 0 or start+count>r.config.max_model_len:
        return None
    bt=torch.as_tensor(blocks,device=r.device,dtype=torch.long).reshape(-1)
    if (start+count-1)//r.block_size>=len(bt):
        return None
    pages=bt[pos//r.block_size]
    if bool((pages<0).any()):
        return None
    return pages*r.block_size+pos%r.block_size


def equal_select(ar,width,f,depth_cap,remaining,future_rounds,R,**kwargs):
    ix=torch.arange(ar.capacity,device=ar.device)
    ok=(ix<ar.n)&ar.valid&(ar.state==0)&(ar.depth==f)&(ar.depth<depth_cap)
    mask=ok[:,None]&(ar.root[:,None]==torch.arange(R,device=ar.device))
    values=torch.where(mask,ar.logpri[:,None],-torch.inf)
    selected=values.argmax(0); valid=mask.gather(0,selected[None]).reshape(-1)
    return selected,valid


def root_subset(scores, contexts, rotation):
    active=[i for i,s in enumerate(scores) if s>0]
    if len(active)<R:return []
    if contexts is None:
        return sorted(active,key=lambda i:(-scores[i],i))[:R]
    pools={}
    for i in sorted(active,key=lambda i:(-scores[i],i)):
        pools.setdefault(contexts[i],[]).append(i)
    import random
    keys=sorted(pools);shift=random.Random(rotation).randrange(len(keys))
    if len(keys)>=R:
        return [pools[keys[(j*len(keys)//R+shift)%len(keys)]][0] for j in range(R)]
    selected=[]
    while len(selected)<R:
        for j in range(len(keys)):
            pool=pools[keys[(j+shift)%len(keys)]]
            if pool:selected.append(pool.pop(0))
            if len(selected)==R:break
    return selected


def install():
    from ssd.engine.draft_runner import DraftRunner
    from ssd.engine.verifier import Verifier
    from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor
    from ssd.engine.helpers import p2_tree as pt
    from ssd.engine.helpers.batch_tree_forward import BatchedTreeForward
    from ssd.engine.helpers.batch_tree_common import build_forward_inputs,capacity
    from ssd.utils.context import set_context,reset_context
    import numpy as np

    select_original=pt._arena_select_global
    def select(ar,*args,**kwargs):
        if getattr(ar,'figure_arm','duet')=='duet':
            return select_original(ar,*args,**kwargs)
        return equal_select(ar,*args,**kwargs)
    pt._arena_select_global=select

    cache_original=DraftRunner._cache_response_topology
    @torch.inference_mode()
    def cache(self,views,phase):
        cache_original(self,views,phase)
        stride=int(os.environ.get('FIGURE_STRIDE','1'))
        if (int(self._request_step_id)-1)%stride:
            views['_fig']=dict(eligible=False,reason='not_sampled_generation')
            return
        ex=(next((e for e in self._p1_execs.values() if e.cell_logits is views['cell_logits']),None)
            if phase==1 else self._p2_exec)
        if ex is None:raise RuntimeError('Native executor identity missing')
        score=views['root_score'].float().cpu().tolist()
        ctx=views.get('root_context_ids')
        ctx=ctx.cpu().tolist() if ctx is not None else None
        rotation=int(os.environ['PAPER_SEED'])*1000003+int(self._figure_seq)*503+int(self._request_step_id)*101
        chosen=root_subset(score,ctx,rotation)
        stats=dict(seq=int(self._figure_seq),phase=phase,build_step=int(self._request_step_id),
                   native_roots=sum(s>0 for s in score),eligible=len(chosen)==R)
        if not chosen:
            views['_fig']=dict(eligible=False,reason='fewer_than_ten_roots')
            append('allocation.jsonl.gz',dict(stats,reason='fewer_than_ten_roots'));return
        gw=int(ex.in_glue_w);prefix=int(ex.in_prefix_len);start=prefix+gw
        physical=slots(self,ex.in_block_tables,start,K*W)
        if physical is None:
            views['_fig']=dict(eligible=False,reason='draft_scratch_capacity')
            append('allocation.jsonl.gz',dict(stats,eligible=False,reason='draft_scratch_capacity'));return
        index=torch.tensor(chosen,device=self.device,dtype=torch.long)
        before_cpu=torch.get_rng_state();before_gpu=torch.cuda.get_rng_state(self.device)
        saved=flat_kv(self).index_select(2,physical)
        variants={};q_tables={};root_tokens=ex.in_root_tok[index].cpu().tolist();root_q_tv={}
        try:
            for arm in ARMS:
                if not hasattr(self,'_figure_executors'):self._figure_executors={};self._figure_graph_pool=None
                key=(phase,arm)
                if key not in self._figure_executors:
                    cfg=copy(ex.cfg);cfg.duet_tree_c_tensor=1 if arm=='chain' else C
                    cfg.duet_tree_policy=cfg.duet_p1_tree_allocation_policy='dynamic'
                    shadow=P2TreeExecutor(ex.model,ex.compute_logits,cfg,ex.dev,ex.bs,
                        ex.max_blocks,ex.V,ex.H,ex.HKV,ex.D,dtype=ex.dtype,
                        phase=ex.phase,width=W,root_count=R,depth=K,max_nodes=G,
                        glue_width=max(self.config.duet_p1_tree_verify_nodes,self.config.duet_p2_tree_verify_nodes)+1,
                        materialize_backbone_logits=False)
                    shadow._workspace_bytes=16*1024*1024
                    shadow.arena.figure_arm=arm;shadow.debug_buffers_enabled=True
                    shadow.gen.manual_seed(int(os.environ['PAPER_SEED'])+370001*phase)
                    self._figure_executors[key]=shadow
                shadow=self._figure_executors[key]
                shadow.in_root_tok.copy_(ex.in_root_tok[index]);shadow.in_root_piv.copy_(ex.in_root_piv[index])
                shadow.in_rope_base.copy_(ex.in_rope_base[index]);shadow.in_glue.zero_()
                shadow.in_glue[:,:ex.gw_max].copy_(ex.in_glue[index])
                shadow.in_glue_w.copy_(ex.in_glue_w);shadow.in_prefix_len.copy_(ex.in_prefix_len)
                shadow.in_temps.fill_(float(ex.in_temps[0]));shadow.in_block_tables.copy_(ex.in_block_tables)
                final=start+K*W
                from ssd.utils.async_helpers.async_spec_helpers import compute_tree_canvas_page_plan
                plan=compute_tree_canvas_page_plan(start+W,final,ex.bs,ex.max_blocks,shadow.canvas_extra_pages)
                if plan is None:raise RuntimeError('Shadow page plan invalid after scratch preflight')
                p0,live,need=plan
                if p0 not in shadow.wrappers:shadow.prepare_bucket(p0)
                pages=shadow.in_page_ids[:need];pages[:live].copy_(ex.in_block_tables[0,:live])
                if live<need:pages[live:].copy_(pages[:1].expand(need-live))
                for f in range(K):
                    shadow.in_slot[f].copy_(physical[f*W:(f+1)*W].int())
                    shadow.in_ctx_len[f].fill_(start+(f+1)*W)
                    shadow.wrappers[p0][f]._paged_kv_indices_buf[:need].copy_(pages)
                if p0 not in shadow.graphs:
                    graph=shadow.capture(p0,graph_pool=self._figure_graph_pool)
                    if self._figure_graph_pool is None:self._figure_graph_pool=graph.pool()
                shadow.graphs[p0].replay()
                if not bool(shadow.dbg_selv.all()):raise RuntimeError('Shadow has padded forward inputs')
                if int(shadow.dbg_fan.sum())!=K*W*shadow.C:
                    raise RuntimeError('Generation cap or reservation unexpectedly bound')
                owners=shadow.arena.root[shadow.dbg_sel].cpu().reshape(-1).tolist()
                rows=[];counts=[owners.count(j) for j in range(R)]
                vals=shadow.out_valid.cpu().tolist()
                par=shadow.view_par.cpu().tolist();tok=shadow.view_tok.cpu().tolist()
                sib=shadow.view_sib.cpu().tolist();pcell=shadow.view_pcell.cpu().tolist()
                for j in range(R):
                    n=int(vals[j]);cells=[];refs=[]
                    for cell in pcell[j][:n]:
                        if cell not in cells:cells.append(cell)
                        refs.append(cells.index(cell))
                    if n!=counts[j]*shadow.C:raise RuntimeError('Forward/node accounting mismatch')
                    rows.append(dict(tokens=tok[j][:n],parents=par[j][:n],siblings=sib[j][:n],
                                     qrefs=refs,cells=cells,inputs=counts[j]))
                native_cells=views['parent_q_cells'][index,0].long()
                new_cells=torch.tensor([t['cells'][0] for t in rows],device=self.device)
                native_q=torch.softmax(views['cell_logits'][native_cells].float()/.7,-1)
                shadow_q=torch.softmax(shadow.cell_logits[new_cells].float()/.7,-1)
                root_q_tv[arm]=float((native_q-shadow_q).abs().sum(-1).max()/2)
                if root_q_tv[arm]>.03:
                    diag=Path(os.environ['SSD_PROFILE_DIR'])/'ROOT_PREFIX_DIAG.pt'
                    if not diag.exists():
                        from ssd.utils.async_helpers.async_spec_helpers import compute_tree_canvas_page_plan
                        plan0=compute_tree_canvas_page_plan(int(ex.in_ctx_len[0]),start+ex.total_cells,
                            ex.bs,ex.max_blocks,ex.canvas_extra_pages)
                        wr0=ex.wrappers[plan0[0]][0];wr1=shadow.wrappers[p0][0]
                        native_mask=np.unpackbits(wr0._custom_mask_buf.cpu().numpy(),bitorder='little').reshape(ex.round_widths[0],-1)
                        new_mask=np.unpackbits(wr1._custom_mask_buf.cpu().numpy(),bitorder='little').reshape(W,-1)
                        mask_equal=np.array_equal(native_mask[native_cells.cpu().numpy(),:start],new_mask[new_cells.cpu().numpy(),:start])
                        torch.save(dict(native=views['cell_logits'][native_cells].cpu(),
                        shadow=shadow.cell_logits[new_cells].cpu(),native_ids=ex.in_root_tok[index].cpu(),
                        shadow_ids=shadow.dbg_ids[0,new_cells].cpu(),native_rope=ex.in_rope_base[index].cpu(),
                        shadow_rope=shadow.dbg_rope[0,new_cells].cpu(),phase=phase,
                        native_cells=native_cells.cpu(),new_cells=new_cells.cpu(),
                        original_glue=ex.in_glue[index].cpu(),new_glue=shadow.in_glue.cpu(),
                        prefix_glue_masks_equal=mask_equal),diag)
                if not torch.equal(shadow.cell_logits,shadow.cell_logits.to(torch.bfloat16).float()):
                    raise RuntimeError('BF16 temporary q storage would change logits')
                variants[arm]=rows
                q_tables[arm]=shadow.cell_logits.to(torch.bfloat16).cpu()
        finally:
            flat_kv(self).index_copy_(2,physical,saved)
        diag=Path(os.environ['SSD_PROFILE_DIR'])/'ROOT_PREFIX_DIAG.pt'
        if diag.exists() and not getattr(self,'_figure_native_q_checked',False):
            data=torch.load(diag,weights_only=True)
            nc=data['native_cells'].to(self.device)
            # _sel points at the most recently CAPTURED page bucket, which
            # need not be the bucket most recently replayed. Reconstruct
            # current round-zero lane order from the current output q cells.
            width0=ex.round_widths[0]
            ids=torch.zeros(width0,dtype=torch.long,device=self.device)
            rope=ex.in_rope_base[0].expand(width0).clone()
            real=[j for j,s in enumerate(score) if s>0 and int(views['valid'][j])>0]
            ri=torch.tensor(real,device=self.device)
            lanes=views['parent_q_cells'][ri,0].long()
            ids[lanes]=ex.in_root_tok[ri];rope[lanes]=ex.in_rope_base[ri]
            from ssd.utils.async_helpers.async_spec_helpers import compute_tree_canvas_page_plan
            oldplan=compute_tree_canvas_page_plan(int(ex.in_ctx_len[0]),start+ex.total_cells,ex.bs,ex.max_blocks,ex.canvas_extra_pages)
            set_context(is_prefill=False,slot_mapping=ex.in_slot[0],context_lens=ex.in_ctx_len[0],
                        block_tables=ex.in_block_tables,active_mq_len=ex.round_widths[0],
                        active_wrappers={1:ex.wrappers[oldplan[0]][0]})
            try:ref=ex.compute_logits(ex.model(ids,rope),False).float()[nc]
            finally:reset_context();flat_kv(self).index_copy_(2,physical,saved)
            tv=(torch.softmax(ref/.7,-1)-torch.softmax(data['native'].to(self.device)/.7,-1)).abs().sum(-1).max()/2
            data['native_width_replay_max_tv']=float(tv)
            torch.save(data,diag);self._figure_native_q_checked=True
            if not data['prefix_glue_masks_equal'] or float(tv)>.001:
                raise RuntimeError(('Native-width prefix replay audit failed',data['prefix_glue_masks_equal'],float(tv)))
        if not torch.equal(before_cpu,torch.get_rng_state()) or not torch.equal(before_gpu,torch.cuda.get_rng_state(self.device)):
            raise RuntimeError('Shadow changed native global RNG')
        if not torch.equal(flat_kv(self).index_select(2,physical),saved):
            raise RuntimeError('Shadow did not restore native draft KV')
        metadata=dict(stats,roots=[dict(native_index=i,token=int(root_tokens[j]),score=score[i],
                    context=None if ctx is None else ctx[i],inputs=variants['duet'][j]['inputs'],
                    nodes=len(variants['duet'][j]['tokens'])) for j,i in enumerate(chosen)],
                    control_inputs={arm:[t['inputs'] for t in variants[arm]] for arm in ARMS},
                    physical_inputs=K*W,valid_inputs=K*W,generation_pruned=False,
                    root_q_max_tv=root_q_tv,sampling_stride=stride)
        views['_fig']=dict(eligible=True,chosen=chosen,variants=variants,q=q_tables,metadata=metadata)
        append('allocation.jsonl.gz',metadata)
    DraftRunner._cache_response_topology=cache

    respond_original=DraftRunner.hit_cache_and_respond
    @torch.inference_mode()
    def respond(self,request_keys,*args,**kwargs):
        self._figure_seq=int(request_keys[0,0])
        out=respond_original(self,request_keys,*args,**kwargs)
        seq,context,token=request_keys[0].cpu().tolist();step=int(self._request_step_id)
        payload=dict(seq=int(seq),step=step,hit=False,eligible=False)
        phase=getattr(self,'_tree_hit_phase',None);views=getattr(self,'_tree_hit_views',None)
        if phase in (1,2) and views is not None:
            root=int(self._tree_hit_root);payload.update(hit=True,phase=phase,native_root=root)
            data=views.get('_fig')
            if data is None:raise RuntimeError('Cache hit has no shadow provenance')
            if not data['eligible']:payload['reason']=data['reason']
            elif root not in data['chosen']:payload['reason']='hit_outside_frozen_pool'
            else:
                j=data['chosen'].index(root)
                trees={arm:data['variants'][arm][j] for arm in ARMS}
                q={arm:data['q'][arm].index_select(0,torch.tensor(trees[arm]['cells'],dtype=torch.long)) for arm in ARMS}
                shm=Path('/dev/shm')/('duet_v14_figures_'+os.environ['PAPER_ARM']);shm.mkdir(exist_ok=True)
                qfile=shm/f'{seq}_{step}.pt';torch.save(q,qfile)
                payload.update(eligible=True,root=j,root_token=int(token),trees=trees,qfile=str(qfile),
                    allocation=data['metadata'],union=merge(trees))
        inbox=Path(os.environ['SSD_PROFILE_DIR'])/'inbox';inbox.mkdir(exist_ok=True)
        dest=inbox/f'{seq}_{step}.json';tmp=dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload,separators=(',',':')));tmp.replace(dest)
        return out
    DraftRunner.hit_cache_and_respond=respond

    verify_original=Verifier.verify
    @torch.inference_mode()
    def verify(self,seqs,result,*args,**kwargs):
        seq=seqs[0];step=int(result.step_id);r=self.target_model_runner
        path=Path(os.environ['SSD_PROFILE_DIR'])/'inbox'/f'{seq.seq_id}_{step}.json'
        payload=json.loads(path.read_text());path.unlink()
        probe=None
        if payload['eligible']:
            if payload['root_token']!=int(result.speculations[0,0]):raise RuntimeError('Shadow hit token mismatch')
            qfile=Path(payload.pop('qfile'))
            tables=torch.load(qfile,weights_only=True);qfile.unlink()
            union=payload['union'];width=len(union['tokens'])+1
            prefix=seq.num_tokens-result.speculations.shape[1]
            physical=slots(r,seq.block_table,prefix,width)
            if physical is None:
                payload.update(eligible=False,reason='target_union_scratch_capacity')
            else:
                saved=flat_kv(r).index_select(2,physical)
                before=torch.cuda.get_rng_state(r.device);before_cpu=torch.get_rng_state()
                previous=r._duet_proxy_fn;r._duet_proxy_fn=None
                rows=[dict(tokens=[payload['root_token']]+union['tokens'],parents=union['parents'],
                           prefix=prefix,blocks=list(seq.block_table))]
                try:
                    # Forward-only shadow graphs share a private graph pool;
                    # online verification stays on its original B1 route.
                    if not hasattr(r,'_figure_target'):r._figure_target=BatchedTreeForward(r,split=True)
                    forward=r._figure_target
                    pages=min(capacity((prefix+width+r.block_size-1)//r.block_size),r.config.max_blocks)
                    bucket=capacity(width)
                    key=(1,bucket,pages)
                    g=forward.capture(*key)
                    packed=build_forward_inputs(rows,bucket,1,pages,r.block_size)
                    for name,array in packed.items():g[name].copy_(torch.from_numpy(array),non_blocking=True)
                    forward._context(g)
                    g['pre'].replay();g['post'].replay()
                    logits=r.model.compute_logits(g['output'][:width],False).float()
                    outcomes={}
                    for arm in ARMS:
                        mapping=union['mapping'][arm]
                        ix=torch.tensor([0]+[x+1 for x in mapping],device=r.device)
                        outcomes[arm]=expected(payload['trees'][arm],logits.index_select(0,ix),
                            tables[arm].to(r.device),seq.temperature,self.sampler_x,self.async_fan_out,
                            int(r.hf_config.eos_token_id),payload['root_token'])
                        if payload['phase'] not in getattr(r,'_figure_mc_phases',set()):
                            outcomes[arm]['native_mc']=check_native(payload['trees'][arm],logits.index_select(0,ix),
                                tables[arm],outcomes[arm],seq.temperature,self.sampler_x,self.async_fan_out,
                                10837+payload['phase'])
                    if not hasattr(r,'_figure_mc_phases'):r._figure_mc_phases=set()
                    r._figure_mc_phases.add(payload['phase'])
                    probe=dict(seq=int(seq.seq_id),step=step,phase=payload['phase'],root=payload['root'],
                        allocation=payload['allocation'],prefix=prefix,outcomes=outcomes)
                finally:
                    reset_context();r._duet_proxy_fn=previous
                    flat_kv(r).index_copy_(2,physical,saved)
                if not torch.equal(before,torch.cuda.get_rng_state(r.device)) or not torch.equal(before_cpu,torch.get_rng_state()):
                    raise RuntimeError('Target shadow changed native RNG')
                if not torch.equal(flat_kv(r).index_select(2,physical),saved):raise RuntimeError('Target shadow KV restore failed')
        out=verify_original(self,seqs,result,*args,**kwargs)
        event=self.metrics['phase_events'][-1]
        if payload['hit'] and payload.get('phase')!=event['source']:
            raise RuntimeError('Shadow hit/source join mismatch')
        append('choices.jsonl.gz',dict(seq=int(seq.seq_id),step=step,source=event['source'],
               eligible=payload['eligible'],reason=payload.get('reason'),native_al=event['accepted_len']))
        if probe is not None:
            probe['native_al']=event['accepted_len'];append('paired.jsonl.gz',probe)
        return out
    Verifier.verify=verify
