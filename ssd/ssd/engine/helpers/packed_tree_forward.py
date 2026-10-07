"""Ragged ancestor attention with a separate sequence for query alignment.

FlashInfer graph mode fixes total query capacity and batch capacity. The
per-request query boundaries may change after re-planning into persistent
integer buffers. Assert the captured plan's structural fields stay fixed.
"""
import numpy as np
import torch
import flashinfer
from ssd.utils.context import set_context,reset_context
from ssd.engine.helpers.batch_tree_common import capacity
from ssd.engine.helpers.batch_tree_forward import BatchedTreeForward
from ssd.engine.helpers.tree_host_topology import context_topology


def layout(rows,width,batch_cap,query_cap,pages,block_size):
    cols=pages*block_size;batch=batch_cap+1
    ids=np.zeros(query_cap,dtype=np.int64);rope=ids.copy()
    slots=np.full(query_cap,-1,dtype=np.int32)
    indices=np.zeros((batch,pages),dtype=np.int32)
    mask=np.zeros((query_cap,cols),dtype=np.uint8)
    qo=[0];dense=[]
    for b,row in enumerate(rows):
        p=row['prefix'];tokens=row['tokens'];n=len(tokens);off=qo[-1]
        depth,_,visible=context_topology(row['parents'])
        if n>width or n!=len(row['parents'])+1 or off+n>query_cap or p+n>cols:
            raise ValueError('Packed tree capacity exceeded')
        bt=row['blocks'];live=(p+n+block_size-1)//block_size
        if len(bt)<live or any(k<0 for k in bt[:live]):raise ValueError('Unallocated tree page')
        indices[b,:live]=bt[:live];indices[b,live:]=bt[0]
        ids[off:off+n]=tokens;rope[off:off+n]=p+np.asarray([0]+[d+1 for d in depth])
        pos=p+np.arange(n);slots[off:off+n]=np.asarray(bt)[pos//block_size]*block_size+pos%block_size
        mask[off:off+n,:p]=1;mask[off:off+n,p:p+n]=visible
        dense.extend([off+j if j<n else off for j in range(width)])
        qo.append(off+n)
    indices[len(rows):]=indices[0]
    # No alignment query is appended to a real sequence: it owns a separate
    # finite prefix and never writes KV. Empty batch lanes have zero queries.
    mask[qo[-1]:,:max(1,rows[0]['prefix'])]=1
    qo.extend([qo[-1]]*(batch_cap-len(rows)))
    qo.append(query_cap)
    return dict(ids=ids,rope=rope,slots=slots,pages=indices.reshape(-1),
                mask=np.packbits(mask.reshape(-1),bitorder='little'),
                qo=np.asarray(qo,dtype=np.int32),dense=np.asarray(dense,dtype=np.int64))


class PackedTreeForward(BatchedTreeForward):
    def _context(self,g):
        kw=dict(is_prefill=False,slot_mapping=g['slots'],context_lens=g['lens'],block_tables=g['blocks'])
        if self.split:kw['tree_verify_wrapper']=g['wrapper']
        else:kw.update(active_mq_len=1,active_wrappers={g['query_cap']:g['wrapper']})
        set_context(**kw)

    def _plan(self,g,qo):
        r=self.r;hf=r.hf_config;tp=max(1,r.num_tp_gpus);dtype=hf.torch_dtype
        qp=torch.as_tensor(qo,dtype=torch.int32,device='cpu')
        g['wrapper'].plan(qp,g['kv_host'],g['pages'],g['last_host'],
            hf.num_attention_heads//tp,hf.num_key_value_heads//tp,hf.head_dim,r.block_size,
            packed_custom_mask=g['mask'],q_data_type=dtype,kv_data_type=dtype)
        # This FlashInfer version computes bit offsets for pre-packed input;
        # its kernel consumes byte offsets, as segment_packbits would return.
        g['wrapper']._mask_indptr_buf.copy_(qp*(g['cols']//8))
        info=tuple(int(x) for x in g['wrapper']._plan_info)
        if 'plan_info' in g and g['plan_info']!=info:
            raise RuntimeError('Packed FlashInfer plan changed captured structural fields')
        g['plan_info']=info

    @torch.inference_mode()
    def capture_packed(self,batch_cap,query_cap,pages):
        key=batch_cap,query_cap,pages
        if key in self.graphs:return self.graphs[key]
        r=self.r;dev=r.device;hf=r.hf_config;dt=hf.torch_dtype;batch=batch_cap+1;cols=pages*r.block_size
        qo=torch.zeros(batch+1,dtype=torch.int32,device=dev);qo[-1]=query_cap
        kv=torch.arange(batch+1,dtype=torch.int32,device=dev)*pages
        page_ids=torch.zeros(batch*pages,dtype=torch.int32,device=dev)
        last=torch.full((batch,),r.block_size,dtype=torch.int32,device=dev)
        mask=torch.full((query_cap*cols//8,),255,dtype=torch.uint8,device=dev)
        wr=flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace(),'NHD',backend='fa2',use_cuda_graph=True,
            qo_indptr_buf=qo,paged_kv_indptr_buf=kv,paged_kv_indices_buf=page_ids,
            paged_kv_last_page_len_buf=last,custom_mask_buf=mask,mask_indptr_buf=qo.clone())
        g=dict(batch=batch,query_cap=query_cap,cols=cols,wrapper=wr,kv_host=kv.cpu(),last_host=last.cpu(),
            ids=torch.zeros(query_cap,dtype=torch.int64,device=dev),rope=torch.zeros(query_cap,dtype=torch.int64,device=dev),
            slots=torch.full((query_cap,),-1,dtype=torch.int32,device=dev),
            lens=torch.full((batch,),cols,dtype=torch.int32,device=dev),
            blocks=torch.zeros(batch,r.config.max_blocks,dtype=torch.int32,device=dev),
            pages=page_ids,mask=mask,output=torch.zeros(query_cap,hf.hidden_size,dtype=dt,device=dev))
        self._plan(g,qo.cpu());self._context(g)
        if self.split:
            g['hidden']=torch.zeros_like(g['output']);g['residual']=torch.zeros_like(g['output'])
            def pre():
                h,res=r.model(g['ids'],g['rope'],end_layer=r.config.duet_exit_layer+1)
                g['hidden'].copy_(h);g['residual'].copy_(res)
            def post():
                x=r.model(g['ids'],g['rope'],start_layer=r.config.duet_exit_layer+1,
                    init_hidden_states=g['hidden'],init_residual=g['residual'])
                g['output'].copy_(x[0] if isinstance(x,tuple) else x)
            pre();post();torch.cuda.synchronize(dev)
            g['pre']=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g['pre'],pool=self.pool):pre()
            if self.pool is None:self.pool=g['pre'].pool()
            g['post']=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g['post'],pool=self.pool):post()
        else:
            def full():g['output'].copy_(r.model(g['ids'],g['rope']))
            full();torch.cuda.synchronize(dev);g['full']=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g['full'],pool=self.pool):full()
            if self.pool is None:self.pool=g['full'].pool()
        reset_context();self.graphs[key]=g
        return g

    @torch.inference_mode()
    def run(self,rows,width,proxy=None):
        r=self.r;b=len(rows);bc=capacity(b)
        pages=min(capacity(max((x['prefix']+len(x['tokens'])+r.block_size-1)//r.block_size for x in rows)),r.config.max_blocks)
        total=sum(len(x['tokens']) for x in rows)
        qc=((total+7)//8)*8
        g=self.capture_packed(bc,qc,pages)
        packed=layout(rows,width,bc,qc,pages,r.block_size)
        for name in ['ids','rope','slots','pages','mask']:g[name].copy_(torch.from_numpy(packed[name]),non_blocking=True)
        self._plan(g,packed['qo'])
        dense=torch.from_numpy(packed['dense']).to(r.device)
        self._context(g)
        try:
            if self.split:
                g['pre'].replay()
                if r.config.duet_exit_replica:
                    replica=getattr(r,'_duet_lm_head_replica',None)
                    if proxy is not None and replica is not None:
                        hs=g['hidden'].index_select(0,dense);res=g['residual'].index_select(0,dense)
                        proxy(torch.nn.functional.linear(r.model.model.norm(hs+res,None),replica).view(b,width,-1),b)
                else:
                    normed=r.model.model.norm(g['hidden']+g['residual'],None)
                    exit_logits=r.model.compute_logits(normed,False)
                    if proxy is not None:proxy(exit_logits.index_select(0,dense).view(b,width,-1),b)
                g['post'].replay()
            else:g['full'].replay()
            logits=r.model.compute_logits(g['output'],False)
            return None if logits is None else logits.index_select(0,dense).view(b,width,-1)
        finally:reset_context()
