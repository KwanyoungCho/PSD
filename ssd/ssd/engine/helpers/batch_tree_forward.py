"""Captured ancestor attention shared by batched target verify and draft glue."""
import os
import torch
import flashinfer
from ssd.utils.context import set_context, reset_context
from ssd.engine.helpers.cudagraph_helpers import duet_record, duet_close
from ssd.engine.helpers.batch_tree_common import capacity, build_forward_inputs


class BatchedTreeForward:
    def __init__(self, runner, split):
        self.r = runner
        self.split = split
        self.graphs = {}
        self.pool = None
        self.proxy_side = None

    def workspace(self):
        # All ancestor forwards and forest rounds run on this runner's
        # compute stream. Float scratch contains no persistent plan state.
        r=self.r
        if not hasattr(r,'_batch_tree_float_workspace'):
            r._batch_tree_float_workspace=torch.empty(
                int(os.getenv('SSD_BATCH_TREE_WORKSPACE_MB','128'))*2**20,
                dtype=torch.uint8,device=r.device)
        return r._batch_tree_float_workspace

    def _context(self, g):
        kwargs = dict(is_prefill=False, slot_mapping=g['slots'],
                      context_lens=g['lens'], block_tables=g['blocks'])
        if self.split:
            kwargs['tree_verify_wrapper'] = g['wrapper']
        else:
            kwargs.update(active_mq_len=g['width'],
                          active_wrappers={g['batch']:g['wrapper']})
        set_context(**kwargs)

    @torch.inference_mode()
    def capture(self, batch, width, pages):
        key = batch,width,pages
        if key in self.graphs:
            return self.graphs[key]
        r = self.r
        dev,dtype,hf = r.device,r.hf_config.torch_dtype,r.hf_config
        cols = pages*r.block_size
        ws = self.workspace()
        qo = torch.arange(batch+1,dtype=torch.int32,device=dev)*width
        kv = torch.arange(batch+1,dtype=torch.int32,device=dev)*pages
        indices = torch.zeros(batch*pages,dtype=torch.int32,device=dev)
        last = torch.full((batch,),r.block_size,dtype=torch.int32,device=dev)
        mask = torch.empty(batch*width*cols//8,dtype=torch.uint8,device=dev)
        wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws,'NHD',backend='fa2',use_cuda_graph=True,
            qo_indptr_buf=qo,paged_kv_indptr_buf=kv,
            paged_kv_indices_buf=indices,paged_kv_last_page_len_buf=last,
            custom_mask_buf=mask,
            mask_indptr_buf=torch.arange(batch+1,dtype=torch.int32,device=dev)*width*cols)
        tp = max(1,r.num_tp_gpus)
        wr.plan(qo,kv,indices,last,hf.num_attention_heads//tp,
                hf.num_key_value_heads//tp,hf.head_dim,r.block_size,
                custom_mask=torch.ones(batch*width*cols,dtype=torch.bool,device=dev),
                q_data_type=dtype,kv_data_type=dtype)
        g = dict(batch=batch,width=width,pages_count=pages,wrapper=wr,workspace=ws,
                 ids=torch.zeros(batch*width,dtype=torch.int64,device=dev),
                 rope=torch.zeros(batch*width,dtype=torch.int64,device=dev),
                 slots=torch.full((batch*width,),-1,dtype=torch.int32,device=dev),
                 lens=torch.full((batch,),cols,dtype=torch.int32,device=dev),
                 blocks=torch.zeros(batch,r.config.max_blocks,dtype=torch.int32,device=dev),
                 pages=wr._paged_kv_indices_buf,mask=wr._custom_mask_buf,
                 output=torch.zeros(batch*width,hf.hidden_size,dtype=dtype,device=dev))
        self._context(g)
        if self.split:
            g['hidden'] = torch.zeros_like(g['output'])
            g['residual'] = torch.zeros_like(g['output'])
            def pre():
                h,res = r.model(g['ids'],g['rope'],end_layer=r.config.duet_exit_layer+1)
                g['hidden'].copy_(h)
                g['residual'].copy_(res)
            def post():
                out = r.model(g['ids'],g['rope'],start_layer=r.config.duet_exit_layer+1,
                              init_hidden_states=g['hidden'],init_residual=g['residual'])
                g['output'].copy_(out[0] if isinstance(out,tuple) else out)
            pre(); post()
            torch.cuda.synchronize(dev)
            g['pre'] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g['pre'],pool=self.pool): pre()
            if self.pool is None: self.pool = g['pre'].pool()
            g['post'] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g['post'],pool=self.pool): post()
        else:
            def full():
                g['output'].copy_(r.model(g['ids'],g['rope']))
            full()
            torch.cuda.synchronize(dev)
            g['full'] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g['full'],pool=self.pool): full()
            if self.pool is None: self.pool = g['full'].pool()
        reset_context()
        self.graphs[key] = g
        return g

    @torch.inference_mode()
    def run(self, rows, width, proxy=None):
        r = self.r
        b = len(rows)
        if proxy is not None and os.getenv('SSD_BATCH_TREE_PROXY_STREAM','0')=='1':
            if self.proxy_side is None:
                from ssd.engine.helpers.tree_proxy_stream import TreeProxySideStream
                self.proxy_side=TreeProxySideStream(r.device)
            callback=proxy
            proxy=lambda logits,batch:self.proxy_side.launch(logits,batch,callback)
        pages = capacity(max((row['prefix']+len(row['tokens'])+r.block_size-1)//r.block_size
                             for row in rows))
        pages = min(pages,r.config.max_blocks)
        g = self.capture(capacity(b),width,pages)
        packed = build_forward_inputs(rows,width,g['batch'],pages,r.block_size)
        for name, array in packed.items():
            g[name].copy_(torch.from_numpy(array),non_blocking=True)
        self._context(g)
        try:
            if self.split:
                _pre=duet_record('batch_target_pre')
                g['pre'].replay()
                duet_close('batch_target_pre',_pre)
                _proxy=duet_record('batch_target_proxy')
                hs = g['hidden'][:b*width]
                res = g['residual'][:b*width]
                if r.config.duet_exit_replica:
                    replica = getattr(r,'_duet_lm_head_replica',None)
                    if proxy is not None and replica is not None:
                        normed = r.model.model.norm(hs+res,None)
                        proxy(torch.nn.functional.linear(normed,replica).view(b,width,-1),b)
                else:
                    normed = r.model.model.norm(hs+res,None)
                    exit_logits = r.model.compute_logits(normed,False)
                    if proxy is not None:
                        proxy(exit_logits.view(b,width,-1),b)
                duet_close('batch_target_proxy',_proxy)
                _post=duet_record('batch_target_post')
                g['post'].replay()
                duet_close('batch_target_post',_post)
            else:
                g['full'].replay()
            _logits=duet_record('batch_target_final_logits' if self.split else 'batch_glue_logits')
            logits = r.model.compute_logits(g['output'][:b*width],False)
            duet_close('batch_target_final_logits' if self.split else 'batch_glue_logits',_logits)
            return None if logits is None else logits.reshape(b,width,-1)
        finally:
            if self.proxy_side is not None:self.proxy_side.finish()
            reset_context()
