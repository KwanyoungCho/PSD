"""Batch independent dynamic-tree arenas into one draft forward per round.

This primitive preserves each request's top-W budget and node namespace.
Serving must still bind each arena to its own cache, wire, and accepted KV
path. It is deliberately not advertised as end-to-end B>1 tree support.
"""
import torch
import flashinfer
from ssd.utils.context import set_context, reset_context


class BatchedTreeExecutor:
    def __init__(self, executors):
        if not executors:
            raise ValueError("empty tree batch")
        self.executors = tuple(executors)
        self.first = executors[0]
        a = self.first
        for ex in executors:
            if (ex.round_widths, ex.F, ex.V, ex.bs, ex.H, ex.HKV, ex.D,
                ex.dtype, ex.dev) != (
                    a.round_widths, a.F, a.V, a.bs, a.H, a.HKV, a.D,
                    a.dtype, a.dev):
                raise ValueError("batched arenas require matching model and round shapes")
            if ex.model is not a.model:
                raise ValueError("batched arenas must share the same model instance")
        self.wrappers = {}
        self.graphs = {}
        self.workspaces = {}

    def prepare(self, page_bucket):
        if page_bucket in self.wrappers:
            return
        a, B = self.first, len(self.executors)
        for ex in self.executors:
            if page_bucket not in ex.wrappers:
                ex.prepare_bucket(page_bucket)
            if page_bucket not in ex._local_idx_by_bucket:
                ex._local_idx_by_bucket[page_bucket] = torch.full_like(ex._local_idx, -1)
            ex._local_idx = ex._local_idx_by_bucket[page_bucket]
        workspace = torch.empty(a._workspace_bytes, dtype=torch.uint8, device=a.dev)
        self.workspaces[page_bucket] = workspace
        pages = page_bucket + a.canvas_extra_pages
        canvas = pages * a.bs
        wrappers, by_width = [], {}
        for W in a.round_widths:
            qo = torch.arange(B + 1, device=a.dev, dtype=torch.int32) * W
            kv = torch.arange(B + 1, device=a.dev, dtype=torch.int32) * pages
            ids = torch.zeros(B * pages, device=a.dev, dtype=torch.int32)
            last = torch.full((B,), a.bs, device=a.dev, dtype=torch.int32)
            # Each sequence starts on a byte boundary (page size is a
            # multiple of eight), so independent packed masks concatenate.
            if (W * canvas) % 8:
                raise ValueError("tree mask rows must end on a byte boundary")
            mask = torch.zeros(B * W * canvas // 8, device=a.dev, dtype=torch.uint8)
            mi = torch.arange(B + 1, device=a.dev, dtype=torch.int32) * W * canvas
            wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, "NHD", backend="fa2", use_cuda_graph=True,
                qo_indptr_buf=qo, paged_kv_indptr_buf=kv,
                paged_kv_indices_buf=ids, paged_kv_last_page_len_buf=last,
                custom_mask_buf=mask, mask_indptr_buf=mi)
            if W in by_width:
                old = by_width[W]
                wr._int_workspace_buffer = old._int_workspace_buffer
                wr._pin_memory_int_workspace_buffer = old._pin_memory_int_workspace_buffer
            else:
                by_width[W] = wr
            wr.plan(qo, kv, ids, last, a.H, a.HKV, a.D, a.bs,
                    custom_mask=torch.zeros(B * W * canvas,
                                            device=a.dev, dtype=torch.bool),
                    q_data_type=a.dtype, kv_data_type=a.dtype)
            wrappers.append(wr)
        self.wrappers[page_bucket] = wrappers

    @torch.inference_mode()
    def run_once(self, page_bucket, finalize=True):
        a, B = self.first, len(self.executors)
        if page_bucket not in self.wrappers:
            self.prepare(page_bucket)
        for ex in self.executors:
            ex._local_idx = ex._local_idx_by_bucket[page_bucket]
        streams = [ex.iter_rounds(page_bucket, finalize=False) for ex in self.executors]
        requests = [next(stream) for stream in streams]
        for f, W in enumerate(a.round_widths):
            wr = self.wrappers[page_bucket][f]
            # These copies remain inside the graph: page IDs and masks vary
            # on every replay even when the capacity bucket is unchanged.
            torch.cat([ex.wrappers[page_bucket][f]._paged_kv_indices_buf
                       for ex in self.executors], out=wr._paged_kv_indices_buf)
            mask = torch.cat([ex.wrappers[page_bucket][f]._custom_mask_buf
                              for ex in self.executors])
            wr._custom_mask_buf[:mask.numel()].copy_(mask)
            ids = torch.cat([r[0] for r in requests])
            rope = torch.cat([r[1] for r in requests])
            slot = torch.cat([r[2]["slot_mapping"] for r in requests])
            lens = torch.cat([r[2]["context_lens"] for r in requests])
            bt = torch.cat([r[2]["block_tables"] for r in requests])
            set_context(is_prefill=False, slot_mapping=slot,
                        context_lens=lens, block_tables=bt,
                        active_mq_len=W, active_wrappers={B: wr})
            try:
                hidden = a.model(ids, rope)
                logits = a.compute_logits(hidden, False).reshape(B, W, a.V).float()
            finally:
                reset_context()
            new_requests = []
            for b, stream in enumerate(streams):
                try:
                    new_requests.append(stream.send(logits[b]))
                except StopIteration:
                    if f != a.F - 1:
                        raise RuntimeError("tree arena terminated before the final round")
            requests = new_requests
        if finalize:
            for ex in self.executors:
                ex._finalize_outputs()

    @torch.inference_mode()
    def capture(self, page_bucket, graph_pool=None):
        self.prepare(page_bucket)
        for _ in range(2):
            self.run_once(page_bucket, finalize=False)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        for ex in self.executors:
            g.register_generator_state(ex.gen)
        with torch.cuda.graph(g, pool=graph_pool):
            self.run_once(page_bucket, finalize=False)
        self.graphs[page_bucket] = g

    @torch.inference_mode()
    def replay(self, page_bucket):
        self.graphs[page_bucket].replay()
        for ex in self.executors:
            ex._finalize_outputs()
