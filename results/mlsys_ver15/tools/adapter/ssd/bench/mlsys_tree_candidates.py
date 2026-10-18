"""Review-only packed-tree execution; not a production default.

Each query is represented as a one-query attention sequence. All queries
still see their own request's prefix and ancestors. This keeps the attention
plan fixed when the distribution of real nodes among requests changes.
"""
import os
import numpy as np
import torch
from ssd.engine.helpers.batch_tree_forward import BatchedTreeForward
from ssd.engine.helpers.batch_tree_common import capacity
from ssd.engine.helpers.packed_tree_forward import layout
from ssd.engine.helpers.cudagraph_helpers import duet_record, duet_close
from ssd.utils.context import reset_context


def query_layout(rows, width, query_capacity, pages, block_size):
    packed = layout(rows, width, capacity(len(rows)), query_capacity, pages, block_size)
    request_pages = packed['pages'].reshape(-1, pages)
    # One fixed query per wrapper sequence, including zero-write padding.
    packed['pages'] = np.repeat(request_pages, np.diff(packed['qo']), axis=0).reshape(-1)
    return packed


class FixedPlanPackedTree(BatchedTreeForward):
    @torch.inference_mode()
    def run(self, rows, width, proxy=None):
        r = self.r
        b = len(rows)
        pages = min(capacity(max((x['prefix'] + len(x['tokens']) + r.block_size - 1)
                                 // r.block_size for x in rows)), r.config.max_blocks)
        q = ((sum(len(x['tokens']) for x in rows) + 7) // 8) * 8
        if q >= capacity(b) * width:
            # Packing cannot remove work for this shape. Retain the original
            # grouped attention instead of repeating its page table per query.
            if not hasattr(self, '_dense_fallback'):
                self._dense_fallback = BatchedTreeForward(r, self.split)
            return self._dense_fallback.run(rows, width, proxy)
        # Cached by total query bucket and page bucket, not per-request tree shape.
        g = self.capture(q, 1, pages)
        packed = query_layout(rows, width, q, pages, r.block_size)
        for name in ('ids', 'rope', 'slots', 'pages', 'mask'):
            g[name].copy_(torch.from_numpy(packed[name]), non_blocking=True)
        dense = torch.from_numpy(packed['dense']).to(r.device)
        side = None
        if proxy is not None and os.getenv('SSD_BATCH_TREE_PROXY_STREAM', '0') == '1':
            from ssd.engine.helpers.tree_proxy_stream import TreeProxySideStream
            if self.proxy_side is None:
                self.proxy_side = TreeProxySideStream(r.device)
            side = self.proxy_side
        self._context(g)
        try:
            if self.split:
                event = duet_record('batch_target_pre')
                g['pre'].replay()
                duet_close('batch_target_pre', event)
                event = duet_record('batch_target_proxy')
                normed = r.model.model.norm(g['hidden'] + g['residual'], None)
                def project(hidden, batch):
                    if r.config.duet_exit_replica:
                        logits = torch.nn.functional.linear(hidden, r._duet_lm_head_replica)
                    else:
                        logits = r.model.compute_logits(hidden, False)
                    if proxy is not None and logits is not None:
                        proxy(logits.index_select(0, dense).view(batch, width, -1), batch)
                if side is not None and (r.num_tp_gpus == 1 or r.config.duet_exit_replica):
                    dense.record_stream(side.stream)
                    side.launch(normed, b, project)
                else:
                    project(normed, b)
                duet_close('batch_target_proxy', event)
                event = duet_record('batch_target_post')
                g['post'].replay()
                duet_close('batch_target_post', event)
            else:
                g['full'].replay()
            event = duet_record('batch_target_final_logits' if self.split else 'batch_glue_logits')
            logits = r.model.compute_logits(g['output'], False)
            duet_close('batch_target_final_logits' if self.split else 'batch_glue_logits', event)
            return None if logits is None else logits.index_select(0, dense).view(b, width, -1)
        finally:
            if side is not None:
                side.finish()
            reset_context()


def install_packed_review():
    mode = os.getenv('DUET_REVIEW_PACKED', '')
    if not mode:
        return
    if mode not in ('replan', 'fixed'):
        raise ValueError('Unknown packed review arm')
    from ssd.engine.model_runner import ModelRunner
    from ssd.engine.helpers.packed_tree_forward import PackedTreeForward
    implementation = FixedPlanPackedTree if mode == 'fixed' else PackedTreeForward
    def run_batched_tree(self, rows, width):
        if not hasattr(self, '_review_packed_forward'):
            self._review_packed_forward = implementation(self, split=True)
        actual=sum(len(row['tokens']) for row in rows)
        self._review_physical_queries=min(((actual+7)//8)*8,capacity(len(rows))*width)
        return self._review_packed_forward.run(rows, width, self._duet_proxy_fn)
    ModelRunner.run_batched_tree = torch.inference_mode()(run_batched_tree)


def install_mixed_miss_review():
    """Review the exact production service with only miss handling changed.

    Keep this policy out of production until AL/TPS and law checks are reviewed.
    The compiled function is captured in the benchmark's adapter source hashes.
    """
    if os.getenv('DUET_REVIEW_MIXED_MISS_AR', '0') != '1':
        return
    import inspect
    import textwrap
    import ssd.engine.helpers.batch_tree_draft as module
    source = textwrap.dedent(inspect.getsource(module.BatchedDuetDraft.serve))
    source = source.replace('    miss_k=miss_depth*miss_width',
        '''    skip_miss = B > 1 and 0 < len(miss) < B and len(miss) <= int(os.getenv("DUET_REVIEW_MISS_MAX", str(B-1)))
    miss_k = 0 if skip_miss else miss_depth*miss_width''')
    # Empty proposals use the existing mixed tree/chain/zero-node verifier.
    source = source.replace('    if miss:\n', '    if miss and not skip_miss:\n', 1)
    source = source.replace('1<=miss_k<=cfg.duet_response_token_width',
                            '(0 if skip_miss else 1)<=miss_k<=cfg.duet_response_token_width')
    source = source.replace('len(miss),cfg.duet_response_token_width', 'len(miss),miss_k')
    assert 'skip_miss =' in source and 'if miss and not skip_miss:' in source
    scope = {}
    exec(compile(source, '<review-tree-mixed-miss>', 'exec'), module.__dict__, scope)
    module.BatchedDuetDraft.serve = scope['serve']
